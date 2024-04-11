#!/usr/bin/env python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2020-2023, Universitätsklinikum Erlangen"


import monai
import os


import pytorch_lightning as pl
from pytorch_lightning.metrics.utils import to_categorical
import pandas as pd

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from skimage import color
from skimage.io import imsave

import numpy as np

from src.image_masker import thorax_masker
from src.sitk_normalizer import sitk_normalizer

class BackboneModel(pl.LightningModule):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, inference_mode=True, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # is multiple GPU?
        #self.multiple_gpus = True if self.hparams.n_gpus > 1 else False

        # len dataloader
        self.len_traindl = None

        # created inference dataframe
        self.inference_results = pd.DataFrame(
            columns=[
                "id",
                "y_pred_prob_positive",
                "y_pred_class"
            ]
        )

        if "fold_number" in kwargs:
            self.fold_number = kwargs["fold_number"]

        # load model architecture:
        self.model = monai.networks.nets.densenet.densenet121(
            pretrained=self.hparams.pretrained,
            spatial_dims=2,
            in_channels=1,
            out_channels=self.hparams.num_classes,
            bn_size=self.hparams.bn_size,
            dropout_prob=self.hparams.dropout_prob
        )


        # instantiate monai transforms to log misclassifies test images
        self.img_resize = monai.transforms.Compose([
            monai.transforms.Resize(spatial_size=[256, 256])
        ])
        self.img_rescale = monai.transforms.Compose([
            monai.transforms.ScaleIntensity(minv=0, maxv=1)
        ])

        # transformation to save jpeg files
        self.img_jpeg_prep = monai.transforms.Compose([
            monai.transforms.ScaleIntensity(minv=0, maxv=255),
            monai.transforms.CastToType(dtype=np.uint8)
        ])

        # set test_img num
        self.test_img_num = 0

        # set test iteration
        self.test_iteration = 0

        # instantiate class activation map function
        self.cam_models = [
            "densenet121"
        ]

        target_layer = "class_layers.relu"

        self.gradcampp = monai.visualize.GradCAMpp(
            nn_module=self.model,
            target_layers=target_layer
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "test")

    def _shared_step(self, batch, batch_idx, prefix):
        #x = batch["image"]
        x = batch["image_ellipsis_mask"] # image_squared_mask

        logits = self(x)

        bs = torch.tensor([len(x)], dtype=torch.int16).type_as(x)

        # prepare targets
        y_preds = self._y_to_cat(
            logits=logits,
            targets=1
        )

        # append inference data
        append_row = pd.DataFrame(
            data={
                "id": batch["id"],
                "y_pred_prob_positive": list(
                    torch.softmax(logits, dim=1).cpu(
                    ).detach().numpy()[:, 1]
                ),
                "y_pred_class": list(
                    y_preds.cpu().detach().numpy()
                ),
                "fold": self.fold_number
            }
        )
        self.inference_results = self.inference_results.append(
            other=append_row,
            ignore_index=True
        )

        self._log_images(
            data=x,
            y_preds=y_preds,
            ids=batch["id"]
        )

        return {"logits": logits, "batch_size": bs}

    def _shared_epoch_end(self, outputs, prefix):
        self.test_iteration += 1

    def test_end(self, outputs):
        # save inference results as csv
        self.inference_results.to_csv(
            path_or_buf=os.path.join(
                self.trainer.default_root_dir,
                "test_images",
                self.fold_number,
                "inference_results.csv"
            ),
            index=False
        )

        return self.inference_results


    def _log_images(self, data, y_preds, ids):

        # create output dir
        test_img_basedir = os.path.join(
            self.trainer.default_root_dir,
            "test_images",
            self.fold_number
        )

        if not os.path.exists(test_img_basedir):
            os.makedirs(test_img_basedir)

        y_preds = y_preds.cpu().detach().numpy()

        for raw_image, _id in zip(data, ids):

            cm_images = {}

            if self.hparams.model_name in self.cam_models:
                with torch.set_grad_enabled(True):
                    # with class_idx
                    for _cls in range(self.hparams.num_classes):
                        cm_images[_cls] = self.gradcampp(
                            x=raw_image[None],
                            class_idx=_cls
                        )

            # dict of images to plot later
            img2plot = {}

            # get colormap
            cm = plt.get_cmap("jet_r")

            # iterate over images that require color map
            for _img_key, _img_val in cm_images.items():
                # build filename to save numpy
                _fname = os.path.join(
                    test_img_basedir,
                    _id + "_cam_" +
                    str(_img_key) + ".npy"
                )
                np.save(
                    file=_fname,
                    arr=_img_val
                )

                # apply colormap
                _img = cm(np.squeeze(_img_val))
                # create 3-channel image for displaying in tensorboard
                img2plot[_img_key] = np.transpose(
                    color.rgba2rgb(_img),
                    (2, 0, 1)
                )

            # process original image
            # add axis, required for transformations (c, h, w)
            img = np.squeeze(raw_image.cpu().detach().numpy())[None]
            # rescale original image
            img = self.img_rescale(img)  # intensity=0-1, c, h, w
            # create 3-channel grayscale image for displaying in tensorboard
            img2plot["original_image"] = np.transpose(
                color.gray2rgb(np.squeeze(img)),
                (2, 0, 1)
            )

            img_keys = [i for i in range(self.hparams.num_classes)] + \
                ["original_image"]
            for _index, _img in enumerate(img_keys):
                _img2tb = self.img_resize(img2plot[_img])  # c, h, w
                _step = _index + 1

                if self.hparams.inference_mode:
                    # tag our cam images
                    if _img != "original_image":
                        suffix = "_cam"
                    else:
                        if self.fold_number != "fold_0":
                            continue
                        else:
                            suffix = ""

                    # build filename
                    _fname = os.path.join(
                        test_img_basedir,
                        _id + suffix +
                        "_" + str(_img) + ".jpeg"
                    )

                    # convert image before saving
                    _img2tb = self.img_jpeg_prep(_img2tb)  # c, h, w
                    
                    # save image
                    imsave(
                        fname=_fname,
                        arr=np.transpose(  # h, w, c
                            _img2tb,
                            (1, 2, 0)
                        )
                    )

            # increment image number
            self.test_img_num += 1

    @ staticmethod
    def _y_to_cat(logits, targets):
        """
        This method must be defined by the user!
        """
        raise Exception("Please customize the method '_y_to_cat' \
            to fit your data.")


class Net(BackboneModel):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def loss(preds, targets, pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(
            input=preds,
            target=targets.type_as(preds),
            pos_weight=pos_weight.type_as(preds)
        )
        return loss

    @staticmethod
    def _y_to_cat(logits, targets):
        y_preds = to_categorical(
            tensor=torch.softmax(logits, dim=1),
            argmax_dim=1
        )
        return y_preds

class NetDataModuleTest(pl.LightningDataModule):
    def __init__(
        self,
        test_data: pd.DataFrame,
        conf,
        num_classes
    ):
        super().__init__()

        self.test_data = test_data

        # get config
        self.batch_size = conf["batch_size"]
        self.base_dir = conf["base_dir"]
        self.num_classes = num_classes
        self.dl_workers = conf["dl_workers"]

    def setup(self, stage=None):
        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_ds = MipDatasetBinary(
                dataframe=self.test_data,
                base_dir=self.base_dir,
                num_class=self.num_classes
            )

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        test_dl = DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dl_workers
        )
        return test_dl


class MipDatasetBinary(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_dir: str,
        num_class: int,
        transform=None
    ):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.num_class = num_class
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]["img_file"]
        img = np.load(img_path).astype(np.float32)

        img_square_masked, img_ellipsis_masked = thorax_masker(img)

        sample = {
            "image": torch.from_numpy(img[np.newaxis]).float(),
            "image_squared_mask": torch.from_numpy(sitk_normalizer(img_square_masked)[np.newaxis]).float(),
            "image_ellipsis_mask": torch.from_numpy(sitk_normalizer(img_ellipsis_masked)[np.newaxis]).float(),
            # reduce dimension (remove first dimension)
            "id": self.dataframe.iloc[idx]["img_fn"].replace("_mip.npy", "")
        }

        return sample
