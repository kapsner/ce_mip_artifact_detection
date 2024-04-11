#!/usr/bin/python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import os
from matplotlib import pyplot as plt
from skimage import color
from skimage.io import imsave
from PIL import Image
import monai
import numpy as np

cm = plt.get_cmap("jet_r")
img_resize = monai.transforms.Compose([
    monai.transforms.Resize(spatial_size=[256, 256])
])
img_jpeg_prep = monai.transforms.Compose([
    monai.transforms.ScaleIntensity(minv=0, maxv=255),
    monai.transforms.CastToType(dtype=np.uint8)
])
jpeg_loader = monai.transforms.Compose(
    [monai.transforms.LoadPNG()]
)

def generate_avg_cams(
        img_id: str,
        orig_img_path: str,
        base_path: str,
        results_dir: str,
        detected_artifact: bool
    ):
    
    orig_img_path = os.path.join(orig_img_path, img_id + "_mip.npy")

    # cam-image-path
    cam_img_path = os.path.join(base_path, "test_images")

    # select cam:
    selcam = "_1" #if detected_artifact else "_0"
    _artifact = "artifacts" if detected_artifact else "no_artifacts"

    cam_img_fn = os.path.join(cam_img_path, "fold_0", img_id + "_cam" + selcam + ".npy")
    cam_img_fn_list = [cam_img_fn.replace("fold_0", "fold_" + str(_i)) for _i in range(0, 5)]

    img_list = [np.load(f) for f in cam_img_fn_list]
    
    img_prep = []
    for _im in img_list:
        img_prep.append(np.squeeze(_im))

    img_stack = np.dstack(
        img_prep
    )

    avg_img = np.mean(
        a=img_stack,
        axis=2
    )

    # apply colormap
    _img = cm(avg_img)
    # create 3-channel image for displaying in tensorboard
    colorized_img = np.transpose(
        color.rgba2rgb(_img),
        (2, 0, 1)
    )

    # convert image before saving
    _img2tb = img_resize(colorized_img)
    _img2tb = img_jpeg_prep(_img2tb)  # c, h, w

    _cam_to_save = np.transpose(  # h, w, c
        _img2tb,
        (1, 2, 0)
    )

    # save image
    imsave(
        fname=os.path.join(
            results_dir,
            _artifact,
            img_id + "_avg_cam.jpeg"
        ),
        arr=_cam_to_save
    )

    # generate overlay
    # convert image before saving
    orig_img_np = np.transpose(
        color.gray2rgb(np.squeeze(np.load(orig_img_path))),
        (2, 0, 1)
    )
    
    _img2tb = img_jpeg_prep(orig_img_np)  # c, h, w

    _orig_to_save = np.transpose(  # h, w, c
        _img2tb,
        (1, 2, 0)
    )
    orig_img = Image.fromarray(_orig_to_save).convert("RGBA")
    overlay_img = Image.fromarray(_cam_to_save).convert("RGBA")

    final_overlay = Image.blend(orig_img, overlay_img, 0.3)
    final_overlay.convert("RGB").save(
        fp=os.path.join(
            results_dir,
            _artifact,
            img_id + "_overlay.jpeg"
        )
    )

    # save original mip
    orig_img.convert("RGB").save(
        fp=os.path.join(
            results_dir,
            _artifact,
            img_id + "_original.jpeg"
        )
    )
