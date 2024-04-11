#!/usr/bin/python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import os
import pandas as pd
from glob import glob
import pytorch_lightning as pl
import logging
from datetime import datetime
from src.backbone import Net, NetDataModuleTest
import torch

def generate_inference_dataset(
        mip_dir,
        fpattern: str="*.npy",
        debug: bool=False
    ):
    # list all mips here
    mip_files = glob(os.path.join(mip_dir, fpattern))
    if debug:
        mip_files = mip_files[0:20]

    inference_info = pd.DataFrame({"img_file": mip_files})
    mip_files_fn = inference_info.img_file.str.split(os.path.sep, expand=True)
    subject_info = mip_files_fn[len(mip_files_fn.columns)-1].str.split("_", expand=True)

    inference_info["subject_id_pseudonymized"] = subject_info.loc[:, 0].astype(str)
    inference_info["study_date"] = subject_info.loc[:, 1].astype(str)
    inference_info["img_fn"] = mip_files_fn[len(mip_files_fn.columns)-1]

    return inference_info

def inference_loop(
        inference_info: pd.DataFrame,
        checkpoints: dict,
        base_path: str,
        checkpoint_dir: str,
        mip_dir: str,
        gpus: str = "0"
    ):

    # save inference results for each fold here:
    final_results = pd.DataFrame()

    for _fold, _chkpath in checkpoints.items():
        chkpath = os.path.join(
            checkpoint_dir,
            _fold,
            _chkpath
        )
        
        # load data module parameters
        dm_params = {
            "batch_size": 1024,  # 128 if os.cpu_count() > 32 else 8,
            #"test_split": 0.15,
            "base_dir": mip_dir.strip(),
            "dl_workers": 32 # 8 if os.cpu_count() > 32 else 4  # dataloader cpu workers
        }

        # seed everything
        pl.seed_everything(0)

        # load the model from checkpoint
        model = Net.load_from_checkpoint(
            checkpoint_path=chkpath,
            # args.hyperparameters (not needed since checkpoint saves hparams)
            hparams_file=None,
            map_location=None,
            #map_location={'cuda:0': 'cpu'}, # to cpu
            inference_mode=True,
            fold_number=_fold
        )

        # instantiate the data module
        test_dm = NetDataModuleTest(
            test_data=inference_info,
            conf=dm_params,
            num_classes=model.hparams.num_classes # 2
        )

        # define name for file saving
        nameparts = chkpath.split("/checkpoints")[0].split(os.path.sep)[-2:]
        run_name = nameparts[0] + "/" + nameparts[1]
        run_version = run_name + "_" + \
            str(datetime.now().strftime("%Y%m%d-%H%M"))
        logging.info("run_version: {}".format(run_version))

        # instantiate the trainer
        # test trainer
        trainer = pl.Trainer(
            accelerator="ddp",
            logger=[pl.loggers.TensorBoardLogger(
                save_dir=base_path,
                name="tensorboard",
                version=run_version,
                default_hp_metric=False
            )],
            default_root_dir=base_path,
            deterministic=True,
            gpus=gpus
            #gpus=None
        )

        try:
            # test the model
            trainer.test(
                model=model,
                datamodule=test_dm
            )

            final_results = pd.concat([final_results, model.inference_results], ignore_index=True)
        except ValueError as e:
            logging.error(e)
        
        # clear cache
        del model
        torch.cuda.empty_cache()
    
    return final_results
