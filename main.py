#!/usr/bin/python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import os
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
from skimage import color

append_path = os.path.join(
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../"
    )),
    "mip_classifier/model"
)
sys.path.append(append_path)

from src.utils import __initialize__, flag_low_contrast_imgs
from src.avg_cams import generate_avg_cams, img_jpeg_prep
from src.inference_utils import generate_inference_dataset, inference_loop
from src.image_masker import thorax_masker

mip_dir = "/home/user/data/tech_artifacts/km_mips/artifact_segmentor_whole_mips/preprocessed_230811"

checkpoint_dir = "/home/user/development/trainings/km_mips/tech_artifacts_final_210921/tensorboard/densenet121_adam_1042"

checkpoints = {
    "fold_0": "checkpoints/loss/valid=0.3856-epoch=176.ckpt",
    "fold_1": "checkpoints/loss/valid=0.3680-epoch=145.ckpt",
    "fold_2": "checkpoints/loss/valid=0.4217-epoch=100.ckpt",
    "fold_3": "checkpoints/loss/valid=0.3827-epoch=189.ckpt",
    "fold_4": "checkpoints/loss/valid=0.4020-epoch=180.ckpt"
}

base_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "experiments"
)


if __name__ == "__main__":

    # initialize dirs
    postprocessing_dir, results_dir = __initialize__(base_path=base_path)

    # generate inference dataset
    inference_info = generate_inference_dataset(
        mip_dir=mip_dir,
        fpattern="*.npy",
        debug=False
    )
    inference_info["low_contrast"] = None
    
    # flag low-contrast or other issues with images
    for _row in tqdm(inference_info.itertuples(index=True), total=len(inference_info)):
        orig_img_np = np.load(_row.img_file)
        inference_info = inference_info.copy()
        try:
            _, masked_img = thorax_masker(orig_img_np)
            masked_gray = np.transpose(
                color.gray2rgb(masked_img),
                (2, 0, 1)
            )
            inference_info.iloc[_row.Index]["low_contrast"] = flag_low_contrast_imgs(img_jpeg_prep(masked_gray))
        except Exception as e:
            logging.error(e)
            inference_info.iloc[_row.Index]["low_contrast"] = True

    final_results = pd.DataFrame({
        "id": inference_info.img_fn.str.replace("_mip.npy", "", regex=False),
        "low_contrast": inference_info.low_contrast.values,
        "artifact_probability_mean": None,
        "artifact_probability_median": None,
        "majority_vote": None
    })
    final_results.index = final_results.id.values

    try:

        artifact_detection_results = inference_loop(
            inference_info=inference_info[~inference_info.low_contrast.isin([True])],
            checkpoints=checkpoints,
            base_path=base_path,
            checkpoint_dir=checkpoint_dir,
            mip_dir=mip_dir,
            gpus="0"
        )

    except Exception as e:
        logging.error(e)
    
    
    for _id in tqdm(artifact_detection_results.id.unique(), total=len(artifact_detection_results.id.unique())):

        res = artifact_detection_results[artifact_detection_results.id == _id]

        try:
            # majority vote of classifiers:
            if sum(res.y_pred_class) >= 3:
                _artifact = True
            else:
                _artifact = False

            # final results
            generate_avg_cams(
                img_id=_id,
                orig_img_path=mip_dir,
                base_path=base_path,
                results_dir=results_dir,
                detected_artifact=_artifact
            )

            append_row = {
                "artifact_probability_mean": res.y_pred_prob_positive.mean(),
                "artifact_probability_median": res.y_pred_prob_positive.median(),
                "majority_vote": "artifact" if _artifact else "no artifact"
            }
            
        except Exception as e:
            logging.error(e)

            append_row = {
                "majority_vote": "ERROR"
            }
        
        final_results.loc[final_results.index == _id, append_row.keys()] = pd.DataFrame(append_row, index=[_id])
    
    # save final results
    final_results.to_csv(
        path_or_buf=os.path.join(
            results_dir,
            "artifact_detection_results.csv"
        ),
        index=False
    )
        
