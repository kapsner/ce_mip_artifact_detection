#!/usr/bin/python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import os
from skimage.exposure import is_low_contrast

def __initialize__(base_path):
    # generate postprocessing dir
    postprocessing_dir = os.path.join(base_path, "postprocessing")
    results_dir = os.path.join(base_path, "results")
    results_dira = os.path.join(results_dir, "artifacts")
    os.makedirs(results_dira, exist_ok=True)
    results_dirb = os.path.join(results_dir, "no_artifacts")
    os.makedirs(results_dirb, exist_ok=True)

    return postprocessing_dir, results_dir


def prepare_postprocessing_folder(base_path: str, postprocessing_dir: str):
    for _fold in range(0, 5):
        _foldn = "fold_" + str(_fold)
        
        post_fold_dir = os.path.join(postprocessing_dir, _foldn)
        os.makedirs(post_fold_dir, exist_ok=True)

def flag_low_contrast_imgs(img):
    return is_low_contrast(img)
