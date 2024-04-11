#!/usr/bin/python

__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import SimpleITK as sitk
from tirutils.preprocessing.transformations import Normalize

def sitk_normalizer(np_img):
    sitk_img = sitk.GetImageFromArray(np_img)
    proc_img = Normalize()(sitk_img)
    return sitk.GetArrayFromImage(proc_img)
