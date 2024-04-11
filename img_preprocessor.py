__author__ = "Lorenz A. Kapsner"
__copyright__ = "Copyright 2023, Universitätsklinikum Erlangen"

import os
from glob import glob
from tqdm import tqdm
from tirutils.preprocessing.sequential import Sequential
import copy

"""
Perprocessor for artifact segmentor to bring image into shape x * y instead of x * y * z
(where z was 1)
"""

mip_dir = "/home/user/data/tech_artifacts/km_mips_230609/artifact_segmentor_whole_mips/dce_mips"
output_path = "/home/user/data/tech_artifacts/km_mips/artifact_segmentor_whole_mips/preprocessed_230811"

in_out_sequence = [
    {"imageinput.NiftiIN": {
        "filepath": None
    }},
    {"transformations.SitkSqueeze": {}},
    {"transformations.Sitk2Numpy": {}},
    {"imageoutput.NumpyOUT": {
        "identifier": None,
        "outdir": output_path
    }}
]

seq = Sequential()

if __name__ == "__main__":
    # list all mips here
    mip_files = glob(os.path.join(mip_dir, "*.nii.gz"))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for _mip_path in tqdm(mip_files):
        mip_fn = _mip_path.split(os.path.sep)[-1]
        identifier = mip_fn.replace(".nii.gz", "")

        # modify input sequence
        params = copy.deepcopy(in_out_sequence)
        params[0]["imageinput.NiftiIN"]["filepath"] = _mip_path
        params[-1]["imageoutput.NumpyOUT"]["identifier"] = identifier
        seq(*params)
