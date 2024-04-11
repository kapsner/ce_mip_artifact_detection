# ce-mip-artifact-detection

## Setup

Requires python 3.8.5

```bash
conda init bash
conda create -n artseg python=3.8.5
conda activate artseg
cd ../tirutils/tirutils
pip install -e .
cd ../../artifact_segmentor
pip install -r requirements.txt
```

## Prepare Numpy MIPs

```bash
python img_preprocessor.py
```

## Apply Thorax Masking and Run Inference

```bash
python main.py
```
