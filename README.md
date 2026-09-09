# 260909_Teratoma_inf

Inference-only pipeline for teratoma virtual H&E staining (VS) followed by x4 super-resolution (SR). Training code is not included.

## System Requirements

You need Pytorch_with_CUDA for this experiments.
And following additional packages are needed:

-    torchvision
-    numpy
-    opencv-python
-    Pillow
-    einops

## Installation guide

Set environment with following:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place test images (png) in `dataset/`. Images are used at native resolution (no resize).

```
dataset/
  19Aug794_i000_016.png
  19Aug794_i008_003.png
  19Aug794_i010_016.png
  19Aug794_i013_005.png
```

## Demo Introduction

### Step 0. Checkpoint Preparation

Place VS and SR weights in `checkpoints/`.

```
checkpoints/
  latest_net_G.pth   # VS generator
  hat.pth            # SR (HAT x4)
```

### Step 1. Test

Run virtual staining and super-resolution with following code.

```bash
python test.py --dataroot ./dataset
```

Expected output:
- `results/LR_VHE/{NAME}.png` (VS, e.g. 1024x1024)
- `results/HR_VHE/{NAME}.png` (SR x4, e.g. 4096x4096)
