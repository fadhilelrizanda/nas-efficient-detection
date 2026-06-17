# Dataset Utilities

This subdirectory contains dataset-specific tools.

## VisDrone ground-truth video preview

The `visdrone_gt_video/` package downloads an official VisDrone MOT split, overlays the ground-truth boxes on one sequence, and exports a preview video capped at 60 seconds by default.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

### Usage

```bash
python3 code/datasets/make_visdrone_gt_video.py \
  --split mot_val \
  --data-root ./data \
  --output ./outputs/visdrone_mot_val_preview.mp4 \
  --max-seconds 60
```

To render a specific sequence:

```bash
python3 code/datasets/make_visdrone_gt_video.py \
  --split mot_val \
  --sequence uav0000086_00000_v \
  --data-root ./data \
  --output ./outputs/uav0000086_00000_v.mp4
```

### Notes

- The default dataset URLs point to the official VisDrone Google Drive links published in the VisDrone dataset repository.
- The script targets the MOT layout with `sequences/` and `annotations/` directories.
- Output FPS defaults to `20`. You can override it with `--fps`.
