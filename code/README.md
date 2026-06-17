# Code Directory

This directory is the main project code area for `DistillNas-YOLO26-Visdrone`.

## Current layout

- `datasets/`: dataset-specific utilities and preparation scripts
- `visdrone_det/`: shared VisDrone-DET preparation, YOLO26x train/eval, latency profiling, and video export logic
- `make_visdrone_det_yolo26x_benchmark.py`: code-side benchmark task entrypoint, matched to `kaggle-nb/visdrone-det-yolo26x-benchmark/`
- `make_visdrone_det_yolo26x_finetuning.py`: code-side fine-tuning task entrypoint, matched to `kaggle-nb/visdrone-det-yolo26x-finetuning/`
- `requirements.txt`: shared Python dependencies for the currently checked-in utilities

## Task structure

This repository is Kaggle-first for execution. Code entrypoints under `code/` should mirror Kaggle task folders one-to-one when a task has different runtime defaults or outputs.

Current task pairings:

- `kaggle-nb/visdrone-det-yolo26x-benchmark/` -> `code/make_visdrone_det_yolo26x_benchmark.py`
- `kaggle-nb/visdrone-det-yolo26x-finetuning/` -> `code/make_visdrone_det_yolo26x_finetuning.py`

Shared implementation lives in `visdrone_det/`. Task-specific defaults such as batch size, GPU selection, run name, report name, and tags belong in the task-specific wrapper so the code path matches the Kaggle path.

## Current utilities

The first concrete utility lives under `datasets/` and handles one dataset preparation task for VisDrone MOT:

- download an official VisDrone MOT split
- overlay ground-truth boxes on one sequence
- export a preview video capped at 60 seconds by default

See `code/datasets/README.md` for usage.

The main detection workflow now lives under `visdrone_det/` and is exposed through task-specific wrappers.

Benchmark example:

```bash
python3 code/make_visdrone_det_yolo26x_benchmark.py \
  --dataset-root /path/to/VisDrone_Dataset \
  --workspace ./outputs/visdrone_det
```

Fine-tuning example:

```bash
python3 code/make_visdrone_det_yolo26x_finetuning.py \
  --dataset-root /path/to/VisDrone_Dataset \
  --workspace ./outputs/visdrone_det_finetuning
```

That shared pipeline:

- converts `VisDrone2019-DET-*` splits into YOLO labels
- trains `yolo26x.pt` on train/val unless `--skip-train` is used
- logs per-epoch scalar training metrics to `wandb` by default, with console-visible `[wandb]` lines for remote monitoring
- evaluates every split that has labels
- measures latency and GPU usage on a chosen split
- exports a prediction video for `test-challenge` when that split exists
