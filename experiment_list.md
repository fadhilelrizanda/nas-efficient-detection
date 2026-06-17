# Experiment List

This file is written by `code-agent` and consumed by `hermes`.

## Status key

- [ ] todo
- [/] running
- [x] done
- [!] blocked
- [-] cancelled

## Template

### Experiment: <name>
- Status: [ ]
- Purpose: <hypothesis or reason>
- Task: <kaggle-nb/task-folder or other orchestration target>
- Code state: <branch, commit, or local state requirement>
- Accelerator: <for example NvidiaTeslaT4>
- Inputs: <dataset assumptions, weights, env vars>
- Outputs to download: <weights, csv, json, mp4, logs>
- Notes: <extra instructions for hermes>

## Active experiments

### Experiment: visdrone-det-yolo26x-finetuning
- Status: [/]
- Purpose: Fine-tune the pretrained `YOLO26x` model on `VisDrone-DET` for 5 epochs and capture per-epoch metrics in W&B plus Kaggle live logs.
- Task: `kaggle-nb/visdrone-det-yolo26x-finetuning`
- Code state: Current local repository state after the dedicated fine-tuning task creation; push the latest main workspace changes before launch.
- Accelerator: `2xT4`
- Inputs: VisDrone dataset from `banuprasadb/visdrone-dataset`; starting weights `yolo26x.pt`; env vars `VISDRONE_EPOCHS=5`, `VISDRONE_WANDB=1`, `VISDRONE_WANDB_PROJECT=DistillNas-YOLO26-Visdrone`, optional `WANDB_API_KEY` via Kaggle secrets, `VISDRONE_BATCH=8`, `VISDRONE_IMGSZ=640`, and `VISDRONE_DEVICE=0,1`.
- Outputs to download: `outputs/visdrone_det_finetuning/runs/yolo26x-visdrone-det-finetune/weights/best.pt`, `outputs/visdrone_det_finetuning/runs/yolo26x-visdrone-det-finetune/weights/last.pt`, `outputs/visdrone_det_finetuning/reports/finetuning_report.json`, the full Kaggle log, and any generated plots under the run directory.
- Notes: Use `kaggle_kernels_push_live(...)` with explicit `2xT4` selection, confirm the run actually starts, watch the live logs for `[wandb]` epoch lines, then download artifacts into `kaggle-nb/output/visdrone-det-yolo26x-finetuning/`.
