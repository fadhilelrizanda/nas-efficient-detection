from __future__ import annotations

import sys

from visdrone_det.cli import main

DEFAULT_ARGS = [
    "--epochs", "5",
    "--batch", "8",
    "--device", "0,1",
    "--run-name", "yolo26x-visdrone-det-finetune",
    "--report-name", "finetuning_report.json",
    "--wandb-tags", "visdrone-det,yolo26x,finetune,kaggle",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGS, *sys.argv[1:]])
