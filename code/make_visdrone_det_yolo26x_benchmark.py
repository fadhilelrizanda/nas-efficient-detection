from __future__ import annotations

import sys

from visdrone_det.cli import main

DEFAULT_ARGS = [
    "--epochs", "5",
    "--batch", "2",
    "--device", "cuda:0",
    "--run-name", "yolo26x-visdrone-det-benchmark",
    "--report-name", "benchmark_report.json",
    "--wandb-tags", "visdrone-det,yolo26x,benchmark",
]


if __name__ == "__main__":
    main([*DEFAULT_ARGS, *sys.argv[1:]])
