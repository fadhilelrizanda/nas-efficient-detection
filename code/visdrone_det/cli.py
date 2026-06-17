from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .benchmark import TrainingArtifacts, benchmark_latency_and_gpu, evaluate_model, train_model
from .data import PreparedDataset, prepare_visdrone_det_dataset
from .video import build_prediction_video

DEFAULT_PROJECT_NAME = "DistillNas-YOLO26-Visdrone"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare VisDrone-DET for YOLO, fine-tune/evaluate YOLO26x, benchmark latency/GPU, "
            "and render a test-challenge prediction video."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root that contains VisDrone2019-DET-* folders.")
    parser.add_argument("--workspace", type=Path, default=Path("outputs/visdrone_det"))
    parser.add_argument("--prepared-root", type=Path, default=None, help="Optional explicit YOLO-prepared dataset path.")
    parser.add_argument("--model", type=str, default="yolo26x.pt", help="Ultralytics weight or model name.")
    parser.add_argument("--weights", type=Path, default=None, help="Existing weights to evaluate/predict instead of training output.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--run-name", type=str, default="yolo26x-visdrone-det-finetune")
    parser.add_argument("--latency-split", type=str, default="val")
    parser.add_argument("--latency-max-images", type=int, default=128)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--video-split", type=str, default="test-challenge")
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--video-max-images", type=int, default=360)
    parser.add_argument("--report-name", type=str, default="benchmark_report.json")
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging.")
    parser.set_defaults(wandb=True)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="visdrone-det,yolo26x,finetune")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args(argv)


def _split_summary(prepared: PreparedDataset) -> list[dict[str, Any]]:
    return [asdict(item) for item in prepared.splits]


def _resolve_weights(args: argparse.Namespace, trained_weights: Path | None) -> Path:
    if args.weights is not None:
        return args.weights
    if trained_weights is not None:
        return trained_weights
    raise FileNotFoundError("No weights available. Provide --weights or run training without --skip-train.")


def _find_prepared_split(prepared: PreparedDataset, split_name: str) -> Path:
    for split in prepared.splits:
        if split.split_name == split_name:
            return Path(split.prepared_image_dir)
    raise FileNotFoundError(f"Requested split '{split_name}' not found in prepared dataset.")


def _build_report_stub(args: argparse.Namespace, prepared: PreparedDataset) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "config": {
            "dataset_root": str(args.dataset_root),
            "workspace": str(args.workspace),
            "prepared_root": prepared.prepared_root,
            "model": args.model,
            "weights": str(args.weights) if args.weights else None,
            "skip_train": args.skip_train,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "cache": args.cache,
            "run_name": args.run_name,
            "latency_split": args.latency_split,
            "latency_max_images": args.latency_max_images,
            "latency_warmup": args.latency_warmup,
            "video_split": args.video_split,
            "video_fps": args.video_fps,
            "video_max_images": args.video_max_images,
            "wandb": args.wandb,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_run_name": args.wandb_run_name,
            "wandb_tags": args.wandb_tags,
            "wandb_mode": args.wandb_mode,
        },
        "dataset": {
            "dataset_yaml": prepared.dataset_yaml,
            "train_yaml": prepared.train_yaml,
            "val_yaml": prepared.val_yaml,
            "test_yaml": prepared.test_yaml,
            "challenge_split": prepared.challenge_split,
            "splits": _split_summary(prepared),
        },
    }


def _training_report(artifacts: TrainingArtifacts) -> dict[str, Any]:
    payload = {
        "best_weights": str(artifacts.best_weights),
        "run_dir": str(artifacts.run_dir),
    }
    if artifacts.wandb is not None:
        payload["wandb"] = artifacts.wandb
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    prepared_root = args.prepared_root.resolve() if args.prepared_root else workspace / "prepared_dataset"
    runs_root = workspace / "runs"
    reports_root = workspace / "reports"
    video_root = workspace / "videos"
    reports_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)

    prepared = prepare_visdrone_det_dataset(args.dataset_root.resolve(), prepared_root)
    report = _build_report_stub(args, prepared)

    trained_weights: Path | None = None
    if not args.skip_train:
        training_artifacts = train_model(
            weights=args.model,
            data_yaml=prepared.dataset_yaml,
            output_root=runs_root,
            run_name=args.run_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            wandb_enabled=args.wandb and args.wandb_mode != "disabled",
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
            wandb_mode=args.wandb_mode,
        )
        trained_weights = training_artifacts.best_weights
        report["training"] = _training_report(training_artifacts)
    else:
        report["training"] = {
            "skipped": True,
        }

    weights_path = _resolve_weights(args, trained_weights).resolve()
    report["weights_used"] = str(weights_path)

    evaluation: dict[str, Any] = {}
    for split_name, split_yaml in prepared.split_yamls.items():
        evaluation[split_name] = evaluate_model(
            weights=weights_path,
            split_yaml=split_yaml,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    report["evaluation"] = evaluation

    latency_image_dir = _find_prepared_split(prepared, args.latency_split)
    report["latency_gpu"] = benchmark_latency_and_gpu(
        weights=weights_path,
        image_dir=latency_image_dir,
        device=args.device,
        imgsz=args.imgsz,
        warmup=args.latency_warmup,
        max_images=args.latency_max_images,
    )

    video_status: dict[str, Any] = {"skipped": True}
    try:
        video_image_dir = _find_prepared_split(prepared, args.video_split)
        video_path = video_root / f"{args.video_split}_predictions.mp4"
        video_status = build_prediction_video(
            weights=weights_path,
            image_dir=video_image_dir,
            output_path=video_path,
            device=args.device,
            imgsz=args.imgsz,
            fps=args.video_fps,
            max_images=args.video_max_images,
        )
    except FileNotFoundError as exc:
        video_status = {"skipped": True, "reason": str(exc)}
    report["video"] = video_status

    report_path = reports_root / args.report_name
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[done] benchmark report written to {report_path}")


if __name__ == "__main__":
    main()
