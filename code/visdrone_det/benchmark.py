from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    import pynvml
except ImportError:  # pragma: no cover - optional at import time
    pynvml = None

try:
    import torch
except ImportError:  # pragma: no cover - optional at import time
    torch = None


def _split_device_tokens(device: str) -> list[str]:
    return [token.strip() for token in device.split(",") if token.strip()]


def parse_training_device(device: str) -> str | int | list[int]:
    stripped = device.strip()
    if not stripped:
        return "cuda:0"
    tokens = _split_device_tokens(stripped)
    if len(tokens) > 1 and all(token.isdigit() for token in tokens):
        return [int(token) for token in tokens]
    if stripped.isdigit():
        return int(stripped)
    return stripped


def normalize_runtime_device(device: str) -> str:
    stripped = device.strip()
    if not stripped:
        return "cuda:0"
    tokens = _split_device_tokens(stripped)
    if tokens and all(token.isdigit() for token in tokens):
        return f"cuda:{tokens[0]}"
    if stripped.isdigit():
        return f"cuda:{stripped}"
    return stripped


def require_runtime_dependencies() -> tuple[Any, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies from code/requirements.txt."
        ) from exc

    if torch is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "PyTorch is not installed. Kaggle GPU images already include it; install it locally if needed."
        )
    return YOLO, torch


def load_yolo_model(weights: str | Path) -> Any:
    YOLO, _ = require_runtime_dependencies()
    return YOLO(str(weights))


@dataclass
class TrainingArtifacts:
    best_weights: Path
    run_dir: Path
    wandb: dict[str, Any] | None = None


def _require_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "wandb is not installed. Install dependencies from code/requirements.txt."
        ) from exc
    return wandb


def _parse_wandb_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _clean_results_dict(results_dict: dict[str, Any]) -> dict[str, float | int | str]:
    cleaned: dict[str, float | int | str] = {}
    for key, value in results_dict.items():
        numeric = _as_float(value)
        cleaned[key] = numeric if numeric is not None else str(value)
    return cleaned


def _build_wandb_epoch_logger(wandb_module: Any) -> Any:
    def _callback(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        payload: dict[str, float | int] = {"epoch": epoch}

        loss_names = list(getattr(trainer, "loss_names", []) or [])
        raw_losses = getattr(trainer, "tloss", None)
        if raw_losses is not None:
            if hasattr(raw_losses, "detach"):
                raw_losses = raw_losses.detach().cpu().tolist()
            elif hasattr(raw_losses, "tolist"):
                raw_losses = raw_losses.tolist()
            elif not isinstance(raw_losses, (list, tuple)):
                raw_losses = [raw_losses]

            numeric_losses = [_as_float(item) for item in raw_losses]
            cleaned_losses = [item for item in numeric_losses if item is not None]
            for name, value in zip(loss_names, numeric_losses):
                if value is not None:
                    payload[f"train/{name}"] = value
            if cleaned_losses:
                payload["train/loss"] = sum(cleaned_losses)

        trainer_metrics = getattr(trainer, "metrics", {}) or {}
        if isinstance(trainer_metrics, dict):
            for key, value in trainer_metrics.items():
                numeric = _as_float(value)
                if numeric is None:
                    continue
                metric_key = key if "/" in key else f"val/{key}"
                payload[metric_key] = numeric

        learning_rates = getattr(trainer, "lr", None)
        if isinstance(learning_rates, dict):
            for key, value in learning_rates.items():
                numeric = _as_float(value)
                if numeric is not None:
                    payload[f"lr/{key}"] = numeric

        wandb_module.log(payload, step=epoch)
        print(f"[wandb] {json.dumps(payload, sort_keys=True)}", flush=True)

    return _callback


def _prepare_wandb(
    *,
    enabled: bool,
    output_root: Path,
    run_name: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_mode: str,
    config: dict[str, Any],
) -> tuple[Any | None, dict[str, Any] | None]:
    if not enabled:
        return None, None

    wandb_module = _require_wandb()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb_module.login(key=api_key)

    effective_mode = wandb_mode
    if effective_mode == "online" and not api_key:
        effective_mode = "offline"
        print("[wandb] WANDB_API_KEY not set; falling back to offline mode.", flush=True)

    tags = _parse_wandb_tags(wandb_tags)
    run = wandb_module.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name or run_name,
        tags=tags,
        dir=str(output_root),
        config=config,
        mode=effective_mode,
        resume="allow",
    )
    return wandb_module, {
        "enabled": True,
        "project": wandb_project,
        "entity": wandb_entity,
        "run_name": run.name if run is not None else wandb_run_name or run_name,
        "run_id": getattr(run, "id", None) if run is not None else None,
        "mode": effective_mode,
        "tags": tags,
        "url": getattr(run, "url", None) if run is not None else None,
    }


def _upload_best_checkpoint_to_wandb(
    wandb_module: Any | None,
    best_weights: Path,
    run_name: str,
) -> None:
    if wandb_module is None:
        return

    artifact = wandb_module.Artifact(f"{run_name}-best", type="model")
    artifact.add_file(str(best_weights), name="best.pt")
    wandb_module.log_artifact(artifact)


def train_model(
    weights: str | Path,
    data_yaml: str | Path,
    output_root: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    cache: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_mode: str,
) -> TrainingArtifacts:
    training_device = parse_training_device(device)
    YOLO, _ = require_runtime_dependencies()
    model = YOLO(str(weights))
    wandb_module, wandb_info = _prepare_wandb(
        enabled=wandb_enabled,
        output_root=output_root,
        run_name=run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_mode=wandb_mode,
        config={
            "model": str(weights),
            "data_yaml": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "cache": cache,
            "run_name": run_name,
            "output_root": str(output_root),
        },
    )
    if wandb_module is not None:
        try:
            from ultralytics import settings

            # Disable Ultralytics W&B autologging so this repo controls a single clean run
            # with scalar-only logging instead of path-derived project names and media uploads.
            settings.update({"wandb": False})
        except Exception:  # pragma: no cover - best effort only
            pass
        model.add_callback("on_fit_epoch_end", _build_wandb_epoch_logger(wandb_module))

    try:
        model.train(
            data=str(data_yaml),
            project=str(output_root),
            name=run_name,
            exist_ok=True,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=training_device,
            workers=workers,
            cache=cache,
            pretrained=True,
            verbose=True,
            plots=False,
        )
        run_dir = output_root / run_name
        best_weights = run_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"Training completed but best weights were not found: {best_weights}")
        _upload_best_checkpoint_to_wandb(wandb_module, best_weights, run_name)
        return TrainingArtifacts(best_weights=best_weights, run_dir=run_dir, wandb=wandb_info)
    finally:
        if wandb_module is not None:
            wandb_module.finish()


def extract_validation_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    results_dict = _clean_results_dict(getattr(metrics, "results_dict", {}) or {})
    per_class_map = None
    maps = getattr(box, "maps", None)
    if maps is not None:
        per_class_map = [float(item) for item in maps]

    return {
        "map50_95": _as_float(getattr(box, "map", None)),
        "map50": _as_float(getattr(box, "map50", None)),
        "map75": _as_float(getattr(box, "map75", None)),
        "precision": _as_float(getattr(box, "mp", None)),
        "recall": _as_float(getattr(box, "mr", None)),
        "fitness": _as_float(getattr(metrics, "fitness", None)),
        "speed_ms_per_image": getattr(metrics, "speed", None),
        "per_class_map50_95": per_class_map,
        "results_dict": results_dict,
    }


def evaluate_model(
    weights: str | Path,
    split_yaml: str | Path,
    device: str,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    runtime_device = normalize_runtime_device(device)
    model = load_yolo_model(weights)
    metrics = model.val(
        data=str(split_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=runtime_device,
        plots=False,
        verbose=False,
    )
    return extract_validation_metrics(metrics)


def _load_sample_images(image_dir: Path, max_images: int) -> list[tuple[Path, Any]]:
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:max_images]
    samples: list[tuple[Path, Any]] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        samples.append((path, image))
    if not samples:
        raise FileNotFoundError(f"No readable images found for benchmarking in {image_dir}")
    return samples


def _setup_nvml(device: str) -> tuple[Any | None, int | None]:
    runtime_device = normalize_runtime_device(device)
    if pynvml is None or not runtime_device.startswith("cuda"):
        return None, None
    pynvml.nvmlInit()
    index = 0
    if ":" in runtime_device:
        index = int(runtime_device.split(":", 1)[1])
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    return handle, index


def benchmark_latency_and_gpu(
    weights: str | Path,
    image_dir: Path,
    device: str,
    imgsz: int,
    warmup: int,
    max_images: int,
) -> dict[str, Any]:
    runtime_device = normalize_runtime_device(device)
    model = load_yolo_model(weights)
    _, torch_runtime = require_runtime_dependencies()
    samples = _load_sample_images(image_dir, max_images=max_images)

    cuda_enabled = runtime_device.startswith("cuda") and torch_runtime.cuda.is_available()
    torch_device = torch_runtime.device(runtime_device if cuda_enabled else "cpu")
    nvml_handle, nvml_index = _setup_nvml(runtime_device)

    if cuda_enabled:
        torch_runtime.cuda.reset_peak_memory_stats(torch_device)

    for _, image in samples[:warmup]:
        model.predict(source=image, imgsz=imgsz, device=runtime_device, verbose=False)
    if cuda_enabled:
        torch_runtime.cuda.synchronize(torch_device)

    timings_ms: list[float] = []
    gpu_utilization: list[int] = []
    gpu_memory_mb: list[float] = []

    for _, image in samples:
        if cuda_enabled:
            start = torch_runtime.cuda.Event(enable_timing=True)
            end = torch_runtime.cuda.Event(enable_timing=True)
            torch_runtime.cuda.synchronize(torch_device)
            start.record()
            model.predict(source=image, imgsz=imgsz, device=runtime_device, verbose=False)
            end.record()
            torch_runtime.cuda.synchronize(torch_device)
            elapsed_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            model.predict(source=image, imgsz=imgsz, device=runtime_device, verbose=False)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        timings_ms.append(elapsed_ms)

        if nvml_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_utilization.append(int(util.gpu))
            gpu_memory_mb.append(mem.used / (1024 ** 2))

    if nvml_handle is not None:
        pynvml.nvmlShutdown()

    average_latency = statistics.mean(timings_ms)
    median_latency = statistics.median(timings_ms)

    peak_allocated_gb = None
    peak_reserved_gb = None
    if cuda_enabled:
        peak_allocated_gb = torch_runtime.cuda.max_memory_allocated(torch_device) / (1024 ** 3)
        peak_reserved_gb = torch_runtime.cuda.max_memory_reserved(torch_device) / (1024 ** 3)

    return {
        "device": runtime_device,
        "gpu_index": nvml_index,
        "num_images": len(samples),
        "warmup_images": min(warmup, len(samples)),
        "latency_ms_mean": average_latency,
        "latency_ms_median": median_latency,
        "latency_ms_min": min(timings_ms),
        "latency_ms_max": max(timings_ms),
        "throughput_fps_mean": 1000.0 / average_latency if average_latency else None,
        "gpu_utilization_percent_mean": statistics.mean(gpu_utilization) if gpu_utilization else None,
        "gpu_utilization_percent_peak": max(gpu_utilization) if gpu_utilization else None,
        "gpu_memory_mb_mean": statistics.mean(gpu_memory_mb) if gpu_memory_mb else None,
        "gpu_memory_mb_peak": max(gpu_memory_mb) if gpu_memory_mb else None,
        "torch_peak_allocated_gb": peak_allocated_gb,
        "torch_peak_reserved_gb": peak_reserved_gb,
    }
