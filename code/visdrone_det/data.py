from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import yaml

from .constants import RAW_TO_CANONICAL_SPLIT, VISDRONE_CLASS_NAMES


@dataclass(frozen=True)
class SplitInfo:
    raw_name: str
    split_name: str
    source_dir: str
    image_dir: str
    annotation_dir: str | None
    source_label_dir: str | None
    prepared_dir: str
    prepared_image_dir: str
    prepared_label_dir: str | None
    image_count: int
    labeled_image_count: int
    box_count: int


@dataclass(frozen=True)
class PreparedDataset:
    raw_root: str
    prepared_root: str
    dataset_yaml: str
    train_yaml: str
    val_yaml: str
    test_yaml: str | None
    challenge_split: str | None
    split_yamls: dict[str, str]
    splits: list[SplitInfo]


def _canonical_split_name(path: Path) -> str:
    return RAW_TO_CANONICAL_SPLIT.get(path.name, path.name.lower().replace("_", "-"))


def _iter_candidate_split_dirs(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if (dataset_root / "images").exists():
        candidates.append(dataset_root)

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").exists():
            candidates.append(child)
            continue
        nested = list(sorted(grand for grand in child.iterdir() if grand.is_dir() and (grand / "images").exists()))
        candidates.extend(nested)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def discover_split_dirs(dataset_root: Path) -> dict[str, Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    split_dirs: dict[str, Path] = {}
    for candidate in _iter_candidate_split_dirs(dataset_root):
        split_name = _canonical_split_name(candidate)
        split_dirs.setdefault(split_name, candidate)

    if not split_dirs:
        raise FileNotFoundError(
            f"Could not find VisDrone-DET split directories under {dataset_root}. "
            "Expected folders such as VisDrone2019-DET-train/images."
        )
    return split_dirs


def _link_or_copy_tree(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), target, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)


def _iter_annotation_rows(annotation_path: Path) -> list[list[str]]:
    with annotation_path.open("r", newline="") as handle:
        return [row for row in csv.reader(handle) if row]


def _load_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image during annotation conversion: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _convert_annotation(annotation_path: Path, image_path: Path, label_path: Path) -> tuple[int, int]:
    rows = _iter_annotation_rows(annotation_path)
    width, height = _load_image_size(image_path)
    labels: list[str] = []

    for row in rows:
        padded = row + ["0"] * max(0, 8 - len(row))
        x = float(padded[0])
        y = float(padded[1])
        w = float(padded[2])
        h = float(padded[3])
        category_id = int(float(padded[5]))

        if category_id < 1 or category_id > len(VISDRONE_CLASS_NAMES):
            continue
        if w <= 0 or h <= 0:
            continue

        x1 = max(0.0, min(x, float(width)))
        y1 = max(0.0, min(y, float(height)))
        x2 = max(0.0, min(x + w, float(width)))
        y2 = max(0.0, min(y + h, float(height)))
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 0 or clipped_h <= 0:
            continue

        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        nw = clipped_w / width
        nh = clipped_h / height
        labels.append(
            f"{category_id - 1} "
            f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""))
    return len(rows), len(labels)


def _write_split_yaml(prepared_split_dir: Path, output_path: Path) -> None:
    payload = {
        "path": str(prepared_split_dir),
        "train": "images",
        "val": "images",
        "names": VISDRONE_CLASS_NAMES,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _count_existing_yolo_labels(label_dir: Path, image_paths: list[Path]) -> tuple[int, int]:
    labeled_images = 0
    box_count = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        if lines:
            labeled_images += 1
            box_count += len(lines)
    return labeled_images, box_count


def prepare_visdrone_det_dataset(dataset_root: Path, prepared_root: Path) -> PreparedDataset:
    split_dirs = discover_split_dirs(dataset_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    split_infos: list[SplitInfo] = []
    split_yamls: dict[str, str] = {}

    for split_name, source_dir in sorted(split_dirs.items()):
        image_dir = source_dir / "images"
        annotation_dir = source_dir / "annotations"
        source_label_dir = source_dir / "labels"
        has_annotations = annotation_dir.exists()
        has_yolo_labels = source_label_dir.exists()
        has_labels = has_annotations or has_yolo_labels

        prepared_split_dir = prepared_root / split_name
        prepared_image_dir = prepared_split_dir / "images"
        prepared_label_dir = prepared_split_dir / "labels" if has_labels else None
        prepared_split_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy_tree(image_dir, prepared_image_dir)

        image_paths = sorted(
            path for path in prepared_image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        labeled_image_count = 0
        box_count = 0

        if has_yolo_labels and prepared_label_dir is not None:
            _link_or_copy_tree(source_label_dir, prepared_label_dir)
            labeled_image_count, box_count = _count_existing_yolo_labels(prepared_label_dir, image_paths)
        elif has_annotations and prepared_label_dir is not None:
            for image_path in image_paths:
                annotation_path = annotation_dir / f"{image_path.stem}.txt"
                label_path = prepared_label_dir / f"{image_path.stem}.txt"
                if not annotation_path.exists():
                    label_path.write_text("")
                    continue
                _, kept_boxes = _convert_annotation(annotation_path, image_path, label_path)
                labeled_image_count += 1
                box_count += kept_boxes

        if has_labels:
            split_yaml = prepared_root / f"{split_name}.yaml"
            _write_split_yaml(prepared_split_dir, split_yaml)
            split_yamls[split_name] = str(split_yaml)

        split_infos.append(
            SplitInfo(
                raw_name=source_dir.name,
                split_name=split_name,
                source_dir=str(source_dir),
                image_dir=str(image_dir),
                annotation_dir=str(annotation_dir) if has_annotations else None,
                source_label_dir=str(source_label_dir) if has_yolo_labels else None,
                prepared_dir=str(prepared_split_dir),
                prepared_image_dir=str(prepared_image_dir),
                prepared_label_dir=str(prepared_label_dir) if prepared_label_dir else None,
                image_count=len(image_paths),
                labeled_image_count=labeled_image_count,
                box_count=box_count,
            )
        )

    train_split = next((item for item in split_infos if item.split_name == "train"), None)
    val_split = next((item for item in split_infos if item.split_name == "val"), None)
    if train_split is None or train_split.prepared_label_dir is None:
        raise FileNotFoundError("VisDrone train split with annotations is required.")
    if val_split is None or val_split.prepared_label_dir is None:
        raise FileNotFoundError("VisDrone val split with annotations is required.")

    test_yaml = split_yamls.get("test-dev")
    challenge_split = next((item.split_name for item in split_infos if item.split_name == "test-challenge"), None)
    dataset_yaml = prepared_root / "visdrone_det.yaml"
    dataset_payload = {
        "path": str(prepared_root),
        "train": "train/images",
        "val": "val/images",
        "names": VISDRONE_CLASS_NAMES,
    }
    if test_yaml is not None:
        dataset_payload["test"] = "test-dev/images"
    dataset_yaml.write_text(yaml.safe_dump(dataset_payload, sort_keys=False))

    manifest = prepared_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "raw_root": str(dataset_root),
                "prepared_root": str(prepared_root),
                "dataset_yaml": str(dataset_yaml),
                "splits": [asdict(item) for item in split_infos],
            },
            indent=2,
        )
    )

    return PreparedDataset(
        raw_root=str(dataset_root),
        prepared_root=str(prepared_root),
        dataset_yaml=str(dataset_yaml),
        train_yaml=split_yamls["train"],
        val_yaml=split_yamls["val"],
        test_yaml=test_yaml,
        challenge_split=challenge_split,
        split_yamls=split_yamls,
        splits=split_infos,
    )
