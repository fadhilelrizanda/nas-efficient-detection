from __future__ import annotations

VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

RAW_TO_CANONICAL_SPLIT = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test-dev",
    "VisDrone2019-DET-test-challenge": "test-challenge",
}

