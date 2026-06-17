from __future__ import annotations

from pathlib import Path

import cv2

from .benchmark import load_yolo_model


def build_prediction_video(
    weights: str | Path,
    image_dir: Path,
    output_path: Path,
    device: str,
    imgsz: int,
    fps: int,
    max_images: int,
    batch_size: int = 16,
) -> dict[str, int | str]:
    model = load_yolo_model(weights)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found for prediction video under {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_count = 0
    writer_size: tuple[int, int] | None = None

    try:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = [str(path) for path in image_paths[start:start + batch_size]]
            results = model.predict(
                source=batch_paths,
                imgsz=imgsz,
                device=device,
                verbose=False,
                stream=False,
            )
            for image_path, result in zip(image_paths[start:start + batch_size], results):
                frame = result.plot()
                cv2.rectangle(frame, (12, 12), (720, 48), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"VisDrone test-challenge prediction: {image_path.name}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if writer is None:
                    height, width = frame.shape[:2]
                    writer_size = (width, height)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        writer_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {output_path}")
                elif writer_size is not None and (frame.shape[1], frame.shape[0]) != writer_size:
                    frame = cv2.resize(frame, writer_size, interpolation=cv2.INTER_LINEAR)
                writer.write(frame)
                frame_count += 1
    finally:
        if writer is not None:
            writer.release()

    return {
        "output_path": str(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps else 0,
    }
