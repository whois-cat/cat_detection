#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "numpy>=2.2.0",
#   "pillow>=11.1.0",
#   "ultralytics>=8.3.0",
# ]
# ///

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
from ultralytics import YOLO

from pipeline_db import (
    DEFAULT_CROPS_DIR,
    PROJECT_ROOT,
    connect_db,
    make_uid,
    print_crop_stats,
    relative_to_project,
    select_inference_device,
)


COCO_CAT_CLASS_ID = 15
MODEL_NAME = "yolov8n.pt"
UNSORTED_LABEL = "unsorted"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_PADDING = 0.15
DEFAULT_MIN_SIZE = 50
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class FrameRow:
    frame_uid: str
    frame_name: str
    frame_path: str


def apply_padding_bbox(
    bbox_x: float,
    bbox_y: float,
    bbox_width: float,
    bbox_height: float,
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    pad_x = bbox_width * padding
    pad_y = bbox_height * padding

    left = max(0, int(round(bbox_x - pad_x)))
    top = max(0, int(round(bbox_y - pad_y)))
    right = min(frame_width, int(round(bbox_x + bbox_width + pad_x)))
    bottom = min(frame_height, int(round(bbox_y + bbox_height + pad_y)))

    return left, top, right, bottom


def load_frames_from_db(connection: Any, limit: int | None) -> list[FrameRow]:
    query = """
        SELECT frame_uid, frame_name, frame_path
        FROM frames
        WHERE COALESCE(is_duplicate, false) = false
        ORDER BY frame_name
    """
    parameters: list[object] = []
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(int(limit))

    rows = connection.execute(query, parameters).fetchall()
    return [
        FrameRow(frame_uid=row[0], frame_name=row[1], frame_path=row[2])
        for row in rows
    ]


@click.command()
@click.option(
    "--confidence",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_CONFIDENCE,
    show_default=True,
    help="Minimum YOLO confidence for cat detections.",
)
@click.option(
    "--padding",
    type=click.FloatRange(min=0.0),
    default=DEFAULT_PADDING,
    show_default=True,
    help="Relative padding added to each side of the bbox before cropping.",
)
@click.option(
    "--min-size",
    type=click.IntRange(min=1),
    default=DEFAULT_MIN_SIZE,
    show_default=True,
    help="Skip bboxes whose width or height are below this size (pixels).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="YOLO inference batch size.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N frames after sorting.",
)
def main(
    confidence: float,
    padding: float,
    min_size: int,
    batch_size: int,
    limit: int | None,
) -> None:
    """Auto-crop cats from extracted frames using YOLO."""

    run_auto_crop_cats(
        confidence=confidence,
        padding=padding,
        min_size=min_size,
        batch_size=batch_size,
        limit=limit,
    )


def run_auto_crop_cats(
    confidence: float = DEFAULT_CONFIDENCE,
    padding: float = DEFAULT_PADDING,
    min_size: int = DEFAULT_MIN_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> None:
    from PIL import Image

    connection = connect_db()
    output_dir = DEFAULT_CROPS_DIR / UNSORTED_LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames_from_db(connection, limit=limit)
    if not frames:
        click.echo("auto-crop: no frames in database (is_duplicate = false)")
        return

    detector = YOLO(MODEL_NAME)
    inference_device = select_inference_device()
    detector.to(inference_device)

    click.echo(
        f"auto-crop config: device={inference_device}, frames={len(frames)}, "
        f"batch_size={batch_size}, confidence={confidence}, padding={padding}, min_size={min_size}"
    )

    total_crops = 0
    total_detections = 0
    total_skipped_small = 0
    total_skipped_missing = 0
    crop_inserts: list[list[object]] = []
    started_at = time.perf_counter()

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]
        batch_arrays: list[np.ndarray] = []
        batch_images: list[tuple[FrameRow, Image.Image]] = []

        for frame in batch:
            resolved_path = PROJECT_ROOT / frame.frame_path
            if not resolved_path.exists():
                total_skipped_missing += 1
                continue

            pil_image = Image.open(resolved_path).convert("RGB")
            batch_arrays.append(np.array(pil_image))
            batch_images.append((frame, pil_image))

        if not batch_arrays:
            continue

        results = detector.predict(
            source=batch_arrays,
            verbose=False,
            conf=confidence,
            classes=[COCO_CAT_CLASS_ID],
            device=inference_device,
        )

        for (frame, pil_image), result in zip(batch_images, results, strict=True):
            try:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.tolist()
                confs = boxes.conf.tolist()
                frame_width, frame_height = pil_image.size

                for bbox_index, (coords, box_conf) in enumerate(zip(xyxy, confs)):
                    total_detections += 1
                    x1, y1, x2, y2 = (float(value) for value in coords)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    if bbox_width < min_size or bbox_height < min_size:
                        total_skipped_small += 1
                        continue

                    left, top, right, bottom = apply_padding_bbox(
                        bbox_x=x1,
                        bbox_y=y1,
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        padding=padding,
                    )
                    if right <= left or bottom <= top:
                        total_skipped_small += 1
                        continue

                    crop_image = pil_image.crop((left, top, right, bottom))
                    crop_filename = f"{Path(frame.frame_name).stem}__box_{bbox_index:02d}.jpg"
                    crop_path = output_dir / crop_filename
                    crop_image.save(crop_path, format="JPEG", quality=95)
                    crop_width, crop_height = crop_image.size
                    crop_image.close()

                    crop_uid = make_uid(
                        "crp",
                        frame.frame_uid,
                        bbox_index,
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    )

                    crop_inserts.append(
                        [
                            crop_uid,
                            frame.frame_uid,
                            frame.frame_name,
                            relative_to_project(crop_path),
                            UNSORTED_LABEL,
                            float(box_conf),
                            x1,
                            y1,
                            bbox_width,
                            bbox_height,
                            padding,
                            crop_width,
                            crop_height,
                        ]
                    )
                    total_crops += 1
            finally:
                pil_image.close()

        batch_index = batch_start // batch_size + 1
        batch_last = min(batch_start + batch_size, len(frames))
        click.echo(
            f"auto-crop batch {batch_index}: processed frames "
            f"{batch_start + 1}..{batch_last}"
        )

    if crop_inserts:
        connection.executemany(
            """
            INSERT INTO crops (
                crop_uid,
                frame_uid,
                frame_name,
                crop_path,
                label,
                confidence,
                bbox_x,
                bbox_y,
                bbox_width,
                bbox_height,
                padding,
                crop_width,
                crop_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (crop_uid) DO UPDATE SET
                frame_uid = EXCLUDED.frame_uid,
                frame_name = EXCLUDED.frame_name,
                crop_path = EXCLUDED.crop_path,
                confidence = EXCLUDED.confidence,
                bbox_x = EXCLUDED.bbox_x,
                bbox_y = EXCLUDED.bbox_y,
                bbox_width = EXCLUDED.bbox_width,
                bbox_height = EXCLUDED.bbox_height,
                padding = EXCLUDED.padding,
                crop_width = EXCLUDED.crop_width,
                crop_height = EXCLUDED.crop_height
            """,
            crop_inserts,
        )

    elapsed_seconds = time.perf_counter() - started_at
    click.echo(
        f"auto-crop: saved {total_crops} crops from {total_detections} detections"
    )
    click.echo(
        f"auto-crop: skipped small={total_skipped_small} missing_frames={total_skipped_missing}"
    )
    click.echo(f"auto-crop: output directory: {output_dir}")
    click.echo(f"auto-crop: elapsed {elapsed_seconds:.1f}s")
    print_crop_stats(DEFAULT_CROPS_DIR)


if __name__ == "__main__":
    main()
