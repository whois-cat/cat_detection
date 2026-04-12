#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "pillow>=11.1.0",
# ]
# ///

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import click

from pipeline_db import (
    DEFAULT_CROPS_DIR,
    PROJECT_ROOT,
    connect_db,
    make_uid,
    relative_to_project,
)


DEFAULT_PADDING = 0.1
DEFAULT_MIN_SIZE = 32


@dataclass(frozen=True)
class AnnotationRow:
    annotation_uid: str
    frame_uid: str
    label: str
    frame_path: str
    frame_name: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", label.strip(), flags=re.UNICODE)
    return cleaned or "unknown"


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


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_CROPS_DIR,
    show_default=True,
    help="Directory for cropped cat images.",
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
def main(output_dir: Path, padding: float, min_size: int) -> None:
    """Export annotated cat crops into per-label folders for classifier training."""

    run_export_cat_crops(output_dir=output_dir, padding=padding, min_size=min_size)


def run_export_cat_crops(
    output_dir: Path = DEFAULT_CROPS_DIR,
    padding: float = DEFAULT_PADDING,
    min_size: int = DEFAULT_MIN_SIZE,
) -> None:
    from PIL import Image

    connection = connect_db()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = connection.execute(
        """
        SELECT a.annotation_uid,
               a.frame_uid,
               a.label,
               f.frame_path,
               f.frame_name,
               a.bbox_x,
               a.bbox_y,
               a.bbox_width,
               a.bbox_height
        FROM annotations a
        JOIN frames f ON f.frame_uid = a.frame_uid
        WHERE a.bbox_x IS NOT NULL
          AND a.bbox_y IS NOT NULL
          AND a.bbox_width IS NOT NULL
          AND a.bbox_height IS NOT NULL
        ORDER BY a.label, f.frame_name, a.annotation_uid
        """
    ).fetchall()

    if not rows:
        click.echo("export-crops: no annotations with bbox found")
        return

    annotations = [
        AnnotationRow(
            annotation_uid=row[0],
            frame_uid=row[1],
            label=row[2],
            frame_path=row[3],
            frame_name=row[4],
            bbox_x=float(row[5]),
            bbox_y=float(row[6]),
            bbox_width=float(row[7]),
            bbox_height=float(row[8]),
        )
        for row in rows
    ]

    saved_crops = 0
    skipped_small = 0
    skipped_missing = 0
    label_counts: dict[str, int] = {}
    crop_inserts: list[list[object]] = []

    for annotation in annotations:
        if annotation.bbox_width < min_size or annotation.bbox_height < min_size:
            skipped_small += 1
            continue

        resolved_frame_path = PROJECT_ROOT / annotation.frame_path
        if not resolved_frame_path.exists():
            skipped_missing += 1
            continue

        label_dir_name = sanitize_label(annotation.label)
        label_dir = output_dir / label_dir_name
        label_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f"{Path(annotation.frame_name).stem}__{annotation.annotation_uid}.jpg"
        crop_path = label_dir / crop_filename

        with Image.open(resolved_frame_path) as frame_image:
            frame_image_rgb = frame_image.convert("RGB")
            frame_width, frame_height = frame_image_rgb.size
            left, top, right, bottom = apply_padding_bbox(
                bbox_x=annotation.bbox_x,
                bbox_y=annotation.bbox_y,
                bbox_width=annotation.bbox_width,
                bbox_height=annotation.bbox_height,
                frame_width=frame_width,
                frame_height=frame_height,
                padding=padding,
            )

            if right <= left or bottom <= top:
                skipped_small += 1
                continue

            crop_image = frame_image_rgb.crop((left, top, right, bottom))
            crop_image.save(crop_path, format="JPEG", quality=95)
            crop_width, crop_height = crop_image.size

        crop_uid = make_uid("crp", annotation.annotation_uid, padding)
        crop_inserts.append(
            [
                crop_uid,
                annotation.annotation_uid,
                annotation.frame_uid,
                annotation.label,
                relative_to_project(crop_path),
                annotation.bbox_x,
                annotation.bbox_y,
                annotation.bbox_width,
                annotation.bbox_height,
                padding,
                crop_width,
                crop_height,
            ]
        )

        saved_crops += 1
        label_counts[label_dir_name] = label_counts.get(label_dir_name, 0) + 1

    if crop_inserts:
        connection.executemany(
            """
            INSERT OR REPLACE INTO crops (
                crop_uid,
                annotation_uid,
                frame_uid,
                label,
                crop_path,
                bbox_x,
                bbox_y,
                bbox_width,
                bbox_height,
                padding,
                crop_width,
                crop_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            crop_inserts,
        )

    click.echo(f"export-crops: saved {saved_crops} crops")
    click.echo(f"export-crops: skipped small={skipped_small} missing_frames={skipped_missing}")
    for label_dir_name, count in sorted(label_counts.items()):
        click.echo(f"  {label_dir_name}: {count}")
    click.echo(f"export-crops: output directory: {output_dir}")


if __name__ == "__main__":
    main()
