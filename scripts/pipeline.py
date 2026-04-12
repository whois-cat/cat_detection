#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "av>=14.4.0",
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "imagehash>=4.3.0",
#   "pillow>=11.1.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "ultralytics>=8.3.0",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Project pipeline commands."""


@cli.command("prepare")
@click.option(
    "--start-at",
    type=str,
    default=None,
    help="Resume scanning from the specified video file name before running later stages.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the next N videos during scan.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="YOLO inference batch size for scan.",
)
@click.option(
    "--detect-max-side",
    type=click.IntRange(min=64),
    default=512,
    show_default=True,
    help="Longest side used for detection scan frames.",
)
@click.option(
    "--max-seconds-per-video",
    type=float,
    default=None,
    help="Scan only the first N seconds of each video.",
)
@click.option(
    "--dedup-threshold",
    type=click.IntRange(min=0),
    default=6,
    show_default=True,
    help="Maximum Hamming distance between neighbouring pHashes to treat frames as duplicates.",
)
@click.option(
    "--rescan",
    is_flag=True,
    help="Re-process videos that already have detections for this model.",
)
@click.option(
    "--force-frames",
    is_flag=True,
    help="Re-extract frames for intervals that already have them on disk.",
)
def prepare_command(
    start_at: str | None,
    limit: int | None,
    batch_size: int,
    detect_max_side: int,
    max_seconds_per_video: float | None,
    dedup_threshold: int,
    rescan: bool,
    force_frames: bool,
) -> None:
    """Run scan -> intervals -> frames -> deduplicate."""

    from build_cat_intervals import run_build_cat_intervals
    from deduplicate_frames import run_deduplicate_frames
    from extract_interval_frames import run_extract_interval_frames
    from scan_cat_detections import run_scan_cat_detections

    click.echo("scan: start")
    run_scan_cat_detections(
        start_at=start_at,
        limit=limit,
        batch_size=batch_size,
        detect_max_side=detect_max_side,
        max_seconds_per_video=max_seconds_per_video,
        rescan=rescan,
    )
    click.echo("intervals: start")
    run_build_cat_intervals()
    click.echo("frames: start")
    run_extract_interval_frames(force=force_frames)
    click.echo("deduplicate: start")
    run_deduplicate_frames(threshold=dedup_threshold)
    click.echo("prepare: done")


@cli.command("scan")
@click.option(
    "--start-at",
    type=str,
    default=None,
    help="Resume scanning from the specified video file name.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the next N videos after --start-at.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="YOLO inference batch size.",
)
@click.option(
    "--detect-max-side",
    type=click.IntRange(min=64),
    default=512,
    show_default=True,
    help="Longest side used for detection scan frames.",
)
@click.option(
    "--max-seconds-per-video",
    type=float,
    default=None,
    help="Scan only the first N seconds of each video.",
)
@click.option(
    "--rescan",
    is_flag=True,
    help="Re-process videos that already have detections for this model.",
)
def scan_command(
    start_at: str | None,
    limit: int | None,
    batch_size: int,
    detect_max_side: int,
    max_seconds_per_video: float | None,
    rescan: bool,
) -> None:
    """Run only detection scan."""

    from scan_cat_detections import run_scan_cat_detections

    run_scan_cat_detections(
        start_at=start_at,
        limit=limit,
        batch_size=batch_size,
        detect_max_side=detect_max_side,
        max_seconds_per_video=max_seconds_per_video,
        rescan=rescan,
    )


@cli.command("intervals")
def intervals_command() -> None:
    """Run only interval building."""

    from build_cat_intervals import run_build_cat_intervals

    run_build_cat_intervals()


@cli.command("frames")
@click.option(
    "--force",
    is_flag=True,
    help="Re-extract frames for intervals that already have frames on disk.",
)
def frames_command(force: bool) -> None:
    """Run only frame extraction."""

    from extract_interval_frames import run_extract_interval_frames

    run_extract_interval_frames(force=force)


@cli.command("import-annotations")
@click.option(
    "--export-json",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to COCO annotations exported from CVAT.",
)
def import_annotations_command(export_json: Path) -> None:
    """Import CVAT COCO annotations into DuckDB."""

    from import_cvat_annotations import run_import_cvat_annotations

    run_import_cvat_annotations(export_json)


@cli.command("deduplicate")
@click.option(
    "--mode",
    type=click.Choice(["frames", "crops"]),
    default="frames",
    show_default=True,
    help="What to deduplicate: extracted frames or unsorted crops.",
)
@click.option(
    "--threshold",
    type=click.IntRange(min=0),
    default=6,
    show_default=True,
    help="Maximum Hamming distance between neighbouring pHashes to treat as duplicates.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only report counts without modifying files or database.",
)
def deduplicate_command(mode: str, threshold: int, dry_run: bool) -> None:
    """Deduplicate frames or crops using perceptual hashing."""

    from deduplicate_frames import run_deduplicate_crops, run_deduplicate_frames

    if mode == "frames":
        run_deduplicate_frames(threshold=threshold, dry_run=dry_run)
    else:
        run_deduplicate_crops(threshold=threshold, dry_run=dry_run)


@cli.command("export-crops")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Directory for cropped cat images. Defaults to data/crops.",
)
@click.option(
    "--padding",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Relative padding added to each side of the bbox before cropping.",
)
@click.option(
    "--min-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Skip bboxes whose width or height are below this size (pixels).",
)
def export_crops_command(output_dir: Path | None, padding: float, min_size: int) -> None:
    """Export annotated cat crops for classifier training."""

    from export_cat_crops import run_export_cat_crops
    from pipeline_db import DEFAULT_CROPS_DIR

    resolved_output_dir = output_dir if output_dir is not None else DEFAULT_CROPS_DIR
    run_export_cat_crops(
        output_dir=resolved_output_dir,
        padding=padding,
        min_size=min_size,
    )


@cli.command("auto-crop")
@click.option(
    "--confidence",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.3,
    show_default=True,
    help="Minimum YOLO confidence for cat detections.",
)
@click.option(
    "--padding",
    type=click.FloatRange(min=0.0),
    default=0.15,
    show_default=True,
    help="Relative padding added to each side of the bbox before cropping.",
)
@click.option(
    "--min-size",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Skip bboxes whose width or height are below this size (pixels).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="YOLO inference batch size.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N frames after sorting.",
)
def auto_crop_command(
    confidence: float,
    padding: float,
    min_size: int,
    batch_size: int,
    limit: int | None,
) -> None:
    """Auto-crop cats from extracted frames using YOLO."""

    from auto_crop_cats import run_auto_crop_cats

    run_auto_crop_cats(
        confidence=confidence,
        padding=padding,
        min_size=min_size,
        batch_size=batch_size,
        limit=limit,
    )


@cli.command("assign-labels")
@click.option(
    "--crops-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root directory with crop subfolders. Defaults to data/crops.",
)
def assign_labels_command(crops_dir: Path | None) -> None:
    """Assign cat labels to crops based on their current subfolder layout."""

    from assign_labels_from_folders import run_assign_labels_from_folders
    from pipeline_db import DEFAULT_CROPS_DIR

    resolved_crops_dir = crops_dir if crops_dir is not None else DEFAULT_CROPS_DIR
    run_assign_labels_from_folders(crops_dir=resolved_crops_dir)


@cli.command("group-crops")
@click.option(
    "--gap",
    type=float,
    default=10.0,
    show_default=True,
    help="Maximum gap in seconds between neighbouring crops within a cluster.",
)
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Directory with unsorted crops. Defaults to data/crops/unsorted.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Directory to create group subfolders in. Defaults to data/crops/groups.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only report how many groups would be created without moving files.",
)
def group_crops_command(
    gap: float,
    input_dir: Path | None,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Group unsorted crops into time-based clusters."""

    from group_crops import run_group_crops

    run_group_crops(
        gap_seconds=gap,
        input_dir=input_dir,
        output_dir=output_dir,
        dry_run=dry_run,
    )


@cli.command("scatter-groups")
@click.option(
    "--groups-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Directory containing group subfolders. Defaults to data/crops/groups.",
)
@click.option(
    "--crops-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Root directory for per-cat crop folders. Defaults to data/crops.",
)
def scatter_groups_command(groups_dir: Path | None, crops_dir: Path | None) -> None:
    """Move crops from named group folders into per-cat label folders."""

    from scatter_groups import run_scatter_groups

    run_scatter_groups(groups_dir=groups_dir, crops_dir=crops_dir)


@cli.command("train")
@click.option(
    "--crops-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Root directory with per-class crop subfolders. Defaults to data/crops.",
)
@click.option(
    "--epochs",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Maximum number of training epochs.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--lr",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate for the classifier head.",
)
@click.option(
    "--patience",
    type=click.IntRange(min=1),
    default=7,
    show_default=True,
    help="Early stopping patience.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Path to save the best model checkpoint. Defaults to data/models/cat_classifier_best.pt.",
)
def train_command(
    crops_dir: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    output: Path | None,
) -> None:
    """Train a cat identity classifier on labelled crops."""

    from pipeline_db import DEFAULT_CROPS_DIR
    from train_classifier import DEFAULT_OUTPUT, run_train_classifier

    run_train_classifier(
        crops_dir=crops_dir if crops_dir is not None else DEFAULT_CROPS_DIR,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        output=output if output is not None else DEFAULT_OUTPUT,
    )


@cli.command("predict")
@click.option(
    "--model",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="Path to the trained model checkpoint. Defaults to data/models/cat_classifier_best.pt.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to an image or a directory of images.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="Minimum confidence to accept a prediction (below = unknown).",
)
def predict_command(model: Path | None, input_path: Path, threshold: float) -> None:
    """Predict cat identity on image(s) using a trained classifier."""

    from predict_cat import DEFAULT_MODEL_PATH, run_predict_cat

    run_predict_cat(
        model_path=model if model is not None else DEFAULT_MODEL_PATH,
        input_path=input_path,
        threshold=threshold,
    )


@cli.command("live")
@click.option("--source", required=True, help="RTSP URL or camera device index.")
@click.option(
    "--model",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=None,
    help="Path to the trained classifier checkpoint. Defaults to data/models/cat_classifier_best.pt.",
)
@click.option(
    "--yolo",
    type=str,
    default="yolov8n.pt",
    show_default=True,
    help="YOLO model name or path.",
)
@click.option(
    "--interval",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Seconds between detection checks.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.6,
    show_default=True,
    help="Minimum classifier confidence to accept a prediction.",
)
@click.option("--webhook", type=str, default=None, help="POST JSON to this URL on match.")
@click.option("--show", is_flag=True, help="Show video window with bboxes.")
@click.option("--save-log", is_flag=True, help="Append detections to DuckDB.")
def live_command(
    source: str,
    model: Path | None,
    yolo: str,
    interval: float,
    threshold: float,
    webhook: str | None,
    show: bool,
    save_log: bool,
) -> None:
    """Live cat detection from camera."""

    from live_detect import DEFAULT_CLASSIFIER, run_live_detect

    run_live_detect(
        source=source,
        model_path=model if model is not None else DEFAULT_CLASSIFIER,
        yolo_model=yolo,
        interval=interval,
        threshold=threshold,
        webhook=webhook,
        show=show,
        save_log=save_log,
    )


if __name__ == "__main__":
    cli()
