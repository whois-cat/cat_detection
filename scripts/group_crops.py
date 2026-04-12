#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
# ]
# ///

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import click

from pipeline_db import DEFAULT_CROPS_DIR


UNSORTED_LABEL = "unsorted"
DEFAULT_GAP_SECONDS = 10.0
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class CropFileInfo:
    path: Path
    video_stem: str
    timestamp_ms: int


def parse_crop_filename(path: Path) -> CropFileInfo | None:
    stem = path.stem
    match = re.match(r"^(.+?)__ts_(\d+)__box_\d+$", stem)
    if match is None:
        return None
    return CropFileInfo(
        path=path,
        video_stem=match.group(1),
        timestamp_ms=int(match.group(2)),
    )


def cluster_by_gap(
    crops: list[CropFileInfo],
    gap_ms: int,
) -> list[list[CropFileInfo]]:
    if not crops:
        return []

    clusters: list[list[CropFileInfo]] = [[crops[0]]]
    for crop in crops[1:]:
        if crop.timestamp_ms - clusters[-1][-1].timestamp_ms <= gap_ms:
            clusters[-1].append(crop)
        else:
            clusters.append([crop])

    return clusters


@click.command()
@click.option(
    "--gap",
    type=float,
    default=DEFAULT_GAP_SECONDS,
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
def main(
    gap: float,
    input_dir: Path | None,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Group unsorted crops into time-based clusters."""

    resolved_input = input_dir if input_dir is not None else DEFAULT_CROPS_DIR / UNSORTED_LABEL
    resolved_output = output_dir if output_dir is not None else DEFAULT_CROPS_DIR / "groups"
    run_group_crops(
        gap_seconds=gap,
        input_dir=resolved_input,
        output_dir=resolved_output,
        dry_run=dry_run,
    )


def run_group_crops(
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    resolved_input = input_dir if input_dir is not None else DEFAULT_CROPS_DIR / UNSORTED_LABEL
    resolved_output = output_dir if output_dir is not None else DEFAULT_CROPS_DIR / "groups"

    if not resolved_input.exists():
        raise click.ClickException(f"input directory not found: {resolved_input}")

    all_files = sorted(
        path for path in resolved_input.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not all_files:
        click.echo("group-crops: no crop files found")
        return

    parsed: list[CropFileInfo] = []
    skipped_parse = 0
    for path in all_files:
        info = parse_crop_filename(path)
        if info is None:
            skipped_parse += 1
            continue
        parsed.append(info)

    if not parsed:
        click.echo("group-crops: no parseable crop filenames found")
        return

    by_video: dict[str, list[CropFileInfo]] = {}
    for info in parsed:
        by_video.setdefault(info.video_stem, []).append(info)
    for crops in by_video.values():
        crops.sort(key=lambda c: c.timestamp_ms)

    gap_ms = int(gap_seconds * 1000)
    all_clusters: list[list[CropFileInfo]] = []
    for video_stem in sorted(by_video):
        clusters = cluster_by_gap(by_video[video_stem], gap_ms)
        all_clusters.extend(clusters)

    cluster_sizes = [len(c) for c in all_clusters]
    min_size = min(cluster_sizes) if cluster_sizes else 0
    max_size = max(cluster_sizes) if cluster_sizes else 0
    avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0

    click.echo(
        f"group-crops: {len(all_clusters)} groups from {len(parsed)} crops "
        f"(gap={gap_seconds}s)"
    )
    click.echo(f"group-crops: crops per group: min={min_size} max={max_size} avg={avg_size:.1f}")
    if skipped_parse:
        click.echo(f"group-crops: skipped unparseable filenames: {skipped_parse}")

    if dry_run:
        click.echo("group-crops: dry run, no files moved")
        return

    resolved_output.mkdir(parents=True, exist_ok=True)

    for group_index, cluster in enumerate(all_clusters, start=1):
        group_name = f"group_{group_index:03d}"
        group_dir = resolved_output / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for crop_info in cluster:
            dest = group_dir / crop_info.path.name
            shutil.move(str(crop_info.path), str(dest))

        preview_src = group_dir / cluster[0].path.name
        preview_dst = group_dir / "_preview.jpg"
        shutil.copy2(str(preview_src), str(preview_dst))

    click.echo(f"group-crops: created {len(all_clusters)} group folders in {resolved_output}")


if __name__ == "__main__":
    main()
