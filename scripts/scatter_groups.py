#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
# ]
# ///

from __future__ import annotations

import shutil
from pathlib import Path

import click

from pipeline_db import DEFAULT_CROPS_DIR, print_crop_stats


PREVIEW_FILENAME = "_preview.jpg"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESERVED_LABELS = {"unsorted", "groups"}


def discover_cat_labels(crops_dir: Path) -> list[str]:
    if not crops_dir.exists():
        return []
    return sorted(
        child.name
        for child in crops_dir.iterdir()
        if child.is_dir() and child.name not in RESERVED_LABELS
    )


def match_label(folder_name: str, cat_labels: list[str]) -> str | None:
    lowered = folder_name.lower()
    matches = [label for label in cat_labels if label.lower() in lowered]
    if not matches:
        return None
    return max(matches, key=len)


@click.command()
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
def main(groups_dir: Path | None, crops_dir: Path | None) -> None:
    """Move crops from named group folders into per-cat label folders."""

    resolved_groups = groups_dir if groups_dir is not None else DEFAULT_CROPS_DIR / "groups"
    resolved_crops = crops_dir if crops_dir is not None else DEFAULT_CROPS_DIR
    run_scatter_groups(groups_dir=resolved_groups, crops_dir=resolved_crops)


def run_scatter_groups(
    groups_dir: Path | None = None,
    crops_dir: Path | None = None,
) -> None:
    resolved_groups = groups_dir if groups_dir is not None else DEFAULT_CROPS_DIR / "groups"
    resolved_crops = crops_dir if crops_dir is not None else DEFAULT_CROPS_DIR

    if not resolved_groups.exists():
        raise click.ClickException(f"groups directory not found: {resolved_groups}")

    cat_labels = discover_cat_labels(resolved_crops)
    if not cat_labels:
        click.echo(f"scatter-groups: no cat label folders found in {resolved_crops}")
        return

    click.echo(f"scatter-groups: known cat labels: {cat_labels}")

    subfolders = sorted(
        folder for folder in resolved_groups.iterdir() if folder.is_dir()
    )

    moved_by_label: dict[str, int] = {}
    removed_dirs: list[str] = []
    skipped_dirs: list[str] = []

    for folder in subfolders:
        label = match_label(folder.name, cat_labels)
        if label is None:
            skipped_dirs.append(folder.name)
            continue

        dest_dir = resolved_crops / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for child in sorted(folder.iterdir()):
            if not child.is_file():
                continue
            if child.name == PREVIEW_FILENAME:
                continue
            if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            dest_path = dest_dir / child.name
            shutil.move(str(child), str(dest_path))
            moved_count += 1

        moved_by_label[label] = moved_by_label.get(label, 0) + moved_count

        remaining = [
            item for item in folder.iterdir()
            if item.name != PREVIEW_FILENAME
        ]
        if not remaining:
            preview = folder / PREVIEW_FILENAME
            if preview.exists():
                preview.unlink()
            folder.rmdir()
            removed_dirs.append(folder.name)

    total_moved = sum(moved_by_label.values())
    click.echo(f"scatter-groups: moved {total_moved} crops")
    if removed_dirs:
        click.echo(f"scatter-groups: removed {len(removed_dirs)} empty folders")
    if skipped_dirs:
        click.echo(f"scatter-groups: skipped {len(skipped_dirs)} folders without label match")
    print_crop_stats(resolved_crops)


if __name__ == "__main__":
    main()
