#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click

from pipeline_db import DEFAULT_CROPS_DIR, connect_db, print_crop_stats, relative_to_project


UNSORTED_LABEL = "unsorted"


@click.command()
@click.option(
    "--crops-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=DEFAULT_CROPS_DIR,
    show_default=True,
    help="Root directory with crop subfolders.",
)
def main(crops_dir: Path) -> None:
    """Assign cat labels to crops based on their current subfolder."""

    run_assign_labels_from_folders(crops_dir=crops_dir)


def run_assign_labels_from_folders(crops_dir: Path = DEFAULT_CROPS_DIR) -> None:
    connection = connect_db()

    if not crops_dir.exists():
        raise click.ClickException(f"crops directory not found: {crops_dir}")

    existing_rows = connection.execute(
        "SELECT crop_uid, crop_path FROM crops"
    ).fetchall()
    crop_uid_by_basename: dict[str, str] = {}
    for crop_uid, crop_path in existing_rows:
        basename = Path(crop_path).name
        crop_uid_by_basename[basename] = crop_uid

    label_dirs = sorted(
        path
        for path in crops_dir.iterdir()
        if path.is_dir() and path.name != UNSORTED_LABEL
    )
    if not label_dirs:
        click.echo("assign-labels: no label subfolders found")
        return

    updates: list[list[object]] = []
    updated_by_label: dict[str, int] = {}
    unmatched_files = 0

    for label_dir in label_dirs:
        label = label_dir.name
        for crop_file in sorted(label_dir.iterdir()):
            if not crop_file.is_file():
                continue

            crop_uid = crop_uid_by_basename.get(crop_file.name)
            if crop_uid is None:
                unmatched_files += 1
                continue

            updates.append(
                [label, relative_to_project(crop_file), crop_uid]
            )
            updated_by_label[label] = updated_by_label.get(label, 0) + 1

    if updates:
        connection.executemany(
            "UPDATE crops SET label = ?, crop_path = ? WHERE crop_uid = ?",
            updates,
        )

    total_updated = sum(updated_by_label.values())
    click.echo(f"assign-labels: updated {total_updated} crops")
    if unmatched_files:
        click.echo(
            f"assign-labels: {unmatched_files} files without matching db row"
        )
    print_crop_stats(crops_dir)


if __name__ == "__main__":
    main()
