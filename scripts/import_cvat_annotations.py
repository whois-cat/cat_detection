#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
# ]
# ///

from __future__ import annotations

import json
from pathlib import Path

import click

from pipeline_db import connect_db, make_uid, relative_to_project


@click.command()
@click.option(
    "--export-json",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to COCO annotations exported from CVAT.",
)
def main(
    export_json: Path,
) -> None:
    """Link CVAT COCO annotations back to extracted frames in DuckDB."""

    run_import_cvat_annotations(export_json)


def run_import_cvat_annotations(export_json: Path) -> None:
    connection = connect_db()
    export_path = relative_to_project(export_json)
    task_name = export_json.parent.name
    connection.execute("DELETE FROM annotations WHERE export_path = ?", [export_path])

    with export_json.open("r", encoding="utf-8") as input_file:
        export_payload = json.load(input_file)

    categories = {
        int(category["id"]): str(category["name"])
        for category in export_payload.get("categories", [])
    }
    image_rows = export_payload.get("images", [])
    annotation_rows = export_payload.get("annotations", [])

    frame_lookup = {
        frame_name: frame_uid
        for frame_name, frame_uid in connection.execute(
            "SELECT frame_name, frame_uid FROM frames"
        ).fetchall()
    }

    image_lookup: dict[int, tuple[str, str] | None] = {}
    matched_frames = 0
    missing_frames = 0

    for image in image_rows:
        image_id = int(image["id"])
        image_name = Path(str(image["file_name"])).name
        frame_uid = frame_lookup.get(image_name)

        if frame_uid is None:
            image_lookup[image_id] = None
            missing_frames += 1
            continue

        image_lookup[image_id] = (frame_uid, image_name)
        matched_frames += 1

        if task_name is not None:
            connection.execute(
                "UPDATE frames SET cvat_task_name = ? WHERE frame_uid = ?",
                [task_name, frame_uid],
            )

    imported_annotations = 0
    skipped_annotations = 0

    for annotation in annotation_rows:
        image_id = int(annotation["image_id"])
        image_link = image_lookup.get(image_id)
        if image_link is None:
            skipped_annotations += 1
            continue

        frame_uid, image_name = image_link
        bbox = annotation.get("bbox", [None, None, None, None])
        category_name = categories.get(int(annotation["category_id"]), "unknown")

        connection.execute(
            """
            INSERT INTO annotations (
                annotation_uid,
                frame_uid,
                image_name,
                label,
                bbox_x,
                bbox_y,
                bbox_width,
                bbox_height,
                area,
                is_crowd,
                cvat_task_name,
                export_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (annotation_uid) DO UPDATE SET
                frame_uid = EXCLUDED.frame_uid,
                image_name = EXCLUDED.image_name,
                label = EXCLUDED.label,
                bbox_x = EXCLUDED.bbox_x,
                bbox_y = EXCLUDED.bbox_y,
                bbox_width = EXCLUDED.bbox_width,
                bbox_height = EXCLUDED.bbox_height,
                area = EXCLUDED.area,
                is_crowd = EXCLUDED.is_crowd,
                cvat_task_name = EXCLUDED.cvat_task_name,
                export_path = EXCLUDED.export_path
            """,
            [
                make_uid("ann", export_path, annotation["id"]),
                frame_uid,
                image_name,
                category_name,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                annotation.get("area"),
                bool(annotation.get("iscrowd", 0)),
                task_name,
                export_path,
            ],
        )
        imported_annotations += 1

    click.echo(f"matched frames: {matched_frames}")
    click.echo(f"missing frames: {missing_frames}")
    click.echo(f"imported annotations: {imported_annotations}")
    click.echo(f"skipped annotations: {skipped_annotations}")


if __name__ == "__main__":
    main()
