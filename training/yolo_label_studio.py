"""Turnkey Label Studio bridge for YOLO box review.

Goal: the operator never thinks about COCO, storage config or task formats. One
command pushes collected frames (with the model's boxes pre-filled) into a Label
Studio project; one command pulls the human-corrected boxes straight back into
the catalog ``annotations`` table that ``yolo_build_version`` reads.

Images are served by Label Studio directly from the catalog folder (local-files
serving), so nothing is copied. A submitted annotation with no boxes is a
human-confirmed empty frame (negative); a task never submitted stays unreviewed.

The pure conversions (box <-> Label Studio percent geometry, task/annotation
shapes) are split out and unit-tested; the SDK-touching orchestration is thin.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from training.streamhub_dataset import open_catalog

LABEL_NAME = "cat"
PROJECT_TITLE = "cat-detection YOLO boxes"
MODEL_VERSION = "yolo-suggestion"
# Must sit under LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT in docker-compose.label.yml;
# every catalog frame lives under images/ (streamhub_dataset).
LOCAL_STORAGE_PATH = "/label-studio/files/images"


def build_label_config() -> str:
    """Single-class rectangle labelling config."""
    return (
        '<View>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f'    <Label value="{LABEL_NAME}" background="#FF4136"/>\n'
        '  </RectangleLabels>\n'
        '</View>\n'
    )


def image_url(image_relpath: str) -> str:
    """Local-files URL served by Label Studio from the dataset document root."""
    return f"/data/local-files/?d={image_relpath}"


def prediction_results(dets: Iterable[dict]) -> list[dict]:
    """Model boxes (model-orientation fractions) -> Label Studio rectangle results."""
    results = []
    for det in dets or []:
        try:
            x, y, w, h = (float(value) for value in det["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        results.append({
            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
            "value": {
                "x": max(0.0, x * 100), "y": max(0.0, y * 100),
                "width": w * 100, "height": h * 100,
                "rotation": 0, "rectanglelabels": [LABEL_NAME],
            },
        })
    return results


def sample_task(sample: dict, dets: Iterable[dict]) -> dict:
    """One Label Studio task: the frame plus the model's boxes as a prediction."""
    task: dict[str, Any] = {
        "data": {
            "image": image_url(sample["image_relpath"]),
            "sample_id": sample["sample_id"],
            "camera": sample["camera"],
            "wall_ms": sample["wall_ms"],
            "reasons": sample.get("reasons"),
        }
    }
    results = prediction_results(dets)
    task["predictions"] = [{"model_version": MODEL_VERSION, "result": results}]
    return task


def annotation_boxes(result: Iterable[dict], width: int, height: int) -> list[tuple]:
    """Label Studio rectangle results -> pixel ``(x, y, w, h)`` boxes for the catalog."""
    boxes = []
    for item in result or []:
        if item.get("type") not in (None, "rectanglelabels"):
            continue
        value = item.get("value") or {}
        try:
            x = float(value["x"]) / 100 * width
            y = float(value["y"]) / 100 * height
            w = float(value["width"]) / 100 * width
            h = float(value["height"]) / 100 * height
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((x, y, w, h))
    return boxes


# --------------------------------------------------------------------------- #
# SDK-touching orchestration (thin; imports Label Studio lazily).
# --------------------------------------------------------------------------- #

def _connect(url: str, api_key: str):
    try:
        from label_studio_sdk import Client
    except ImportError as exc:  # pragma: no cover - env guard
        raise SystemExit(
            "label-studio-sdk is not installed. Run via the label extra:\n"
            "  uv run --project training --extra label python -m training.yolo_label_studio ..."
        ) from exc
    if not api_key:
        raise SystemExit(
            "LABEL_STUDIO_API_KEY is empty. Open the Label Studio URL, log in, then copy "
            "Account & Settings -> Access Token into .env as LABEL_STUDIO_API_KEY."
        )
    client = Client(url=url, api_key=api_key)
    client.check_connection()
    return client


def _ensure_project(client):
    for project in client.get_projects():
        if project.get_params().get("title") == PROJECT_TITLE:
            break
    else:
        project = client.start_project(title=PROJECT_TITLE, label_config=build_label_config())
    _ensure_local_storage(project)
    return project


def _ensure_local_storage(project) -> None:
    # Label Studio refuses /data/local-files/ URLs unless the project has a Local
    # Storage whose path covers the file. It is never synced: tasks are imported
    # by push() with their own URLs, the storage only authorises serving.
    storages = project.make_request(
        "GET", "/api/storages/localfiles", params={"project": project.id}
    ).json()
    if any(LOCAL_STORAGE_PATH.startswith(s.get("path", "").rstrip("/") or "\0")
           for s in storages):
        return
    project.make_request("POST", "/api/storages/localfiles", json={
        "project": project.id, "path": LOCAL_STORAGE_PATH, "title": "catalog frames",
        "use_blob_urls": True, "regex_filter": "",
    })


def _pending_tasks(conn) -> list[dict]:
    # Everything not yet human-verified, so frames left in 'exported' by an
    # earlier CVAT export are picked up too (not only fresh 'unreviewed' ones).
    tasks = []
    for row in conn.execute("SELECT * FROM samples WHERE status!='verified' ORDER BY wall_ms"):
        pred = conn.execute(
            "SELECT detections_json FROM predictions WHERE sample_id=?", (row["sample_id"],)
        ).fetchone()
        dets = json.loads(pred[0]) if pred and pred[0] else []
        tasks.append(sample_task(dict(row), dets))
    return tasks


def push(url: str, api_key: str, catalog: Path, limit: int | None = None) -> dict:
    """Create/refresh the project and import unreviewed frames (skip already pushed)."""
    client = _connect(url, api_key)
    project = _ensure_project(client)
    existing = {
        task["data"].get("sample_id")
        for task in project.get_tasks(only_ids=False)
    }
    conn = open_catalog(catalog)
    try:
        tasks = [t for t in _pending_tasks(conn)
                 if t["data"]["sample_id"] not in existing]
    finally:
        conn.close()
    if limit is not None:
        tasks = tasks[:limit]
    if tasks:
        project.import_tasks(tasks)
    return {"project_id": project.id, "pushed": len(tasks),
            "already_present": len(existing)}


def pull(url: str, api_key: str, catalog: Path) -> dict:
    """Write submitted annotations back into the catalog (verified; empty=negative)."""
    client = _connect(url, api_key)
    project = _ensure_project(client)
    conn = open_catalog(catalog)
    now = int(time.time() * 1000)
    updated = objects = negatives = 0
    try:
        for task in project.get_labeled_tasks(only_ids=False):
            annotations = [a for a in task.get("annotations") or []
                           if not a.get("was_cancelled")]
            if not annotations:
                continue
            sid = task["data"].get("sample_id")
            row = conn.execute(
                "SELECT width,height,status FROM samples WHERE sample_id=?", (sid,)
            ).fetchone()
            if row is None:
                continue
            boxes = annotation_boxes(annotations[-1].get("result"), row["width"], row["height"])
            conn.execute("DELETE FROM annotations WHERE sample_id=?", (sid,))
            for index, (x, y, w, h) in enumerate(boxes, 1):
                conn.execute(
                    "INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?)",
                    (sid, index, LABEL_NAME, x, y, w, h, "label_studio", now),
                )
            conn.execute(
                "UPDATE samples SET status='verified',protected=1 WHERE sample_id=?", (sid,)
            )
            updated += 1
            objects += len(boxes)
            negatives += int(not boxes)
        conn.commit()
    finally:
        conn.close()
    return {"verified": updated, "objects": objects, "confirmed_empty": negatives}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/yolo_dataset/catalog.sqlite3"))
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("LABEL_STUDIO_API_KEY", ""))
    sub = parser.add_subparsers(dest="command", required=True)
    push_cmd = sub.add_parser("push", help="import unreviewed frames with model suggestions")
    push_cmd.add_argument("--limit", type=int, default=None)
    sub.add_parser("pull", help="write submitted boxes back into the catalog")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "push":
        result = push(args.url, args.api_key, args.catalog, args.limit)
    else:
        result = pull(args.url, args.api_key, args.catalog)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
