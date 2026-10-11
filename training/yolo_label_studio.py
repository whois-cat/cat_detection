"""Turnkey Label Studio bridge: where the cats are and which cat each one is.

Goal: the operator never thinks about COCO, storage config or task formats. One
command pushes collected frames into a Label Studio project with the live
models' answers pre-filled (the detector's boxes and, per box, the classifier's
cat name, both from the streamhub sidecars); one command pulls the
human-corrected boxes and names straight back into the catalog ``annotations``
table, which feeds both the detector (``yolo_build_version``) and the identity
classifier (``catalog_crops``). Model answers are only suggestions: nothing is
trained on until a person submits the frame.

Frames the models are least sure about come first in the queue. A push can send
a Telegram message with how many frames wait (``notify``).

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

from training.streamhub_dataset import box_iou, open_catalog

LABEL_NAME = "cat"
NAME_CONTROL = "cat_name"
UNKNOWN_NAME = "unknown"       # a cat, but not sure which one: not trained on
PROJECT_TITLE = "cat-detection YOLO boxes"
# Tasks whose suggestion is older than this are refreshed while still unlabelled.
MODEL_VERSION = "boxes+names"
# A suggested name below this classifier probability puts the frame up front.
NAME_UNSURE_BELOW = 0.8
# Must sit under LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT in docker-compose.label.yml;
# every catalog frame lives under images/ (streamhub_dataset).
LOCAL_STORAGE_PATH = "/label-studio/files/images"


def name_choices(cat_names: Iterable[str]) -> list[str]:
    """The cat names offered per box: the configured cats, then ``unknown``."""
    names: list[str] = []
    for name in cat_names:
        name = name.strip()
        if name and name != UNKNOWN_NAME and name not in names:
            names.append(name)
    if not names:
        raise SystemExit("no cat names: set CAT_NAMES=name1,name2,... in .env")
    return names + [UNKNOWN_NAME]


def build_label_config(cat_names: Iterable[str]) -> str:
    """One `cat` rectangle class; every box also gets a (required) cat name,
    picked with keys 1..n while the box is selected."""
    choices = "".join(
        f'      <Choice value="{name}" hotkey="{i}"/>\n'
        for i, name in enumerate(name_choices(cat_names), 1)
    )
    return (
        '<View>\n'
        '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f'    <Label value="{LABEL_NAME}" background="#FF4136" hotkey="c"/>\n'
        '  </RectangleLabels>\n'
        '  <View visibleWhen="region-selected">\n'
        '    <Header value="Which cat?"/>\n'
        f'    <Choices name="{NAME_CONTROL}" toName="image" perRegion="true" required="true"\n'
        '             choice="single-radio" showInline="true">\n'
        f'{choices}'
        '    </Choices>\n'
        '  </View>\n'
        '</View>\n'
    )


def image_url(image_relpath: str) -> str:
    """Local-files URL served by Label Studio from the dataset document root."""
    return f"/data/local-files/?d={image_relpath}"


def suggested_name(cats: dict | None, names: Iterable[str]) -> tuple[str, float] | None:
    """The classifier's most likely cat among ``names``, with its probability."""
    scored = [(float(p), n) for n, p in (cats or {}).items() if n in set(names)]
    if not scored:
        return None
    prob, name = max(scored)
    return name, prob


def prediction_results(dets: Iterable[dict], names: Iterable[str] = ()) -> list[dict]:
    """Model boxes (model-orientation fractions) -> Label Studio rectangle
    results, each with the classifier's suggested cat name when it has one."""
    names = list(names)
    results = []
    for det in dets or []:
        try:
            x, y, w, h = (float(value) for value in det["box"])
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        region_id = f"r{len(results)}"
        geometry = {"x": max(0.0, x * 100), "y": max(0.0, y * 100),
                    "width": w * 100, "height": h * 100, "rotation": 0}
        results.append({
            "id": region_id, "from_name": "label", "to_name": "image",
            "type": "rectanglelabels", "value": {**geometry, "rectanglelabels": [LABEL_NAME]},
        })
        suggestion = suggested_name(det.get("cats"), names)
        if suggestion is not None:
            results.append({
                "id": region_id, "from_name": NAME_CONTROL, "to_name": "image",
                "type": "choices", "value": {**geometry, "choices": [suggestion[0]]},
            })
    return results


# Review order: frames the models are most likely wrong about first. Lower =
# sooner. name_unsure: the classifier is unsure which cat a box is. names_only:
# boxes already verified before names existed, only the names need a look.
REASON_PRIORITY = {
    "low_confidence": 0, "name_unsure": 1, "multiple_boxes": 2, "visual_change": 3,
    "visit_sample": 4, "regular": 5, "names_only": 6,
}
URGENT_PRIORITY = REASON_PRIORITY["multiple_boxes"]   # "needs a look" in notifications


def sample_reasons(sample: dict) -> list[str]:
    try:
        return list(json.loads(sample.get("reasons_json") or "[]"))
    except (TypeError, ValueError):
        return []


def task_priority(reasons: Iterable[str]) -> int:
    """Best (lowest) priority among a frame's selection reasons."""
    return min((REASON_PRIORITY.get(r, len(REASON_PRIORITY)) for r in reasons),
               default=len(REASON_PRIORITY))


def name_unsure(dets: Iterable[dict], names: Iterable[str]) -> bool:
    names = list(names)
    for det in dets or []:
        suggestion = suggested_name(det.get("cats"), names)
        if suggestion is not None and suggestion[1] < NAME_UNSURE_BELOW:
            return True
    return False


def sample_task(sample: dict, dets: Iterable[dict], names: Iterable[str] = (),
                extra_reasons: Iterable[str] = ()) -> dict:
    """One Label Studio task: the frame plus the models' boxes and names as a
    prediction."""
    names = list(names)
    reasons = sample_reasons(sample) + list(extra_reasons)
    if name_unsure(dets, names):
        reasons.append("name_unsure")
    task: dict[str, Any] = {
        "data": {
            "image": image_url(sample["image_relpath"]),
            "sample_id": sample["sample_id"],
            "camera": sample["camera"],
            "wall_ms": sample["wall_ms"],
            "reasons": ",".join(reasons),
            "priority": task_priority(reasons),
        }
    }
    results = prediction_results(dets, names)
    task["predictions"] = [{"model_version": MODEL_VERSION, "result": results}]
    return task


def annotation_boxes(result: Iterable[dict], width: int, height: int) -> list[tuple]:
    """Label Studio results -> pixel ``(x, y, w, h, cat_name)`` boxes for the
    catalog; ``cat_name`` is None when the box was not named."""
    result = list(result or [])
    names = {
        item.get("id"): (item.get("value") or {}).get("choices", [None])[0]
        for item in result if item.get("from_name") == NAME_CONTROL
    }
    boxes = []
    for item in result:
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
        boxes.append((x, y, w, h, names.get(item.get("id"))))
    return boxes


def verified_dets(boxes: Iterable[tuple], width: int, height: int,
                  model_dets: Iterable[dict]) -> list[dict]:
    """A verified frame's own boxes as prediction dets (fractions), each with the
    cat probabilities of the model box it overlaps most (IoU >= 0.5), so frames
    reviewed before names existed get their names suggested too."""
    model_dets = [d for d in model_dets or [] if d.get("cats") and len(d.get("box") or ()) == 4]
    dets = []
    for x, y, w, h in boxes:
        box = (x / width, y / height, w / width, h / height)
        best = max(model_dets, key=lambda d: box_iou(box, tuple(map(float, d["box"]))), default=None)
        cats = best["cats"] if best and box_iou(box, tuple(map(float, best["box"]))) >= 0.5 else {}
        dets.append({"box": list(box), "score": 1.0, "cats": cats})
    return dets


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


def _ensure_project(client, cat_names: Iterable[str] | None = None):
    """The project, created if missing; with ``cat_names``, its labelling config
    is brought up to date (e.g. a cat added to CAT_NAMES)."""
    config = build_label_config(cat_names) if cat_names is not None else None
    for project in client.get_projects():
        if project.get_params().get("title") == PROJECT_TITLE:
            if config is not None and project.get_params().get("label_config") != config:
                project.set_params(label_config=config)
            break
    else:
        if config is None:
            raise SystemExit("no Label Studio project yet: run `just detector-label` first")
        project = client.start_project(title=PROJECT_TITLE, label_config=config)
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


def _model_dets(conn, sample_id: str) -> list[dict]:
    pred = conn.execute(
        "SELECT detections_json FROM predictions WHERE sample_id=?", (sample_id,)
    ).fetchone()
    return json.loads(pred[0]) if pred and pred[0] else []


def _pending_tasks(conn, names: list[str]) -> list[dict]:
    # Everything not yet human-verified, so frames left in 'exported' by an
    # earlier CVAT export are picked up too (not only fresh 'unreviewed' ones).
    tasks = [sample_task(dict(row), _model_dets(conn, row["sample_id"]), names)
             for row in conn.execute(
                 "SELECT * FROM samples WHERE status IN ('unreviewed','exported') ORDER BY wall_ms")]
    tasks.sort(key=lambda t: t["data"]["priority"])   # stable: chronological within
    return tasks


def _unnamed_tasks(conn, names: list[str]) -> list[dict]:
    """Verified frames with a box nobody has named yet (reviewed before names
    existed): their own boxes, with the classifier's names suggested."""
    tasks = []
    for row in conn.execute(
        """SELECT * FROM samples WHERE status='verified' AND sample_id IN (
             SELECT sample_id FROM annotations WHERE cat_name IS NULL) ORDER BY wall_ms"""
    ):
        boxes = [tuple(b) for b in conn.execute(
            "SELECT x,y,w,h FROM annotations WHERE sample_id=? ORDER BY annotation_id",
            (row["sample_id"],))]
        dets = verified_dets(boxes, row["width"], row["height"],
                             _model_dets(conn, row["sample_id"]))
        tasks.append(sample_task(dict(row), dets, names, extra_reasons=["names_only"]))
    return tasks


def _is_labelled(task: dict) -> bool:
    return bool(task.get("total_annotations") or task.get("annotations"))


def _suggestion_version(task: dict) -> str | None:
    predictions = task.get("predictions") or []
    return predictions[-1].get("model_version") if predictions else None


def _backup(tasks: list[dict], catalog: Path) -> Path:
    """Label Studio's copy of tasks about to be deleted (the catalog stays the
    source of truth; this is only for peace of mind)."""
    dest = catalog.parent / "label_studio_backups" / f"tasks-{time.strftime('%Y%m%d-%H%M%S')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    return dest


def push(url: str, api_key: str, catalog: Path, names: Iterable[str],
         limit: int | None = None, reopen_unnamed: bool = False) -> dict:
    """Bring the project up to date with the catalog, hardest frames first.

    - new unreviewed frames are imported with the models' boxes and names;
    - unreviewed tasks still holding an older suggestion (no names) are
      re-imported with the current one; other tasks get their priority patched;
    - ``reopen_unnamed``: frames verified before names existed go back into the
      queue with their own boxes and suggested names (their old Label Studio
      task is backed up and replaced; the catalog keeps the boxes until the new
      task is submitted)."""
    names = name_choices(names)
    client = _connect(url, api_key)
    project = _ensure_project(client, names[:-1])
    existing = {
        task["data"].get("sample_id"): task
        for task in project.get_tasks(only_ids=False)
    }
    conn = open_catalog(catalog)
    try:
        pending = _pending_tasks(conn, names)
        unnamed = _unnamed_tasks(conn, names)
    finally:
        conn.close()

    replace: list[dict] = []    # existing tasks to delete before re-importing
    fresh: list[dict] = []
    reprioritised = 0
    for task in pending:
        old = existing.get(task["data"]["sample_id"])
        if old is None:
            fresh.append(task)
        elif not _is_labelled(old) and _suggestion_version(old) != MODEL_VERSION:
            replace.append(old)
            fresh.append(task)
        elif old["data"].get("priority") != task["data"]["priority"]:
            data = {**old["data"], "priority": task["data"]["priority"],
                    "reasons": task["data"]["reasons"]}
            project.make_request("PATCH", f"/api/tasks/{old['id']}", json={"data": data})
            reprioritised += 1
    def queued(task: dict) -> bool:   # already reopened, waiting for review
        old = existing.get(task["data"]["sample_id"])
        return old is not None and not _is_labelled(old) and _suggestion_version(old) == MODEL_VERSION

    unnamed = [task for task in unnamed if not queued(task)]
    reopened = 0
    if reopen_unnamed:
        for task in unnamed:
            old = existing.get(task["data"]["sample_id"])
            if old is not None:
                replace.append(old)
            fresh.append(task)
            reopened += 1
    fresh.sort(key=lambda t: t["data"]["priority"])   # stable: chronological within
    if limit is not None:
        fresh = fresh[:limit]
        keep = {t["data"]["sample_id"] for t in fresh}
        replace = [t for t in replace if t["data"].get("sample_id") in keep]
    backup = None
    if replace:
        backup = _backup(replace, catalog)
        for old in replace:
            project.make_request("DELETE", f"/api/tasks/{old['id']}")
    if fresh:
        # Label Studio's "Label All Tasks" stream walks tasks in import order.
        project.import_tasks(fresh)

    waiting = [t for t in project.get_tasks(only_ids=False) if not _is_labelled(t)]
    result = {
        "project_id": project.id, "pushed": len(fresh), "replaced": len(replace),
        "reprioritised": reprioritised, "reopened": reopened,
        "waiting": len(waiting),
        "needs_a_look": sum(1 for t in waiting
                            if int(t["data"].get("priority", 99)) <= URGENT_PRIORITY),
        "unnamed_reviewed": 0 if reopen_unnamed else len(unnamed),
    }
    if backup:
        result["backup"] = str(backup)
    return result


def pull(url: str, api_key: str, catalog: Path) -> dict:
    """Write submitted boxes and cat names back into the catalog (verified;
    no boxes = a confirmed empty frame)."""
    client = _connect(url, api_key)
    project = _ensure_project(client)
    conn = open_catalog(catalog)
    now = int(time.time() * 1000)
    updated = objects = negatives = unnamed = 0
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
            if row is None or row["status"] == "retired":   # thinned out, image gone
                continue
            boxes = annotation_boxes(annotations[-1].get("result"), row["width"], row["height"])
            conn.execute("DELETE FROM annotations WHERE sample_id=?", (sid,))
            for index, (x, y, w, h, name) in enumerate(boxes, 1):
                conn.execute(
                    "INSERT INTO annotations(sample_id,annotation_id,class_name,x,y,w,h,"
                    "source,updated_at_ms,cat_name) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (sid, index, LABEL_NAME, x, y, w, h, "label_studio", now, name),
                )
                unnamed += int(name is None)
            conn.execute(
                "UPDATE samples SET status='verified',protected=1 WHERE sample_id=?", (sid,)
            )
            updated += 1
            objects += len(boxes)
            negatives += int(not boxes)
        conn.commit()
    finally:
        conn.close()
    return {"verified": updated, "objects": objects, "confirmed_empty": negatives,
            "boxes_without_name": unnamed}


def push_message(result: dict, link: str) -> str:
    """Telegram text for a push: how many frames wait and how many need a look."""
    if not result["waiting"]:
        return "Label Studio: nothing to label right now."
    lines = [f"Label Studio: {result['waiting']} frames to label"
             + (f" ({result['pushed']} new)" if result["pushed"] else "") + "."]
    if result["needs_a_look"]:
        lines.append(f"{result['needs_a_look']} need a look: the models are unsure "
                     "(boxes or cat names). They come first.")
    lines.append(link)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/yolo_dataset/catalog.sqlite3"))
    parser.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.environ.get("LABEL_STUDIO_API_KEY", ""))
    sub = parser.add_subparsers(dest="command", required=True)
    push_cmd = sub.add_parser("push", help="import frames with the models' boxes and names")
    push_cmd.add_argument("--names", default=os.environ.get("CAT_NAMES", ""),
                          help="comma-separated cat names (default: $CAT_NAMES)")
    push_cmd.add_argument("--limit", type=int, default=None)
    push_cmd.add_argument("--reopen-unnamed", action="store_true",
                          help="queue frames reviewed before names existed, to name their cats")
    push_cmd.add_argument("--notify", action="store_true",
                          help="send the result to Telegram (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)")
    sub.add_parser("pull", help="write submitted boxes and names back into the catalog")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "push":
        result = push(args.url, args.api_key, args.catalog, args.names.split(","),
                      args.limit, reopen_unnamed=args.reopen_unnamed)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["unnamed_reviewed"]:
            print(f"{result['unnamed_reviewed']} reviewed frames have boxes without a cat "
                  "name; queue them with --reopen-unnamed")
        if args.notify:
            from training.notify import telegram
            public = os.environ.get("LABEL_STUDIO_PUBLIC_URL", args.url).rstrip("/")
            telegram(push_message(result, f"{public}/projects/{result['project_id']}/data"))
    else:
        print(json.dumps(pull(args.url, args.api_key, args.catalog), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
