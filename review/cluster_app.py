"""Cluster-review web app for cold-start bulk labelling."""
from __future__ import annotations

import io
import json
import os
import sys
import random
import sqlite3
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw

from training import CropRef, CropUnavailable, decode_one_crop
from training.build_cluster_manifest import kmeans
from training.db import Box
from training.segments import SegmentIndex

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

MANIFEST = Path(os.environ.get("CLUSTER_MANIFEST", ROOT / "data/review/clusters.json"))
RECORDINGS = Path(os.environ.get("RECORDINGS_ROOT", ROOT / "data/recordings"))
REVIEW_DB = Path(os.environ.get("REVIEW_DB", ROOT / "data/review/reviews.db"))
LABELS_ENV = [c.strip() for c in os.environ.get("REVIEW_LABELS", "").split(",") if c.strip()]
EXTRA_LABELS = ["unknown", "discard"]


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        # Actionable failure instead of a bare traceback. The manifest is either
        # not built yet, or was archived by `just review-reset` into a sibling
        # _backup_<ts>/ dir — surface both fixes, and any backup we can see.
        lines = [
            f"cluster manifest not found: {MANIFEST}",
            "",
            "Build it first (writes data/review/clusters.json):",
            "    just cluster-manifest",
        ]
        backups = sorted(MANIFEST.parent.glob("_backup_*/clusters.json"), reverse=True)
        if backups:
            lines += [
                "",
                "Or restore the most recent review-reset backup:",
                f"    mv {backups[0]} {MANIFEST}",
            ]
        msg = "\n".join(lines)
        # Print to stderr too so the guidance is visible even when the import
        # error is wrapped (e.g. by uvicorn's app loader).
        print(f"\n[cluster-review] {msg}\n", file=sys.stderr, flush=True)
        raise SystemExit(msg)
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


MAN = _load_manifest()
ITEMS = MAN["items"]
BASE_CLUSTERS = MAN["clusters"]
BY_ID = {it["crop_id"]: it for it in ITEMS}
LABELS = LABELS_ENV or MAN.get("labels") or ["alisa", "chuzh", "ellie", "felisis"]

_db_lock = threading.Lock()
REVIEW_DB.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(REVIEW_DB), check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute(
    """CREATE TABLE IF NOT EXISTS reviews (
           src_event_key INTEGER PRIMARY KEY,
           crop_id       TEXT,
           label         TEXT NOT NULL,
           predicted     TEXT,
           conf          REAL,
           camera        TEXT,
           wall_ms       INTEGER,
           reviewed_at   TEXT NOT NULL
       )"""
)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS cluster_reviews (
           cluster_id  INTEGER PRIMARY KEY,
           status      TEXT NOT NULL,
           label       TEXT,
           reviewed_at TEXT NOT NULL
       )"""
)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS cluster_splits (
           child_id     INTEGER PRIMARY KEY,
           parent_id    INTEGER NOT NULL,
           item_indices TEXT NOT NULL,
           created_at   TEXT NOT NULL
       )"""
)
_conn.commit()


def _reviews_map() -> dict[str, str]:
    with _db_lock:
        rows = _conn.execute("SELECT crop_id, label FROM reviews").fetchall()
    return {cid: label for cid, label in rows}


def _cluster_status() -> dict[int, dict]:
    with _db_lock:
        rows = _conn.execute(
            "SELECT cluster_id, status, label, reviewed_at FROM cluster_reviews"
        ).fetchall()
    return {
        int(cid): {"status": status, "label": label, "reviewed_at": reviewed_at}
        for cid, status, label, reviewed_at in rows
    }


def _split_rows() -> list[dict]:
    with _db_lock:
        rows = _conn.execute(
            "SELECT child_id, parent_id, item_indices, created_at "
            "FROM cluster_splits ORDER BY parent_id, child_id"
        ).fetchall()
    return [
        {
            "child_id": int(child_id),
            "parent_id": int(parent_id),
            "item_indices": json.loads(item_indices),
            "created_at": created_at,
        }
        for child_id, parent_id, item_indices, created_at in rows
    ]


def _descendant_child_ids(conn: sqlite3.Connection, cluster_id: int) -> list[int]:
    """All split-children of ``cluster_id``, recursively (children, grandchildren,
    …). Used to wipe a previous partition before re-splitting. Caller holds the
    DB lock."""
    rows = conn.execute("SELECT child_id, parent_id FROM cluster_splits").fetchall()
    by_parent: dict[int, list[int]] = {}
    for child_id, parent_id in rows:
        by_parent.setdefault(int(parent_id), []).append(int(child_id))
    out: list[int] = []
    stack = list(by_parent.get(int(cluster_id), []))
    while stack:
        cid = stack.pop()
        out.append(cid)
        stack.extend(by_parent.get(cid, []))
    return out


def _label_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Per-label crop counts. Every row in ``reviews`` is one concretely-labelled
    crop (mixed/skip never insert here), so a plain GROUP BY is the live count of
    labelled CROPS — not clusters. Caller holds the DB lock."""
    rows = conn.execute(
        "SELECT label, COUNT(*) FROM reviews GROUP BY label"
    ).fetchall()
    return {label: int(n) for label, n in rows}


def _all_clusters() -> list[dict]:
    children: dict[int, list[dict]] = {}
    for row in _split_rows():
        members = [int(i) for i in row["item_indices"]]
        children.setdefault(row["parent_id"], []).append({
            "cluster_id": row["child_id"],
            "parent_id": row["parent_id"],
            "size": len(members),
            "item_indices": members,
            "representatives": [ITEMS[i]["crop_id"] for i in members[:24]],
            "split_child": True,
            "created_at": row["created_at"],
        })

    out: list[dict] = []

    def append_with_children(cluster: dict) -> None:
        out.append(cluster)
        for child in children.get(int(cluster["cluster_id"]), []):
            append_with_children(child)

    for base in BASE_CLUSTERS:
        append_with_children(base)
    return out


def _cluster_by_id(cluster_id: int) -> dict | None:
    for cluster in _all_clusters():
        if int(cluster["cluster_id"]) == cluster_id:
            return cluster
    return None


_idx_cache: dict[str, SegmentIndex] = {}
_idx_lock = threading.Lock()


def _index_for(camera: str) -> SegmentIndex:
    with _idx_lock:
        idx = _idx_cache.get(camera)
        if idx is None:
            idx = SegmentIndex.from_dir(RECORDINGS / camera)
            _idx_cache[camera] = idx
        return idx


@lru_cache(maxsize=1024)
def _crop_rgb(crop_id: str) -> np.ndarray:
    item = BY_ID[crop_id]
    b = item["box"]
    ref = CropRef(
        camera_id=item["camera"],
        wall_ms=item["wall_ms"],
        box=Box(
            x=b["x"], y=b["y"], w=b["w"], h=b["h"],
            cat=None, score=item.get("score", 0.0), track_id=None,
            rowid=item.get("src_event_key"),
        ),
        rotate_deg=int(item.get("rotate_deg", 0)),
    )
    crop_bgr = decode_one_crop(
        ref,
        RECORDINGS,
        pad_frac=item.get("pad_frac", MAN.get("params", {}).get("pad_frac", 0.15)),
        index=_index_for(item["camera"]),
    )
    return np.ascontiguousarray(crop_bgr[..., ::-1])


def _select_cluster_items(cluster: dict, mode: str, limit: int | None = None) -> list[dict]:
    members = [ITEMS[i] for i in cluster["item_indices"]]
    if mode == "outliers":
        members = list(reversed(members))
    elif mode == "random":
        rng = random.Random(int(cluster["cluster_id"]))
        members = members[:]
        rng.shuffle(members)
    return members if limit is None else members[:limit]


def _placeholder(text: str, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (36, 20, 28))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text[:36], fill=(255, 130, 160))
    return img


def _thumb(crop_id: str, size: int) -> Image.Image:
    try:
        img = Image.fromarray(_crop_rgb(crop_id))
    except CropUnavailable:
        return _placeholder("missing recording", size)
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (12, 14, 20))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


app = FastAPI(title="cat cluster review")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "cluster.html").read_text(encoding="utf-8")


@app.get("/api/clusters")
def api_clusters() -> JSONResponse:
    reviews = _reviews_map()
    status = _cluster_status()
    queue = []
    clusters = _all_clusters()
    for c in clusters:
        cid = int(c["cluster_id"])
        crop_ids = [ITEMS[i]["crop_id"] for i in c["item_indices"]]
        reviewed_items = sum(1 for crop_id in crop_ids if crop_id in reviews)
        queue.append({
            "cluster_id": cid,
            "parent_id": c.get("parent_id"),
            "split_child": bool(c.get("split_child")),
            "size": c["size"],
            "reviewed_items": reviewed_items,
            "representatives": c.get("representatives", [])[:8],
            "status": status.get(cid),
        })
    done_clusters = sum(1 for c in queue if c["status"])
    return JSONResponse({
        "labels": LABELS,
        "extra_labels": EXTRA_LABELS,
            "queue": queue,
            "done_clusters": done_clusters,
            "total_clusters": len(queue),
            "total_items": len(ITEMS),
        "reviewed_items": len(reviews),
        "params": MAN.get("params", {}),
    })


@app.get("/api/cluster/{cluster_id}/sheet")
def api_sheet(cluster_id: int, mode: str = "representative",
              limit: int = 24, cols: int = 6, thumb: int = 128) -> Response:
    cluster = _cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="unknown cluster")
    if mode not in {"representative", "random", "outliers"}:
        raise HTTPException(status_code=400, detail="bad mode")
    limit = max(1, min(48, int(limit)))
    cols = max(1, min(8, int(cols)))
    thumb = max(80, min(180, int(thumb)))
    selected = _select_cluster_items(cluster, mode, limit)
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, max(1, rows) * thumb), (12, 14, 20))
    for i, item in enumerate(selected):
        tile = _thumb(item["crop_id"], thumb)
        sheet.paste(tile, ((i % cols) * thumb, (i // cols) * thumb))
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/cluster/{cluster_id}/items")
def api_cluster_items(cluster_id: int, mode: str = "representative") -> JSONResponse:
    cluster = _cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="unknown cluster")
    if mode not in {"representative", "random", "outliers"}:
        raise HTTPException(status_code=400, detail="bad mode")
    reviews = _reviews_map()
    items = []
    for item in _select_cluster_items(cluster, mode, limit=None):
        items.append({
            "crop_id": item["crop_id"],
            "camera": item["camera"],
            "wall_ms": item["wall_ms"],
            "score": item.get("score"),
            "label": reviews.get(item["crop_id"]),
        })
    return JSONResponse({"cluster_id": cluster_id, "mode": mode, "items": items})


@app.get("/api/crop/{crop_id:path}")
def api_crop(crop_id: str, thumb: int = 144) -> Response:
    if crop_id not in BY_ID:
        raise HTTPException(status_code=404, detail="unknown crop")
    thumb = max(80, min(260, int(thumb)))
    img = _thumb(crop_id, thumb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/cluster_review")
async def api_cluster_review(payload: dict) -> JSONResponse:
    cluster_id = int(payload.get("cluster_id"))
    label = payload.get("label")
    cluster = _cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="unknown cluster")
    allowed = set(LABELS) | set(EXTRA_LABELS) | {"mixed", "skip"}
    if label not in allowed:
        raise HTTPException(status_code=400, detail=f"invalid label {label!r}")

    now = datetime.now(timezone.utc).isoformat()
    status = "mixed" if label == "mixed" else "skipped" if label == "skip" else "labeled"
    with _db_lock:
        if label in set(LABELS) | set(EXTRA_LABELS):
            for idx in cluster["item_indices"]:
                item = ITEMS[idx]
                _conn.execute(
                    """INSERT INTO reviews
                           (src_event_key, crop_id, label, predicted, conf, camera, wall_ms, reviewed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(src_event_key) DO UPDATE SET
                           label=excluded.label, crop_id=excluded.crop_id,
                           predicted=excluded.predicted, conf=excluded.conf,
                           camera=excluded.camera, wall_ms=excluded.wall_ms,
                           reviewed_at=excluded.reviewed_at""",
                    (
                        item["src_event_key"], item["crop_id"], label, None,
                        item.get("score"), item["camera"], item["wall_ms"], now,
                    ),
                )
        _conn.execute(
            """INSERT INTO cluster_reviews (cluster_id, status, label, reviewed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(cluster_id) DO UPDATE SET
                   status=excluded.status, label=excluded.label,
                   reviewed_at=excluded.reviewed_at""",
            (cluster_id, status, label, now),
        )
        _conn.commit()
    return JSONResponse({"ok": True, "cluster_id": cluster_id, "label": label, "status": status})


@app.post("/api/cluster_split")
async def api_cluster_split(payload: dict) -> JSONResponse:
    cluster_id = int(payload.get("cluster_id"))
    parts = int(payload.get("parts", 2))
    cluster = _cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="unknown cluster")
    parts = max(2, min(8, parts))
    members = [int(i) for i in cluster["item_indices"]]
    if len(members) < parts:
        raise HTTPException(status_code=400, detail="cluster is too small to split")
    if any("embedding" not in ITEMS[i] for i in members):
        raise HTTPException(
            status_code=400,
            detail="manifest has no stored embeddings; rebuild with build_cluster_manifest",
        )

    x = np.asarray([ITEMS[i]["embedding"] for i in members], dtype=np.float32)
    labels, distances = kmeans(x, parts, seed=cluster_id + len(members), max_iter=30)
    groups: dict[int, list[int]] = {}
    for pos, lab in enumerate(labels.tolist()):
        groups.setdefault(int(lab), []).append(pos)
    ordered_groups = []
    for _lab, positions in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        positions.sort(key=lambda pos: float(distances[pos]))
        ordered_groups.append([members[pos] for pos in positions])

    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        # Re-split = REPLACE the previous partition (no 409). Drop any existing
        # descendant children (recursively) and their cluster_reviews, then redo
        # the split below. The per-crop `reviews` rows are cleared for all of this
        # cluster's members further down — children only ever partition those, so
        # that single delete also removes labels applied to old grandchildren.
        replaced_children = _descendant_child_ids(_conn, cluster_id)
        if replaced_children:
            placeholders = ",".join("?" * len(replaced_children))
            _conn.execute(
                f"DELETE FROM cluster_splits WHERE child_id IN ({placeholders})",
                replaced_children,
            )
            _conn.execute(
                f"DELETE FROM cluster_reviews WHERE cluster_id IN ({placeholders})",
                replaced_children,
            )
        used_ids = [int(c["cluster_id"]) for c in BASE_CLUSTERS]
        used_ids.extend(
            int(r[0]) for r in _conn.execute("SELECT child_id FROM cluster_splits").fetchall()
        )
        next_id = max(used_ids, default=-1) + 1
        _conn.executemany(
            "DELETE FROM reviews WHERE src_event_key=?",
            [(ITEMS[i]["src_event_key"],) for i in members],
        )
        children = []
        for group in ordered_groups:
            child_id = next_id
            next_id += 1
            children.append({
                "cluster_id": child_id,
                "parent_id": cluster_id,
                "size": len(group),
            })
            _conn.execute(
                """INSERT INTO cluster_splits
                       (child_id, parent_id, item_indices, created_at)
                   VALUES (?, ?, ?, ?)""",
                (child_id, cluster_id, json.dumps(group, separators=(",", ":")), now),
            )
        _conn.execute(
            """INSERT INTO cluster_reviews (cluster_id, status, label, reviewed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(cluster_id) DO UPDATE SET
                   status=excluded.status, label=excluded.label,
                   reviewed_at=excluded.reviewed_at""",
            (cluster_id, "split", None, now),
        )
        _conn.commit()
    return JSONResponse({
        "ok": True, "cluster_id": cluster_id, "children": children,
        "replaced": len(replaced_children),
    })


@app.post("/api/crop_label")
async def api_crop_label(payload: dict) -> JSONResponse:
    """Label individual crops (fallback for hopelessly mixed clusters). Writes one
    `reviews` row per crop — same per-crop grain as a cluster label, just scoped
    to an explicit selection instead of the whole cluster."""
    label = payload.get("label")
    crop_ids = payload.get("crop_ids") or []
    if label not in set(LABELS) | set(EXTRA_LABELS):
        raise HTTPException(status_code=400, detail=f"invalid label {label!r}")
    if not isinstance(crop_ids, list) or not crop_ids:
        raise HTTPException(status_code=400, detail="no crops selected")
    unknown = [cid for cid in crop_ids if cid not in BY_ID]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown crop(s): {unknown[:5]}")

    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        for cid in crop_ids:
            item = BY_ID[cid]
            _conn.execute(
                """INSERT INTO reviews
                       (src_event_key, crop_id, label, predicted, conf, camera, wall_ms, reviewed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(src_event_key) DO UPDATE SET
                       label=excluded.label, crop_id=excluded.crop_id,
                       predicted=excluded.predicted, conf=excluded.conf,
                       camera=excluded.camera, wall_ms=excluded.wall_ms,
                       reviewed_at=excluded.reviewed_at""",
                (
                    item["src_event_key"], cid, label, None,
                    item.get("score"), item["camera"], item["wall_ms"], now,
                ),
            )
        _conn.commit()
    return JSONResponse({"ok": True, "labeled": len(crop_ids), "label": label})


@app.get("/api/counts")
def api_counts() -> JSONResponse:
    """Live per-class crop counts (labelled crops, GROUP BY label)."""
    with _db_lock:
        counts = _label_counts(_conn)
    return JSONResponse({
        "counts": counts,
        "total": sum(counts.values()),
        "order": [*LABELS, *EXTRA_LABELS],
    })


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "clusters": len(_all_clusters()), "items": len(ITEMS)}
