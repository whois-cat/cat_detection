"""Stage B — label-review web app.

A tiny FastAPI service for hand-correcting cat-crop labels (the alisa↔felisis
confusion). It depends on `av` + the `training` package for on-the-fly decode,
NOT on openvino/torch — all inference already happened in Stage A.

Design constraints (see the task brief):
  - NO image files, ever. Each crop is decoded from the recordings into memory
    on request (training.decode_one_crop) → JPEG in memory → streamed. A bounded
    in-memory LRU avoids re-decoding on back/forward navigation.
  - events.db and recordings are READ-ONLY; in fact events.db is never opened
    here — the manifest already carries the box coords. Corrections are written
    NON-DESTRUCTIVELY to a SEPARATE reviews.db keyed by src_event_key, so the
    detector's rows are untouched and the work is fully resumable.

Config (env vars; the `just review` recipe sets sane defaults):
  REVIEW_MANIFEST   manifest.jsonl from Stage A      (default data/review/manifest.jsonl)
  RECORDINGS_ROOT   recordings root                  (default data/recordings)
  REVIEW_DB         writable corrections DB          (default data/review/reviews.db)
  REVIEW_CONFUSE    comma pair to highlight in UI    (default "" — names from data)
  REVIEW_PAD_FRAC   fallback crop padding            (default 0.15)
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from training import CropRef, CropUnavailable, decode_one_crop
from training.db import Box
from training.segments import SegmentIndex

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

MANIFEST = Path(os.environ.get("REVIEW_MANIFEST", ROOT / "data/review/manifest.jsonl"))
RECORDINGS = Path(os.environ.get("RECORDINGS_ROOT", ROOT / "data/recordings"))
REVIEW_DB = Path(os.environ.get("REVIEW_DB", ROOT / "data/review/reviews.db"))
CONFUSE = [c.strip() for c in os.environ.get("REVIEW_CONFUSE", "").split(",") if c.strip()]
PAD_FRAC_DEFAULT = float(os.environ.get("REVIEW_PAD_FRAC", "0.15"))

# Labels the reviewer can assign beyond the model's classes.
EXTRA_LABELS = ["unknown", "discard"]


# ---- manifest (load once) ---------------------------------------------------

def _load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        raise RuntimeError(
            f"manifest not found: {MANIFEST}. Run Stage A "
            "(python -m training.build_review_manifest ...) first."
        )
    items = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


ITEMS = _load_manifest()
BY_ID = {it["crop_id"]: it for it in ITEMS}
CLASSES = sorted({k for it in ITEMS for k in it.get("probs", {})})


# ---- reviews DB (separate, writable; events.db is never touched) ------------

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
_conn.commit()


def _reviews_map() -> dict[str, str]:
    with _db_lock:
        rows = _conn.execute("SELECT crop_id, label FROM reviews").fetchall()
    return {cid: label for cid, label in rows}


# ---- segment-index cache + crop decode --------------------------------------

_idx_cache: dict[str, SegmentIndex] = {}
_idx_lock = threading.Lock()


def _index_for(camera: str) -> SegmentIndex:
    with _idx_lock:
        idx = _idx_cache.get(camera)
        if idx is None:
            idx = SegmentIndex.from_dir(RECORDINGS / camera)
            _idx_cache[camera] = idx
        return idx


@lru_cache(maxsize=64)
def _crop_jpeg(crop_id: str) -> bytes:
    """Decode one crop from the recordings into an in-memory JPEG. LRU-cached so
    back/forward navigation doesn't re-decode. Never writes to disk."""
    item = BY_ID[crop_id]
    b = item["box"]
    ref = CropRef(
        camera_id=item["camera"],
        wall_ms=item["wall_ms"],
        box=Box(x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                cat=None, score=0.0, track_id=None, rowid=item.get("src_event_key")),
    )
    crop_bgr = decode_one_crop(
        ref, RECORDINGS,
        pad_frac=item.get("pad_frac", PAD_FRAC_DEFAULT),
        index=_index_for(item["camera"]),
    )
    rgb = np.ascontiguousarray(crop_bgr[..., ::-1])  # BGR -> RGB
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---- app --------------------------------------------------------------------

app = FastAPI(title="cat crop review")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/manifest")
def api_manifest() -> JSONResponse:
    """Queue (already contentious-first from Stage A) + current corrections."""
    reviews = _reviews_map()
    queue = [
        {
            "crop_id": it["crop_id"],
            "src_event_key": it["src_event_key"],
            "camera": it["camera"],
            "wall_ms": it["wall_ms"],
            "predicted": it["predicted"],
            "conf": it["conf"],
            "probs": it["probs"],
        }
        for it in ITEMS
    ]
    return JSONResponse({
        "classes": CLASSES,
        "extra_labels": EXTRA_LABELS,
        "confuse": CONFUSE,
        "queue": queue,
        "reviews": reviews,
    })


@app.get("/api/crop/{crop_id}")
def api_crop(crop_id: str) -> Response:
    if crop_id not in BY_ID:
        raise HTTPException(status_code=404, detail="unknown crop_id")
    try:
        data = _crop_jpeg(crop_id)
    except CropUnavailable as exc:
        # Recording pruned / gap — tell the client so it can grey the slot out.
        raise HTTPException(status_code=410, detail=str(exc))
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/review")
async def api_review(payload: dict) -> JSONResponse:
    crop_id = payload.get("crop_id")
    label = payload.get("label")
    if crop_id not in BY_ID:
        raise HTTPException(status_code=404, detail="unknown crop_id")
    if not label or (label not in CLASSES and label not in EXTRA_LABELS):
        raise HTTPException(status_code=400, detail=f"invalid label {label!r}")
    item = BY_ID[crop_id]
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        _conn.execute(
            """INSERT INTO reviews
                   (src_event_key, crop_id, label, predicted, conf, camera, wall_ms, reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(src_event_key) DO UPDATE SET
                   label=excluded.label, crop_id=excluded.crop_id,
                   predicted=excluded.predicted, conf=excluded.conf,
                   camera=excluded.camera, wall_ms=excluded.wall_ms,
                   reviewed_at=excluded.reviewed_at""",
            (item["src_event_key"], crop_id, label, item["predicted"],
             item["conf"], item["camera"], item["wall_ms"], now),
        )
        _conn.commit()
        n_done = _conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    return JSONResponse({"ok": True, "crop_id": crop_id, "label": label,
                         "reviewed": n_done, "total": len(ITEMS)})


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "items": len(ITEMS), "reviewed": len(_reviews_map())}
