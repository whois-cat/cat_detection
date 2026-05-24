"""SQLite event queries for training-data extraction.

The detector writes one row per detected box; multiple boxes in one frame
share `wall_ms` (and `pts`). For training we usually want them re-grouped
back into per-frame records.

Schema reference: live2/detector/storage.py
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(slots=True)
class Box:
    x: int; y: int; w: int; h: int
    cat: str | None
    score: float
    track_id: int | None


@dataclass(slots=True)
class FrameRecord:
    camera_id: str
    model: str
    wall_ms: int
    pts: int | None
    media_t: float | None
    frame_w: int
    frame_h: int
    boxes: list[Box]


def open_db_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection — safe to use while the detector is writing."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    return conn


def iter_frames(conn: sqlite3.Connection, *,
                camera_id: str | None = None,
                model: str | None = None,
                cat: str | None = None,
                t_from: int | None = None,
                t_to: int | None = None,
                min_score: float | None = None) -> Iterator[FrameRecord]:
    """Stream FrameRecords in wall_ms order, regrouping per-box rows.

    Sorted by (camera_id, wall_ms, pts) so all boxes for one frame appear
    contiguously. We yield one FrameRecord per (camera_id, model, wall_ms, pts).
    """
    sql = [
        "SELECT camera_id, model, wall_ms, pts, media_t, frame_w, frame_h,",
        "       cat, box_x, box_y, box_w, box_h, score, track_id",
        "FROM events WHERE 1=1",
    ]
    params: list = []
    if camera_id is not None: sql.append("AND camera_id = ?");  params.append(camera_id)
    if model     is not None: sql.append("AND model     = ?");  params.append(model)
    if cat       is not None: sql.append("AND cat       = ?");  params.append(cat)
    if t_from    is not None: sql.append("AND wall_ms  >= ?");  params.append(t_from)
    if t_to      is not None: sql.append("AND wall_ms  <= ?");  params.append(t_to)
    if min_score is not None: sql.append("AND score    >= ?");  params.append(min_score)
    sql.append("ORDER BY camera_id, wall_ms, pts")

    cur = conn.execute(" ".join(sql), params)
    current: FrameRecord | None = None
    for row in cur:
        (cam, model_, wall_ms, pts, media_t, fw, fh,
         cat_, bx, by, bw, bh, score, track_id) = row
        key = (cam, model_, wall_ms, pts)
        if current is None or (current.camera_id, current.model,
                               current.wall_ms, current.pts) != key:
            if current is not None:
                yield current
            current = FrameRecord(
                camera_id=cam, model=model_, wall_ms=wall_ms, pts=pts,
                media_t=media_t, frame_w=fw, frame_h=fh, boxes=[],
            )
        current.boxes.append(Box(
            x=bx, y=by, w=bw, h=bh, cat=cat_, score=score, track_id=track_id,
        ))
    if current is not None:
        yield current


def distinct_cats(conn: sqlite3.Connection, *,
                  camera_id: str | None = None,
                  model: str | None = None) -> list[str]:
    sql = ["SELECT DISTINCT cat FROM events WHERE cat IS NOT NULL"]
    params: list = []
    if camera_id is not None: sql.append("AND camera_id = ?"); params.append(camera_id)
    if model     is not None: sql.append("AND model     = ?"); params.append(model)
    sql.append("ORDER BY cat")
    return [r[0] for r in conn.execute(" ".join(sql), params)]
