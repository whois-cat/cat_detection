"""Persistent SQLite index for MediaMTX recording segments.

The detector events DB remains the source of truth for what happened. This
table only indexes where already-recorded media files live, so it is safe to
rebuild from ``data/recordings`` at any time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SCHEMA = """
CREATE TABLE IF NOT EXISTS recording_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    path TEXT NOT NULL UNIQUE,
    size_bytes INTEGER,
    mtime_ms INTEGER,
    created_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready'
);

CREATE INDEX IF NOT EXISTS idx_recording_segments_lookup
ON recording_segments(camera_id, start_ms, end_ms);

CREATE INDEX IF NOT EXISTS idx_recording_segments_camera_start
ON recording_segments(camera_id, start_ms);
"""

DEFAULT_SEGMENT_DURATION_MS = 30_000


@dataclass(frozen=True, slots=True)
class RecordingSegment:
    camera_id: str
    start_ms: int
    end_ms: int
    path: str
    size_bytes: int | None
    mtime_ms: int | None
    status: str = "ready"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _recording_tz():
    raw = os.environ.get("RECORDING_TZ") or os.environ.get("TZ") or "UTC"
    if raw.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(raw)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"invalid RECORDING_TZ/TZ value: {raw!r}") from exc


def parse_segment_start_ms(name: str, *, tz=None) -> int | None:
    stem = name.rsplit(".", 1)[0]
    try:
        dt = datetime.strptime(stem, "%Y-%m-%d_%H-%M-%S-%f")
    except ValueError:
        return None
    zone = tz if tz is not None else _recording_tz()
    return int(dt.replace(tzinfo=zone).timestamp() * 1000)


def discover_recording_files(recordings_root: Path) -> dict[str, list[tuple[Path, int]]]:
    """Return camera_id -> [(path, start_ms)] discovered below recordings_root."""
    root = Path(recordings_root)
    found: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    if not root.exists():
        return {}
    tz = _recording_tz()
    for path in root.rglob("*.mp4"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        start_ms = parse_segment_start_ms(path.name, tz=tz)
        if start_ms is None:
            continue
        found[rel.parts[0]].append((path, start_ms))
    for items in found.values():
        items.sort(key=lambda item: (item[1], str(item[0])))
    return dict(found)


def _segments_from_files(
    files_by_camera: dict[str, list[tuple[Path, int]]],
    *,
    segment_duration_ms: int,
) -> list[RecordingSegment]:
    out: list[RecordingSegment] = []
    for camera_id, files in files_by_camera.items():
        for i, (path, start_ms) in enumerate(files):
            default_end_ms = start_ms + segment_duration_ms
            if i + 1 < len(files):
                next_start_ms = files[i + 1][1]
                end_ms = min(default_end_ms, max(start_ms + 1, next_start_ms))
            else:
                end_ms = default_end_ms
            try:
                st = path.stat()
                size_bytes = int(st.st_size)
                mtime_ms = int(st.st_mtime * 1000)
            except FileNotFoundError:
                size_bytes = None
                mtime_ms = None
            out.append(
                RecordingSegment(
                    camera_id=camera_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    path=str(path),
                    size_bytes=size_bytes,
                    mtime_ms=mtime_ms,
                )
            )
    return out


def refresh_index(
    conn: sqlite3.Connection,
    recordings_root: Path,
    *,
    segment_duration_ms: int = DEFAULT_SEGMENT_DURATION_MS,
    now_ms: int | None = None,
) -> dict:
    """Incrementally upsert discovered files and mark absent paths missing."""
    ensure_schema(conn)
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    segments = _segments_from_files(
        discover_recording_files(recordings_root),
        segment_duration_ms=segment_duration_ms,
    )
    seen_paths = {s.path for s in segments}
    ready_before = conn.execute(
        "SELECT path FROM recording_segments WHERE status='ready'"
    ).fetchall()
    missing_paths = [row[0] for row in ready_before if row[0] not in seen_paths]

    with conn:
        for seg in segments:
            conn.execute(
                """
                INSERT INTO recording_segments
                  (camera_id, start_ms, end_ms, path, size_bytes, mtime_ms, created_at_ms, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ready')
                ON CONFLICT(path) DO UPDATE SET
                  camera_id=excluded.camera_id,
                  start_ms=excluded.start_ms,
                  end_ms=excluded.end_ms,
                  size_bytes=excluded.size_bytes,
                  mtime_ms=excluded.mtime_ms,
                  status='ready'
                """,
                (
                    seg.camera_id,
                    seg.start_ms,
                    seg.end_ms,
                    seg.path,
                    seg.size_bytes,
                    seg.mtime_ms,
                    now_ms,
                ),
            )
        for path in missing_paths:
            conn.execute(
                "UPDATE recording_segments SET status='missing' WHERE path=?",
                (path,),
            )

    return {
        "ready": len(segments),
        "marked_missing": len(missing_paths),
        "cameras": sorted({s.camera_id for s in segments}),
    }


def lookup_segment(
    conn: sqlite3.Connection,
    *,
    camera_id: str,
    wall_ms: int,
) -> RecordingSegment | None:
    ensure_schema(conn)
    row = conn.execute(
        """
        SELECT camera_id, start_ms, end_ms, path, size_bytes, mtime_ms, status
        FROM recording_segments
        WHERE camera_id = ?
          AND start_ms <= ?
          AND end_ms >= ?
          AND status = 'ready'
        ORDER BY start_ms DESC
        LIMIT 1
        """,
        (camera_id, wall_ms, wall_ms),
    ).fetchone()
    return _row_to_segment(row) if row else None


def query_ranges(
    conn: sqlite3.Connection,
    *,
    camera_id: str,
    from_ms: int,
    to_ms: int,
) -> list[list[int]]:
    """Return compact merged availability ranges for a camera/time window.

    This intentionally does *not* expose segment paths or file metadata. The
    WebUI timeline only needs "media exists here" intervals, and thousands of
    per-segment rows with absolute paths make the payload much larger than the
    signal it carries.
    """
    ensure_schema(conn)
    rows = conn.execute(
        """
        SELECT start_ms, end_ms
        FROM recording_segments
        WHERE camera_id = ?
          AND start_ms <= ?
          AND end_ms >= ?
          AND status = 'ready'
        ORDER BY start_ms, end_ms
        """,
        (camera_id, to_ms, from_ms),
    ).fetchall()
    return merge_availability_ranges(rows, from_ms=from_ms, to_ms=to_ms)


def merge_availability_ranges(
    rows,
    *,
    from_ms: int,
    to_ms: int,
    merge_gap_ms: int = 1000,
) -> list[list[int]]:
    merged: list[list[int]] = []
    for start_ms, end_ms in rows:
        start = max(int(start_ms), int(from_ms))
        end = min(int(end_ms), int(to_ms))
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + merge_gap_ms:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _row_to_segment(row) -> RecordingSegment:
    return RecordingSegment(
        camera_id=row[0],
        start_ms=row[1],
        end_ms=row[2],
        path=row[3],
        size_bytes=row[4],
        mtime_ms=row[5],
        status=row[6],
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    ensure_schema(conn)
    return conn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rebuild = sub.add_parser("rebuild", help="rebuild/refresh recording_segments from files")
    rebuild.add_argument("--db", type=Path, default=Path("data/events/events.db"))
    rebuild.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    rebuild.add_argument("--segment-duration-sec", type=float, default=30.0)
    args = parser.parse_args(argv)

    if args.cmd == "rebuild":
        conn = _connect(args.db)
        stats = refresh_index(
            conn,
            args.recordings,
            segment_duration_ms=int(args.segment_duration_sec * 1000),
        )
        print(
            "recordings index: "
            f"{stats['ready']} ready, {stats['marked_missing']} marked missing, "
            f"cameras={','.join(stats['cameras']) or '-'}"
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
