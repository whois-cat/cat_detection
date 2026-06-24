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

# Incremental ("live") indexer defaults. The live indexer only looks at files
# whose segment-start falls inside the last ``DEFAULT_INDEX_WINDOW_MS`` so a
# single cycle never stats/upserts the whole 30k+ file archive.
DEFAULT_INDEX_WINDOW_MS = 10 * 60_000
# A segment file whose mtime is younger than this is assumed to still be open
# for writing by mediamtx and is skipped until the next cycle (avoids indexing
# partial mp4 files). It is indexed once it stops changing.
DEFAULT_MIN_STABLE_AGE_MS = 10_000


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


def discover_recording_files(
    recordings_root: Path,
    *,
    cameras: set[str] | None = None,
    since_ms: int | None = None,
) -> dict[str, list[tuple[Path, int]]]:
    """Return camera_id -> [(path, start_ms)] discovered below recordings_root.

    ``cameras`` restricts discovery to that allow-list (top-level dir == camera
    id), which keeps a deleted/typo camera dir from ever being re-indexed.
    ``since_ms`` keeps only files whose segment-start is at/after the cutoff —
    the cheap filename parse runs before any ``stat``, so callers can look at
    recent files only without walking metadata for the whole archive.
    """
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
        camera_id = rel.parts[0]
        if cameras is not None and camera_id not in cameras:
            continue
        start_ms = parse_segment_start_ms(path.name, tz=tz)
        if start_ms is None:
            continue
        if since_ms is not None and start_ms < since_ms:
            continue
        found[camera_id].append((path, start_ms))
    for items in found.values():
        items.sort(key=lambda item: (item[1], str(item[0])))
    return dict(found)


def scan_recent_files(
    recordings_root: Path,
    *,
    since_ms: int,
    cameras: set[str] | None = None,
    tz=None,
) -> dict[str, list[tuple[Path, int]]]:
    """Like ``discover_recording_files`` but walks only top-level camera dirs
    with ``os.scandir`` and parses the filename timestamp before doing anything
    expensive. Used by the live indexer so a 30s cycle never globs the archive.

    Recordings are written flat as ``<root>/<camera>/<ts>.mp4`` (mediamtx
    recordPath ``/recordings/%path/...``); we fall back to the recursive
    ``discover_recording_files`` only if a nested layout is ever introduced.
    """
    root = Path(recordings_root)
    if not root.exists():
        return {}
    tz = tz if tz is not None else _recording_tz()
    found: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    try:
        camera_dirs = list(os.scandir(root))
    except OSError:
        return {}
    nested = False
    for cam_entry in camera_dirs:
        if not cam_entry.is_dir():
            continue
        camera_id = cam_entry.name
        if cameras is not None and camera_id not in cameras:
            continue
        try:
            entries = os.scandir(cam_entry.path)
        except OSError:
            continue
        with entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(".mp4"):
                    if entry.is_dir():
                        nested = True
                    continue
                start_ms = parse_segment_start_ms(name, tz=tz)
                if start_ms is None or start_ms < since_ms:
                    continue
                found[camera_id].append((Path(entry.path), start_ms))
    if nested and not found:
        # Unexpected nested layout: fall back to the recursive walk so we never
        # silently miss recordings.
        return discover_recording_files(root, cameras=cameras, since_ms=since_ms)
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
    cameras: set[str] | None = None,
) -> dict:
    """Full rebuild: upsert every discovered file and mark absent paths missing.

    This globs the whole archive and is meant for recovery/backfill (the
    ``rebuild`` CLI), not the steady-state hot path — use ``incremental_update``
    for that.
    """
    ensure_schema(conn)
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    segments = _segments_from_files(
        discover_recording_files(recordings_root, cameras=cameras),
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


def last_indexed_to_ms(
    conn: sqlite3.Connection,
    *,
    cameras: set[str] | None = None,
) -> int | None:
    """Largest ``end_ms`` of any ready segment — i.e. how far the index has
    caught up. Used to size the restart catch-up window."""
    ensure_schema(conn)
    if cameras:
        placeholders = ",".join("?" for _ in cameras)
        row = conn.execute(
            f"SELECT MAX(end_ms) FROM recording_segments "
            f"WHERE status='ready' AND camera_id IN ({placeholders})",
            tuple(cameras),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(end_ms) FROM recording_segments WHERE status='ready'"
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def incremental_update(
    conn: sqlite3.Connection,
    recordings_root: Path,
    *,
    now_ms: int | None = None,
    since_ms: int | None = None,
    window_ms: int = DEFAULT_INDEX_WINDOW_MS,
    min_stable_age_ms: int = DEFAULT_MIN_STABLE_AGE_MS,
    segment_duration_ms: int = DEFAULT_SEGMENT_DURATION_MS,
    cameras: set[str] | None = None,
) -> dict:
    """Index recently-finalized segments and reconcile recent deletions.

    Only files whose segment-start is at/after ``since_ms`` (defaults to
    ``now - window_ms``) are considered, so a cycle is O(recent files), never
    O(archive). Files modified within ``min_stable_age_ms`` (or still empty) are
    treated as "still being written by mediamtx" and skipped until they settle.

    Deletions of *old* files are the pruner's job (it removes index rows for
    what it unlinks). This function additionally reconciles deletions *inside*
    the scan window: a ready row whose file is gone from disk is marked missing.
    """
    ensure_schema(conn)
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if since_ms is None:
        since_ms = now_ms - int(window_ms)
    since_ms = int(since_ms)

    by_cam = scan_recent_files(
        recordings_root, since_ms=since_ms, cameras=cameras
    )

    inserted = updated = skipped_unstable = marked_missing = 0
    last_indexed: dict[str, int] = {}
    on_disk_recent: set[str] = set()

    with conn:
        for camera_id, files in by_cam.items():
            starts = [start for _, start in files]
            for i, (path, start_ms) in enumerate(files):
                spath = str(path)
                on_disk_recent.add(spath)
                # End is capped by the next segment's start so adjacent 30s
                # files don't double-count the boundary second. The newest file
                # in the window has no successor yet → assume a full segment.
                default_end = start_ms + segment_duration_ms
                if i + 1 < len(files):
                    end_ms = min(default_end, max(start_ms + 1, starts[i + 1]))
                else:
                    end_ms = default_end
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                size_bytes = int(st.st_size)
                mtime_ms = int(st.st_mtime * 1000)
                if size_bytes == 0 or (now_ms - mtime_ms) < min_stable_age_ms:
                    skipped_unstable += 1
                    continue
                existed = conn.execute(
                    "SELECT 1 FROM recording_segments WHERE path=?", (spath,)
                ).fetchone()
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
                    (camera_id, start_ms, end_ms, spath, size_bytes, mtime_ms, now_ms),
                )
                if existed:
                    updated += 1
                else:
                    inserted += 1
                last_indexed[camera_id] = max(last_indexed.get(camera_id, 0), end_ms)

        # Reconcile deletions inside the window. Files we deliberately skipped
        # as "still writing" are in on_disk_recent, so they're never flagged.
        ready_rows = conn.execute(
            "SELECT path, camera_id FROM recording_segments "
            "WHERE status='ready' AND start_ms >= ?",
            (since_ms,),
        ).fetchall()
        for spath, cam in ready_rows:
            if cameras is not None and cam not in cameras:
                continue
            if spath in on_disk_recent:
                continue
            if not Path(spath).exists():
                conn.execute(
                    "UPDATE recording_segments SET status='missing' WHERE path=?",
                    (spath,),
                )
                marked_missing += 1

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped_unstable": skipped_unstable,
        "marked_missing": marked_missing,
        "since_ms": since_ms,
        "last_indexed": last_indexed,
        "cameras": sorted(by_cam.keys()),
    }


def delete_paths_from_index(conn: sqlite3.Connection, paths) -> int:
    """Remove index rows for files that were intentionally deleted (pruner).

    Hard delete (not 'missing') keeps the table bounded for the steady stream of
    pruned segments; transient/unexpected disappearances are marked 'missing' by
    ``incremental_update`` instead so they aren't silently lost.
    """
    rows = [(str(p),) for p in paths]
    if not rows:
        return 0
    ensure_schema(conn)
    with conn:
        cur = conn.executemany(
            "DELETE FROM recording_segments WHERE path=?", rows
        )
    return cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(rows)


def reconcile_deletions(
    conn: sqlite3.Connection,
    *,
    cameras: set[str] | None = None,
) -> dict:
    """Drop index rows for files that no longer exist on disk.

    The incremental scan only sees its recent window, so deletions of *old*
    segments (e.g. by the detection-aware pruner, which runs independently) are
    caught here. This walks index *rows* and stats each path — it never globs
    the archive — so it's cheap enough to run every few minutes.

    Safety: if every ready row's file is missing we assume the recordings volume
    is unmounted/empty rather than truly emptied, and make no changes (mirrors
    the pruner's "no events → skip" guard).
    """
    ensure_schema(conn)
    if cameras:
        placeholders = ",".join("?" for _ in cameras)
        rows = conn.execute(
            f"SELECT path FROM recording_segments "
            f"WHERE status='ready' AND camera_id IN ({placeholders})",
            tuple(cameras),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT path FROM recording_segments WHERE status='ready'"
        ).fetchall()
    gone = [row[0] for row in rows if not os.path.exists(row[0])]
    if rows and len(gone) == len(rows):
        return {"checked": len(rows), "deleted": 0, "skipped_all_missing": True}
    if gone:
        with conn:
            conn.executemany(
                "DELETE FROM recording_segments WHERE path=?",
                [(p,) for p in gone],
            )
    return {"checked": len(rows), "deleted": len(gone), "skipped_all_missing": False}


def open_index_db(db_path: Path) -> sqlite3.Connection:
    """Open (and schema-init) the recording index DB for read/write access.

    The index table lives inside the detector events DB; with WAL mode the
    detector keeps reading ranges while the indexer writes."""
    return _connect(Path(db_path))


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
    rebuild = sub.add_parser("rebuild", help="full rebuild of recording_segments from files (recovery/backfill)")
    rebuild.add_argument("--db", type=Path, default=Path("data/events/events.db"))
    rebuild.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    rebuild.add_argument("--segment-duration-sec", type=float, default=30.0)
    rebuild.add_argument("--cameras", default="", help="comma-separated allow-list (default: all)")

    incr = sub.add_parser("incremental", help="one-shot incremental index of recent files")
    incr.add_argument("--db", type=Path, default=Path("data/events/events.db"))
    incr.add_argument("--recordings", type=Path, default=Path("data/recordings"))
    incr.add_argument("--window-sec", type=float, default=DEFAULT_INDEX_WINDOW_MS / 1000)
    incr.add_argument("--stable-age-sec", type=float, default=DEFAULT_MIN_STABLE_AGE_MS / 1000)
    incr.add_argument("--segment-duration-sec", type=float, default=30.0)
    incr.add_argument("--cameras", default="", help="comma-separated allow-list (default: all)")

    args = parser.parse_args(argv)
    cameras = {c.strip() for c in args.cameras.split(",") if c.strip()} or None

    if args.cmd == "rebuild":
        conn = _connect(args.db)
        stats = refresh_index(
            conn,
            args.recordings,
            segment_duration_ms=int(args.segment_duration_sec * 1000),
            cameras=cameras,
        )
        print(
            "recordings index: "
            f"{stats['ready']} ready, {stats['marked_missing']} marked missing, "
            f"cameras={','.join(stats['cameras']) or '-'}"
        )
        return 0
    if args.cmd == "incremental":
        conn = _connect(args.db)
        stats = incremental_update(
            conn,
            args.recordings,
            window_ms=int(args.window_sec * 1000),
            min_stable_age_ms=int(args.stable_age_sec * 1000),
            segment_duration_ms=int(args.segment_duration_sec * 1000),
            cameras=cameras,
        )
        print(
            "recordings index (incremental): "
            f"+{stats['inserted']} new, {stats['updated']} updated, "
            f"{stats['skipped_unstable']} unstable, {stats['marked_missing']} missing, "
            f"cameras={','.join(stats['cameras']) or '-'}"
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
