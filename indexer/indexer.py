"""Live recording indexer.

Keeps ``recording_segments`` (inside the detector events DB) in sync with the
mp4 files mediamtx writes to ``data/recordings``, so the WebUI timeline keeps
growing automatically — ``just recordings-index-rebuild`` becomes a
recovery/backfill tool, not part of normal operation.

Design (see recordings_index.py for the primitives):

  * Startup catch-up: index everything from where we left off
    (``MAX(end_ms)`` already in the DB) up to now, capped to STARTUP_LOOKBACK_SEC
    so a long-down indexer never walks the whole archive. Then one full
    reconcile to drop rows for files deleted while we were down.
  * Every INDEX_INTERVAL_SEC: incrementally index the last INDEX_WINDOW_SEC of
    files. Only recent files are scanned, never the 30k+ archive. Files modified
    within MIN_STABLE_AGE_SEC are assumed to still be open for writing by
    mediamtx and are skipped until they settle (no partial mp4s indexed).
  * Every RECONCILE_INTERVAL_SEC: a full reconcile pass drops index rows whose
    files are gone — this is how pruner deletions of *old* segments (outside the
    incremental window) stop being advertised on the timeline.

This service is the single writer of ``recording_segments``; the detector only
reads it for /recordings/ranges. It does not touch mediamtx, the pruner, or
WebRTC. Reads/writes use WAL so the detector keeps serving ranges concurrently.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from recordings_index import (
    DEFAULT_MIN_STABLE_AGE_MS,
    incremental_update,
    last_indexed_to_ms,
    open_index_db,
    reconcile_deletions,
)

# Repo-root-relative by default (WORKDIR=/app) so stored segment paths match
# `just recordings-index-rebuild` exactly (the path column is UNIQUE).
RECORDINGS_DIR        = Path(os.environ.get("RECORDINGS_DIR", "data/recordings"))
EVENTS_DB             = Path(os.environ.get("EVENTS_DB", "/data/events/events.db"))
SEGMENT_DURATION_SEC  = float(os.environ.get("SEGMENT_DURATION_SEC", "30"))
INDEX_INTERVAL_SEC    = float(os.environ.get("INDEX_INTERVAL_SEC", "30"))
INDEX_WINDOW_SEC      = float(os.environ.get("INDEX_WINDOW_SEC", "900"))     # 15 min
RECONCILE_INTERVAL_SEC = float(os.environ.get("RECONCILE_INTERVAL_SEC", "300"))  # 5 min
STARTUP_LOOKBACK_SEC  = float(os.environ.get("STARTUP_LOOKBACK_SEC", str(24 * 3600)))
MIN_STABLE_AGE_SEC    = float(os.environ.get("MIN_STABLE_AGE_SEC", str(DEFAULT_MIN_STABLE_AGE_MS / 1000)))

# Optional allow-list. Empty => index every camera dir found. Defends against a
# deleted/typo camera dir ever being re-indexed (e.g. the old "beidge" typo).
RECORDINGS_CAMERAS = {
    c.strip() for c in os.environ.get("RECORDINGS_CAMERAS", "").split(",") if c.strip()
} or None

SEGMENT_DURATION_MS = int(SEGMENT_DURATION_SEC * 1000)
INDEX_WINDOW_MS     = int(INDEX_WINDOW_SEC * 1000)
MIN_STABLE_AGE_MS   = int(MIN_STABLE_AGE_SEC * 1000)
STARTUP_LOOKBACK_MS = int(STARTUP_LOOKBACK_SEC * 1000)
# Re-check the last couple of segments on catch-up so a boundary end_ms that was
# left open (newest file at last run) gets corrected.
CATCHUP_OVERLAP_MS  = 2 * SEGMENT_DURATION_MS


def _fmt_last_indexed(last: dict[str, int]) -> str:
    if not last:
        return "-"
    return ",".join(f"{cam}@{ts}" for cam, ts in sorted(last.items()))


def run_incremental(conn, *, since_ms=None, cameras=None, label="cycle") -> dict:
    started = time.perf_counter()
    stats = incremental_update(
        conn,
        RECORDINGS_DIR,
        since_ms=since_ms,
        window_ms=INDEX_WINDOW_MS,
        min_stable_age_ms=MIN_STABLE_AGE_MS,
        segment_duration_ms=SEGMENT_DURATION_MS,
        cameras=cameras if cameras is not None else RECORDINGS_CAMERAS,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    # Only log cycles that did something (or the explicit catch-up), so steady
    # idle operation doesn't spam a line every 30s.
    if label != "cycle" or stats["inserted"] or stats["marked_missing"]:
        print(
            f"[indexer] {label}: +{stats['inserted']} new "
            f"{stats['updated']} updated {stats['skipped_unstable']} unstable "
            f"{stats['marked_missing']} missing "
            f"since={stats['since_ms']} last_indexed={_fmt_last_indexed(stats['last_indexed'])} "
            f"elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )
    return stats


def run_reconcile(conn) -> dict:
    started = time.perf_counter()
    stats = reconcile_deletions(conn, cameras=RECORDINGS_CAMERAS)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if stats.get("skipped_cameras"):
        print(
            f"[indexer] reconcile: skipped cameras {stats['skipped_cameras']} "
            f"(all their ready rows missing — dir unmounted/empty?); "
            f"deleted={stats['deleted']} checked={stats['checked']}",
            flush=True,
        )
    elif stats["deleted"]:
        print(
            f"[indexer] reconcile: checked={stats['checked']} "
            f"deleted={stats['deleted']} elapsed_ms={elapsed_ms:.1f}",
            flush=True,
        )
    return stats


def catchup_since_ms(conn, camera: str, now_ms: int) -> int:
    """Per-camera restart catch-up: resume from that camera's own MAX(end_ms),
    never going back further than STARTUP_LOOKBACK_SEC.

    Per camera (not a global MAX) so a camera that lagged or was added later
    still gets fully caught up even if a busier camera is already current."""
    floor_ms = now_ms - STARTUP_LOOKBACK_MS
    last = last_indexed_to_ms(conn, cameras={camera})
    if last is None:
        return floor_ms
    return max(floor_ms, last - CATCHUP_OVERLAP_MS)


def resolve_cameras(conn) -> list[str]:
    """Cameras to catch up at startup. Prefer the explicit allow-list; otherwise
    fall back to the union of on-disk camera dirs and already-indexed cameras.
    With the allow-list set (RECORDINGS_CAMERAS=beige,grey) a deleted typo dir
    like ``beidge`` is never touched."""
    if RECORDINGS_CAMERAS:
        return sorted(RECORDINGS_CAMERAS)
    cams: set[str] = set()
    try:
        cams |= {e.name for e in os.scandir(RECORDINGS_DIR) if e.is_dir()}
    except OSError:
        pass
    cams |= {
        row[0]
        for row in conn.execute("SELECT DISTINCT camera_id FROM recording_segments")
    }
    return sorted(cams)


def main() -> None:
    print(
        f"[indexer] started: recordings={RECORDINGS_DIR} db={EVENTS_DB} "
        f"interval={INDEX_INTERVAL_SEC}s window={INDEX_WINDOW_SEC}s "
        f"reconcile={RECONCILE_INTERVAL_SEC}s stable_age={MIN_STABLE_AGE_SEC}s "
        f"cameras={','.join(sorted(RECORDINGS_CAMERAS)) if RECORDINGS_CAMERAS else 'all'}",
        flush=True,
    )
    conn = open_index_db(EVENTS_DB)

    now_ms = int(time.time() * 1000)
    try:
        for camera in resolve_cameras(conn):
            since = catchup_since_ms(conn, camera, now_ms)
            print(f"[indexer] startup catch-up {camera} from {since} (now {now_ms})", flush=True)
            run_incremental(conn, since_ms=since, cameras={camera}, label=f"catch-up[{camera}]")
        run_reconcile(conn)
    except Exception as e:  # never let startup hiccups kill the loop
        print(f"[indexer] startup error: {e!r}", flush=True)

    last_reconcile = time.monotonic()
    while True:
        time.sleep(INDEX_INTERVAL_SEC)
        try:
            run_incremental(conn)
            if time.monotonic() - last_reconcile >= RECONCILE_INTERVAL_SEC:
                run_reconcile(conn)
                last_reconcile = time.monotonic()
        except Exception as e:
            print(f"[indexer] error: {e!r}", flush=True)


if __name__ == "__main__":
    main()
