"""
Detection-aware pruner. Periodically scans recording segments and deletes any
that don't contain (or surround) at least one detection event.

A segment is *kept* if any detection event's wall_ms falls within:
    [segment_start - KEEP_PRE_ROLL_SEC, segment_end + KEEP_POST_ROLL_SEC]

…OR if it's newer than KEEP_RECENT_HOURS (default 24): recent video stays
intact so the timeline scrubbing experience is predictable, even for boring
periods. Detection-based pruning only kicks in once a segment ages out of
the "recent" window. mediamtx still enforces a hard cap via recordDeleteAfter.

Otherwise the segment is deleted from disk. mediamtx's /list will reflect the
change on its next periodic re-scan.

Segment file naming follows mediamtx's recordPath template:
    /recordings/<path>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4
The timestamp in the filename is the segment START in local time.

Event source is now the detector's SQLite DB (one row per detection). We open
it read-only via URI; with WAL mode the detector can keep writing concurrently.

Environment:
    RECORDINGS_DIR        — root of recordings (default /recordings)
    EVENTS_DB             — path to detector's SQLite DB (default /data/events/events.db)
    SEGMENT_DURATION_SEC  — must match mediamtx recordSegmentDuration (default 30)
    KEEP_PRE_ROLL_SEC     — keep this much before each detection (default 30)
    KEEP_POST_ROLL_SEC    — keep this much after each detection  (default 30)
    KEEP_RECENT_HOURS     — never prune segments newer than this (default 24)
    PRUNE_INTERVAL_SEC    — how often to run the pruner (default 3600 = 1h)
    DRY_RUN               — set to "1" to log what would be deleted, no-op
"""
import os
import sqlite3
import time
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

RECORDINGS_DIR     = Path(os.environ.get("RECORDINGS_DIR", "/recordings"))
EVENTS_DB          = Path(os.environ.get("EVENTS_DB", "/data/events/events.db"))
SEGMENT_DURATION   = int(os.environ.get("SEGMENT_DURATION_SEC", "30"))
KEEP_PRE_ROLL_SEC  = int(os.environ.get("KEEP_PRE_ROLL_SEC", "30"))
KEEP_POST_ROLL_SEC = int(os.environ.get("KEEP_POST_ROLL_SEC", "30"))
KEEP_RECENT_HOURS  = int(os.environ.get("KEEP_RECENT_HOURS", "24"))
PRUNE_INTERVAL_SEC = int(os.environ.get("PRUNE_INTERVAL_SEC", "3600"))
DRY_RUN            = os.environ.get("DRY_RUN", "0") == "1"


def parse_segment_timestamp_ms(name: str) -> int | None:
    """mediamtx recordPath: <YYYY-MM-DD_HH-MM-SS-ffffff>.mp4 (local time)."""
    stem = name.rsplit(".", 1)[0]
    try:
        dt = datetime.strptime(stem, "%Y-%m-%d_%H-%M-%S-%f")
    except ValueError:
        return None
    # mediamtx writes in LOCAL time. `.timestamp()` interprets naive datetimes
    # as local for the host where the pruner runs — both containers share the
    # same TZ if compose sets it (default UTC).
    return int(dt.timestamp() * 1000)


def load_event_times() -> list[int]:
    """Read all detection wall_ms from the SQLite DB. Returns sorted list."""
    if not EVENTS_DB.exists():
        return []
    # Read-only URI lets us coexist with the detector's WAL writes.
    uri = f"file:{EVENTS_DB}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return []
    try:
        cur = conn.execute("SELECT wall_ms FROM events ORDER BY wall_ms")
        return [r[0] for r in cur]
    finally:
        conn.close()


def segment_has_event(start_ms: int, end_ms: int, events_ms: list[int]) -> bool:
    """True if any event lies within the segment ± roll margin."""
    lo = bisect_left(events_ms, start_ms - KEEP_PRE_ROLL_SEC * 1000)
    hi = bisect_left(events_ms, end_ms + KEEP_POST_ROLL_SEC * 1000)
    return lo != hi


def prune_once() -> tuple[int, int, int, int]:
    """Returns (kept, deleted, unparseable, skipped_recent)."""
    events_ms = load_event_times()
    if not events_ms:
        # Sanity: if there are zero detections we don't want to nuke
        # everything (likely a config/data issue rather than truly nothing
        # happening). Skip and warn.
        print("[pruner] no events found; skipping prune as a safety measure", flush=True)
        return (0, 0, 0, 0)

    seg_dur_ms = SEGMENT_DURATION * 1000
    recent_cutoff_ms = int(time.time() * 1000) - KEEP_RECENT_HOURS * 3600 * 1000
    kept = deleted = unparseable = skipped_recent = 0
    for seg in sorted(RECORDINGS_DIR.rglob("*.mp4")):
        ts = parse_segment_timestamp_ms(seg.name)
        if ts is None:
            unparseable += 1
            continue
        # Recent segments are always kept — no detection-based pruning until
        # they age out of the protection window.
        if ts >= recent_cutoff_ms:
            kept += 1
            skipped_recent += 1
            continue
        if segment_has_event(ts, ts + seg_dur_ms, events_ms):
            kept += 1
            continue
        if DRY_RUN:
            print(f"[pruner] would delete {seg}", flush=True)
        else:
            try:
                seg.unlink()
            except OSError as e:
                print(f"[pruner] delete failed {seg}: {e}", flush=True)
                continue
        deleted += 1
    return (kept, deleted, unparseable, skipped_recent)


def main():
    print(
        f"[pruner] started: segment={SEGMENT_DURATION}s "
        f"pre={KEEP_PRE_ROLL_SEC}s post={KEEP_POST_ROLL_SEC}s "
        f"interval={PRUNE_INTERVAL_SEC}s dry_run={DRY_RUN}",
        flush=True,
    )
    while True:
        try:
            kept, deleted, bad, recent = prune_once()
            print(
                f"[pruner] pass complete: kept={kept} (of which {recent} were "
                f"protected as recent) deleted={deleted} unparseable={bad}",
                flush=True,
            )
        except Exception as e:
            print(f"[pruner] error: {e!r}", flush=True)
        time.sleep(PRUNE_INTERVAL_SEC)


if __name__ == "__main__":
    main()
