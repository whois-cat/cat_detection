"""Feed + door journal — one shared SQLite log for every feeder.

ONE database for all feeders (env FEED_JOURNAL_DB, default
/data/feed_journal/journal.db), mounted into each feeder container. Two tables:

  - feed_events  — one row per feeding decision: a scheduled-slot feed/maintenance
                   or an empty-bowl refill. Unified feed metrics across both modes
                   and the double-feed guard for the scheduled path (UNIQUE per
                   slot).
  - door_sessions — one row per door open→close session (a "meal"): which cat the
                   door opened for, when it opened/closed, how long it stayed open,
                   the presence-based meal estimate, whether the meal "counted"
                   (>= MIN_MEAL_SEC), and the close reason. Recorded on close
                   across all close paths (normal, backstop, fail-safe).

Style mirrors detector/storage.py: WAL + synchronous=NORMAL + busy_timeout so
concurrent feeders on a shared (possibly CIFS) volume don't fail under
contention; mkdir parents; idempotent additive _migrate(). All DB access lives
here behind FeedJournal, so a future journal migration touches only this file.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

UTC = dt.timezone.utc


SCHEMA = """
CREATE TABLE IF NOT EXISTS feed_events (
    id        INTEGER PRIMARY KEY,
    feeder_id TEXT    NOT NULL,
    mode      TEXT    NOT NULL,        -- "scheduled" | "empty_bowl"
    status    TEXT    NOT NULL,        -- "fed" | "maintenance"
    slot_date TEXT,                    -- "YYYY-MM-DD" local tz (scheduled); NULL for empty_bowl
    slot_time TEXT,                    -- "HH:MM" (scheduled); NULL for empty_bowl
    grain_num REAL,                    -- dispensed amount; 0/NULL for maintenance
    fed_at    TEXT    NOT NULL,        -- ISO UTC of the write
    acked     INTEGER NOT NULL DEFAULT 0
);
-- One row per (feeder, slot): a slot can be marked at most once. empty_bowl rows
-- carry slot_date=slot_time=NULL, and SQLite treats NULLs as distinct in a UNIQUE
-- index, so empty_bowl rows never collide — exactly what we want.
CREATE UNIQUE INDEX IF NOT EXISTS idx_feed_slot
    ON feed_events(feeder_id, slot_date, slot_time);
CREATE INDEX IF NOT EXISTS idx_feed_feeder_time
    ON feed_events(feeder_id, fed_at);

CREATE TABLE IF NOT EXISTS door_sessions (
    id           INTEGER PRIMARY KEY,
    feeder_id    TEXT    NOT NULL,
    cat          TEXT,                 -- cat the door opened for (NULL if unknown)
    opened_at    TEXT,                 -- ISO UTC of door open (NULL if unknown)
    closed_at    TEXT    NOT NULL,     -- ISO UTC of door close
    open_sec     REAL,                 -- door physically-open duration (wall)
    meal_sec     REAL,                 -- presence-based meal estimate
    counted_meal INTEGER NOT NULL DEFAULT 0,  -- 1 if meal_sec >= MIN_MEAL_SEC
    close_reason TEXT    NOT NULL      -- cat_left | multi_cat | identity_change | stream_lost | backstop | ...
);
CREATE INDEX IF NOT EXISTS idx_door_feeder_time
    ON door_sessions(feeder_id, closed_at);
"""


def _utc_now_iso() -> str:
    return dt.datetime.now(UTC).isoformat()


def _wall_to_iso(wall_t: float) -> str:
    """Epoch-seconds wall time (feeder events use wall_ms/1000) → ISO UTC."""
    return dt.datetime.fromtimestamp(wall_t, tz=UTC).isoformat()


class FeedJournal:
    def __init__(self, db_path: Path | str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit, like detector/storage.py.
        self._conn = sqlite3.connect(str(path), isolation_level=None,
                                     check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent, additive migrations for pre-existing journals. No columns
        have changed yet; this is the hook so future changes stay in this file."""
        # cols = {row[1] for row in self._conn.execute("PRAGMA table_info(feed_events)")}
        # (add ALTER TABLE ... ADD COLUMN guarded by `if "<col>" not in cols` here)
        return

    # ---- scheduled mode ----

    def record_scheduled(self, feeder_id: str, slot_date: str, slot_time: str,
                         status: str, grain_num: float, acked: bool) -> bool:
        """Mark a scheduled slot. INSERT OR IGNORE on the UNIQUE(feeder, slot)
        index: returns True if THIS call inserted the row (slot was unmarked),
        False if the slot was already marked (by an earlier tick / another racing
        tick). The caller dispenses only when this returns True for a "fed"."""
        cur = self._conn.execute(
            """INSERT OR IGNORE INTO feed_events
                   (feeder_id, mode, status, slot_date, slot_time, grain_num, fed_at, acked)
               VALUES (?, 'scheduled', ?, ?, ?, ?, ?, ?)""",
            (feeder_id, status, slot_date, slot_time,
             grain_num, _utc_now_iso(), 1 if acked else 0),
        )
        return cur.rowcount > 0

    def was_slot_marked(self, feeder_id: str, slot_date: str, slot_time: str) -> bool:
        row = self._conn.execute(
            """SELECT 1 FROM feed_events
               WHERE feeder_id=? AND slot_date=? AND slot_time=? LIMIT 1""",
            (feeder_id, slot_date, slot_time),
        ).fetchone()
        return row is not None

    # ---- empty-bowl mode ----

    def record_empty_bowl(self, feeder_id: str, grain_num: float, acked: bool) -> None:
        """Append an empty-bowl refill. slot_date/slot_time are NULL (no slot),
        so the UNIQUE index never blocks these."""
        self._conn.execute(
            """INSERT INTO feed_events
                   (feeder_id, mode, status, slot_date, slot_time, grain_num, fed_at, acked)
               VALUES (?, 'empty_bowl', 'fed', NULL, NULL, ?, ?, ?)""",
            (feeder_id, grain_num, _utc_now_iso(), 1 if acked else 0),
        )

    # ---- door sessions (meals) ----

    def record_door_session(self, feeder_id: str, *, cat: str | None,
                            opened_at_wall: float | None, closed_at_wall: float,
                            open_sec: float | None, meal_sec: float | None,
                            counted_meal: bool, close_reason: str) -> None:
        """Append one door open→close session. Wall times are epoch seconds
        (feeder events use wall_ms/1000) and are stored as ISO UTC. One row per
        session — written once, on close."""
        self._conn.execute(
            """INSERT INTO door_sessions
                   (feeder_id, cat, opened_at, closed_at, open_sec, meal_sec,
                    counted_meal, close_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (feeder_id, cat,
             _wall_to_iso(opened_at_wall) if opened_at_wall is not None else None,
             _wall_to_iso(closed_at_wall), open_sec, meal_sec,
             1 if counted_meal else 0, close_reason),
        )

    def close(self) -> None:
        self._conn.close()
