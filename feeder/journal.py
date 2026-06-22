"""Feed journal — one shared SQLite log of all dispense events, every feeder.

ONE database for all feeders (env FEED_JOURNAL_DB, default
/data/feed_journal/journal.db), mounted into each feeder container. Each row is
one feeding decision: a scheduled-slot feed/maintenance, or an empty-bowl refill.
This gives unified feed metrics across both modes and is the natural
double-feed guard for the scheduled path (UNIQUE per slot).

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
"""


def _utc_now_iso() -> str:
    return dt.datetime.now(UTC).isoformat()


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

    def close(self) -> None:
        self._conn.close()
