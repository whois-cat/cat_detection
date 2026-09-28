"""Feed + door journal — one shared SQLite log for every feeder.

ONE database for all feeders (env FEED_JOURNAL_DB, default
/data/feed_journal/journal.db), mounted into each feeder container. Two tables:

  - feed_events  — one row per feeding decision: a scheduled-slot feed/maintenance
                   or an empty-bowl refill. Unified feed metrics across both modes
                   and the double-feed guard for the scheduled path (UNIQUE per
                   slot).
  - door_sessions — one row per door open→close session (a "meal"). The row is
                   INSERTed at OPEN (opened_at + cat known; closed_at/duration/
                   meal NULL) and UPDATEd at CLOSE (finalize). A session left open
                   by a crashed process is marked incomplete='interrupted' by the
                   next process's startup recovery, with the close fields kept NULL
                   (honestly unknown).

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
    cat          TEXT,                 -- cat the door opened for (known at open)
    opened_at    TEXT    NOT NULL,     -- ISO UTC, written AT OPEN
    closed_at    TEXT,                 -- NULL while open / if interrupted by a crash
    duration_sec REAL,                 -- NULL while open / if interrupted
    meal_sec     REAL,                 -- NULL while open / if interrupted
    counted_meal INTEGER NOT NULL DEFAULT 0,  -- 1 if meal_sec >= MIN_MEAL_SEC
    incomplete   INTEGER NOT NULL DEFAULT 0,  -- 1 if the session did not close cleanly
    close_reason TEXT                  -- NULL while open; cat_left|multi_cat|stream_lost|backstop|interrupted
);
CREATE INDEX IF NOT EXISTS idx_door_feeder_opened
    ON door_sessions(feeder_id, opened_at);
CREATE INDEX IF NOT EXISTS idx_door_cat_opened
    ON door_sessions(cat, opened_at);
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
        """Idempotent migrations for pre-existing journals.

        door_sessions changed shape between an earlier (reverted) close-only
        version — `closed_at TEXT NOT NULL`, no `incomplete` column — and the
        current open→finalize version. CREATE TABLE IF NOT EXISTS leaves an
        existing old table untouched, so we detect the old form and rebuild it.
        Dev-only data, so DROP+CREATE is acceptable; this guarantees a fresh DB
        and a DB from the reverted commit converge to the same schema."""
        info = {row[1]: row for row in
                self._conn.execute("PRAGMA table_info(door_sessions)")}
        if not info:
            return                                   # SCHEMA already made the new table
        # row = (cid, name, type, notnull, dflt_value, pk); [3] == 1 → NOT NULL.
        old_form = ("incomplete" not in info) or (info["closed_at"][3] == 1)
        if old_form:
            self._conn.execute("DROP TABLE door_sessions")
            self._conn.executescript(SCHEMA)         # recreate under the new schema

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

    # ---- door sessions (meals): open → finalize, with crash recovery ----

    def open_session(self, feeder_id: str, cat: str | None,
                     opened_at_wall: float) -> int:
        """Insert a row AT door open and return its id. closed_at / duration_sec /
        meal_sec / close_reason are left NULL until finalize_session()."""
        cur = self._conn.execute(
            """INSERT INTO door_sessions (feeder_id, cat, opened_at)
               VALUES (?, ?, ?)""",
            (feeder_id, cat, _wall_to_iso(opened_at_wall)),
        )
        return int(cur.lastrowid)

    def finalize_session(self, session_id: int, closed_at_wall: float,
                         meal_sec: float | None, close_reason: str,
                         min_meal_sec: float) -> None:
        """Complete the open row: fill closed_at, duration (from the stored
        opened_at), meal_sec and counted_meal, and the close reason. `incomplete`
        stays 0. No-op if the row vanished (defensive)."""
        row = self._conn.execute(
            "SELECT opened_at FROM door_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if row is None:
            return
        opened_epoch = dt.datetime.fromisoformat(row[0]).timestamp()
        duration = max(0.0, closed_at_wall - opened_epoch)
        counted = 1 if (meal_sec is not None and meal_sec >= min_meal_sec) else 0
        self._conn.execute(
            """UPDATE door_sessions
                   SET closed_at=?, duration_sec=?, meal_sec=?,
                       counted_meal=?, close_reason=?
               WHERE id=?""",
            (_wall_to_iso(closed_at_wall), duration, meal_sec,
             counted, close_reason, session_id),
        )

    def recover_interrupted(self, feeder_id: str) -> int:
        """Mark every still-open session for this feeder (closed_at IS NULL —
        left behind by a crashed process) as incomplete='interrupted'. closed_at /
        duration_sec / meal_sec stay NULL (honestly unknown); counted_meal stays
        0. Returns the number of rows recovered (for logging)."""
        # closed_at stays NULL, so also require incomplete=0 to skip rows already
        # recovered by an earlier call — otherwise re-running would re-match them.
        cur = self._conn.execute(
            """UPDATE door_sessions
                   SET incomplete=1, close_reason='interrupted'
               WHERE feeder_id=? AND closed_at IS NULL AND incomplete=0""",
            (feeder_id,),
        )
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
