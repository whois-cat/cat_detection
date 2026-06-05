"""Per-cat cooldown persistence in SQLite.

Port of donor feeder_state.py — replaces JSON file with SQLite for
atomic writes and concurrent-reader safety.

Cooldown is recorded at the END of a meal (door close), not at open.
The DB column last_opened_at is kept as-is to preserve existing rows;
semantically it now holds the timestamp of the last meal end.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

UTC = dt.timezone.utc


def _utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


class CooldownState:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cooldowns (
                   cat            TEXT PRIMARY KEY,
                   last_opened_at TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def record_meal_end(self, cat: str, ended_at: dt.datetime | None = None) -> None:
        ts = (ended_at or _utc_now()).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO cooldowns (cat, last_opened_at) VALUES (?, ?)",
            (cat, ts),
        )
        self._conn.commit()

    def last_opened_at(self, cat: str) -> dt.datetime | None:
        row = self._conn.execute(
            "SELECT last_opened_at FROM cooldowns WHERE cat=?", (cat,)
        ).fetchone()
        if row is None:
            return None
        return dt.datetime.fromisoformat(row[0]).replace(tzinfo=UTC)

    def remaining_hours(
        self, cat: str, cooldown_hours: float, now: dt.datetime | None = None
    ) -> float:
        last = self.last_opened_at(cat)
        if last is None:
            return 0.0
        elapsed_h = ((now or _utc_now()) - last).total_seconds() / 3600.0
        return max(0.0, cooldown_hours - elapsed_h)

    def is_ready(
        self, cat: str, cooldown_hours: float, now: dt.datetime | None = None
    ) -> bool:
        return self.remaining_hours(cat, cooldown_hours, now=now) <= 0.0
