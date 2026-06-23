"""Show how a cat has been eating: door_sessions for one cat over the last N days.

Standalone, stdlib-only (mirrors tools/configure.py — no project imports), so it
runs anywhere with python3 and a path to the shared feed journal. Read-only.

    python3 tools/feed_log.py alisa --days 7 --db data/feed_journal/journal.db

Columns: opened (local time), feeder, dur (duration_sec; "open"/"interrupted"
when the session has no clean close), meal (meal_sec), counted (✓ if the meal was
long enough to count), reason (close_reason). A session is a single door
open→close episode (see feeder/journal.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

UTC = dt.timezone.utc
DEFAULT_DB = "data/feed_journal/journal.db"


def _resolve_tz(name: str | None) -> dt.tzinfo:
    """IANA tz name → tzinfo, or the local tz when not given/unknown."""
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name)
        except Exception:
            print(f"[feed-log] unknown tz {name!r}; using local time", file=sys.stderr)
    local = dt.datetime.now().astimezone().tzinfo
    assert local is not None
    return local


def _fmt_local(iso_utc: str, tz: dt.tzinfo) -> str:
    """ISO-UTC opened_at → 'MM-DD HH:MM' in display tz (best-effort)."""
    try:
        return dt.datetime.fromisoformat(iso_utc).astimezone(tz).strftime("%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(iso_utc)


def _dur_cell(duration_sec, incomplete: int) -> str:
    if incomplete:
        return "interrupted"
    if duration_sec is None:
        return "open"
    return f"{int(round(duration_sec))}s"


def _num_cell(value, suffix: str = "s") -> str:
    return "—" if value is None else f"{int(round(value))}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cat", help="cat name to report on")
    ap.add_argument("--days", type=int, default=3, help="look-back window (default 3)")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"journal DB (default {DEFAULT_DB})")
    ap.add_argument("--tz", default=None, help="IANA tz for display (default: local)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"no journal yet at {args.db} — nothing to show "
              "(the feeder writes it once it runs)")
        return                                   # exit 0, not a stacktrace

    tz = _resolve_tz(args.tz)
    threshold = (dt.datetime.now(UTC) - dt.timedelta(days=args.days)).isoformat()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """SELECT opened_at, feeder_id, duration_sec, meal_sec, counted_meal,
                      incomplete, close_reason
               FROM door_sessions
               WHERE cat=? AND opened_at >= ?
               ORDER BY opened_at""",
            (args.cat, threshold),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"no sessions for {args.cat} in last {args.days} days")
        return

    print(f"door sessions for {args.cat} — last {args.days} days "
          f"(times in {tz})")
    print(f"  {'opened':<12} {'feeder':<10} {'dur':>11} {'meal':>6} "
          f"{'cnt':>3}  reason")
    total = 0
    counted_meal_sum = 0.0
    interrupted = 0
    for opened_at, feeder_id, duration_sec, meal_sec, counted_meal, incomplete, reason in rows:
        total += 1
        if incomplete:
            interrupted += 1
        if counted_meal and meal_sec is not None:
            counted_meal_sum += meal_sec
        print(f"  {_fmt_local(opened_at, tz):<12} {str(feeder_id or '—'):<10} "
              f"{_dur_cell(duration_sec, incomplete):>11} "
              f"{_num_cell(meal_sec):>6} {('✓' if counted_meal else '—'):>3}  "
              f"{reason or '—'}")

    print(f"  ── {total} session(s); counted meal total "
          f"{int(round(counted_meal_sum))}s; {interrupted} interrupted")


if __name__ == "__main__":
    main()
