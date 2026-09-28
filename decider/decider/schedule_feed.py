"""Scheduled feeding plan — pure, unit-testable (no env, no network, no clock).

`now` and the journal are injected, so the same plan() decides both the normal
tick and the start-of-process catch-up (no separate catch-up code needed).

Plan logic (the cap):
  pending = today's slots whose time has ARRIVED (now_local >= slot today) and
            which the journal has NOT yet marked, oldest→newest.
  If pending is empty → [].
  Otherwise the LAST `catchup_max` pending slots (the latest by time, closest to
  now) become action="fed"; every OLDER pending slot becomes
  action="maintenance" (marked in the journal but NOT dispensed).

Why: after a restart or a stretch of downtime we don't dump a whole day's worth
of food at once — only the most recent `catchup_max` missed slots actually feed;
the rest are recorded as "maintenance" (skipped-by-cap) so the day's record is
complete and they won't be retried. A single missed slot → 1 fed; two → 2 fed;
five with catchup_max=2 → the two latest fed, the three earliest maintenance.

Day boundary is in `tz`: only TODAY's arrived slots are pending — yesterday's
slots are never in pending, and after local midnight today's slots reset (the
journal keys on the local slot_date).

plan() never touches the feeder; it returns [(slot, action)] and the main.py
driver performs the dispense + journal writes.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    """A scheduled feeding slot for a specific local day."""
    date: str   # "YYYY-MM-DD" (local tz)
    time: str   # "HH:MM"


def parse_times(times: list[str]) -> list[dt.time]:
    """Validate + parse "HH:MM" strings into time objects (sorted, deduped).

    Raises ValueError on a malformed entry so the process fails loudly at start
    rather than silently skipping a feeding."""
    parsed: list[dt.time] = []
    for raw in times:
        s = raw.strip()
        if not s:
            continue
        try:
            hh, mm = s.split(":")
            t = dt.time(int(hh), int(mm))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"invalid FEED_TIMES entry {raw!r} (expected HH:MM)"
            ) from e
        parsed.append(t)
    return sorted(set(parsed))


class ScheduleFeeder:
    def __init__(self, *, times: list[str], grain_num: int, tz: dt.tzinfo,
                 catchup_max: int, journal, feeder_id: str) -> None:
        self.times = parse_times(times)        # may be empty (caller warns)
        self.grain_num = grain_num
        self.tz = tz
        self.catchup_max = max(0, int(catchup_max))
        self.journal = journal
        self.feeder_id = feeder_id

    def plan(self, now: dt.datetime) -> list[tuple[Slot, str]]:
        """Return [(slot, action)] for this instant; action ∈ {"fed","maintenance"}.

        `now` may be tz-aware or naive; it is interpreted in `self.tz`."""
        if not self.times:
            return []
        now_local = now.astimezone(self.tz) if now.tzinfo else now.replace(tzinfo=self.tz)
        today = now_local.date()
        date_str = today.isoformat()

        # Today's slots whose time has arrived and that aren't already journaled.
        pending: list[Slot] = []
        for t in self.times:                                   # times is sorted
            slot_dt = dt.datetime.combine(today, t, tzinfo=self.tz)
            if now_local < slot_dt:
                continue                                       # not arrived yet
            slot = Slot(date=date_str, time=t.strftime("%H:%M"))
            if self.journal.was_slot_marked(self.feeder_id, slot.date, slot.time):
                continue                                       # already handled
            pending.append(slot)

        if not pending:
            return []

        # Cap: the latest `catchup_max` pending slots feed; older ones maintenance.
        cut = len(pending) - self.catchup_max
        out: list[tuple[Slot, str]] = []
        for i, slot in enumerate(pending):
            out.append((slot, "fed" if i >= cut else "maintenance"))
        return out
