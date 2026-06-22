"""Scheduled-feed planner: the catch-up cap, idempotency, day boundary, and
feed-failure retry. The planner is pure — `now` and the journal are injected, so
no clock, env, or network is involved.

Cap rule under test: of the arrived-but-unmarked slots today, the latest
`catchup_max` become "fed" and every older one becomes "maintenance".
"""
import datetime as dt

import pytest

from schedule_feed import ScheduleFeeder, Slot, parse_times

TZ = dt.timezone.utc


class FakeJournal:
    """In-memory stand-in: marks slots and answers was_slot_marked, mirroring the
    real FeedJournal's INSERT OR IGNORE semantics (record returns False on dup)."""
    def __init__(self):
        self.marked: dict[tuple[str, str, str], dict] = {}

    def was_slot_marked(self, feeder_id, slot_date, slot_time):
        return (feeder_id, slot_date, slot_time) in self.marked

    def record_scheduled(self, feeder_id, slot_date, slot_time, status, grain_num, acked):
        key = (feeder_id, slot_date, slot_time)
        if key in self.marked:
            return False
        self.marked[key] = {"status": status, "grain_num": grain_num, "acked": acked}
        return True


def _feeder(journal, *, times, catchup_max=2, grain=1):
    return ScheduleFeeder(times=times, grain_num=grain, tz=TZ,
                          catchup_max=catchup_max, journal=journal, feeder_id="f1")


def _at(h, m=0):
    return dt.datetime(2026, 6, 22, h, m, tzinfo=TZ)


def _apply(plan, journal, feeder_id="f1", *, fed_ok=True):
    """Mimic the main.py driver: dispense (fed) / journal (both), honoring the
    fed-failure rule (a failed feed does NOT mark its slot)."""
    for slot, action in plan:
        if action == "fed":
            if not fed_ok:
                continue                          # leave pending for retry
            journal.record_scheduled(feeder_id, slot.date, slot.time, "fed",
                                     1, acked=True)
        else:
            journal.record_scheduled(feeder_id, slot.date, slot.time,
                                     "maintenance", 0, acked=False)


# ---- parse_times ----

def test_parse_times_sorts_dedupes_and_skips_blank():
    assert parse_times(["13:00", "08:00", "13:00", ""]) == [dt.time(8), dt.time(13)]


def test_parse_times_rejects_malformed():
    with pytest.raises(ValueError):
        parse_times(["08:00", "9am"])


# ---- the cap ----

def test_single_miss_feeds_once():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "13:00", "18:00"], catchup_max=2)
    # now = 13:30 → 08:00 and 13:00 have arrived; both unmarked.
    plan = f.plan(_at(13, 30))
    # Latest catchup_max(=2) of 2 pending → both fed (no maintenance here).
    assert [(s.time, a) for s, a in plan] == [("08:00", "fed"), ("13:00", "fed")]


def test_five_missed_with_cap_two():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "10:00", "13:00", "15:00", "18:00"], catchup_max=2)
    plan = f.plan(_at(18, 30))            # all five arrived
    got = [(s.time, a) for s, a in plan]
    assert got == [
        ("08:00", "maintenance"),
        ("10:00", "maintenance"),
        ("13:00", "maintenance"),
        ("15:00", "fed"),                 # latest 2 feed
        ("18:00", "fed"),
    ]


def test_two_missed_feeds_two():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "18:00"], catchup_max=2)
    plan = f.plan(_at(18, 30))
    assert [a for _s, a in plan] == ["fed", "fed"]


def test_not_yet_arrived_slots_excluded():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "13:00", "18:00"], catchup_max=2)
    plan = f.plan(_at(9, 0))             # only 08:00 arrived
    assert [(s.time, a) for s, a in plan] == [("08:00", "fed")]


# ---- idempotency / no re-catchup ----

def test_replan_after_marking_is_empty():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "13:00"], catchup_max=2)
    first = f.plan(_at(13, 30))
    _apply(first, j)
    assert f.plan(_at(13, 35)) == []     # nothing left pending


def test_marked_slots_are_not_redispensed_after_partial_run():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "10:00", "13:00"], catchup_max=1)
    # First pass at 13:30: 13:00 fed, 08:00+10:00 maintenance.
    _apply(f.plan(_at(13, 30)), j)
    assert j.marked[("f1", "2026-06-22", "13:00")]["status"] == "fed"
    assert j.marked[("f1", "2026-06-22", "10:00")]["status"] == "maintenance"
    # A later tick must not touch any already-marked slot.
    assert f.plan(_at(14, 0)) == []


# ---- day boundary ----

def test_yesterday_slots_not_pending():
    j = FakeJournal()
    # Mark yesterday's 18:00 as fed; today it must not reappear.
    j.record_scheduled("f1", "2026-06-21", "18:00", "fed", 1, True)
    f = _feeder(j, times=["08:00", "18:00"], catchup_max=2)
    plan = f.plan(_at(0, 30))            # just after local midnight, no slot arrived
    assert plan == []


def test_today_slots_reset_after_midnight():
    j = FakeJournal()
    f = _feeder(j, times=["08:00"], catchup_max=2)
    # Yesterday 08:00 was fed.
    j.record_scheduled("f1", "2026-06-21", "08:00", "fed", 1, True)
    # Today 08:30 — today's 08:00 is a fresh, unmarked slot → feeds again.
    plan = f.plan(_at(8, 30))
    assert [(s.date, s.time, a) for s, a in plan] == [("2026-06-22", "08:00", "fed")]


# ---- feed failure ----

def test_feed_failure_keeps_slot_pending():
    j = FakeJournal()
    f = _feeder(j, times=["13:00"], catchup_max=2)
    plan = f.plan(_at(13, 30))
    _apply(plan, j, fed_ok=False)        # dispense failed → slot NOT marked
    assert ("f1", "2026-06-22", "13:00") not in j.marked
    # Next tick re-plans the same slot for retry.
    assert [(s.time, a) for s, a in f.plan(_at(13, 31))] == [("13:00", "fed")]


def test_maintenance_marked_even_when_a_fed_fails():
    j = FakeJournal()
    f = _feeder(j, times=["08:00", "10:00", "18:00"], catchup_max=1)
    # 18:00 = fed (fails), 08:00 + 10:00 = maintenance (hardware-independent).
    _apply(f.plan(_at(18, 30)), j, fed_ok=False)
    assert j.marked[("f1", "2026-06-22", "08:00")]["status"] == "maintenance"
    assert j.marked[("f1", "2026-06-22", "10:00")]["status"] == "maintenance"
    assert ("f1", "2026-06-22", "18:00") not in j.marked   # failed fed retries
    # Retry tick: only the failed fed slot remains.
    assert [(s.time, a) for s, a in f.plan(_at(18, 31))] == [("18:00", "fed")]


# ---- empty times ----

def test_empty_times_is_noop():
    j = FakeJournal()
    f = _feeder(j, times=[], catchup_max=2)
    assert f.plan(_at(23, 59)) == []
