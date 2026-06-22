"""Door-session journaling wired through the real main.py: every close path
(normal, fail-safe, backstop) records exactly one session, and counted_meal uses
MIN_MEAL_SEC. Container-only deps (websockets) are stubbed; the journal is a real
temp SQLite.
"""
import datetime as dt
import os
import sys
import types
from pathlib import Path

import pytest

# conftest puts both feeder/ and detector/ on sys.path; "main" is ambiguous and
# detector/main (needs aiohttp) would win. Force feeder/ ahead for this module.
_FEEDER = str(Path(__file__).resolve().parent.parent / "feeder")
sys.path.insert(0, _FEEDER)
# feeder/main reads required env at import and imports `websockets` (container-only).
sys.modules.setdefault("websockets", types.ModuleType("websockets"))
os.environ.setdefault("CAMERA_ID", "g")
os.environ.setdefault("FEEDER_ID", "f1")
os.environ.setdefault("FEEDER_API_BASE_URL", "http://x")
os.environ.setdefault("FEEDER_SERIAL_NUMBER", "SN1")

import main  # noqa: E402
from door_fsm import DoorFSM  # noqa: E402
from journal import FeedJournal  # noqa: E402

UTC = dt.timezone.utc


class FakeClient:
    def __init__(self, ok=True):
        self.ok = ok
        self.state = "open"
        self.calls = []

    def set_door(self, desired, reason):
        self.calls.append((desired, reason))
        self.state = desired
        return self.ok


def _wall(h, m=0):
    return dt.datetime(2026, 6, 22, h, m, tzinfo=UTC).timestamp()


@pytest.fixture()
def journal(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    main._journal = j
    return j


def _sessions(j):
    return j._conn.execute(
        "SELECT cat, open_sec, meal_sec, counted_meal, close_reason "
        "FROM door_sessions ORDER BY id").fetchall()


def _open_door(cat, opened_wall):
    main._fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    main._fsm.confirm_open(cat, opened_wall)


def test_fail_safe_close_records_one_session(journal):
    _open_door("alisa", _wall(9, 0))
    main._last_event_wall_t = _wall(9, 0) + 30.0
    c = FakeClient(ok=True)
    main._fail_safe_close(c, "stream_lost")
    rows = _sessions(journal)
    assert len(rows) == 1
    cat, open_sec, meal_sec, counted, reason = rows[0]
    assert cat == "alisa" and reason == "stream_lost"
    assert open_sec == pytest.approx(30.0)
    assert counted == 1                       # 30s >= MIN_MEAL_SEC(10) default
    assert main._fsm.state == "closed"


def test_backstop_records_one_session(journal):
    _open_door("ellie", _wall(10, 0))
    main._last_event_wall_t = _wall(10, 0) + 5.0
    main._reset_close_backstop()
    # Trip the backstop (CLOSE_BACKSTOP_MAX_ATTEMPTS failures in a row).
    for _ in range(main.CLOSE_BACKSTOP_MAX_ATTEMPTS):
        main._close_backstop_after_failure("ellie")
    rows = _sessions(journal)
    assert len(rows) == 1                      # recorded once, at the trip
    cat, open_sec, meal_sec, counted, reason = rows[0]
    assert cat == "ellie" and reason == "backstop"
    assert counted == 0                        # 5s < MIN_MEAL_SEC(10)
    assert main._fsm.state == "closed"


def test_normal_close_records_session_with_presence_meal(journal):
    import feed_control
    from zone_state import ZoneState
    main._zone = ZoneState(window_sec=5, door_close_timeout_sec=30,
                           classifier_min_conf=0.5)
    main._feed = feed_control.FeedController(enabled=False, grain_num=1,
        empty_consecutive=1, min_interval_sec=0, confirm_timeout_sec=120)
    _open_door("alisa", _wall(11, 0))
    c = FakeClient(ok=True)
    # A clear event well past the presence TTL → decide() returns close/no_cat.
    main._handle_event({"wall_ms": int((_wall(11, 0) + 60.0) * 1000), "boxes": []}, c)
    rows = _sessions(journal)
    assert len(rows) == 1
    cat, open_sec, meal_sec, counted, reason = rows[0]
    # FSM verdict "no_cat" (cat left) is journaled as the clearer "cat_left".
    assert cat == "alisa" and reason == "cat_left"
    assert open_sec == pytest.approx(60.0)
    assert main._fsm.state == "closed"


def test_no_session_when_door_was_not_open(journal):
    # Fail-safe with a CLOSED fsm + non-open client → guard returns, nothing logged.
    main._fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    c = FakeClient(ok=True)
    c.state = "closed"
    main._fail_safe_close(c, "stream_lost")
    assert _sessions(journal) == []
