"""Door-session journaling wired through the real main.py: open creates the row,
each close path (normal, fail-safe, backstop) finalizes it, and a session with no
open is a no-op. Container-only deps (websockets) stubbed; real temp SQLite.
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
    main._open_session_id = None
    return j


def _rows(j):
    return j._conn.execute(
        "SELECT cat, opened_at, closed_at, duration_sec, meal_sec, counted_meal, "
        "incomplete, close_reason FROM door_sessions ORDER BY id").fetchall()


def _open_door(j, cat, opened_wall):
    """Mimic the OPEN hook: confirm_open + open_session, set module state."""
    main._fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    main._fsm.confirm_open(cat, opened_wall)
    main._open_session_id = j.open_session(main.FEEDER_ID, cat, opened_wall)


def test_open_then_normal_close_finalizes(journal):
    import feed_control
    from zone_state import ZoneState
    main._zone = ZoneState(window_sec=5, door_close_timeout_sec=30,
                           classifier_min_conf=0.5)
    main._feed = feed_control.FeedController(enabled=False, grain_num=1,
        empty_consecutive=1, min_interval_sec=0, confirm_timeout_sec=120)
    _open_door(journal, "alisa", _wall(11, 0))
    # Row exists, still open.
    cat, opened_at, closed_at, *_ = _rows(journal)[0]
    assert cat == "alisa" and closed_at is None

    c = FakeClient(ok=True)
    main._handle_event({"wall_ms": int((_wall(11, 0) + 60.0) * 1000), "boxes": []}, c)
    rows = _rows(journal)
    assert len(rows) == 1
    cat, opened_at, closed_at, dur, meal, counted, incomplete, reason = rows[0]
    assert closed_at is not None and dur == pytest.approx(60.0)
    assert reason == "cat_left"               # no_cat → cat_left mapping
    # meal_sec is the presence-based estimate from the snapshot — 0 here since the
    # test only sent a clear event (no presence accumulated), so the door duration
    # (60s) and the meal estimate (0s) are intentionally different.
    assert meal == pytest.approx(0.0) and counted == 0
    assert incomplete == 0
    assert main._open_session_id is None       # finalized → cleared
    assert main._fsm.state == "closed"


def test_fail_safe_close_finalizes(journal):
    _open_door(journal, "ellie", _wall(9, 0))
    main._last_event_wall_t = _wall(9, 0) + 30.0
    c = FakeClient(ok=True)
    main._fail_safe_close(c, "stream_lost")
    cat, _o, closed_at, dur, meal, counted, incomplete, reason = _rows(journal)[0]
    assert closed_at is not None and dur == pytest.approx(30.0)
    assert reason == "stream_lost"
    assert meal is None and counted == 0       # no snapshot → meal_sec NULL
    assert incomplete == 0
    assert main._open_session_id is None


def test_backstop_finalizes_once(journal):
    _open_door(journal, "alisa", _wall(10, 0))
    main._last_event_wall_t = _wall(10, 0) + 5.0
    main._reset_close_backstop()
    for _ in range(main.CLOSE_BACKSTOP_MAX_ATTEMPTS):
        main._close_backstop_after_failure("alisa")
    rows = _rows(journal)
    assert len(rows) == 1
    _c, _o, closed_at, dur, meal, counted, incomplete, reason = rows[0]
    assert closed_at is not None and reason == "backstop"
    assert dur == pytest.approx(5.0) and counted == 0   # 5s < MIN_MEAL_SEC
    assert incomplete == 0
    assert main._open_session_id is None


def test_finalize_is_idempotent(journal):
    _open_door(journal, "alisa", _wall(9, 0))
    c = FakeClient(ok=True)
    main._fail_safe_close(c, "stream_lost")
    # A second close attempt with no open session writes nothing more.
    main._finalize_open_session(_wall(9, 30), "cat_left")
    assert len(_rows(journal)) == 1


def test_no_open_session_close_is_noop(journal):
    # Door never opened → _open_session_id None → finalize writes nothing.
    main._fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    main._open_session_id = None
    main._finalize_open_session(_wall(9, 0), "cat_left", meal_sec=20.0)
    assert _rows(journal) == []
