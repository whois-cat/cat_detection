"""Feed journal DB: per-slot UNIQUE (double-feed guard), empty-bowl appends, and
door-session (meal) rows."""
import datetime as dt

from journal import FeedJournal


def test_scheduled_slot_marked_once(tmp_path):
    j = FeedJournal(tmp_path / "fj" / "journal.db")   # mkdir parents
    assert j.record_scheduled("f1", "2026-06-22", "08:00",
                              status="fed", grain_num=1, acked=True) is True
    # Second mark of the same slot is ignored (INSERT OR IGNORE → False).
    assert j.record_scheduled("f1", "2026-06-22", "08:00",
                              status="fed", grain_num=1, acked=True) is False
    assert j.was_slot_marked("f1", "2026-06-22", "08:00") is True
    assert j.was_slot_marked("f1", "2026-06-22", "13:00") is False


def test_same_slot_different_feeder_is_independent(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    assert j.record_scheduled("f1", "2026-06-22", "08:00", "fed", 1, True) is True
    assert j.record_scheduled("f2", "2026-06-22", "08:00", "fed", 1, True) is True


def test_empty_bowl_rows_never_collide(tmp_path):
    # slot_date/slot_time are NULL → UNIQUE index treats them as distinct, so
    # repeated empty-bowl refills all insert.
    j = FeedJournal(tmp_path / "journal.db")
    j.record_empty_bowl("f1", grain_num=1, acked=True)
    j.record_empty_bowl("f1", grain_num=1, acked=True)
    n = j._conn.execute(
        "SELECT COUNT(*) FROM feed_events WHERE mode='empty_bowl'").fetchone()[0]
    assert n == 2


def test_reopen_existing_db_is_idempotent(tmp_path):
    path = tmp_path / "journal.db"
    j1 = FeedJournal(path)
    j1.record_scheduled("f1", "2026-06-22", "08:00", "fed", 1, True)
    j1.close()
    # Re-open (runs SCHEMA + _migrate again) without error; data persists.
    j2 = FeedJournal(path)
    assert j2.was_slot_marked("f1", "2026-06-22", "08:00") is True


# ---- door sessions (meals) ----

def _epoch(y, mo, d, h, mi):
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp()


def test_door_session_records_fields_and_iso_times(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    opened = _epoch(2026, 6, 22, 9, 0)
    closed = _epoch(2026, 6, 22, 9, 0) + 42.0
    j.record_door_session("f1", cat="alisa", opened_at_wall=opened,
                          closed_at_wall=closed, open_sec=42.0, meal_sec=42.0,
                          counted_meal=True, close_reason="cat_left")
    row = j._conn.execute(
        "SELECT feeder_id, cat, opened_at, closed_at, open_sec, meal_sec, "
        "counted_meal, close_reason FROM door_sessions").fetchone()
    assert row[0] == "f1" and row[1] == "alisa"
    assert row[2].startswith("2026-06-22T09:00:00")     # ISO UTC
    assert row[4] == 42.0 and row[5] == 42.0
    assert row[6] == 1 and row[7] == "cat_left"


def test_door_session_null_open_when_unknown(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    closed = _epoch(2026, 6, 22, 9, 0)
    j.record_door_session("f1", cat=None, opened_at_wall=None,
                          closed_at_wall=closed, open_sec=None, meal_sec=0.0,
                          counted_meal=False, close_reason="stream_lost")
    row = j._conn.execute(
        "SELECT cat, opened_at, counted_meal FROM door_sessions").fetchone()
    assert row[0] is None and row[1] is None and row[2] == 0


def test_door_sessions_append(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    base = _epoch(2026, 6, 22, 9, 0)
    for i in range(3):
        j.record_door_session("f1", cat="ellie", opened_at_wall=base,
                              closed_at_wall=base + 10, open_sec=10.0, meal_sec=10.0,
                              counted_meal=True, close_reason="cat_left")
    n = j._conn.execute("SELECT COUNT(*) FROM door_sessions").fetchone()[0]
    assert n == 3
