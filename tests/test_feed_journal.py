"""Feed journal DB: per-slot UNIQUE (double-feed guard) + empty-bowl appends.

Door sessions (open→finalize→recover) live in test_door_journal.py."""
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
