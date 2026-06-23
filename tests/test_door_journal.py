"""Door-session journal: open → finalize → crash recovery, plus the old-schema
migration. Journal-level unit tests (tmp SQLite, no feeder process)."""
import datetime as dt
import sqlite3

from journal import FeedJournal

MIN_MEAL = 10.0


def _epoch(h, m=0, s=0):
    return dt.datetime(2026, 6, 22, h, m, s, tzinfo=dt.timezone.utc).timestamp()


def _row(j, sid):
    return j._conn.execute(
        "SELECT cat, opened_at, closed_at, duration_sec, meal_sec, counted_meal, "
        "incomplete, close_reason FROM door_sessions WHERE id=?", (sid,)).fetchone()


def test_open_session_inserts_open_row(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    sid = j.open_session("f1", "alisa", _epoch(9, 0))
    cat, opened_at, closed_at, dur, meal, counted, incomplete, reason = _row(j, sid)
    assert cat == "alisa"
    assert opened_at.startswith("2026-06-22T09:00:00")     # ISO UTC, set at open
    assert closed_at is None and dur is None and meal is None
    assert counted == 0 and incomplete == 0 and reason is None


def test_finalize_fills_close_fields(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    sid = j.open_session("f1", "alisa", _epoch(9, 0))
    j.finalize_session(sid, _epoch(9, 0) + 42.0, meal_sec=42.0,
                       close_reason="cat_left", min_meal_sec=MIN_MEAL)
    cat, opened_at, closed_at, dur, meal, counted, incomplete, reason = _row(j, sid)
    assert closed_at.startswith("2026-06-22T09:00:42")
    assert dur == 42.0                       # computed from stored opened_at
    assert meal == 42.0 and counted == 1     # 42 >= MIN_MEAL
    assert incomplete == 0 and reason == "cat_left"


def test_finalize_counted_meal_boundary(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    # below threshold → not counted
    s1 = j.open_session("f1", "alisa", _epoch(9, 0))
    j.finalize_session(s1, _epoch(9, 0) + 5.0, meal_sec=5.0,
                       close_reason="cat_left", min_meal_sec=MIN_MEAL)
    assert _row(j, s1)[5] == 0
    # exactly at threshold → counted
    s2 = j.open_session("f1", "alisa", _epoch(10, 0))
    j.finalize_session(s2, _epoch(10, 0) + 10.0, meal_sec=10.0,
                       close_reason="cat_left", min_meal_sec=MIN_MEAL)
    assert _row(j, s2)[5] == 1


def test_finalize_without_snapshot_keeps_meal_null_but_duration_set(tmp_path):
    # fail-safe/backstop pass meal_sec=None → meal_sec NULL, counted 0, duration set.
    j = FeedJournal(tmp_path / "journal.db")
    sid = j.open_session("f1", "ellie", _epoch(9, 0))
    j.finalize_session(sid, _epoch(9, 0) + 30.0, meal_sec=None,
                       close_reason="stream_lost", min_meal_sec=MIN_MEAL)
    _, _, closed_at, dur, meal, counted, incomplete, reason = _row(j, sid)
    assert dur == 30.0 and meal is None and counted == 0
    assert incomplete == 0 and reason == "stream_lost"


def test_recover_interrupted_marks_open_rows_only(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    open1 = j.open_session("f1", "alisa", _epoch(9, 0))         # left open (crash)
    open2 = j.open_session("f1", "ellie", _epoch(10, 0))        # left open (crash)
    done = j.open_session("f1", "chuzh", _epoch(8, 0))
    j.finalize_session(done, _epoch(8, 0) + 20.0, meal_sec=20.0,
                       close_reason="cat_left", min_meal_sec=MIN_MEAL)

    n = j.recover_interrupted("f1")
    assert n == 2                              # only the two still-open rows

    for sid in (open1, open2):
        _, _, closed_at, dur, meal, counted, incomplete, reason = _row(j, sid)
        assert incomplete == 1 and reason == "interrupted"
        assert closed_at is None and dur is None and meal is None  # honestly unknown
        assert counted == 0
    # The already-finalized row is untouched.
    assert _row(j, done)[7] == "cat_left" and _row(j, done)[6] == 0


def test_recover_is_idempotent_and_scoped_per_feeder(tmp_path):
    j = FeedJournal(tmp_path / "journal.db")
    j.open_session("f1", "alisa", _epoch(9, 0))
    j.open_session("f2", "chuzh", _epoch(9, 0))
    assert j.recover_interrupted("f1") == 1     # only f1
    assert j.recover_interrupted("f1") == 0     # nothing left open for f1
    assert j.recover_interrupted("f2") == 1     # f2 still pending


def test_migrates_old_close_only_schema(tmp_path):
    """A DB from the reverted close-only commit (closed_at NOT NULL, no
    `incomplete`) is rebuilt to the new schema; fresh and migrated converge."""
    path = tmp_path / "journal.db"
    c = sqlite3.connect(str(path))
    c.executescript(
        """CREATE TABLE door_sessions (
               id INTEGER PRIMARY KEY, feeder_id TEXT NOT NULL, cat TEXT,
               opened_at TEXT, closed_at TEXT NOT NULL, open_sec REAL,
               meal_sec REAL, counted_meal INTEGER NOT NULL DEFAULT 0,
               close_reason TEXT NOT NULL);""")
    c.commit()
    c.close()

    j = FeedJournal(path)
    cols = {r[1]: r for r in j._conn.execute("PRAGMA table_info(door_sessions)")}
    assert "incomplete" in cols and "duration_sec" in cols
    assert cols["closed_at"][3] == 0           # nullable now
    # And the migrated table actually works with the new API.
    sid = j.open_session("f1", "alisa", _epoch(9, 0))
    j.finalize_session(sid, _epoch(9, 0) + 12.0, meal_sec=12.0,
                       close_reason="cat_left", min_meal_sec=MIN_MEAL)
    assert _row(j, sid)[5] == 1

    fresh = FeedJournal(tmp_path / "fresh.db")
    fresh_cols = [r[1] for r in fresh._conn.execute("PRAGMA table_info(door_sessions)")]
    assert list(cols) == fresh_cols            # schemas converge
