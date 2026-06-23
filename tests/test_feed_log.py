"""Smoke test for tools/feed_log.py: a journal with a closed + an interrupted
session prints both rows and a correct summary; missing DB / empty are graceful."""
import datetime as dt
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "feed_log.py"

_SCHEMA = """CREATE TABLE door_sessions (
    id INTEGER PRIMARY KEY, feeder_id TEXT NOT NULL, cat TEXT,
    opened_at TEXT NOT NULL, closed_at TEXT, duration_sec REAL, meal_sec REAL,
    counted_meal INTEGER NOT NULL DEFAULT 0, incomplete INTEGER NOT NULL DEFAULT 0,
    close_reason TEXT);"""


def _make_db(path: Path) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    c = sqlite3.connect(str(path))
    c.executescript(_SCHEMA)
    o1 = (now - dt.timedelta(hours=2)).isoformat()
    o2 = (now - dt.timedelta(hours=1)).isoformat()
    c.execute(
        "INSERT INTO door_sessions (feeder_id,cat,opened_at,closed_at,duration_sec,"
        "meal_sec,counted_meal,incomplete,close_reason) VALUES (?,?,?,?,?,?,?,?,?)",
        ("feeder1", "alisa", o1,
         (now - dt.timedelta(hours=2) + dt.timedelta(seconds=42)).isoformat(),
         42.0, 42.0, 1, 0, "cat_left"))
    c.execute(
        "INSERT INTO door_sessions (feeder_id,cat,opened_at,closed_at,duration_sec,"
        "meal_sec,counted_meal,incomplete,close_reason) VALUES (?,?,?,?,?,?,?,?,?)",
        ("feeder1", "alisa", o2, None, None, None, 0, 1, "interrupted"))
    c.commit()
    c.close()


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


def test_prints_closed_and_interrupted_with_summary(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db)
    r = _run("alisa", "--days", "3", "--db", str(db))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "cat_left" in out and "interrupted" in out
    assert "42s" in out                       # closed session duration + meal
    assert "2 session(s)" in out
    assert "counted meal total 42s" in out
    assert "1 interrupted" in out


def test_missing_db_is_graceful(tmp_path):
    r = _run("alisa", "--db", str(tmp_path / "nope.db"))
    assert r.returncode == 0
    assert "no journal yet" in r.stdout


def test_no_sessions_for_cat(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db)
    r = _run("ellie", "--db", str(db))
    assert r.returncode == 0
    assert "no sessions for ellie" in r.stdout


def test_days_window_excludes_old(tmp_path):
    db = tmp_path / "journal.db"
    c = sqlite3.connect(str(db))
    c.executescript(_SCHEMA)
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat()
    c.execute(
        "INSERT INTO door_sessions (feeder_id,cat,opened_at,closed_at,duration_sec,"
        "meal_sec,counted_meal,incomplete,close_reason) VALUES (?,?,?,?,?,?,?,?,?)",
        ("feeder1", "alisa", old, old, 10.0, 10.0, 1, 0, "cat_left"))
    c.commit()
    c.close()
    r = _run("alisa", "--days", "3", "--db", str(db))
    assert "no sessions for alisa in last 3 days" in r.stdout
