import json
import sqlite3
import subprocess
import sys
from collections import Counter

import training.label_stats as label_stats
from training.label_stats import (
    classify_review_usability,
    episode_camera_counts,
    format_report,
    load_event_scores,
    load_label_counts,
    load_label_episode_rows,
    resolve_events_db,
    resolve_reviews_db,
    summarize_counts,
    usable_for_training_stats,
)


GAP_MS = 60_000  # 60s, matching the training split default


def test_episode_camera_counts_groups_by_gap_and_camera():
    by_label = {
        # grey: two visits (gap > 60s between 1_000 and 200_000); brown: one visit.
        "alisa": [("grey", 1_000), ("grey", 5_000), ("grey", 200_000),
                  ("brown", 9_000)],
        # one crop on one camera → 1 episode, 1 camera.
        "chuzh": [("grey", 1_000)],
    }
    stats = episode_camera_counts(by_label, GAP_MS)
    assert stats["alisa"] == {"episodes": 3, "cameras": 2}  # 2 grey visits + 1 brown
    assert stats["chuzh"] == {"episodes": 1, "cameras": 1}


def test_episode_camera_counts_empty():
    assert episode_camera_counts({}, GAP_MS) == {}


def _reviews_db(tmp_path, rows, *, with_meta=True):
    db = tmp_path / "reviews.db"
    conn = sqlite3.connect(str(db))
    try:
        if with_meta:
            conn.execute("CREATE TABLE reviews (src_event_key INTEGER PRIMARY KEY, "
                         "label TEXT NOT NULL, camera TEXT, wall_ms INTEGER)")
            conn.executemany(
                "INSERT INTO reviews (src_event_key, label, camera, wall_ms) "
                "VALUES (?, ?, ?, ?)", rows)
        else:
            conn.execute("CREATE TABLE reviews (src_event_key INTEGER PRIMARY KEY, "
                         "label TEXT NOT NULL)")
            conn.executemany("INSERT INTO reviews (src_event_key, label) VALUES (?, ?)",
                             [(k, lab) for (k, lab, *_rest) in rows])
        conn.commit()
    finally:
        conn.close()
    return db


def test_load_label_episode_rows_reads_camera_wall_ms(tmp_path):
    db = _reviews_db(tmp_path, [
        (1, "alisa", "grey", 1_000), (2, "alisa", "grey", 2_000),
        (3, "chuzh", "brown", 5_000),
    ])
    rows = load_label_episode_rows(db)
    stats = episode_camera_counts(rows, GAP_MS)
    assert stats["alisa"] == {"episodes": 1, "cameras": 1}
    assert stats["chuzh"] == {"episodes": 1, "cameras": 1}


def test_load_label_episode_rows_none_for_old_schema(tmp_path):
    db = _reviews_db(tmp_path, [(1, "alisa"), (2, "chuzh")], with_meta=False)
    assert load_label_episode_rows(db) is None


def test_format_report_includes_episode_columns():
    summary = summarize_counts(Counter({"alisa": 3, "chuzh": 2}))
    report = format_report(summary, {"alisa": {"episodes": 2, "cameras": 1},
                                     "chuzh": {"episodes": 1, "cameras": 1}})
    assert "episodes" in report and "cameras" in report
    # Without episode stats the columns are absent (back-compat).
    assert "episodes" not in format_report(summary)


def test_load_label_counts_from_review_db(tmp_path):
    db = tmp_path / "reviews.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """CREATE TABLE reviews (
                   src_event_key INTEGER PRIMARY KEY,
                   label TEXT NOT NULL
               )"""
        )
        conn.executemany(
            "INSERT INTO reviews (src_event_key, label) VALUES (?, ?)",
            [
                (1, "alisa"),
                (2, "alisa"),
                (3, "chuzh"),
                (4, "discard"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    counts = load_label_counts(db)

    assert counts == Counter({"alisa": 2, "chuzh": 1, "discard": 1})


def test_summarize_counts_shows_missing_and_dropped_labels():
    summary = summarize_counts(
        Counter({"alisa": 3, "chuzh": 1, "discard": 2, "unknown": 1}),
        expected_labels=["alisa", "chuzh", "ellie"],
        drop_labels=["discard", "unknown"],
    )

    assert summary["total"] == 7
    assert summary["trainable_total"] == 4
    assert summary["dropped_total"] == 3
    assert summary["trainable"] == {"alisa": 3, "chuzh": 1, "ellie": 0}
    assert summary["dropped"] == {"discard": 2, "unknown": 1}
    assert summary["imbalance_ratio"] == float("inf")
    assert summary["missing"] == ["ellie"]

    report = format_report(summary)
    assert "ellie" in report
    assert "missing labels: ellie" in report


def test_resolve_reviews_db_falls_back_to_root_reviews_db(tmp_path):
    db = tmp_path / "reviews.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """CREATE TABLE reviews (
                   src_event_key INTEGER PRIMARY KEY,
                   label TEXT NOT NULL
               )"""
        )
        conn.commit()
    finally:
        conn.close()

    found = resolve_reviews_db(
        tmp_path / "data/review/reviews.db",
        search_roots=[tmp_path],
    )

    assert found == db


def test_label_stats_import_does_not_require_av():
    code = """
import builtins
orig_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "av":
        raise RuntimeError("label_stats should not import av")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import training.label_stats as label_stats
assert label_stats.parse_csv("alisa,chuzh") == ["alisa", "chuzh"]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---- usable-for-training cross-check ----------------------------------------

def _write_events_db(path, rows):
    """events.db with the (id, score) columns the usable cross-check reads."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, score REAL)")
        conn.executemany("INSERT INTO events (id, score) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return path


def _events_db(tmp_path, rows):
    return _write_events_db(tmp_path / "events.db", rows)


def test_classify_review_usability_buckets():
    review_labels = {
        1: "cat_a",    # usable (event exists, score >= min)
        2: "cat_b",    # below min score
        3: "cat_a",    # orphan (no event row)
        4: "discard",  # dropped → not counted at all
        5: "cat_c",    # event score NULL → below
    }
    event_scores = {1: 0.9, 2: 0.4, 5: None}   # key 3 absent → orphan
    r = classify_review_usability(review_labels, event_scores, min_score=0.7)
    assert r["usable_for_training"] == 1
    assert r["below_min_score"] == 2           # key2 (0.4) + key5 (NULL)
    assert r["orphan_reviews"] == 1            # key3
    assert r["min_score"] == 0.7


def test_load_event_scores_returns_only_existing(tmp_path):
    edb = _events_db(tmp_path, [(1, 0.9), (2, 0.4)])
    assert load_event_scores(edb, [1, 2, 99]) == {1: 0.9, 2: 0.4}  # 99 missing


def test_usable_for_training_stats_end_to_end(tmp_path):
    rdb = _reviews_db(tmp_path, [
        (1, "cat_a"), (2, "cat_b"), (3, "cat_a"), (4, "discard"), (5, "cat_c"),
    ], with_meta=False)
    edb = _events_db(tmp_path, [(1, 0.9), (2, 0.4), (5, 0.95)])  # key 3 missing
    assert usable_for_training_stats(rdb, edb, min_score=0.7) == {
        "usable_for_training": 2,   # key1 (0.9) + key5 (0.95)
        "orphan_reviews": 1,        # key3
        "below_min_score": 1,       # key2 (0.4)
        "min_score": 0.7,
    }


# ---- events.db resolution / auto-discovery ----------------------------------

def test_resolve_events_db_autodiscovers_valid_default(tmp_path):
    (tmp_path / "data" / "events").mkdir(parents=True)
    p = _write_events_db(tmp_path / "data/events/events.db", [(1, 0.9)])
    assert resolve_events_db(None, root=tmp_path) == p


def test_resolve_events_db_none_when_default_missing(tmp_path):
    assert resolve_events_db(None, root=tmp_path) is None


def test_resolve_events_db_none_when_default_is_empty_file(tmp_path):
    (tmp_path / "data" / "events").mkdir(parents=True)
    (tmp_path / "data/events/events.db").write_bytes(b"")   # 0-byte, no 'events' table
    assert resolve_events_db(None, root=tmp_path) is None


def test_resolve_events_db_explicit_override(tmp_path):
    p = _write_events_db(tmp_path / "e.db", [(1, 0.9)])
    assert resolve_events_db(p) == p                       # explicit, usable
    assert resolve_events_db(tmp_path / "nope.db") is None  # explicit, missing → skip


def test_main_autodiscovers_default_events_db(tmp_path, monkeypatch, capsys):
    # No --events-db: label-stats finds <root>/data/events/events.db on its own.
    (tmp_path / "data" / "events").mkdir(parents=True)
    _write_events_db(tmp_path / "data/events/events.db", [(1, 0.9), (2, 0.4)])
    rdb = _reviews_db(tmp_path, [(1, "cat_a"), (2, "cat_b"), (3, "cat_a")], with_meta=False)
    monkeypatch.setattr(label_stats, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["label_stats", "--reviews-db", str(rdb), "--json"])
    label_stats.main()
    assert json.loads(capsys.readouterr().out)["usable_for_training"] == {
        "usable_for_training": 1, "orphan_reviews": 1,   # key1 usable, key3 orphan
        "below_min_score": 1, "min_score": 0.7,          # key2 (0.4) below
    }


def test_label_stats_graceful_when_events_db_missing(tmp_path):
    # Missing events.db (explicit, so it's deterministic) → plain report, exit 0.
    rdb = _reviews_db(tmp_path, [(1, "cat_a"), (2, "cat_b")], with_meta=False)
    out = subprocess.run(
        [sys.executable, "-m", "training.label_stats",
         "--reviews-db", str(rdb), "--events-db", str(tmp_path / "nope.db")],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "reviewed labels:" in out
    assert "usable for training" not in out


def test_usable_cli_json_reports_fields(tmp_path):
    rdb = _reviews_db(tmp_path, [(1, "cat_a"), (2, "cat_b"), (3, "cat_a")], with_meta=False)
    edb = _events_db(tmp_path, [(1, 0.9), (2, 0.4)])   # key3 orphan, key2 below
    out = subprocess.run(
        [sys.executable, "-m", "training.label_stats",
         "--reviews-db", str(rdb), "--events-db", str(edb), "--json"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert json.loads(out)["usable_for_training"] == {
        "usable_for_training": 1, "orphan_reviews": 1,
        "below_min_score": 1, "min_score": 0.7,
    }
