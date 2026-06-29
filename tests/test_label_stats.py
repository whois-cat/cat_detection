import sqlite3
import subprocess
import sys
from collections import Counter

from training.label_stats import (
    episode_camera_counts,
    format_report,
    load_label_counts,
    load_label_episode_rows,
    resolve_reviews_db,
    summarize_counts,
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
