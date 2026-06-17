import sqlite3
import subprocess
import sys
from collections import Counter

from training.label_stats import (
    format_report,
    load_label_counts,
    resolve_reviews_db,
    summarize_counts,
)


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
