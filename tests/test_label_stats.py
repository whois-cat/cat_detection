import sqlite3
from collections import Counter

from training.label_stats import format_report, load_label_counts, summarize_counts


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
