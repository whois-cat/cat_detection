import sqlite3

from training.reviews import load_review_rows, load_reviews, resolve_reviews_db


def test_load_reviews_falls_back_to_cwd_reviews_db(tmp_path, monkeypatch):
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
            [(10, "alisa"), (11, "discard")],
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.chdir(tmp_path)

    assert resolve_reviews_db("data/review/reviews.db") == db
    assert load_reviews("data/review/reviews.db") == {10: "alisa", 11: "discard"}
    rows = load_review_rows("data/review/reviews.db")
    assert rows[10].label == "alisa"
    assert rows[10].duplicate_group_id is None
    assert rows[10].suspicious_score == 0.0


def test_load_review_rows_reads_sampling_metadata(tmp_path):
    db = tmp_path / "reviews.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """CREATE TABLE reviews (
                   src_event_key INTEGER PRIMARY KEY,
                   crop_id TEXT,
                   label TEXT NOT NULL,
                   duplicate_group_id TEXT,
                   is_duplicate INTEGER,
                   suspicious_score REAL,
                   sampling_reason TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO reviews
                   (src_event_key, crop_id, label, duplicate_group_id,
                    is_duplicate, suspicious_score, sampling_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (12, "grey:12", "felisis", "0:1", 1, 0.7, "suspicious_duplicate"),
        )
        conn.commit()
    finally:
        conn.close()

    rows = load_review_rows(db)
    assert rows[12].crop_id == "grey:12"
    assert rows[12].label == "felisis"
    assert rows[12].duplicate_group_id == "0:1"
    assert rows[12].is_duplicate is True
    assert rows[12].suspicious_score == 0.7
    assert rows[12].sampling_reason == "suspicious_duplicate"
