import sqlite3

from training.reviews import load_reviews, resolve_reviews_db


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
