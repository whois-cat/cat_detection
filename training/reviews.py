"""Load human label corrections (Stage B reviews.db) for training.

The review web app writes non-destructive corrections to a separate reviews.db
keyed by `src_event_key` (the events-table rowid). This maps that key to a
human label WITHOUT touching events.db. See training/README.md "Recipe: label
review".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def load_reviews(reviews_db: Path | str) -> dict[int, str]:
    """Return {src_event_key: label} from a reviews.db (empty if it doesn't exist).

    Opened read-only; `discard` / `unknown` are returned as-is — it's the
    consumer (CropSource.drop_labels) that decides to skip them.
    """
    path = Path(reviews_db)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT src_event_key, label FROM reviews").fetchall()
    finally:
        conn.close()
    return {int(k): v for k, v in rows}
