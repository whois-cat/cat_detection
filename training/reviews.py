"""Load human label corrections (Stage B reviews.db) for training.

The review web app writes non-destructive corrections to a separate reviews.db
keyed by `src_event_key` (the events-table rowid). This maps that key to a
human label WITHOUT touching events.db. See training/README.md "Recipe: label
review".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_reviews_db(reviews_db: Path | str) -> Path:
    """Resolve review DB path, accepting the canonical and root-level layouts."""
    requested = Path(reviews_db)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add(requested)
    if not requested.is_absolute():
        add(ROOT / requested)
    add(Path.cwd() / "reviews.db")
    add(ROOT / "reviews.db")
    add(ROOT / "data/review/reviews.db")
    for path in candidates:
        if path.exists():
            return path
    return requested


def load_reviews(reviews_db: Path | str) -> dict[int, str]:
    """Return {src_event_key: label} from a reviews.db (empty if it doesn't exist).

    Opened read-only; `discard` / `unknown` are returned as-is — it's the
    consumer (CropSource.drop_labels) that decides to skip them.
    """
    path = resolve_reviews_db(reviews_db)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT src_event_key, label FROM reviews").fetchall()
    finally:
        conn.close()
    return {int(k): v for k, v in rows}
