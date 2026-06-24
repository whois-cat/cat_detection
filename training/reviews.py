"""Load human label corrections (Stage B reviews.db) for training.

The review web app writes non-destructive corrections to a separate reviews.db
keyed by `src_event_key` (the events-table rowid). This maps that key to a
human label WITHOUT touching events.db. See training/README.md "Recipe: label
review".
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReviewRow:
    src_event_key: int
    label: str
    crop_id: str | None = None
    duplicate_group_id: str | None = None
    is_duplicate: bool = False
    suspicious_score: float = 0.0
    sampling_reason: str | None = None


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


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def load_review_rows(reviews_db: Path | str) -> dict[int, ReviewRow]:
    """Return review rows with optional review-sampling metadata.

    Older reviews.db files only have ``src_event_key`` and ``label``; the extra
    columns are treated as optional so training remains backwards-compatible.
    """
    path = resolve_reviews_db(reviews_db)
    if not path.exists():
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = _columns(conn, "reviews")
        select = ["src_event_key", "label"]
        optional = [
            "crop_id",
            "duplicate_group_id",
            "is_duplicate",
            "suspicious_score",
            "sampling_reason",
        ]
        for name in optional:
            select.append(name if name in cols else f"NULL AS {name}")
        rows = conn.execute(
            "SELECT " + ", ".join(select) + " FROM reviews"
        ).fetchall()
    finally:
        conn.close()

    out: dict[int, ReviewRow] = {}
    for (
        key,
        label,
        crop_id,
        duplicate_group_id,
        is_duplicate,
        suspicious_score,
        sampling_reason,
    ) in rows:
        out[int(key)] = ReviewRow(
            src_event_key=int(key),
            label=str(label),
            crop_id=crop_id,
            duplicate_group_id=duplicate_group_id,
            is_duplicate=bool(is_duplicate),
            suspicious_score=float(suspicious_score or 0.0),
            sampling_reason=sampling_reason,
        )
    return out
