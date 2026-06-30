"""Report human-review label counts and class balance."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DROP_LABELS = ("discard", "unknown")
# Matches train_classifier.py's --min-score default: boxes below this YOLO score
# are filtered out by iter_frames before reviews are joined, so a reviewed crop
# under it is never seen by training.
TRAIN_MIN_SCORE_DEFAULT = 0.7


def _has_reviews_table(path: Path) -> bool:
    if not path.exists() or path.name.endswith(("-wal", "-shm")):
        return False
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='reviews'"
        ).fetchone()
        return row is not None
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def resolve_reviews_db(requested: Path, *, search_roots: Iterable[Path] = ()) -> Path:
    """Find the review DB, accepting both the canonical and old root location."""
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    add(requested)
    if not requested.is_absolute():
        add(ROOT / requested)
    for root in search_roots:
        add(root / "reviews.db")
        add(root / "data/review/reviews.db")
    add(Path.cwd() / "reviews.db")
    add(ROOT / "reviews.db")
    add(ROOT / "data/review/reviews.db")

    for candidate in candidates:
        if _has_reviews_table(candidate):
            return candidate

    looked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "reviews db not found. Looked for a SQLite DB with table 'reviews' in: "
        f"{looked}"
    )


def parse_csv(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = raw
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if label and label not in seen:
            out.append(label)
            seen.add(label)
    return out


def load_label_counts(reviews_db: Path) -> Counter[str]:
    if not reviews_db.exists():
        raise FileNotFoundError(f"reviews db not found: {reviews_db}")
    conn = sqlite3.connect(str(reviews_db))
    try:
        try:
            rows = conn.execute(
                "SELECT label, COUNT(*) FROM reviews GROUP BY label"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"{reviews_db} does not look like a review DB") from exc
    finally:
        conn.close()
    return Counter({str(label): int(count) for label, count in rows})


def load_review_labels(reviews_db: Path) -> dict[int, str]:
    """{src_event_key: label} for every review row (read-only)."""
    conn = sqlite3.connect(f"file:{reviews_db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT src_event_key, label FROM reviews").fetchall()
    finally:
        conn.close()
    return {int(k): str(v) for k, v in rows}


def load_event_scores(events_db: Path, keys: Iterable[int]) -> dict[int, float | None]:
    """{events.id: score} for the requested ids that exist in events.db (read-only).

    Chunked IN-queries stay under SQLite's bound-variable limit. Missing ids are
    simply absent from the result (→ treated as orphans by the caller)."""
    key_list = list(dict.fromkeys(int(k) for k in keys))
    out: dict[int, float | None] = {}
    if not key_list:
        return out
    conn = sqlite3.connect(f"file:{events_db}?mode=ro", uri=True)
    try:
        for i in range(0, len(key_list), 900):
            chunk = key_list[i:i + 900]
            placeholders = ",".join("?" * len(chunk))
            for eid, score in conn.execute(
                f"SELECT id, score FROM events WHERE id IN ({placeholders})", chunk
            ):
                out[int(eid)] = None if score is None else float(score)
    finally:
        conn.close()
    return out


def _has_events_table(path: Path) -> bool:
    """True only if `path` is a usable events DB (exists, opens, has 'events').
    A missing / empty (0-byte) / non-SQLite file returns False so the cross-check
    is skipped rather than crashing."""
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()
        return row is not None
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def resolve_events_db(explicit: Path | None, *, root: Path | None = None) -> Path | None:
    """Pick the events.db for the usable cross-check, or None to skip it.

    With no explicit path, auto-discovers the canonical
    ``<root>/data/events/events.db`` (root defaults to the repo root) so
    ``just label-stats`` includes usability stats with no extra flags. Returns
    None when the chosen DB is missing/empty/not an events DB, so label-stats
    degrades to the plain report instead of erroring. An explicit --events-db is
    honoured when usable, else skipped (never a hard failure)."""
    base = root if root is not None else ROOT
    candidate = explicit if explicit is not None else base / "data/events/events.db"
    return candidate if _has_events_table(candidate) else None


def classify_review_usability(
    review_labels: dict[int, str],
    event_scores: dict[int, float | None],
    *,
    min_score: float,
    drop_labels: Iterable[str] = DEFAULT_DROP_LABELS,
) -> dict:
    """Bucket TRAINABLE reviews by whether train_classifier would keep them.

    Mirrors train_classifier's join + score filter (iter_frames `AND score >=
    min_score`, then reviews joined on events.id == src_event_key). A trainable
    review (label not in drop_labels) is:
      - orphan        : its src_event_key has no row in events.db
      - below_min_score: event exists but score < min_score (or NULL)
      - usable        : event exists and score >= min_score
    No recording-availability check (train logs that separately as 'unavailable').
    """
    drop = set(parse_csv(drop_labels))
    usable = orphan = below = 0
    for key, label in review_labels.items():
        if label in drop:
            continue
        if key not in event_scores:
            orphan += 1
        elif event_scores[key] is None or event_scores[key] < min_score:
            below += 1
        else:
            usable += 1
    return {
        "usable_for_training": usable,
        "orphan_reviews": orphan,
        "below_min_score": below,
        "min_score": float(min_score),
    }


def usable_for_training_stats(
    reviews_db: Path, events_db: Path, *,
    min_score: float = TRAIN_MIN_SCORE_DEFAULT,
    drop_labels: Iterable[str] = DEFAULT_DROP_LABELS,
) -> dict:
    """Cross-check reviews.db against events.db: how many trainable reviews are
    actually usable by training vs lost to orphan keys / below-min-score boxes."""
    review_labels = load_review_labels(reviews_db)
    drop = set(parse_csv(drop_labels))
    trainable_keys = [k for k, lab in review_labels.items() if lab not in drop]
    event_scores = load_event_scores(events_db, trainable_keys)
    return classify_review_usability(
        review_labels, event_scores, min_score=min_score, drop_labels=drop_labels)


def episode_camera_counts(
    by_label: dict[str, list[tuple[str, int]]], gap_ms: int
) -> dict[str, dict[str, int]]:
    """Per-label episode + camera counts.

    An episode = same camera, consecutive ``wall_ms`` with no gap > ``gap_ms`` —
    the SAME grouping the train/val/test split uses (train_classifier.build_episodes),
    so "episodes" here means "independent visits" the model can learn from.
    ``by_label`` maps label -> [(camera, wall_ms), ...].
    """
    out: dict[str, dict[str, int]] = {}
    for label, rows in by_label.items():
        by_cam: dict[str, list[int]] = defaultdict(list)
        for camera, wall_ms in rows:
            by_cam[str(camera)].append(int(wall_ms))
        episodes = 0
        for walls in by_cam.values():
            walls.sort()
            prev: int | None = None
            for w in walls:
                if prev is None or w - prev > gap_ms:
                    episodes += 1
                prev = w
        out[label] = {"episodes": episodes, "cameras": len(by_cam)}
    return out


def load_label_episode_rows(reviews_db: Path) -> dict[str, list[tuple[str, int]]] | None:
    """Per-label (camera, wall_ms) rows from reviews.db, or None if the DB predates
    the camera/wall_ms columns (older review DBs only store label). Read-only."""
    conn = sqlite3.connect(f"file:{reviews_db}?mode=ro", uri=True)
    try:
        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(reviews)")}
        if not {"camera", "wall_ms"} <= cols:
            return None
        rows = conn.execute(
            "SELECT label, camera, wall_ms FROM reviews "
            "WHERE camera IS NOT NULL AND wall_ms IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for label, camera, wall_ms in rows:
        by_label[str(label)].append((str(camera), int(wall_ms)))
    return dict(by_label)


def summarize_counts(
    counts: Counter[str],
    *,
    expected_labels: Iterable[str] = (),
    drop_labels: Iterable[str] = DEFAULT_DROP_LABELS,
) -> dict:
    drop = set(parse_csv(drop_labels))
    expected = parse_csv(expected_labels)
    train_seen = sorted(label for label in counts if label not in drop)
    train_order = parse_csv([*expected, *train_seen])
    trainable = {label: int(counts.get(label, 0)) for label in train_order}
    dropped = {label: int(counts[label]) for label in sorted(counts) if label in drop}
    total = int(sum(counts.values()))
    trainable_total = int(sum(counts[label] for label in counts if label not in drop))
    dropped_total = int(sum(dropped.values()))

    values = list(trainable.values())
    max_count = max(values) if values else 0
    min_count = min(values) if values else 0
    missing = [label for label, count in trainable.items() if count == 0]
    if not values or max_count == 0:
        ratio = None
    elif min_count == 0:
        ratio = float("inf")
    else:
        ratio = max_count / min_count

    return {
        "total": total,
        "trainable_total": trainable_total,
        "dropped_total": dropped_total,
        "drop_labels": sorted(drop),
        "trainable": trainable,
        "dropped": dropped,
        "max_count": max_count,
        "min_count": min_count,
        "missing": missing,
        "imbalance_ratio": ratio,
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _balance_verdict(ratio: float | None) -> str:
    if ratio is None:
        return "no trainable labels yet"
    if ratio == float("inf"):
        return "not balanced: at least one expected label is missing"
    if ratio <= 1.25:
        return "well balanced"
    if ratio <= 2.0:
        return "usable, but skewed"
    return "heavily skewed"


def format_report(summary: dict, episode_stats: dict[str, dict[str, int]] | None = None) -> str:
    lines: list[str] = []
    lines.append(
        "reviewed labels: "
        f"total={summary['total']} "
        f"trainable={summary['trainable_total']} "
        f"dropped={summary['dropped_total']}"
    )

    trainable: dict[str, int] = summary["trainable"]
    if trainable:
        width = max(5, max(len(label) for label in trainable))
        lines.append("")
        lines.append("trainable labels (drop labels excluded):")
        # episodes/cameras only when reviews.db carries camera + wall_ms.
        ep_hdr = f"  {'episodes':>8}  {'cameras':>7}" if episode_stats is not None else ""
        lines.append(
            f"{'label':<{width}}  {'count':>7}  {'share':>7}  "
            f"{'vs_max':>7}  {'need_to_max':>11}{ep_hdr}"
        )
        max_count = int(summary["max_count"])
        total = int(summary["trainable_total"])
        for label, count in trainable.items():
            share = count / total if total else 0.0
            vs_max = count / max_count if max_count else 0.0
            deficit = max_count - count if max_count else 0
            ep_cols = ""
            if episode_stats is not None:
                es = episode_stats.get(label, {"episodes": 0, "cameras": 0})
                ep_cols = f"  {es['episodes']:8d}  {es['cameras']:7d}"
            lines.append(
                f"{label:<{width}}  {count:7d}  {_fmt_pct(share):>7}  "
                f"{_fmt_pct(vs_max):>7}  {deficit:11d}{ep_cols}"
            )
    else:
        lines.append("")
        lines.append("trainable labels: none")

    dropped: dict[str, int] = summary["dropped"]
    if dropped:
        width = max(5, max(len(label) for label in dropped))
        lines.append("")
        lines.append("dropped labels:")
        for label, count in dropped.items():
            lines.append(f"{label:<{width}}  {count:7d}")

    lines.append("")
    ratio = summary["imbalance_ratio"]
    if ratio is None:
        ratio_text = "NA"
    elif ratio == float("inf"):
        ratio_text = "inf"
    else:
        ratio_text = f"{ratio:.2f}"
    lines.append(
        f"balance: max/min={ratio_text} - {_balance_verdict(ratio)}"
    )
    if summary["missing"]:
        lines.append("missing labels: " + ", ".join(summary["missing"]))
    elif trainable and summary["min_count"] > 0:
        lines.append(
            f"balanced downsample target: {summary['min_count']} per class"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reviews-db",
        type=Path,
        default=ROOT / "data/review/reviews.db",
        help="review DB written by cluster-review",
    )
    ap.add_argument(
        "--labels",
        default=os.environ.get("REVIEW_LABELS", ""),
        help="comma-separated expected trainable labels; zero-count labels are shown",
    )
    ap.add_argument(
        "--drop-labels",
        default=",".join(DEFAULT_DROP_LABELS),
        help="comma-separated labels excluded from trainable balance",
    )
    ap.add_argument("--episode-gap-sec", type=float, default=60.0,
                    help="episode = same camera, consecutive reviewed crops within "
                         "this wall-clock gap (matches the training split default)")
    ap.add_argument("--events-db", type=Path, default=None,
                    help="events.db for the usable-for-training cross-check. "
                         "Auto-discovered at data/events/events.db when present, so "
                         "no flag is needed for the normal layout; pass this to "
                         "override (tests/custom paths). Missing/empty → skipped.")
    ap.add_argument("--min-score", type=float, default=TRAIN_MIN_SCORE_DEFAULT,
                    help=f"YOLO box score floor for the usable cross-check "
                         f"(default {TRAIN_MIN_SCORE_DEFAULT}, matches train_classifier)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    reviews_db = resolve_reviews_db(args.reviews_db)
    counts = load_label_counts(reviews_db)
    summary = summarize_counts(
        counts,
        expected_labels=parse_csv(args.labels),
        drop_labels=parse_csv(args.drop_labels),
    )
    summary["reviews_db"] = str(reviews_db)

    # Episodes/cameras per label — only when reviews.db has camera + wall_ms.
    ep_rows = load_label_episode_rows(reviews_db)
    episode_stats = (
        episode_camera_counts(ep_rows, int(args.episode_gap_sec * 1000))
        if ep_rows is not None else None
    )
    if episode_stats is not None:
        summary["episode_gap_sec"] = args.episode_gap_sec
        summary["episodes_per_label"] = episode_stats

    # Usable-for-training cross-check. Runs by default when the canonical
    # events.db is present; --events-db overrides; missing/empty → skipped.
    usable = None
    events_db = resolve_events_db(args.events_db)
    if events_db is not None:
        usable = usable_for_training_stats(
            reviews_db, events_db,
            min_score=args.min_score, drop_labels=parse_csv(args.drop_labels),
        )
        summary["usable_for_training"] = usable

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"reviews db: {reviews_db}")
        print(format_report(summary, episode_stats))
        if usable is not None:
            print(
                f"\nusable for training (vs {events_db}): "
                f"{usable['usable_for_training']}   "
                f"(orphan={usable['orphan_reviews']}  "
                f"below_min_score({usable['min_score']:g})={usable['below_min_score']}  "
                f"[recording-unavailable: see train log])"
            )


if __name__ == "__main__":
    main()
