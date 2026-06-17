"""Report human-review label counts and class balance."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DROP_LABELS = ("discard", "unknown")


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


def format_report(summary: dict) -> str:
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
        lines.append(
            f"{'label':<{width}}  {'count':>7}  {'share':>7}  "
            f"{'vs_max':>7}  {'need_to_max':>11}"
        )
        max_count = int(summary["max_count"])
        total = int(summary["trainable_total"])
        for label, count in trainable.items():
            share = count / total if total else 0.0
            vs_max = count / max_count if max_count else 0.0
            deficit = max_count - count if max_count else 0
            lines.append(
                f"{label:<{width}}  {count:7d}  {_fmt_pct(share):>7}  "
                f"{_fmt_pct(vs_max):>7}  {deficit:11d}"
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
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    counts = load_label_counts(args.reviews_db)
    summary = summarize_counts(
        counts,
        expected_labels=parse_csv(args.labels),
        drop_labels=parse_csv(args.drop_labels),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"reviews db: {args.reviews_db}")
        print(format_report(summary))


if __name__ == "__main__":
    main()
