"""Build/update a compact replay set from human-reviewed crops.

The replay set is intentionally small and balanced: by default up to 500 good,
diverse crops per class. It is stored outside recordings as compressed numpy
arrays, so weekly fine-tuning can retain old knowledge even after video pruning.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.build_cluster_manifest import (
    build_extractor,
    dedupe_nearby,
    prepare_embeddings,
)
from training.reviews import load_review_rows, load_reviews

DROP_LABELS = {"discard", "unknown"}


@dataclass
class Candidate:
    row: dict
    image: np.ndarray


def _rel_crop_path(label: str, key: int) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in label)
    return f"crops/{safe}/{int(key)}.npz"


def _load_existing(out_dir: Path) -> list[Candidate]:
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        crop_path = out_dir / row["path"]
        if crop_path.exists():
            with np.load(crop_path) as data:
                rows.append(Candidate(row=row, image=data["image"].astype(np.uint8, copy=False)))
    return rows


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image.astype(np.uint8, copy=False))


def _farthest_first(indices: list[int], embeddings: np.ndarray, quota: int,
                    scores: list[float]) -> list[int]:
    if len(indices) <= quota:
        return indices
    selected = [max(indices, key=lambda i: scores[i])]
    remaining = set(indices)
    remaining.remove(selected[0])
    min_dist = {
        i: 1.0 - float(embeddings[i] @ embeddings[selected[0]])
        for i in remaining
    }
    while remaining and len(selected) < quota:
        nxt = max(remaining, key=lambda i: (min_dist[i], scores[i]))
        selected.append(nxt)
        remaining.remove(nxt)
        for i in remaining:
            d = 1.0 - float(embeddings[i] @ embeddings[nxt])
            if d < min_dist[i]:
                min_dist[i] = d
    selected.sort(key=lambda i: (indices.index(i) if i in indices else i))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--reviews-db", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/replay"))
    ap.add_argument("--per-class", type=int, default=500,
                    help="target replay memory size per class")
    ap.add_argument("--reset", action="store_true",
                    help="ignore and replace the existing replay set")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--default-rotate-deg", type=int, default=0)
    ap.add_argument("--t-from", type=int, default=None)
    ap.add_argument("--t-to", type=int, default=None)
    ap.add_argument("--embedding", choices=("auto", "visual", "efficientnet"),
                    default="auto")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--dedupe-window-sec", type=float, default=2.0)
    ap.add_argument("--dedupe-threshold", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from training import CropSource

    review_rows = load_review_rows(args.reviews_db)
    reviews = {key: row.label for key, row in review_rows.items()}
    if not reviews:
        reviews = load_reviews(args.reviews_db)
    if not reviews:
        raise SystemExit(f"no human reviews found in {args.reviews_db}")

    existing = [] if args.reset else _load_existing(args.out)
    candidates: list[Candidate] = existing[:]
    raw_features: list[np.ndarray] = []

    extractor = build_extractor(
        args.embedding,
        batch_size=args.embedding_batch_size,
        allow_download=args.allow_download,
    )
    print(f"[replay] embedding={extractor.name}")

    for start in range(0, len(existing), max(1, args.embedding_batch_size)):
        batch = [c.image for c in existing[start:start + args.embedding_batch_size]]
        raw_features.extend(extractor.encode_batch(batch))

    src = CropSource(
        db_path=args.db,
        recordings_root=args.recordings,
        camera_id=args.camera,
        model=args.model,
        t_from=args.t_from,
        t_to=args.t_to,
        min_score=args.min_score,
        pad_frac=args.pad_frac,
        default_rotate_deg=args.default_rotate_deg,
    )

    pending_images: list[np.ndarray] = []
    pending_rows: list[dict] = []

    def flush() -> None:
        nonlocal pending_images, pending_rows, raw_features
        if not pending_images:
            return
        feats = extractor.encode_batch(pending_images)
        raw_features.extend(feats)
        for row, image in zip(pending_rows, pending_images):
            candidates.append(Candidate(row=row, image=image))
        pending_images = []
        pending_rows = []

    seen_existing = {int(c.row["src_event_key"]) for c in existing}
    for sample in src:
        sb = sample.src_box
        if sb is None or sb.rowid is None:
            continue
        key = int(sb.rowid)
        if key in seen_existing:
            continue
        review_row = review_rows.get(key)
        label = review_row.label if review_row is not None else reviews.get(key)
        if label is None or label in DROP_LABELS:
            continue
        row = {
            "src_event_key": key,
            "label": label,
            "camera": sample.camera_id,
            "wall_ms": int(sample.wall_ms),
            "model": sample.model,
            "score": float(sb.score),
            "path": _rel_crop_path(label, key),
        }
        if review_row is not None:
            row["duplicate_group_id"] = review_row.duplicate_group_id
            row["is_duplicate"] = review_row.is_duplicate
            row["suspicious_score"] = review_row.suspicious_score
            row["sampling_reason"] = review_row.sampling_reason
        pending_rows.append(row)
        pending_images.append(sample.image)
        if len(pending_images) >= max(1, args.embedding_batch_size):
            flush()
    flush()

    if not candidates:
        raise SystemExit("no replay candidates found")

    items = [c.row for c in candidates]
    emb = prepare_embeddings(np.stack(raw_features), out_dim=args.embedding_dim, seed=args.seed)
    for row, vec in zip(items, emb):
        row["embedding"] = [round(float(v), 5) for v in vec]
    items, emb, deduped = dedupe_nearby(
        items, emb, window_sec=args.dedupe_window_sec, threshold=args.dedupe_threshold,
    )
    by_key = {int(c.row["src_event_key"]): c for c in candidates}
    scores = [
        float(row.get("suspicious_score", 0.0)) * 2.0 + float(row.get("score", 0.0))
        for row in items
    ]

    by_label: dict[str, list[int]] = {}
    for i, row in enumerate(items):
        by_label.setdefault(row["label"], []).append(i)

    selected: list[int] = []
    for label, idxs in sorted(by_label.items()):
        chosen = _farthest_first(idxs, emb, args.per_class, scores)
        selected.extend(chosen)
        print(f"[replay] {label}: kept {len(chosen)} / {len(idxs)}")
    selected_set = set(selected)
    selected_rows = [items[i] for i in sorted(selected_set, key=lambda i: (items[i]["label"], items[i]["wall_ms"]))]

    args.out.mkdir(parents=True, exist_ok=True)
    for row in selected_rows:
        cand = by_key[int(row["src_event_key"])]
        crop_path = args.out / row["path"]
        if not crop_path.exists():
            _save_image(crop_path, cand.image)

    wanted_paths = {row["path"] for row in selected_rows}
    crops_dir = args.out / "crops"
    if crops_dir.exists():
        for path in crops_dir.rglob("*.npz"):
            rel = path.relative_to(args.out).as_posix()
            if rel not in wanted_paths:
                path.unlink()
        for path in sorted(crops_dir.glob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    manifest = args.out / "manifest.jsonl"
    tmp = manifest.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in selected_rows) + "\n",
        encoding="utf-8",
    )
    tmp.replace(manifest)

    meta = {
        "per_class": args.per_class,
        "total": len(selected_rows),
        "deduped": deduped,
        "embedding": extractor.name,
        "embedding_dim": args.embedding_dim,
        "min_score": args.min_score,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.reset:
        # Clean empty label dirs left by old label names after a reset.
        for path in sorted((args.out / "crops").glob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                shutil.rmtree(path, ignore_errors=True)

    print(f"[replay] wrote {len(selected_rows)} crops -> {manifest}")


if __name__ == "__main__":
    main()
