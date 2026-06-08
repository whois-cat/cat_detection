"""Produce a per-cat classifier dataset from recordings + events.db.

Output layout (drop-in for torchvision.datasets.ImageFolder):

    out_dir/
        train/
            alisa/ ... .jpg
            chuzh/ ... .jpg
            ellie/ ... .jpg
            felisis/ ... .jpg
        val/
            alisa/ ... .jpg
            ...

Filename: <wall_ms>_<camera>_<model>_<track_id>.jpg. wall_ms first so a sort
is also a time order. track_id makes it easy to keep all crops from one
track in the same split (a separate `--split-by-track` mode does that).

Usage (from the live2/ directory):

    uv run python -m training.extract_classifier \
        --recordings data/recordings --db data/events/events.db \
        --out data/datasets/classifier --val-frac 0.1 \
        --camera default --model yolov8n --min-score 0.3

By default ALL detector models present in the DB contribute crops. Filter
with --model when you want only one (e.g. only the fine-tuned cat YOLO).
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2

from .reviews import load_reviews
from .sources import CropSource


log = logging.getLogger("training.extract_classifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="path to events.db")
    ap.add_argument("--recordings", type=Path, required=True, help="data/recordings root")
    ap.add_argument("--out", type=Path, required=True, help="output dataset root")
    ap.add_argument("--camera", default=None, help="filter by camera_id (default: all)")
    ap.add_argument("--model", default=None, help="filter by detector model (default: all)")
    ap.add_argument("--cat", default=None, help="filter by cat label (default: all)")
    ap.add_argument("--t-from", type=int, default=None, help="wall_ms lower bound")
    ap.add_argument("--t-to",   type=int, default=None, help="wall_ms upper bound")
    ap.add_argument("--min-score", type=float, default=None, help="drop low-score detections")
    ap.add_argument("--reviews-db", type=Path, default=None,
                    help="reviews.db from the label-review tool — overrides the "
                         "detector label per crop; 'discard'/'unknown' crops are skipped")
    ap.add_argument("--pad-frac", type=float, default=0.15, help="extra context around the box")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction for val")
    ap.add_argument("--split-by-track", action="store_true",
                    help="keep all crops from one track_id in the same split "
                         "(prevents near-duplicate leakage between train and val)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rng = random.Random(args.seed)
    reviews = load_reviews(args.reviews_db) if args.reviews_db else None
    if reviews:
        log.info("loaded %d human label corrections from %s", len(reviews), args.reviews_db)
    src = CropSource(
        db_path=args.db, recordings_root=args.recordings,
        camera_id=args.camera, model=args.model, cat=args.cat,
        t_from=args.t_from, t_to=args.t_to, min_score=args.min_score,
        pad_frac=args.pad_frac, reviews=reviews,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = {}
    track_split: dict[int, str] = {}  # track_id -> 'train'/'val'

    for i, sample in enumerate(src):
        if not sample.boxes:
            continue
        box = sample.boxes[0]
        cat = box.cat
        if not cat:
            continue

        # split decision
        if args.split_by_track and box.track_id is not None:
            split = track_split.get(box.track_id)
            if split is None:
                split = "val" if rng.random() < args.val_frac else "train"
                track_split[box.track_id] = split
        else:
            split = "val" if rng.random() < args.val_frac else "train"

        cls_dir = args.out / split / cat
        cls_dir.mkdir(parents=True, exist_ok=True)
        name = f"{sample.wall_ms:013d}_{sample.camera_id}_{sample.model}_{box.track_id or 0}_{i:06d}.jpg"
        cv2.imwrite(str(cls_dir / name), sample.image)
        counts[(split, cat)] = counts.get((split, cat), 0) + 1

    log.info("done. per-class counts:")
    for (split, cat), n in sorted(counts.items()):
        log.info("  %-5s  %-12s  %d", split, cat, n)


if __name__ == "__main__":
    main()
