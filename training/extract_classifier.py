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
        test/
            alisa/ ... .jpg
            ...

Filename: <wall_ms>_<camera>_<model>_<track_id>.jpg. wall_ms first so a sort
is also a time order. The split unit is always an episode: consecutive crops
from the same camera with no large wall-clock gap. This keeps near-duplicate
frames from one visit/video out of multiple splits.

Usage (from the live2/ directory):

    uv run python -m training.extract_classifier \
        --recordings data/recordings --db data/events/events.db \
        --out data/datasets/classifier --val-frac 0.1 --test-frac 0.1 \
        --camera default --model yolov8n --min-score 0.7

By default, only human-reviewed identity labels are exported. Unreviewed crops
are ignored unless --trust-classifier is explicitly set.
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


def _choose_split(rng: random.Random, val_frac: float, test_frac: float) -> str:
    r = rng.random()
    if r < test_frac:
        return "test"
    if r < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="path to events.db")
    ap.add_argument("--recordings", type=Path, required=True, help="data/recordings root")
    ap.add_argument("--out", type=Path, required=True, help="output dataset root")
    ap.add_argument("--camera", default=None, help="filter by camera_id (default: all)")
    ap.add_argument("--model", default=None, help="filter by detector model (default: all)")
    ap.add_argument("--cat", default=None,
                    help="legacy classifier-label filter before review overlay (default: all)")
    ap.add_argument("--t-from", type=int, default=None, help="wall_ms lower bound")
    ap.add_argument("--t-to",   type=int, default=None, help="wall_ms upper bound")
    ap.add_argument("--min-score", type=float, default=0.7, help="drop low-score detections")
    ap.add_argument("--reviews-db", type=Path, default=None,
                    help="reviews.db from the label-review tool; human labels are "
                         "the default source of identity truth")
    ap.add_argument("--trust-classifier", action="store_true",
                    help="also export unreviewed crops using existing classifier labels")
    ap.add_argument("--trust-detector", dest="trust_classifier", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--pad-frac", type=float, default=0.15, help="extra context around the box")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction for val")
    ap.add_argument("--test-frac", type=float, default=0.1, help="held-out fraction for test")
    ap.add_argument("--episode-gap-sec", type=float, default=60.0,
                    help="gap that starts a new split episode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.val_frac < 0 or args.test_frac < 0 or args.val_frac + args.test_frac >= 1:
        raise SystemExit("--val-frac and --test-frac must be >=0 and sum to < 1")
    rng = random.Random(args.seed)
    reviews = load_reviews(args.reviews_db) if args.reviews_db else {}
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
    episode: list[tuple[int, object, object, str]] = []
    episode_cam: str | None = None
    episode_last_ms: int | None = None
    episode_gap_ms = int(args.episode_gap_sec * 1000)

    def write_sample(split: str, seq: int, sample, box, cat: str) -> None:
        cls_dir = args.out / split / cat
        cls_dir.mkdir(parents=True, exist_ok=True)
        name = (
            f"{sample.wall_ms:013d}_{sample.camera_id}_{sample.model}_"
            f"{box.track_id or 0}_{seq:06d}.jpg"
        )
        cv2.imwrite(str(cls_dir / name), sample.image)
        counts[(split, cat)] = counts.get((split, cat), 0) + 1

    def flush_episode() -> None:
        nonlocal episode
        if not episode:
            return
        split = _choose_split(rng, args.val_frac, args.test_frac)
        for seq, sample, box, cat in episode:
            write_sample(split, seq, sample, box, cat)
        episode = []

    for i, sample in enumerate(src):
        if not sample.boxes:
            continue
        box = sample.boxes[0]
        src_key = sample.src_box.rowid if sample.src_box is not None else None
        human = reviews.get(src_key) if src_key is not None else None
        if human is None and not args.trust_classifier:
            continue
        cat = box.cat
        if not cat:
            continue

        if (
            episode
            and (
                sample.camera_id != episode_cam
                or episode_last_ms is None
                or sample.wall_ms - episode_last_ms > episode_gap_ms
            )
        ):
            flush_episode()
        episode.append((i, sample, box, cat))
        episode_cam = sample.camera_id
        episode_last_ms = sample.wall_ms

    flush_episode()

    log.info("done. per-class counts:")
    for (split, cat), n in sorted(counts.items()):
        log.info("  %-5s  %-12s  %d", split, cat, n)


if __name__ == "__main__":
    main()
