"""Produce a YOLO-format detection dataset from recordings + events.db.

Output layout (drop-in for Ultralytics' YOLO trainer):

    out_dir/
        data.yaml
        images/
            train/ ... .jpg
            val/   ... .jpg
        labels/
            train/ ... .txt   (one line per box: `class cx cy w h` normalised)
            val/   ... .txt

`data.yaml` is written with the classes seen in the events DB (or those
passed via --classes). A single class "cat" is the common case; with
fine-tuned per-cat labels you'd pass --classes alisa,chuzh,ellie,felisis.

Train a fine-tuned YOLO on it:

    yolo train data=out_dir/data.yaml model=yolov8n.pt imgsz=640 epochs=50

Usage (from live2/):

    uv run python -m training.extract_detector \
        --recordings data/recordings --db data/events/events.db \
        --out data/datasets/detector --val-frac 0.1 \
        --camera default --model yolov8n
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2

from .db import open_db_ro, distinct_cats
from .sources import FullFrameSource


log = logging.getLogger("training.extract_detector")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cat", default=None)
    ap.add_argument("--t-from", type=int, default=None)
    ap.add_argument("--t-to",   type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", default=None,
                    help="comma-separated class names. Defaults to all "
                         "distinct cat labels in the DB.")
    ap.add_argument("--collapse-to-single-class", action="store_true",
                    help="map every detected cat label to class 0 ('cat'). "
                         "Useful for training a 'cat or not' detector.")
    ap.add_argument("--include-empty-frac", type=float, default=0.0,
                    help="optional fraction of empty (no-box) frames to also "
                         "sample as hard negatives. NOT YET IMPLEMENTED — "
                         "left as a placeholder; the detector currently only "
                         "logs frames that had detections.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Resolve class list
    if args.collapse_to_single_class:
        class_names = ["cat"]
    elif args.classes:
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        conn = open_db_ro(args.db)
        try:
            class_names = distinct_cats(conn, camera_id=args.camera, model=args.model)
        finally:
            conn.close()
        if not class_names:
            log.error("no cat labels found in DB; pass --classes or "
                      "--collapse-to-single-class")
            return
    class_idx = {name: i for i, name in enumerate(class_names)}
    log.info("classes: %s", class_names)

    rng = random.Random(args.seed)
    src = FullFrameSource(
        db_path=args.db, recordings_root=args.recordings,
        camera_id=args.camera, model=args.model, cat=args.cat,
        t_from=args.t_from, t_to=args.t_to, min_score=args.min_score,
    )

    img_train = args.out / "images" / "train"; img_train.mkdir(parents=True, exist_ok=True)
    img_val   = args.out / "images" / "val";   img_val.mkdir(parents=True, exist_ok=True)
    lbl_train = args.out / "labels" / "train"; lbl_train.mkdir(parents=True, exist_ok=True)
    lbl_val   = args.out / "labels" / "val";   lbl_val.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for i, sample in enumerate(src):
        H, W = sample.image.shape[:2]
        lines = []
        for b in sample.boxes:
            cls = 0 if args.collapse_to_single_class else class_idx.get(b.cat or "", -1)
            if cls < 0:
                continue
            cx = (b.x + b.w * 0.5) / W
            cy = (b.y + b.h * 0.5) / H
            nw = b.w / W
            nh = b.h / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if not lines:
            continue

        split = "val" if rng.random() < args.val_frac else "train"
        img_dir = img_val if split == "val" else img_train
        lbl_dir = lbl_val if split == "val" else lbl_train
        stem = f"{sample.wall_ms:013d}_{sample.camera_id}_{i:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), sample.image)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        if split == "val": n_val += 1
        else:              n_train += 1

    # data.yaml — Ultralytics format
    yaml = [
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(class_names)],
    ]
    (args.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    log.info("done. train=%d val=%d  classes=%s", n_train, n_val, class_names)
    log.info("train: yolo train data=%s model=yolov8n.pt imgsz=640 epochs=50",
             args.out / "data.yaml")


if __name__ == "__main__":
    main()
