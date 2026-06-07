"""Stage A — build the label-review manifest. METADATA ONLY, no image files.

Runs in the detector environment (needs OpenVINO + the classifier IR). It walks
CropSource over the recordings, and for each detected box:

  1. decodes the crop into memory (CropSource),
  2. runs the classifier (CatClassifier.classify_all — same _preprocess + IR as
     production; nothing is re-implemented),
  3. records ONLY metadata and lets the pixels fall out of scope.

The single output is a JSONL manifest — no JPEGs, no crops on disk. Each line:

    {crop_id, src_event_key, wall_ms, camera, model, box{x,y,w,h}, pad_frac,
     predicted, conf, probs{name: p}}

`src_event_key` is the events-table rowid, the non-destructive link the review
UI writes corrections against. `box` is in camera coords so the UI can re-cut
the identical crop on the fly via training.decode_one_crop.

Records are sorted most-contentious-first so the confusable pair surfaces at the
top of the review queue (pass --confuse alisa,felisis — names are data, never
hardcoded):

  tier 0 — the two highest-probability classes ARE the confuse pair: sort by the
           margin |p1 - p2| ascending (a near-tie is maximally contentious);
  tier 1 — predicted is one of the pair (but top-2 aren't both): by conf ascending;
  tier 2 — everything else: by conf ascending.

Usage (from the repo root, detector env):

    python -m training.build_review_manifest \
        --db data/events/events.db \
        --recordings data/recordings \
        --classifier detector/models/cat_classifier_openvino \
        --out data/review/manifest.jsonl \
        --model yolov8n+cat --confuse alisa,felisis --min-score 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_classifier(classifier_dir: str):
    """Import CatClassifier from the detector package (we run in its env)."""
    try:
        from classifier import CatClassifier
    except ImportError:
        # Not on the detector WORKDIR — add detector/ to the path and retry.
        sys.path.insert(0, str(ROOT / "detector"))
        from classifier import CatClassifier
    return CatClassifier(classifier_dir)


def contentiousness_key(rec: dict, confuse: set[str]):
    """Sort key: lower == more contentious (see module docstring)."""
    ranked = sorted(rec["probs"].items(), key=lambda kv: kv[1], reverse=True)
    conf = rec["conf"]
    if confuse and len(ranked) >= 2:
        top2 = {ranked[0][0], ranked[1][0]}
        if top2 == confuse:
            return (0, abs(ranked[0][1] - ranked[1][1]))
    if confuse and rec["predicted"] in confuse:
        return (1, conf)
    return (2, conf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="events.db")
    ap.add_argument("--recordings", type=Path, required=True, help="data/recordings root")
    ap.add_argument("--classifier", required=True,
                    help="OpenVINO IR dir (cat_classifier.xml + classes.json)")
    ap.add_argument("--out", type=Path, required=True, help="manifest.jsonl path")
    ap.add_argument("--camera", default=None, help="filter by camera_id (default: all)")
    ap.add_argument("--model", default=None, help="filter by detector model (default: all)")
    ap.add_argument("--confuse", default=None,
                    help="comma-separated confusable pair to float to the top, "
                         "e.g. alisa,felisis (names from data — not hardcoded)")
    ap.add_argument("--min-score", type=float, default=None, help="drop low-score boxes")
    ap.add_argument("--pad-frac", type=float, default=0.15, help="crop context padding")
    ap.add_argument("--t-from", type=int, default=None, help="wall_ms lower bound")
    ap.add_argument("--t-to", type=int, default=None, help="wall_ms upper bound")
    ap.add_argument("--limit", type=int, default=None, help="cap number of crops")
    args = ap.parse_args()

    import cv2  # BGR -> RGB only

    from training import CropSource

    confuse = {c.strip() for c in args.confuse.split(",") if c.strip()} if args.confuse else set()
    clf = _load_classifier(args.classifier)

    src = CropSource(
        db_path=args.db, recordings_root=args.recordings,
        camera_id=args.camera, model=args.model,
        t_from=args.t_from, t_to=args.t_to, min_score=args.min_score,
        pad_frac=args.pad_frac,
    )

    records: list[dict] = []
    for n, sample in enumerate(src):
        if args.limit is not None and n >= args.limit:
            break
        sb = sample.src_box
        if sb is None or sb.rowid is None:
            continue
        # Classify in memory, then drop the pixels (sample is rebound next loop).
        crop_rgb = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB)
        probs = {name: p for name, p in clf.classify_all(crop_rgb)}
        predicted = max(probs, key=probs.__getitem__)
        records.append({
            "crop_id": f"{sample.camera_id}:{sb.rowid}",
            "src_event_key": int(sb.rowid),
            "wall_ms": int(sample.wall_ms),
            "camera": sample.camera_id,
            "model": sample.model,
            "box": {"x": int(sb.x), "y": int(sb.y), "w": int(sb.w), "h": int(sb.h)},
            "pad_frac": float(args.pad_frac),
            "predicted": predicted,
            "conf": float(probs[predicted]),
            "probs": probs,
        })

    records.sort(key=lambda r: contentiousness_key(r, confuse))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_pair = sum(1 for r in records if contentiousness_key(r, confuse)[0] == 0)
    print(f"wrote {len(records)} records to {args.out} "
          f"({n_pair} in the contentious {sorted(confuse) or '—'} pair, sorted first)")


if __name__ == "__main__":
    main()
