"""Stage A — build the label-review manifest. METADATA ONLY, no image files.

Runs in the detector environment (needs OpenVINO + the classifier IR). It walks
CropSource over the recordings, and for each detected box:

  1. decodes the crop into memory (CropSource),
  2. runs the classifier (CatClassifier.classify_all — same _preprocess + IR as
     production; nothing is re-implemented),
  3. records ONLY metadata and lets the pixels fall out of scope.

The single output is a JSONL manifest — no JPEGs, no crops on disk. Each line:

    {crop_id, src_event_key, wall_ms, camera, model, box{x,y,w,h}, rotate_deg,
     pad_frac, predicted, conf, probs{name: p}}

`src_event_key` is the events-table rowid, the non-destructive link the review
UI writes corrections against. `box` is in camera coords and `rotate_deg` is the
event's recorded rotation, so the UI re-cuts the identical crop on the fly via
training.decode_one_crop (the classifier here is scored on that same rotated crop).

Records are sorted by OVERALL uncertainty — least-confident crops first — so the
genuinely ambiguous ones (any cats, any confused pair) surface at the top:
top-1 probability ascending, tie-broken by the top-1 minus top-2 margin
ascending. `--confuse` is OPTIONAL and does NOT affect ordering or trust; it's
only carried for the UI to highlight (the app reads REVIEW_CONFUSE).

Usage (from the repo root, detector env):

    python -m training.build_review_manifest \
        --db data/events/events.db \
        --recordings data/recordings \
        --classifier detector/models/cat_classifier_openvino \
        --out data/review/manifest.jsonl --min-score 0.7
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


def uncertainty_key(rec: dict):
    """Sort key: lower == more uncertain. Top-1 probability ascending, tie-broken
    by the top-1 − top-2 margin ascending. Any cats, any confused pair — the
    least-confident crops sort first."""
    ps = sorted(rec["probs"].values(), reverse=True)
    top1 = ps[0] if ps else 0.0
    margin = top1 - (ps[1] if len(ps) > 1 else 0.0)
    return (top1, margin)


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
                    help="OPTIONAL confusable pair for UI highlight only (e.g. "
                         "alisa,felisis); does NOT affect ordering. The app reads "
                         "REVIEW_CONFUSE — pass it there to actually highlight.")
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="drop low detector-score boxes before review")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="crop context padding — MUST match detector "
                         "CLASSIFIER_PAD_FRAC and train_classifier --pad-frac")
    ap.add_argument("--default-rotate-deg", type=int, default=0,
                    help="rotation to assume for events recorded BEFORE rotate_deg "
                         "was persisted (set to the camera's rotate_deg then)")
    ap.add_argument("--t-from", type=int, default=None, help="wall_ms lower bound")
    ap.add_argument("--t-to", type=int, default=None, help="wall_ms upper bound")
    ap.add_argument("--limit", type=int, default=None, help="cap number of crops")
    args = ap.parse_args()

    import cv2  # BGR -> RGB only

    from training import CropSource
    from training.db import open_db_ro

    # Auto-detect --model when the DB holds exactly one (the common case); refuse
    # to guess when several are present so crops aren't silently mixed.
    if args.model is None:
        ro = open_db_ro(args.db)
        q = "SELECT DISTINCT model FROM events"
        p: list = []
        if args.camera is not None:
            q += " WHERE camera_id = ?"
            p.append(args.camera)
        models = [r[0] for r in ro.execute(q, p)]
        ro.close()
        if not models:
            sys.exit("no events found (check --db / --camera)")
        if len(models) > 1:
            sys.exit(f"multiple models in DB — pass --model one of: {models}")
        args.model = models[0]
        print(f"[manifest] auto-selected --model {args.model!r}")

    clf = _load_classifier(args.classifier)

    src = CropSource(
        db_path=args.db, recordings_root=args.recordings,
        camera_id=args.camera, model=args.model,
        t_from=args.t_from, t_to=args.t_to, min_score=args.min_score,
        pad_frac=args.pad_frac, default_rotate_deg=args.default_rotate_deg,
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
            "rotate_deg": int(sample.rotate_deg),
            "pad_frac": float(args.pad_frac),
            "predicted": predicted,
            "conf": float(probs[predicted]),
            "probs": probs,
        })

    records.sort(key=uncertainty_key)   # least-confident first

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(records)} records to {args.out} "
          f"(sorted by uncertainty; lowest top-1 confidence first)")


if __name__ == "__main__":
    main()
