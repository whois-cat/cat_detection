# YOLO fine-tune pipeline (single-class `cat`)

Offline, manual pipeline that turns existing **streamhub** recordings into a
hand-labelled dataset and a fine-tuned YOLO detector. It is fully independent of
the old `events.db` classifier path and **never touches** streaming, recording,
feeders, the running models, or the existing web UIs. All reads are of
already-written data; all writes go under `data/yolo_dataset/` and
`models/trained/`.

Run everything from the repo root on the machine that has the streamhub
recordings (the server). `just --list` shows the `[yolo]` group.

## Geometry (important)

Cameras run with `rotate_deg: 90`. The collector saves each frame **as the
detector sees it** — `detect_roi` crop + rotation applied — so labelling,
training and runtime share one coordinate system. Sidecar suggestion boxes are
rotated to match. One shared helper (`training/frame_geometry.py`, mirrored by
`sources.rotate_crop` and `cv_worker/geometry.py`) is the only place this
transform lives.

## 0. One-time dependency warmup

```bash
# base (collect / review / build-version): av, numpy, opencv, pyyaml
uv run --project training python -c "import training.streamhub_dataset"
# training (train / evaluate / export): + ultralytics, psutil
uv run --project training --extra yolo python -c "import ultralytics"
```

## 1. Collect frames for review

Full, un-annotated frames. Sidecar detections are **hints only** (selection
reasons), never labels. Resumable and incremental (safe to re-run / cron).

```bash
just yolo-collect \
  --camera black,grey \
  --from 2026-10-01 --to 2026-10-02 \
  --tag shaved           # optional range tag, e.g. the shaved-cat period
```

What it saves (per frame kept): camera, source segment, exact PTS, wall time,
visit group, and the reasons it was picked (`regular`, `visual_change`,
`low_confidence`, `multiple_boxes`, `visit_sample`). Predictions are stored in a
separate table from any future human labels.

Key knobs (all have defaults; see `--help`):
`--candidate-interval-sec 5`, `--regular-interval-sec 300`, `--visit-gap-sec 90`,
`--visit-sample-interval-sec 20`, `--max-visit-samples 6`, `--dedupe-threshold 3`,
`--low-confidence 0.35`, `--budget-gb 5`, `--report-json run.json`.

A **visit** is a run of activity (a YOLO detection **or** frame-to-frame motion
**or** low confidence **or** multiple boxes) with gaps `< --visit-gap-sec`; so
visits are detected even when YOLO never sees the cat. Near-identical frames in a
visit are dropped; a periodic `regular` sample is always kept so coverage does
not depend on YOLO.

Check the queue any time (read-only):

```bash
just yolo-queue
```

## 2. Label in CVAT (manual import/export)

Export an un-reviewed batch as a CVAT/COCO zip (with model boxes pre-filled as
**suggestions**, clearly not truth):

```bash
just yolo-review-export out/batch-001.zip 200
```

In CVAT: import the zip, then for each frame draw **one box per cat** around the
whole visible cat (head + body together), label it `cat`, fix/add/delete boxes.
A frame with no cat is a negative **only after a human confirms it**. Export from
CVAT as COCO and import back:

```bash
just yolo-review-import path/to/cvat-export.zip               # cat boxes present
just yolo-review-import path/to/cvat-export.zip --confirm-empty  # allow empty frames as negatives
```

Import validates image IDs, dimensions and box bounds, accepts only the `cat`
class, and keeps stable sample IDs. Corrections are non-destructive.

## 3. Build an immutable dataset version

```bash
just yolo-build-version --val-frac 0.15 --test-frac 0.15
```

- Splits by **visit group across all cameras** (related observations of one
  visit never straddle train/val/test).
- Group→split assignment is **stable** (hash of the group id), so adding data
  later does not reshuffle existing groups.
- Exact-duplicate images are de-duplicated before the split; near-duplicates
  across splits are counted and reported.
- Images are **hardlinked** (no second copy of every frame); labels are written
  as `0 cx cy w h`.
- Reports loudly when there are too few independent groups or a required split
  is empty — **honest evaluation is not yet possible**, collect/review more
  visits.

Output: `data/yolo_dataset/versions/<version_id>/` with `data.yaml`,
`manifest.json` (checksummed) and `images|labels/{train,val,test}/`.

## 4. Fine-tune (from a pretrained `.pt`, not an INT8 export)

```bash
just yolo-train \
  --dataset data/yolo_dataset/versions/<version_id> \
  --weights yolov8n.pt \
  --imgsz 640 --epochs 50 --device cpu
```

Writes `models/trained/<run>/` with `weights/best.pt`, Ultralytics plots, and a
durable `report.json`. The runtime model is **not** replaced. Uses the same
`imgsz`/geometry as runtime; keep the current architecture (yolov8n) first so
only one factor changes. Add old examples alongside new ones rather than
training only on the latest look of the shaved cat.

## 5. Evaluate / compare base .pt vs INT8

```bash
just yolo-evaluate --dataset <version> --model models/trained/<run>/weights/best.pt
# tell model vs export error apart:
just yolo-evaluate --dataset <version> --model <openvino_int8_dir>
```

Reports precision/recall/mAP50/mAP50-95, operational TP/FP/FN at a fixed
confidence, errors on confirmed-empty frames, per-subset metrics (tag the shaved
period with `--tag shaved` at collection to get the shaved-cat split separately),
and offline inference timing (explicitly **not** end-to-end feeder latency).

## 6. Export to OpenVINO (parity-gated) — optional, still not deployed

```bash
just yolo-export --dataset <version> --model models/trained/<run>/weights/best.pt --int8
```

Evaluates `.pt` vs the export on the same test split and fails if quality drops
beyond the limits. Exit code 2 = gate failed. It never promotes a runtime model.

## 7. Compare two runs

```bash
just yolo-compare models/trained/<run_a>/report.json models/trained/<run_b>/report.json
```

## Deploying later (separate, deliberate decision)

Not part of this pipeline. When you decide to deploy a single-class model, the
runtime filters the COCO `cat` class id `15` in
`cv-worker/cv_worker/models/yolo.py` — a single-class fine-tune uses id `0`, so
that filter must change (or auto-detect from `model.names`). Changing bounding
boxes also changes the classifier's input crop, so re-check the whole chain.
