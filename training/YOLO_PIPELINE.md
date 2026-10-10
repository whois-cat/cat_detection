# YOLO fine-tune pipeline (`cat`)

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
just box-collect \
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
just box-queue
```

## 2. Label boxes in the browser (Label Studio, under the hood)

No COCO, no zips, no manual config. Label Studio runs locally and serves the
collected frames straight from the catalog folder; the model's boxes are
pre-filled as suggestions.

**First time only** — start it and get an access token:
```bash
just box-label        # starts Label Studio at http://localhost:8080 (localhost only)
# open it (SSH tunnel if remote: ssh -L 8080:localhost:8080 <server>), log in with
# LABEL_STUDIO_USERNAME/PASSWORD from .env, then copy
# Account & Settings -> Access Token into .env as LABEL_STUDIO_API_KEY
```

**Each round:**
```bash
just box-label           # starts LS if needed + pushes unreviewed frames (with suggestions)
# in the project click "Label All Tasks": Submit (Ctrl/Cmd+Enter) then moves to the
# next frame. Opening a single task from the table does NOT auto-advance.
# label in the browser: one box per cat (head + body), class `cat`; fix/add/delete.
# no cat names here — which cat it is gets labelled later with `just cat-label`.
# a frame submitted with no box = confirmed-empty negative. Untouched = stays unreviewed.
just box-sync            # pull submitted boxes back into the catalog (verified)
```

Frames are pushed hardest first (`priority` column: 0 model unsure, 1 several
boxes, 2 visual change, 3 visit sample, 4 regular), so the first few hundred
teach the model the most. Tasks pushed before priorities existed get the column
patched in on the next push; sort the Data Manager by `priority` before "Label
All Tasks" to walk them in that order.

Because the model pre-fills boxes, most frames are "accept/adjust", and you only
draw from scratch where YOLO was wrong. `just box-label --limit 300` pushes a
smaller first batch. Stop the server with `just box-label-stop` (projects and
labels persist in a docker volume).

> Label Studio itself is a third-party service pinned in `docker-compose.label.yml`.
> It is standalone and localhost-only; the main stack is untouched. The CVAT
> COCO export/import path (`training/yolo_review.py`) still exists for anyone who
> prefers it, but is no longer the default.

## 3. Build an immutable dataset version

```bash
just box-build --val-frac 0.15 --test-frac 0.15
```

- Splits by **visit group across all cameras** (related observations of one
  visit never straddle train/val/test).
- Group→split assignment is **stable** (hash of the group id), so adding data
  later does not reshuffle existing groups.
- Exact-duplicate images are de-duplicated before the split; near-duplicates
  across splits are counted and reported.
- Images are **hardlinked** (no second copy of every frame); labels are written
  as `15 cx cy w h`.
- Uses the **COCO class numbering** of the base model and the runtime detector
  (80 names, only `cat` = 15 labelled). The fine-tune keeps the whole pretrained
  head, cat output included, and every model and version agree on the cat id.
  Versions built with the earlier single-class numbering are refused with a
  hint to rebuild.
- Reports loudly when there are too few independent groups or a required split
  is empty — **honest evaluation is not yet possible**, collect/review more
  visits.

Output: `data/yolo_dataset/versions/<version_id>/` with `data.yaml`,
`manifest.json` (checksummed) and `images|labels/{train,val,test}/`.

## 4. Fine-tune (from a pretrained `.pt`, not an INT8 export)

```bash
just box-train \
  --dataset data/yolo_dataset/versions/<version_id> \
  --weights yolov8n.pt \
  --imgsz 640 --epochs 50 --device cpu
```

Writes `models/trained/<run>/` with `weights/best.pt`, Ultralytics plots,
`test_eval/` (test-split plots), and a durable `report.json`. The runtime model is **not** replaced. Uses the same
`imgsz`/geometry as runtime; keep the current architecture (yolov8n) first so
only one factor changes.

This is always **fine-tuning** (transfer learning) from a pretrained `.pt`,
never random-init "from scratch", and never from an INT8 export (rejected). Two
bases:

- from the COCO base: `--weights yolov8n.pt` (first fine-tune);
- from the previous fine-tune (incremental): `--weights models/trained/<prev>/weights/best.pt`.

**Mixing in old versions (anti-forgetting).** Verified samples stay in the
catalog, so a fresh `box-build` already includes old visits. When you
train on a version that does *not* contain older appearances (e.g. a recent-only
version), mix them back with `--replay-version` (repeatable):

```bash
just box-train \
  --dataset data/yolo_dataset/versions/<recent> \
  --weights yolov8n.pt \
  --replay-version data/yolo_dataset/versions/<older> \
  --replay-version data/yolo_dataset/versions/<pre-haircut>
```

Only each replay version's **train** split is added. Replay samples that collide
with the current version's val/test (by sample id, image checksum, or visit
group) are dropped, so mixing never inflates evaluation. The run report records
`replay_added` / `replay_skipped_leakage`.

## 5. Evaluate / compare base .pt vs INT8

```bash
just box-eval --dataset <version> --model models/trained/<run>/weights/best.pt
# tell model vs export error apart:
just box-eval --dataset <version> --model <openvino_int8_dir>
```

Reports precision/recall/mAP50/mAP50-95, operational TP/FP/FN at a fixed
confidence, errors on confirmed-empty frames, per-subset metrics (tag the shaved
period with `--tag shaved` at collection to get the shaved-cat split separately),
and offline inference timing (explicitly **not** end-to-end feeder latency).

## 6. Compare two runs

```bash
just box-compare models/trained/<run_a>/report.json models/trained/<run_b>/report.json
```

## Deploying (separate, deliberate decision)

```bash
just deploy detector <run>     # e.g. yolo-20261010-011816; default: newest run
just models                    # what runs now
just rollback detector         # previous version, or the built-in COCO model
```

`deploy` exports the run's `best.pt` to INT8 OpenVINO in the cv-worker image
(the ultralytics/OpenVINO versions that serve it) and evaluates `.pt` vs export
on the run's own test split. It refuses on a quality drop: mAP50-95
(threshold-free), and recall and false positives at the runtime confidence
(0.25). Ultralytics' own precision/recall are reported, not gated — they sit at
each model's max-F1 threshold, which moves between the two. On a pass it
installs `models/detector/versions/<run>`, switches `current` and restarts
cv-worker; the web UI and sidecars show the detector by its run name. The
classifier deploys the same way (`just deploy classifier <run>`).

A fine-tune has the same 80 COCO classes as the built-in model (cv-worker looks
the `cat` id up in the model's own names), so it is a drop-in replacement.
Changing boxes also changes the classifier's input crops, and per-camera
`cv.yolo_conf` values tuned for the COCO model (e.g. a very low threshold set to
catch missed cats) should be revisited.
