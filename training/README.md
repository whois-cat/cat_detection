# live2 training pipeline

This directory contains the bridge between the live2 detection pipeline
and any model-training workflow you want to run. The live2 pipeline
captures recordings + per-frame detection events; this package turns that
on-disk data into datasets you can feed to a YOLO fine-tune (object
detection) or a classifier (per-cat identity).

> If you are picking this up cold: read [../README.md](../README.md) first
> to understand the rest of the system. The short version: a camera RTSP
> stream is recorded as 30-second fMP4 segments to `data/recordings/`, and
> a Python detector writes per-box rows to a SQLite DB at
> `data/events/events.db`.

---

## What "internal public API" means here

The live2 system has two surfaces:

| Consumer | Video | Labels |
|---|---|---|
| **Browser / UI** (live, interactive) | mediamtx WebRTC + playback HTTP (chunked fMP4) | detector `GET /events` |
| **Training / batch extraction** (this package) | recording files on disk | SQLite directly |

The browser surfaces are streaming-shaped and not suitable for batch
random access. The training surfaces are the **on-disk files and the
SQLite schema** — that's the contract the modeller depends on:

- `data/recordings/<camera_id>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4`
  (mediamtx-written, segment start is the filename in local wall-clock).
- `data/events/events.db` — schema in [`../detector/storage.py`](../detector/storage.py).

If those layouts ever change, this package is the single place that needs
updating.

---

## Joining events to frames — wall-clock, not PTS

See the docstring in [`segments.py`](segments.py) for the full reasoning.
Short version: the detector and mediamtx are two independent consumers of
the RTSP source. Their PTS values do not share an origin. The shared
authoritative key is **wall-clock**: events carry `wall_ms`, segments
carry their start in the filename. So the extractor:

1. Locates the segment that covers `event.wall_ms` (bisect on segment
   filenames).
2. Computes the segment-local seek time `(event.wall_ms − segment.start_ms) / 1000`.
3. PyAV-seeks inside the segment to that media time.

Precision is sub-frame in practice. `event.pts` is preserved in the event
row but unused by the extractor — kept for future cross-system anchoring
work or for in-stream debugging.

---

## Abstract base + two concrete sources

A *Sample* is `(BGR ndarray, list[Box], wall_ms, camera_id, model)`.

```python
from training import FullFrameSource, CropSource

# One Sample per frame, all boxes attached. For training a detector.
for sample in FullFrameSource(db_path, recordings_root,
                              camera_id="default", model="yolov8n"):
    img = sample.image            # H×W×3 BGR uint8 (camera orientation)
    for b in sample.boxes:        # all detected cats in this frame
        ...

# One Sample per BOX, image is the crop (with optional context padding).
# For training a per-cat classifier.
for sample in CropSource(db_path, recordings_root,
                         cat=None,            # all cats; or filter to one
                         pad_frac=0.15):
    crop = sample.image           # padded cat crop, camera orientation
    [box] = sample.boxes          # exactly one, coords relative to crop
    label = box.cat
```

Both classes share `SampleSource` (in [`sources.py`](sources.py)). It
handles the SQLite query, the segment lookup, and the PyAV seek
batching — so segments are opened once each and frames are decoded in
PTS-monotonic order regardless of which subclass you use. The subclass
implements only `_emit(frame, img) -> Iterator[Sample]`. Adding a third
shape (e.g. *fixed-size patches around the box centre*, or *both crop and
full frame in one Sample*) is a five-line subclass.

Why two surfaces:

- Detector training (YOLO, RT-DETR, …) wants the **full frame plus all
  boxes**. The model learns to localise and to score; crops would
  destroy that supervision.
- Classifier training (EfficientNet, ResNet, …) wants the **box crop
  only**, paired with the cat label. The model has been spared
  localisation; the localiser already chose the patch.

Both shapes come from the same `(frame, image)` pair, so we never decode
twice. If you need the third shape (full-frame *and* crop together,
e.g. to train a head that takes context features into account), subclass
`SampleSource` and emit two related samples per frame — still one decode.

---

## In-memory path: skip the disk roundtrip for PyTorch training

The disk-extractor recipes below encode crops as JPEGs because that's the
universal interchange format (CVAT review, hand-off to other workstations,
inspectable on the filesystem). But for a one-laptop PyTorch training run
the JPEG step is a CPU-cost cache that you can side-step entirely.

[`torch_dataset.py`](torch_dataset.py) provides two adapters around the
same `SampleSource`:

- `TorchStreamingDataset` (IterableDataset) — each epoch re-iterates the
  source, re-decoding segments. Right for one-shot training or huge
  streaming datasets.
- `TorchCachedDataset` (map-style Dataset) — materialises the source
  into RAM once, serves random-access lookups thereafter. Right for the
  standard multi-epoch classifier loop. Rule of thumb: ~1 GB RAM per 5k
  ~250² crops.

See the docstring at the top of [`torch_dataset.py`](torch_dataset.py)
for a complete usage sketch (a CropSource feeding torchvision
transforms into an EfficientNet/ResNet/whatever loop).

When to prefer disk JPEGs anyway:

- You want to hand the dataset to CVAT for review.
- You want to share the dataset across machines / persist it for repro.
- The dataset doesn't fit in RAM (tens of thousands of crops, full size).
- You're training Ultralytics YOLO — its trainer hard-wires
  YAML/directory layout, so disk is the only path.

YOLO fine-tune always uses [`extract_detector.py`](extract_detector.py)
(disk). Per-cat classifier training has both options.

## Recipe: per-cat classifier

```bash
cd live2
uv run python -m training.extract_classifier \
    --recordings data/recordings \
    --db data/events/events.db \
    --out data/datasets/classifier \
    --model yolov8n \
    --min-score 0.3 \
    --val-frac 0.1 \
    --split-by-track   # keep all crops from one track in one split
```

Output is a torchvision `ImageFolder`-compatible tree:

```
data/datasets/classifier/
    train/
        alisa/   13_........jpg ...
        chuzh/   ...
        ellie/   ...
        felisis/ ...
    val/
        alisa/  ...
        ...
```

Train a classifier with the parent project's existing EfficientNet-B0
recipe (see [`../../scripts/`](../../scripts/) and [`../../models/`](../../models/)
in the parent repo). The 4-cat label set should be identical.

> `--split-by-track` matters. Within one track, consecutive frames are
> near-duplicates. Random per-image splitting leaks frame n+1 of a track
> into val while frame n is in train, inflating val accuracy. Splitting
> per `track_id` keeps each cat-appearance entirely on one side.

---

## Recipe: label review (correct confusable labels) — no JPEGs on disk

The detector's labels are guesses from the very model we're trying to improve
(it confuses alisa↔felisis), so before re-training we hand-verify them. The
recordings are already the image store and `events.db` already has the box
coordinates — so this workflow writes **no crop files at all**. Crops are cut
from the recordings into memory, shown/scored, and the pixels are dropped; only
metadata (the labels) is persisted.

**Stage A — build the manifest** (runs in the detector image: it has OpenVINO +
the baked classifier IR). Walks `CropSource`, runs the classifier per crop
(reusing `detector/classifier.py::CatClassifier.classify_all` — same `_preprocess`
and IR as production), and writes one JSONL line of metadata per crop, sorted by
**overall uncertainty** (least-confident first). Driven by `just`:

```bash
just review-manifest detector-grey --min-score 0.3
```

`just review-manifest` mounts the repo into the detector container and runs
`python -m training.build_review_manifest`. `--model` is auto-detected when the DB
has a single model. Overridable env: `EVENTS_DB`, `RECORDINGS_ROOT`,
`CLASSIFIER_IR`, `REVIEW_MANIFEST`. Each line: `{crop_id, src_event_key, wall_ms,
camera, model, box, rotate_deg, pad_frac, predicted, conf, probs{name:p}}`.
**Ordering** = top-1 probability ascending, tie-broken by the top-1−top-2 margin —
the genuinely ambiguous crops (any cats, any confused pair) float to the top.
`--confuse` is optional and only affects the UI highlight (not ordering/trust).
For data captured before per-event rotation was recorded, pass
`--default-rotate-deg <deg>` (the camera's rotation then). **The only thing on
disk is `manifest.jsonl`.**

> Bare invocation (outside `just`, in any env with openvino+av+cv2):
> `python -m training.build_review_manifest --db … --recordings … --classifier … --out … --min-score 0.3`

**Stage B — review web app** (host: needs `av` + this package, NOT openvino/torch).

```bash
just review-setup                              # once: venv + fastapi/uvicorn/av/Pillow
REVIEW_CONFUSE=alisa,felisis just review 8095  # → http://localhost:8095
```

Each crop is decoded on demand straight from the recordings
(`training.decode_one_crop`, which reuses the same segment lookup + keyframe seek
as the batch sources), encoded to an in-memory JPEG, and streamed — never written
to a file. The page shows the crop big, the model's guess + confidence, the two
top probabilities (the `REVIEW_CONFUSE` pair highlighted, if set), and one button
per class plus `unknown` / `discard`; hotkeys `1..N`, `←/→` to navigate, a live
`X / N` progress bar. Crops are shown in the detector's inference orientation
(each rotated by its own recorded `rotate_deg`), so you judge exactly what the
model sees.

Corrections are **non-destructive**: they go to a *separate* writable
`data/review/reviews.db` keyed by `src_event_key`, so the detector's rows in
`events.db` are never touched and the database can stay read-only. Progress
survives restart (already-reviewed crops are skipped; any crop can be reopened
and re-labelled).

**Feeding corrections back into training.** `CropSource` carries each box's
`events` rowid (`box.rowid` / `Sample.src_box.rowid`) and accepts a `reviews`
map (`training.load_reviews("data/review/reviews.db")`): where a crop has a human
label it overrides the detector's `box.cat`, and `discard` / `unknown` crops are
dropped — unreviewed crops keep the detector label. Both training paths use it:

```bash
# disk dataset (ImageFolder) with corrected labels:
uv run python -m training.extract_classifier \
    --recordings data/recordings --db data/events/events.db \
    --out data/datasets/classifier --reviews-db data/review/reviews.db \
    --min-score 0.3 --split-by-track
```

```python
# in-memory (no JPEGs), same correction overlay:
from training import CropSource, load_reviews
src = CropSource(db, recordings, reviews=load_reviews("data/review/reviews.db"))
```

---

## Recipe: train OUR classifier (to replace the donor)

`train_classifier.py` trains a fresh EfficientNet-B0 from the recordings +
corrections and writes the **best-by-val** model to a NEW path. It does **not**
touch the donor (`detector/models/cat_classifier.pt`) or the runtime — swapping
is a later, separate step.

```bash
cd live2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m training.train_classifier \
    --db data/events/events.db \
    --recordings data/recordings \
    --reviews-db data/review/reviews.db \
    --confuse alisa,felisis \
    --pad-frac 0.15 \
    --min-recall 0.9
# heavier pass when you have more data:  add --full-finetune
# cheap volume (trust the donor on unreviewed crops):  --trust-detector --trust-conf 0.9
```

What it does, and why each part:

- **Labels (human-only by default).** The donor confuses several pairs, so by
  default NO detector label is trusted — training uses ONLY human-reviewed labels.
  `--trust-detector` opts back in to the donor's label for unreviewed crops when
  its `cat_score >= --trust-conf` (cheap volume, your call). `discard`/`unknown`
  are dropped; class names are the sorted unique surviving labels, saved with the
  model. `--confuse` here only highlights a cell in the confusion matrix.
- **Honest split.** `track_id` is empty, so a random split leaks near-duplicate
  neighbours. Crops are grouped into **episodes** (same camera, `wall_ms` gaps
  `> --episode-gap-sec` start a new one) and whole episodes go to train OR val.
  The split is stratified so val contains **every** class (per-class recall needs it).
- **No JPEGs.** Crops are decoded from the recordings into RAM
  (`CropSource` → `TorchCachedDataset.materialise()`), trained on, discarded.
- **Rotation, per-event.** Each crop is rotated by its own recorded `rotate_deg`
  (the shared `training.sources.rotate_crop`, same convention as the detector), so
  data captured under different — or later-changed — camera rotations trains
  together correctly. For events recorded before `rotate_deg` was persisted, pass
  `--default-rotate-deg <deg>` (one warning is logged). `rotate_deg=0` is a no-op.
- **Crop framing.** `--pad-frac` MUST equal the detector `CLASSIFIER_PAD_FRAC`
  and `build_review_manifest --pad-frac` (default 0.15 everywhere). The eval/val
  transform is byte-identical to `detector/classifier.py::_preprocess`; only train
  augments (h-flip, small rotation, mild brightness/contrast, light
  random-resized-crop — no hard color jitter, since night IR is ~grayscale).

**Reading the output.** The script prints a confusion matrix (rows=true,
cols=pred), per-class precision/recall, overall + macro accuracy, and explicitly
the `alisa↔felisis` cross-error cell — that cell is the whole point; it should be
near zero. It ends with a **PASS/FAIL** line: PASS iff every class (incl. alisa
and felisis) has val recall `>= --min-recall`. On FAIL it warns loudly and does
NOT crash — collect/relabel more crops (especially the confuse pair) and re-run.

**Output.** `models/trained/<timestamp>/cat_classifier.pt` (same
`{state_dict, class_names, num_classes}` format `export_classifier.py` expects)
plus `metadata.json` (class_names, pad_frac, preprocessing spec, val metrics).

**Export + swap the donor (later, deliberate step).**

```bash
# 1) export the new .pt to OpenVINO IR (parity-gated, see ../detector/):
python detector/export_classifier.py \
    --pt models/trained/<timestamp>/cat_classifier.pt \
    --out detector/models/cat_classifier_openvino_NEW \
    --crops <dir of real crops>          # for a meaningful parity check
# 2) point the detector at it (per camera) and rebuild:
#    cameras.yaml: classifier_weights: /opt/models/cat_classifier_openvino_NEW
#    just configure && just rebuild-detectors
# Keep classifier_pad_frac identical to what you trained with.
```

---

## Recipe: YOLO fine-tune (per-cat detector)

```bash
cd live2
uv run python -m training.extract_detector \
    --recordings data/recordings \
    --db data/events/events.db \
    --out data/datasets/detector \
    --model yolov8n \
    --val-frac 0.1
    # add --collapse-to-single-class to train a "cat or not" detector
```

Output is Ultralytics-compatible:

```
data/datasets/detector/
    data.yaml
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt    # `class cx cy w h`, normalised
```

Train:

```bash
uv pip install ultralytics
yolo train data=data/datasets/detector/data.yaml \
          model=yolov8n.pt imgsz=640 epochs=50
```

After training, drop the resulting `best.pt` into the detector container
and point `YOLO_WEIGHTS` at it. You can also export to OpenVINO INT8
for the same ~3-4× CPU speedup the off-the-shelf weights get — see the
Dockerfile in [`../detector/`](../detector/) for the recipe.

---

## What's missing (intentionally) and how to add it

These are deliberate not-yet-done bits. None of them block the v1 dataset:

- **Hard-negative sampling.** The detector only writes rows when it sees
  something. Empty frames aren't logged. For a robust detector you want
  a sampling of empty frames as negatives — easy add: scan segment files
  by media-time stride, decode the frame, and emit a Sample with
  `boxes=[]` if no event from any model is within ±100 ms.

- **Label review / correction UI.** ✅ Done — see "Recipe: label review"
  below. Two stages: a metadata-only manifest (Stage A) and a tiny web app
  that decodes crops on the fly and records non-destructive corrections
  (Stage B). No JPEGs are ever written; recordings + events.db stay the
  only image/label store on disk.

- **Track-aware temporal sampling.** Right now the extractor takes every
  detection. For classifier training you usually want at most a few
  diverse frames per track. Drop-in: read events ordered by
  `(track_id, wall_ms)`, take every Nth, or use a perceptual hash to
  skip near-duplicates.

- **Live evaluation harness.** "Does the new model do better than the
  current one on the last 24 hours of recordings?" Two ways: (a) script
  that runs both models over a date range and produces a confusion
  matrix; (b) the detector already supports running multiple `model`
  rows in parallel, so you can run a candidate alongside production and
  compare in the UI.

---

## Code map

```
training/
├── README.md         (this file)
├── pyproject.toml    (just deps for extraction — no torch in here)
├── __init__.py
├── db.py             (events.db queries; per-row → per-frame regrouping; box.rowid)
├── segments.py       (wall_ms → segment file + offset; see docstring)
├── sources.py        (SampleSource ABC + FullFrameSource + CropSource;
│                      decode_one_crop / CropRef for random-access review)
├── reviews.py        (load human corrections reviews.db → {rowid: label})
├── extract_classifier.py     (one-shot script wrapping CropSource)
├── extract_detector.py       (one-shot script wrapping FullFrameSource)
├── build_review_manifest.py  (Stage A: metadata-only review manifest)
└── train_classifier.py       (train OUR classifier; best-by-val → models/trained/)

../review/            (Stage B: FastAPI label-review web app; `just review`)
├── app.py            (on-demand in-memory crop decode; corrections → reviews.db)
├── static/index.html (one-page vanilla-JS reviewer)
└── requirements.txt  (fastapi/uvicorn/av/numpy/Pillow — no openvino/torch)
```
