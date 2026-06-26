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
  (mediamtx-written, segment start is the filename in the recording
  container timezone; tools parse this with `RECORDING_TZ`, default `UTC`).
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
(disk). Per-cat classifier training should normally use
[`train_classifier.py`](train_classifier.py) directly from reviewed labels;
[`extract_classifier.py`](extract_classifier.py) is an optional ImageFolder/JPG
export for external tools or visual audits.

## Optional: export reviewed classifier crops to ImageFolder

```bash
uv run python -m training.extract_classifier \
    --recordings data/recordings \
    --db data/events/events.db \
    --out data/datasets/classifier \
    --reviews-db data/review/reviews.db \
    --min-score 0.7 \
    --val-frac 0.1 \
    --test-frac 0.1
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
    test/
        alisa/  ...
        ...
```

This is not the normal weekly training path. Use it when you need a portable
JPG/ImageFolder dataset for visual audit, CVAT, or an external trainer. The
standard in-repo classifier workflow below trains directly from recordings and
`reviews.db` without writing crop JPEGs.

> Grouped splitting matters. Consecutive frames from one visit are
> near-duplicates. Random per-image splitting leaks frame n+1 into val/test while
> frame n is in train, inflating accuracy. Splitting is always episode-based:
> consecutive crops are grouped by camera and wall-clock gap before assignment
> to train/val/test.

---

## Recipe: cold-start cluster review — no old identity labels

For a new classifier, do not trust the old model's names or probabilities.
Use the detector score only as a gate for "there is probably a cat in this box",
then cluster visually similar crops and label whole clusters. This writes no
crop files: crops are decoded from recordings on demand and labels go into the
separate `data/review/reviews.db`.

Quick review workflow (copy-paste):

```bash
cd ~/compose-services/cat_detection_current

# Start a clean review pass (MOVES the old reviews.db + clusters.json aside).
CONFIRM=1 just label-reset

# Build a compact, review-only manifest (one episode = one cluster, deduped,
# hard-capped at 16 crops per cluster).
REVIEW_LABELS=alisa,chuzh,ellie,felisis \
RECORDING_TZ=UTC \
just label-build \
  --min-score 0.5 \
  --mode time \
  --episode-gap-sec 30 \
  --dedupe-window-sec 10 \
  --max-cluster-size 16

# Bulk-label in the browser at http://localhost:8095
REVIEW_LABELS=alisa,chuzh,ellie,felisis \
RECORDING_TZ=UTC \
just label-review 8095
```

The detailed stages below explain each step and its options.

**Stage A — build the cluster manifest** (runs in a detector image because it
already has PyAV/OpenCV/Numpy, and often torch via ultralytics). Low-confidence
boxes are dropped first (`--min-score`, default 0.7).

`--min-score` is the **detector** confidence ("there is probably a cat in this
box"), NOT identity confidence — at cold start there is no trusted identity
model yet, so the detector score is used only as a gate for which crops enter
review. The manifest stores compact embeddings so mixed clusters can be split
later in the browser.

```bash
REVIEW_LABELS=alisa,chuzh,ellie,felisis \
just label-build --min-score 0.7
```

If the live detector event pool is polluted by static false positives, rebuild
review-only events from the recordings with the non-quantized YOLO path:

```bash
just train-rescan --conf 0.3 --imgsz 512 --sample-interval-sec 1
just label-build --model offline-yolov8n --min-score 0.5 --clusters 100
```

`just label-build` writes `data/review/clusters.json` by default. Useful
overrides: `EVENTS_DB`, `RECORDINGS_ROOT`, `CLUSTER_MANIFEST`, `RECORDING_TZ`,
`--camera`, `--model`, `--clusters`, `--default-rotate-deg`.

The manifest is review-only and compact: `--max-cluster-size` is a **hard cap**
on the crops actually written per cluster (the rest are dropped, not hidden),
and near-identical crops are deduped within a per-camera window first
(`--dedupe-window-sec`, `--dedupe-threshold`, `--max-cluster-size`). So the
manifest size is bounded by `clusters × max-cluster-size`: with ~600 clusters
and `--max-cluster-size 16` expect roughly **5k–10k crops**, not the ~200k of
the old "keep everything, hide most" manifests. The review UI shows and labels
exactly the crops in the manifest. Use `--max-cluster-size 0` only when you
really want every (deduped) crop in the review UI.

Validate a generated manifest with `just label-validate` (hard cap respected,
indices valid, no hidden/collapsed fields).

Static false positives can be masked before review. Add tight
`ignore_regions` to `cameras.yaml` only for hard false-positive objects such as
the feeder body/lid: detections whose box is mostly covered by those
camera-normalized regions are dropped by the detector and by
`build_cluster_manifest`. Do not put the bowl/food area into `ignore_regions`,
because that can remove real cat detections and future review crops. Use
`food_region` for the bowl-level monitor instead; it is a soft observation zone
and is not used by review/training filters. For one-off experiments, pass
`--ignore-region grey:0.35,0.02,0.68,0.34`. Use tight regions so real cats next
to or above the feeder are not removed. Tune the cutoff with
`--ignore-region-min-coverage` (default `0.8`).

Embedding choice:

- `--embedding auto` (default) uses cached ImageNet EfficientNet-B0 features
  when available, otherwise falls back to deterministic visual texture/color
  features. This keeps the command usable offline.
- `--embedding efficientnet --allow-download` lets torchvision fetch ImageNet
  weights if the machine has network access.
- `--embedding visual` is fastest and dependency-light, but mixed clusters are
  more likely. Avoid `--no-store-embeddings` unless you do not need the split
  button in the review UI.

If review crops show unrelated frames or many `410 Gone` responses, first check
timezone consistency. The Docker-side manifest builder and host-side review app
must use the same `RECORDING_TZ` as mediamtx. The bundled `just` recipes default
to `UTC`; override with `RECORDING_TZ=America/New_York` only if mediamtx was
recording filenames in the server's EDT/EST timezone.

**Stage B — bulk-label clusters** (host: needs `av` + FastAPI/Pillow).

```bash
just setup label                                  # once
REVIEW_LABELS=alisa,chuzh,ellie,felisis \
just label-review 8095                           # http://localhost:8095
```

The page shows contact sheets per cluster. Label a pure cluster as one cat, mark
junk as `discard`, mark uncertain crops as `unknown`, and split mixed clusters
with `split x2` / `split x3` before labeling the child clusters. If a mixed
cluster still is not worth splitting, mark it as `mixed` so it is not bulk-written
into labels. Cluster labels are written as ordinary review rows keyed by
`src_event_key`, so the training code consumes them without touching `events.db`.

Corrections are **non-destructive**: they go to a *separate* writable
`data/review/reviews.db` keyed by `src_event_key`, so the detector's rows in
`events.db` are never touched and the database can stay read-only. Progress
survives restart.

Check label balance before training:

```bash
just label-stats
```

The report shows trainable cat labels separately from dropped labels such as
`discard` / `unknown`, includes zero-count expected cats from `REVIEW_LABELS`,
and prints a simple `max/min` balance ratio.

**Feeding corrections back into training.** `CropSource` carries each box's
`events` rowid (`box.rowid` / `Sample.src_box.rowid`) and accepts a `reviews`
map (`training.load_reviews("data/review/reviews.db")`): human labels are the
only trusted identity labels for cold start, and `discard` / `unknown` crops are
dropped. Unreviewed crops are ignored by `train_classifier.py` unless you
explicitly opt into `--trust-classifier` later, after you already have a decent
classifier.

```bash
# optional disk dataset (ImageFolder) with human labels; unreviewed crops are ignored:
uv run python -m training.extract_classifier \
    --recordings data/recordings --db data/events/events.db \
    --out data/datasets/classifier --reviews-db data/review/reviews.db \
    --min-score 0.7
```

```python
# in-memory (no JPEGs), same review overlay:
from training import CropSource, load_reviews
src = CropSource(db, recordings, reviews=load_reviews("data/review/reviews.db"))
```

---

## Recipe: train the identity classifier

`train_classifier.py` trains a fresh EfficientNet-B0 from the recordings +
corrections and writes the **best-by-val** model to a NEW path. It does **not**
touch the runtime model — swapping is a later, separate step.

```bash
just setup train
just train-run \
    --confuse alisa,felisis \
    --pad-frac 0.15 \
    --min-recall 0.9
# heavier pass when you have more data: add --full-finetune
# later active-learning pass only: add --trust-classifier --trust-conf 0.9
```

**Fine-tune modes** (mutually exclusive; the optimizer only ever receives
trainable params, and trainable/frozen param counts are logged at startup):

- default (no flag) — **partial**: head + the last two feature blocks.
- `--head-only` — **CPU-friendly**: only the classifier head trains; the backbone
  is a frozen feature extractor. Recommended low-RAM CPU combo:
  ```bash
  just train-run --head-only --batch-size 4 --batch-max-side 320 --num-workers 0
  ```
- `--full-finetune` — the whole backbone (low LR). Passing `--head-only` with
  `--full-finetune` is rejected.

`just train-run` runs through the training uv project with the classifier
extra, so `numpy`, `av`, `torch`, and `torchvision` are installed by uv instead
of being manually added to the system Python. On Linux, `torch` and
`torchvision` are resolved from PyTorch's CPU-only wheel index; CUDA /
`nvidia-*` wheels are not needed for this project.

The trainer is CPU/server friendly by default: cold-start training indexes only
human-reviewed trainable crop refs, not the whole detector pool, and decodes
pixels only for the current in-RAM batch. Crops are copied out of their decoded
frame (contiguous), so a crop never keeps a full frame alive in RAM. Train batch
crops are capped with `--batch-max-side 384`; val/test still use the
runtime-identical preprocessing. `--batch-size` defaults to `8` (conservative for
CPU); raise it when you have RAM headroom. If the machine is still tight, lower
both:

```bash
just train-run --min-recall 0.85 --batch-size 4 --batch-max-side 320
```

RSS is logged before the dataset, after the first batch, and after each epoch.
The optional `TorchLazyCachedDataset` (training/torch_dataset.py) is **truly
lazy**: it stores only lightweight `CropRef` metadata (camera, wall_ms, box,
rotation) — never decoded images or closures over them — and decodes exactly one
crop per `__getitem__`, serving it through a byte-bounded LRU of resized `uint8`
crops. Cap the cache with `TRAINING_CACHE_MAX_MB` (`0` disables retention, every
access re-decodes). RAM plateaus at the cache budget regardless of dataset size;
see `tests/test_memory_smoke.py`.

**Replay memory is also lazy.** `--replay-set` loads only manifest metadata
(label, camera, `wall_ms`, `.npz` path) into RAM; replay pixels decode per batch
from their `.npz`, so a large replay set no longer materialises in memory.
`--replay-max-items N` caps the set (balanced across classes, deterministic with
`--seed`). Because replay crops are old fresh crops, they are checked against the
val/test split for **leakage** before being added to train: exact (`src_event_key`),
same-frame, near-timestamp (same camera within `--replay-leak-window-sec`), and a
perceptual-hash fallback. Default `--replay-leakage-policy error` fails closed;
`drop-from-replay` drops the offending replay crops; `move-related-episode-to-train`
moves the colliding eval episode into train.

What it does, and why each part:

- **Labels (human-only by default).** Cold start trains ONLY on human-reviewed
  labels. The detector `score` is a crop-quality gate ("probably a cat"), not
  an identity label. `cat_score` is the existing identity classifier confidence
  and is ignored in cold start. `--trust-classifier` is for a later
  active-learning pass, when an existing classifier is already good enough to
  reuse some unreviewed high-confidence labels. `discard`/`unknown` are dropped;
  class names are the sorted unique surviving labels, saved with the model.
  `--confuse` here only highlights a cell in the confusion matrix.
- **Honest split by default.** A random per-image split leaks near-duplicate neighbours.
  Crops are grouped into **episodes** (same camera, `wall_ms` gaps
  `> --episode-gap-sec` start a new one) and whole episodes go to train, val, or
  test. `--val-frac` and `--test-frac` control the ratios. Val selects the best
  epoch; test is held out for the final honest report. Runtime operating
  thresholds, such as opening the feeder only at `cat_score >= 0.9`, should be
  chosen from val behavior and then verified once on test. The split is nudged
  so val/test contain every class when possible.
- **No JPEGs and no full crop cache.** Crops are decoded from recordings only
  for the current batch, trained/evaluated, and discarded.
- **Rotation, per-event.** Each crop is rotated by its own recorded `rotate_deg`
  (the shared `training.sources.rotate_crop`, same convention as the detector), so
  data captured under different — or later-changed — camera rotations trains
  together correctly. For events recorded before `rotate_deg` was persisted, pass
  `--default-rotate-deg <deg>` (one warning is logged). `rotate_deg=0` is a no-op.
- **Crop framing.** `--pad-frac` MUST equal the detector `CLASSIFIER_PAD_FRAC`
  and `build_cluster_manifest --pad-frac` (default 0.15 everywhere). The eval/val
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

**Export + swap the classifier (later, deliberate step).**

```bash
# 1) export the new .pt to OpenVINO IR (parity-gated, see ../detector/):
python detector/export_classifier.py \
    --pt models/trained/<timestamp>/cat_classifier.pt \
    --out detector/models/cat_classifier_openvino_NEW \
    --crops <dir of real crops>          # for a meaningful parity check
# 2) point the detector at it (per camera) and rebuild:
#    cameras.yaml: classifier_weights: /opt/models/cat_classifier_openvino_NEW
#    just configure && just up
# Keep classifier_pad_frac identical to what you trained with.
```

**Compare models before swapping.** Evaluate current vs candidate on the same
human-reviewed crops, with closed-set metrics and thresholded runtime behavior:

```bash
just train-compare \
    --candidate current=/opt/models/cat_classifier_openvino \
    --candidate new=models/trained/<stamp>/cat_classifier.pt \
    --baseline current \
    --thresholds 0.7,0.8,0.9 \
    --out reports/classifier_compare.json
```

The script decodes crops from recordings in memory and trusts only
`reviews.db`. The conservative deploy signal is: candidate macro/min recall does
not regress and high-confidence wrong predictions at the feeder threshold do not
increase. If `--replay-set` is supplied, replay metrics are reported separately
as forgetting/regression memory; they are not blended into the headline
promotion verdict.

**Weekly retraining.** Use the already-trained runtime classifier for
active-learning queues, not as truth. Each week:

1. Build/review new clusters for the recent time range.
2. Update compact replay memory from human-reviewed crops.
3. Train from fresh human labels plus replay memory.
4. Compare current vs candidate with `train-compare`.
5. Export/swap only after comparison is clean.

Replay memory stores a small balanced set of compressed numpy crops (`.npz`),
not JPGs. It is train-only memory: it prevents forgetting, but it is not a
replacement for a held-out test set.

```bash
just train-replay-set --per-class 500
```

The command merges the existing replay set with currently available reviewed
crops, removes near-duplicates, keeps diverse examples per class, and writes
`data/replay/manifest.jsonl` plus `data/replay/crops/.../*.npz`.

To continue from the previous weekly classifier rather than ImageNet-only
initialization, pass:

```bash
just train-run \
    --init-from models/trained/<previous>/cat_classifier.pt \
    --replay-set data/replay
```

For regression checking against old memory:

```bash
just train-compare \
    --candidate current=/opt/models/cat_classifier_openvino \
    --candidate new=models/trained/<stamp>/cat_classifier.pt \
    --baseline current \
    --replay-set data/replay
```

If old recordings were pruned, old labels cannot be decoded for training. For a
long-lived training corpus, either keep reviewed event segments longer
(`PRUNER_KEEP_RECENT_HOURS`, `PRUNER_PRE_ROLL_SEC`, `PRUNER_POST_ROLL_SEC`, or a
separate archive) or deliberately export an approved crop dataset.

---

## Recipe: YOLO fine-tune (per-cat detector)

```bash
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
│                      decode_one_crop / CropRef for random-access review/replay)
├── reviews.py        (load human corrections reviews.db → {rowid: label})
├── extract_classifier.py     (optional ImageFolder/JPG export from CropSource)
├── extract_detector.py       (one-shot script wrapping FullFrameSource)
├── build_cluster_manifest.py (cold-start clustering manifest)
└── train_classifier.py       (train identity classifier; best-by-val → models/trained/)

../review/            (FastAPI bulk label-review app; `just label-review`)
├── cluster_app.py    (bulk cluster labels; corrections → reviews.db)
├── static/cluster.html
└── requirements.txt  (fastapi/uvicorn/av/numpy/Pillow — no openvino/torch)
```
