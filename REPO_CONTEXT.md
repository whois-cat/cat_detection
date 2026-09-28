# Repo Context: `cat_detection`

This file is a compact handoff for the repository state, data flow, and model
workflow. It is written to preserve the project context between local work,
server pulls, and future Codex sessions.

## What This System Does

`cat_detection` is a local, multi-camera cat detection and feeder-control stack.
It records camera video without re-encoding, detects and identifies cats per
frame, serves a live/history web UI with frame-exact detections, stores every
CV result and feeder decision next to the video, and opens each feeder only for
its allowed cats. Human review and classifier training run on top.

Architecture and decisions: `PLAN.md`. How to run: `README.md`.

## Architecture

```text
camera RTSP
  -> streamhub (Go)             ingest, wall-clock PTS, fMP4 segments, API, webui
      -> cv-worker (Python)     every frame over the hub port; YOLO + classifier
      <- results                boxes + per-cat probabilities, every processed frame
      -> decider (Python)       results; door FSM, scheduled feeding, journal
      <- decisions              state/reason/identity, stored and shown
      -> data/streamhub/recordings/<camera>/<date>/<hour>/*.mp4 + .labels.jsonl
      -> browser                live (WebSocket fMP4 + labels), history (segments)
pruner                          sparsifies old recordings, enforces the size cap
training/review                 previous stack's events.db + recordings (for now)
```

## Important Invariants

- Frame identity is `(camera, pts)`; pts is 90 kHz ticks since the Unix epoch,
  strictly increasing per camera. Camera RTP timing is kept (no CFR rewrite);
  RTCP is ignored (the cameras' sender reports are wrong).
- Files are the source of truth: segment names carry start and duration,
  sidecars (`<start>.labels.jsonl`) carry CV results and decisions. Indexes are
  rebuilt from them, so anyone may delete files.
- Detections leave cv-worker in camera-frame fractions (camera orientation);
  `rotate_deg`/`detect_roi` only affect the model input.
- Detector score (`score`: is it a cat) and identity probabilities (`cats`) are
  different things; decider thresholds apply to identity.
- Classifier runtime preprocessing (`cv-worker/cv_worker/models/classifier.py`)
  must stay bit-identical to training; training imports it from there.
- `classifier_pad_frac` (cv-worker `CLASSIFIER_PAD_FRAC`) must match training.

## Data Layout

Local runtime data is intentionally outside git:

```text
data/
  streamhub/recordings/<camera>/<date>/<hour>/<start>_<dur>ms.mp4   segments
  streamhub/recordings/<camera>/<date>/<hour>/<start>.labels.jsonl  CV results + decisions
  streamhub/feed_journal/journal.db  decider journal (door sessions, scheduled feeds)
  streamhub/pins.json                ranges the pruner keeps
  events/events.db                   previous stack's detections (training)
  recordings/<camera>/*.mp4          previous stack's recordings (training)
  review/clusters.json, reviews.db   cold-start clusters and human labels
  replay/                            compact replay memory
  mlflow/                            experiment tracking
models/
  trained/<timestamp>/cat_classifier.pt
  classifier/versions/<id>/, current, previous   runtime classifier (OpenVINO)
```

## Core Services

- `streamhub`: camera ingest, recording, hub port (:9000), API + webui.
- `cv-worker`: CV for all cameras (`CV_MAX_FPS` per camera, default 2).
- `decider`: all feeders; `dry_run` decides without calling the feeder API.
- `pruner`: detection-aware cleanup + size cap.
- `mlflow`: experiment-tracking UI.

All configured in `config.yaml` (template `config.example.yaml`); compose in
`docker-compose.yml`.

## Operator Commands

Run from the repo root.

```bash
just up          # build + start
just status      # per-camera ingest status
just logs cv-worker
just check       # all tests
```

## Cold-Start Labeling Workflow

Cold start means: assume the old identity classifier is not reliable. Use only
detector confidence to decide whether a crop is likely worth reviewing.

Example:

```bash
export REVIEW_LABELS=alisa,chuzh,ellie,felisis
export RECORDING_TZ=America/New_York

just label-build --default-rotate-deg 90 --min-score 0.7 --clusters 80
just setup label
just label-review 8095
```

What happens:

1. `training.build_cluster_manifest` reads `events.db` and recordings.
2. It keeps only detections with detector `score >= --min-score`.
3. It ignores old `cat` / `cat_score` identity predictions for truth.
4. It computes embeddings for crops.
5. It groups similar crops into clusters.
6. The review UI shows contact sheets, not isolated random crops.
7. A human labels a whole cluster as a cat, `unknown`, or `discard`.
8. If a cluster is mixed, use the split button and label the smaller clusters.

Default detector gate is `--min-score 0.7`. For this project that is a good
starting point because obvious bowl/wall false positives should not enter
identity labeling. If real cats are getting filtered out, lower it slightly;
if too much junk remains, raise it.

## Embeddings And Clustering

An embedding is a numeric fingerprint of an image crop. Similar-looking crops
should have similar vectors, so clustering can group them before labeling.

Current embedding options:

- `--embedding visual`: lightweight handcrafted visual features.
- `--embedding efficientnet`: ImageNet EfficientNet-B0 feature vectors.
- `--embedding auto`: try EfficientNet-B0 if its weights are already cached,
  otherwise fall back to visual features.

`clusters.json` stores metadata plus compact embeddings. It does not store crop
images. The UI reconstructs contact-sheet thumbnails from the original videos
when needed. Keep embeddings if you want mixed clusters to be splittable later;
`--no-store-embeddings` makes the manifest smaller but disables recursive split.

CLIP is not the best default for this case because the task is fine-grained
identity recognition in IR/security-camera crops, not text-image matching.
DINO-style self-supervised vision embeddings could be a future quality upgrade,
but EfficientNet/visual embeddings are simpler and practical for now.

## Bulk Labeling

Bulk labeling means labeling a group at once:

- "this whole cluster is `alisa`";
- "this whole cluster is `chuzh`";
- "this whole cluster is junk, discard it";
- "this cluster is mixed, split it".

This is much faster and safer than labeling tens of thousands of single crops
one by one. It also matches the real uncertainty: the model should ask the
human about groups and edge cases, not pretend it knows the cat names at cold
start.

Human labels are stored in `data/review/reviews.db`, keyed by source event.

## Training Workflow

Train only from reviewed human labels by default:

```bash
just train-run \
  --default-rotate-deg 90 \
  --confuse alisa,felisis \
  --val-frac 0.2 \
  --test-frac 0.1 \
  --replay-set data/replay
```

Important behavior:

- The split is group/episode-level by default, not random crop-level.
- Neighboring frames from the same visit should land in only one split.
- This avoids fake validation accuracy from near-duplicate frames leaking from
  train into validation/test.
- The train/validation/test ratio is configurable with `--val-frac` and
  `--test-frac`.
- `--replay-set` examples are used as train-only memory.

Concepts:

- An epoch is one full pass over the training examples.
- Validation data is checked between/after epochs to choose the better model
  state and avoid overfitting.
- Test data is held back until final evaluation. Do not use it to make daily
  training decisions.
- A threshold is the minimum confidence required before the system acts on a
  prediction. For the feeder, the default identity threshold is `0.9`.

There is a `--trust-classifier` mode for later active-learning workflows, but
the cold-start path should not use classifier predictions as truth.

## Weekly Fine-Tuning Workflow

The recommended weekly loop:

1. Let the system collect new recordings and events.
2. Build/update the cluster manifest on recent data.
3. Bulk-label clusters and split mixed clusters.
4. Rebuild/update the compact replay set.
5. Fine-tune from the previous model with replay memory.
6. Compare the candidate model against the current deployed model.
7. Promote only if metrics and threshold behavior are acceptable.

Example fine-tune:

```bash
just train-replay-set --per-class 500

just train-run \
  --init-from models/trained/<previous>/cat_classifier.pt \
  --replay-set data/replay \
  --val-frac 0.2 \
  --test-frac 0.1
```

Fine-tuning from the previous model helps, but it is not enough by itself if
old video has been deleted. Without either old recordings or replay memory, the
model can catastrophically forget older examples. The compact replay set is the
chosen compromise: keep a small, diverse memory of approved crops without
turning the repo into a JPG archive.

## Replay Set

`training.build_replay_set` creates/updates a compact training memory:

```bash
just train-replay-set --per-class 500
```

It stores compressed crop arrays under `data/replay`, plus a manifest. This is
local runtime data, not source code. It should not be committed by default.

Use replay for weekly training so the model sees:

- new reviewed examples from the current week;
- stable older examples for every cat;
- hard/rare examples that should not be forgotten.

## Model Comparison

Compare models on the same reviewed data before promoting a candidate:

```bash
just train-compare \
  --candidate current=models/classifier/current \
  --candidate new=models/trained/<stamp>/cat_classifier.pt \
  --baseline current \
  --thresholds 0.7,0.8,0.9 \
  --replay-set data/replay \
  --out reports/classifier_compare.json
```

Useful checks:

- overall accuracy;
- macro recall;
- worst-class recall;
- high-confidence wrong predictions at the feeder threshold;
- confusion between visually similar cats, especially `alisa` and `felisis`.

The script can produce a verdict, but promotion should still be a human
decision when the model controls a physical feeder.

## Feeder Safety

The feeder uses identity confidence, not detector confidence.

Current default:

```text
CLASSIFIER_MIN_CONF=0.9
```

The intended rule is: the feeder can open only when the classifier is at least
90% confident and the cat is allowed by feeder policy/cooldown state.

Detector confidence answers "is there probably a cat in this crop?" Identity
confidence answers "which cat is it?" Do not mix these gates. Detector-side
identity fallback uses `DETECTOR_UNKNOWN_CONF` (legacy alias:
`classifier_min_conf`) to decide when to record `cat="unknown"`.

## Pruner Behavior

Every `pruner.interval` (10 min): recordings newer than `keep_recent` (3 h) are
kept; older segments are deleted unless a detection is within `event_margin`
(30 s) or a pin covers them; segments CV never processed are deleted
(`delete_unprocessed`); then the oldest unpinned go until the total is under
`max_size` (50 GB). Sidecars go with their segments.

## Important Files

- `PLAN.md`: architecture and decisions of the current stack.
- `config.example.yaml`: all settings, commented.
- `streamhub/internal/…`: `timeline` (RTP → wall-clock), `recorder`, `hub`
  (protocol), `live`, `sidecar`, `prune`, `api`.
- `cv-worker/cv_worker/`: `harness.py` (decode/infer loop), `geometry.py`,
  `models/` (yolo, classifier, blob); `cv-worker/tools/export_classifier.py`.
- `decider/decider/`: `feeder.py` (per-feeder loop), `zone_state.py`,
  `decision.py`, `door_fsm.py`, `journal.py`, `schedule_feed.py`, `display.py`.
- `webui/src/`: `App.svelte`, `Player.svelte`, `Timeline.svelte`, `lib/`.
- `training/…`, `review/…`: labeling and training (unchanged).
- `tools/promote_classifier.py`, `tools/feed_log.py`.

## Gotchas

- A crop showing a wall/bowl with `model says alisa 32%` is not evidence of
  `alisa`; it is usually a detector false positive plus an irrelevant identity
  guess.
- Cold-start manifests should sort/review by detector quality and clusters, not
  old identity names.
- If many `/api/crop/...` requests return `410 Gone`, the source recording is no
  longer on disk. Review sooner, retain video longer, or rely on replay memory
  after approved crops have been exported.
- Keep `classifier_pad_frac` consistent between training, export, and runtime.
- `--no-store-embeddings` makes cluster manifests smaller but removes the data
  needed for later split-mixed-cluster actions.
- `data/` is operational state. Do not commit local DBs, recordings, replay
  memory, reports, or large model weights unless explicitly requested.

## Current Code-State Notes

- Branch `redesign` replaces mediamtx/detector/feeder/indexer with the stack
  above (see `PLAN.md`, `README.md` for migration).
- Training/review still read the previous stack's `data/events` and
  `data/recordings`; adapting them to segments + sidecars is pending.
