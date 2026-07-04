# Repo Context: `cat_detection`

This file is a compact handoff for the repository state, data flow, and model
workflow. It is written to preserve the project context between local work,
server pulls, and future Codex sessions.

## What This System Does

`cat_detection` is a local, multi-camera cat detection and feeder-control stack.
The system records camera video, detects cats, serves a live/history web UI,
stores detection events, supports human review, trains an identity classifier,
and uses the classifier to open a feeder only for allowed cats.

The current branch is focused on a safer cold-start labeling pipeline:

- do not trust the old identity classifier as truth;
- filter crops by detector confidence first;
- cluster visually similar crops;
- let a human bulk-label clusters;
- train from human labels with leakage-safe train/validation/test splits;
- keep a compact replay set so weekly fine-tuning does not forget older cats.

## Architecture

High-level flow:

```text
camera RTSP
  -> mediamtx
      -> browser WebRTC live view
      -> detector RTSP input
      -> local MP4 recordings
      -> playback API for timeline/history

detector
  -> YOLO boxes
  -> optional identity classifier
  -> data/events/events.db
  -> websocket events for webui/feeder

review/training tools
  -> read events.db + recordings directly
  -> decode crops in memory
  -> write review labels and model artifacts

feeder
  -> consumes detector websocket
  -> opens only when an allowed cat is present with enough classifier confidence

pruner
  -> deletes old boring recording segments
  -> keeps recent video and video around detections
```

## Important Invariants

- Do not casually change the WebRTC/mediamtx live-video path. The live stream
  base is from `live2`: mediamtx owns ingest/recording/WebRTC, and detector
  services should not add slow extra video hops.
- Training/review tools read `data/events/events.db` and `data/recordings`
  directly. They should not depend on the web UI APIs.
- `wall_ms` is the authoritative join key between detections and recordings.
  Avoid using raw video PTS as a cross-system identifier.
- Recording filename timezone matters. mediamtx segment names are parsed using
  `RECORDING_TZ` or `TZ`, defaulting to `UTC`.
- The main review/training path does not store crop JPGs. Crops are decoded
  from recordings in memory. The replay set is the intentional exception: it
  stores compressed `.npz` crop arrays for long-term training memory.
- Detector confidence and identity confidence are different things:
  `events.score` means "detector thinks this box is a cat"; `cat_score` means
  "identity classifier thinks this is a specific cat".
- Static false positives such as a feeder or bowl should be configured as
  `ignore_regions` in `cameras.yaml`. Detections whose box center lands inside
  these camera-normalized regions are dropped before events/review/training.

## Data Layout

Local runtime data is intentionally outside git:

```text
data/
  events/events.db                 SQLite detection event store
  recordings/<camera>/*.mp4        mediamtx recording segments
  review/clusters.json             cold-start cluster manifest
  review/reviews.db                human labels and split decisions
  replay/manifest.jsonl            compact replay memory index
  replay/crops/<label>/*.npz       compressed crop arrays

models/
  trained/<timestamp>/cat_classifier.pt
```

Common untracked local files such as `configs/`, `reports/`, `secrets/`,
`data/`, model weights, and generated artifacts should stay out of normal
source commits unless there is a specific reason to version them.

## Core Services

- `mediamtx`: camera ingest, WebRTC live view, playback, recording.
- `detector-<camera_id>`: one detector process per configured camera.
- `webui`: Svelte/nginx UI for live/history.
- `pruner`: detection-aware recording cleanup.
- `feeder`: feeder door control, if feeder config is present.

Generated camera-specific compose/config files come from `tools/configure.py`
and `cameras.yaml`.

## Operator Commands

Run from the repo root.

```bash
just configure
just dev
just up
just down
just ps
just logs detector-grey
just check
```

`just up` starts the production-shaped local stack. `just dev` starts the dev
stack with frontend/backend watch behavior where configured. `just check`
compiles Python packages and runs tests.

## Cold-Start Labeling Workflow

Cold start means: assume the old identity classifier is not reliable. Use only
detector confidence to decide whether a crop is likely worth reviewing.

Example:

```bash
export REVIEW_LABELS=cat_a,cat_b,cat_c,cat_d
export RECORDING_TZ=America/New_York

just label-build detector-grey --default-rotate-deg 90 --min-score 0.7 --clusters 80
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

- "this whole cluster is `cat_a`";
- "this whole cluster is `cat_b`";
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
  --confuse cat_a,cat_d \
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
  --candidate current=/opt/models/cat_classifier_openvino \
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
- confusion between visually similar cats, especially `cat_a` and `cat_d`.

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

The pruner scans recording segments periodically and deletes old segments that
do not contain detections or nearby context.

Current defaults:

- segment duration: `30s`;
- keep pre-roll around detections: `30s`;
- keep post-roll around detections: `30s`;
- always keep the newest `24h`;
- prune interval: `3600s` / one hour;
- dry-run can be enabled with `PRUNER_DRY_RUN=1`.

mediamtx also has its own hard recording retention cap:
`recordDeleteAfter: 720h`, i.e. 30 days. Replay memory is still useful because
approved examples can outlive both pruner cleanup and the 30-day hard cap.

## Important Files

- `docker-compose.yml`: base services: mediamtx, pruner, webui.
- `docker-compose.cameras.yml`: generated camera detector services.
- `mediamtx/mediamtx.yml`: generated mediamtx camera/path config.
- `tools/configure.py`: reads camera config and generates service/config files.
- `justfile`: main operator command surface.
- `detector/main.py`: detector service runtime and websocket events.
- `detector/detectors.py`: YOLO/OpenVINO detector wrappers.
- `detector/classifier.py`: runtime identity classifier.
- `detector/export_classifier.py`: export trained classifier for deployment.
- `detector/storage.py`: SQLite event writes.
- `training/build_cluster_manifest.py`: cold-start clustering manifest.
- `review/cluster_app.py`: cluster review API.
- `review/static/cluster.html`: browser UI for bulk cluster labeling.
- `training/reviews.py`: review DB helpers.
- `training/train_classifier.py`: identity classifier training.
- `training/build_replay_set.py`: compact replay memory builder.
- `training/replay.py`: replay-set loader.
- `training/compare_classifiers.py`: candidate model evaluation/comparison.
- `training/segments.py`: recording segment lookup and timezone parsing.
- `training/extract_classifier.py`: optional ImageFolder/JPG export.
- `pruner/pruner.py`: detection-aware video cleanup.
- `feeder/main.py`: feeder runtime and classifier confidence gate.
- `feeder/zone_state.py`: temporal smoothing for identities/presence.
- `feeder/door_fsm.py`: debounced feeder door state machine.

## Gotchas

- A crop showing a wall/bowl with `model says cat_a 32%` is not evidence of
  `cat_a`; it is usually a detector false positive plus an irrelevant identity
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

- Branch: `feat/cat-classifier-and-feeder`.
- Last source-code commit before this document: `1a75b2c Add cold-start cluster labeling and replay training`.
- Last full check after the implementation: `just check` passed with 17 tests.
- Known local untracked runtime/development artifacts at the time of writing:
  `configs/`, `reports/`, and `yolov8n.pt`.
