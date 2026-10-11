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
training/review                 YOLO catalog (verified boxes) -> classifier crops
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
  decider/feed_journal/journal.db    decider journal (door sessions, scheduled feeds)
  streamhub/pins.json                ranges the pruner keeps
  yolo_dataset/catalog.sqlite3, images/   collected frames, verified boxes
  yolo_dataset/versions/, label_studio_backups/   dataset snapshots, LS task backups
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

## Labeling Workflow

One pass, one tool, one source of frames (the YOLO catalog): in Label Studio
every cat gets a box (trains the detector) and every box gets the cat's name
(trains the identity classifier).

1. `just detector-collect` saves frames into `data/yolo_dataset`, with the live
   models' answers from the streamhub sidecars (boxes, and per box the
   classifier's cat probabilities).
2. `just detector-label` pushes them to Label Studio, hardest first (the
   detector or the classifier unsure, several cats, ...), with the boxes and
   names pre-filled. A human fixes boxes, picks/fixes the name of every box
   (`CAT_NAMES` plus `unknown`, keys 1..n) and submits; a frame without cats is
   submitted empty.
3. `just detector-sync` writes the verified boxes and names into the catalog
   (`annotations.cat_name`).
4. `just detector-build` / `detector-train` for the detector;
   `just classifier-stats` / `classifier-train` for the classifier.

```bash
# .env: CAT_NAMES=alisa,chuzh,ellie,felisis
just detector-round 7        # = detector-collect --days 7 + detector-label --notify
just detector-sync           # after labelling
just classifier-stats
just classifier-train --confuse alisa,felisis
```

Model answers are only suggestions: nothing is trained on until a person
submits the frame, so a confident mistake (e.g. a cat after a haircut taken for
another) never teaches itself. `unknown` (a cat, not sure which) and unnamed
boxes are not trained on. Frames reviewed before names existed are queued again
for their names with `just detector-label --reopen-unnamed` (their boxes stay,
the old Label Studio task is backed up to `data/yolo_dataset/label_studio_backups/`).

Crops come only from human-verified boxes, so there is no detector-score gate
and no ignore-region filtering. Catalog frames are stored in the detector's
input geometry, so crops (`CLASSIFIER_PAD_FRAC` padding) need no rotation.
Reviewed frames have their own share of the storage budget
(`--reviewed-budget-gb`, 4 of 5 GB by default). When it fills up, the least
useful are retired: near-duplicates first, then the oldest of the largest
(camera, cat name, month) group; frames where you corrected the models and a
minimum per cat/camera/month are never retired. `detector-build` keeps the two
newest dataset versions plus the deployed detector's.

`just detector-round` can run from cron (weekly is plenty) and reports in
Telegram how many frames wait and how many need a look, when
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` are set (see `.env.example`).

The previous stack's `events.db` + recordings path (labels from a reviews.db)
still exists in the Python tools, but no `just` recipe uses it.

## Training Workflow

Train only from reviewed human labels:

```bash
just classifier-train \
  --confuse alisa,felisis \
  --val-frac 0.2 \
  --test-frac 0.1
```

Important behavior:

- The split is group/episode-level by default, not random crop-level.
- Neighboring frames from the same visit should land in only one split.
- This avoids fake validation accuracy from near-duplicate frames leaking from
  train into validation/test.
- The train/validation/test ratio is configurable with `--val-frac` and
  `--test-frac`.

Concepts:

- An epoch is one full pass over the training examples.
- Validation data is checked between/after epochs to choose the better model
  state and avoid overfitting.
- Test data is held back until final evaluation. Do not use it to make daily
  training decisions.
- A threshold is the minimum confidence required before the system acts on a
  prediction. For the feeder, the default identity threshold is `0.9`.

## Weekly Fine-Tuning Workflow

1. `just detector-round 7`, label in Label Studio, `just detector-sync`.
2. `just classifier-stats` to check every cat has enough named boxes.
3. Fine-tune from the previous model:
   `just classifier-train --init-from models/trained/<previous>/cat_classifier.pt`.
4. `just classifier-compare` the candidate against the deployed model.
5. `just deploy classifier <run>` only if metrics and threshold behavior are acceptable.

Reviewed catalog frames are kept within their budget (thinned, not dropped
wholesale), so old examples stay trainable after the recordings are pruned;
the replay memory the previous stack needed for that is not part of this
workflow.

## Model Comparison

Compare models on the same reviewed data before promoting a candidate:

```bash
just classifier-compare \
  --candidate current=models/classifier/current \
  --candidate new=models/trained/<stamp>/cat_classifier.pt \
  --baseline current \
  --thresholds 0.7,0.8,0.9 \
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
- `training/…`: labeling and training; `training/catalog_crops.py`
  turns verified catalog boxes into classifier crops.
- `tools/models.py` (deploy/rollback of both models), `tools/feed_log.py`.

## Gotchas

- A pre-filled name on a box around a wall/bowl is not evidence of that cat;
  it is a detector false positive plus an irrelevant identity guess: delete the
  box.
- A box must hold one cat. Two cats touching get two boxes; if one cat cannot
  be told apart, name it `unknown` (not trained on) rather than guess.
- Keep `CLASSIFIER_PAD_FRAC` consistent between training, export, and runtime.
- `data/` is operational state. Do not commit local DBs, recordings, replay
  memory, reports, or large model weights unless explicitly requested.

## Current Code-State Notes

- Branch `redesign` replaces mediamtx/detector/feeder/indexer with the stack
  above (see `PLAN.md`, `README.md` for migration).
- Training/review read the YOLO catalog (`--catalog`); the previous stack's
  `events.db` path remains in the Python tools only.
