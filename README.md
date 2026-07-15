# cat_detection — camera → cat identity → feeder

A self-hosted, multi-camera stack that watches cat feeding spots, recognises
**which** cat is present, and opens a physical feeder only for the allowed
cats. It ingests each camera's H.264 stream once via mediamtx, decodes each
stream once inside a per-camera detector container, and surfaces **live** and
**historical** detections in a browser UI — without ever re-encoding video.

On top of the video pipeline sits a full model lifecycle: human-in-the-loop
labeling of recorded crops, identity-classifier training with leakage-safe
splits, model comparison, promotion/rollback — all reading the same on-disk
recordings and event DB.

```
camera RTSP ──► mediamtx ─┬── WebRTC (WHEP)             ──► browser <video>
                          │
                          ├── RTSP republish            ──► detector-<id> (PyAV decode once)
                          │                                  ├─ YOLO "is that a cat?"
                          │                                  ├─ EfficientNet-B0 "which cat?"
                          │                                  │   └─ open-set gate → unknown
                          │                                  ├─► SQLite (events.db)
                          │                                  └─► WebSocket events ──► feeder-<id>
                          │                                                            ├─ zone/identity smoothing
                          │                                                            ├─ door state machine
                          │                                                            └─► feeder REST API + journal.db
                          │
                          └── fMP4 recording (30 s segs)──► data/recordings/<camera>/*.mp4
                                                            ├─► indexer  (keeps segment index fresh)
                                                            ├─► pruner   (deletes "boring" old segments)
                                                            ├─► mediamtx playback ──► browser history view
                                                            └─► review / training tools (offline)
```

Key properties:

- Decode happens **once**, inside the detector — the model needs pixels.
  Live and recording paths are bitstream copies of the camera's H.264;
  no re-encode anywhere.
- The browser switches between live (WebRTC) and history (native fMP4 from
  mediamtx playback) based on the timeline playhead.
- The feeder acts on **identity confidence** (`cat_score`, default gate 0.9),
  never on raw detector confidence, with debouncing, multi-cat detection, and
  fail-closed behaviour on stream loss.

Docs map:

- **This file** — what the system is and how to run it.
- [NOTES.md](NOTES.md) — design rationale ("why does it look like that").
- [REPO_CONTEXT.md](REPO_CONTEXT.md) — compact handoff: invariants, data
  layout, workflows.
- [training/README.md](training/README.md) — labeling, training, open-set
  calibration, model promotion, in depth.

## Run

```bash
cp .env.example .env                 # WEB_PORT + pruner/indexer knobs
cp cameras.yaml.example cameras.yaml
# Edit cameras.yaml — at minimum: webrtc_host, and one RTSP URL per camera.
just dev        # configure + dev stack in the foreground (Vite HMR, watchfiles)
just up         # configure + production-shaped stack, detached
# Open http://localhost:8090  (WEB_PORT)
```

`cameras.yaml` is the **single source of truth**. Each camera becomes:

- one mediamtx path (`paths.<id>`),
- one detector container (`detector-<id>`),
- optionally one feeder container (`feeder-<id>`, if the camera has a
  `feeder:` block),
- one entry in the UI camera picker,
- a per-camera recordings directory (`data/recordings/<id>/`).

Generated files (`mediamtx/mediamtx.yml`, `docker-compose.cameras*.yml`,
`webui/nginx.conf`, `webui/public/cameras.json`, `secrets/cameras.env`) are
**not committed** — they are server-specific outputs of `just configure`;
rerun it after editing `cameras.yaml`.

### Camera credentials

Camera RTSP URLs carry passwords, so they never land in a committed file.
`just configure` writes the credentialed URLs to **`secrets/cameras.env`**
(gitignored, chmod 600) as `MTX_PATHS_<ID>_SOURCE=…` env vars, which
docker-compose loads into the mediamtx container. The committed
`mediamtx/mediamtx.yml` has no `source:` lines. Camera ids must be
**hyphen-free** (`[a-z0-9]+`) because they become mediamtx path keys addressed
via env vars. Rotate a camera password by editing `cameras.yaml` and rerunning
`just configure`.

## Services

| Service | Role | Ports |
|---|---|---|
| `mediamtx` | RTSP ingest, WebRTC (WHEP) egress, RTSP republish, fMP4 recording, playback server | 8554, 8889, 9996, 9997 (host network) |
| `detector-<id>` | PyAV decode → detector (`blob`/`yolo`/`yolo_cat`) → identity classifier → SQLite + WS. One per camera. | internal 8091/8092 (proxied by webui) |
| `feeder-<id>` | Consumes detector WS, decides open/close via zone smoothing + door FSM, drives the feeder REST API, logs meals to a journal DB. One per camera with a `feeder:` block. | — |
| `indexer` | Keeps the `recording_segments` SQLite index in sync with mediamtx's mp4 files, so the UI timeline grows without manual rebuilds. Single writer of the index. | — |
| `pruner` | Detection-aware deletion of recording segments older than `KEEP_RECENT_HOURS` (mediamtx `recordDeleteAfter` is the hard cap) | — |
| `mlflow` | Experiment-tracking UI over the local file store the training scripts write to (`data/mlflow`) | `${MLFLOW_PORT}` (default 5000) |
| `webui` | Svelte 5 SPA (Vite dev or nginx prod); proxies WS + mediamtx + detector(s) | `${WEB_PORT}` (default 8090) |

## Detection and identity

The detector is swappable per camera via `detector_type` in `cameras.yaml`:

- `blob` — bright-blob filter (threshold + connected components). Tiny,
  zero-dep, useful for end-to-end testing without a model (shine a flashlight
  at the camera).
- `yolo` — Ultralytics YOLO, cat class only. Default weights are a
  **pre-quantised INT8 OpenVINO IR** baked into the image at build time
  (`yolo26n`), ~3–4× faster than torch CPU. Swap via `yolo_weights`.
- `yolo_cat` — YOLO finds cats, then an EfficientNet-B0 identity classifier
  names them. The runtime model is served from a shared read-only volume
  (`/opt/models/classifier/current`) switched by `just classifier-promote` /
  `classifier-rollback`, so upgrades don't rebuild images.

Two confidences, never mixed:

- `score` — detector confidence: "there is probably a cat in this box".
- `cat_score` — identity confidence: "this is probably *that specific* cat".
  The feeder gate (`CLASSIFIER_MIN_CONF`, default 0.9) uses this one.

**Open-set gate (optional).** A closed-set softmax head will confidently name
*any* animal as one of the known cats. Training therefore also stores a
per-class embedding **prototype**; at serve time a crop whose embedding is
farther (cosine distance) than a calibrated ceiling from every prototype is
rejected as `unknown` — regardless of softmax confidence — so a stranger cat
never opens the feeder. The gate is off by default and activates only when the
model ships `prototypes.json` **and** `DETECTOR_MAX_PROTOTYPE_DISTANCE` is set.
See [training/README.md](training/README.md) for calibration.

**Regions** (all in camera/UI-normalized coordinates, per camera):

- `ignore_regions` — hard masks for static false positives (feeder body/lid).
  Detections mostly covered by them are dropped before events/review/training.
  Never put the food bowl here — that removes real cat detections.
- `decision_polygon` — where feeder decisions are allowed; detections outside
  are still shown/recorded but never open the door.
- `food_region` — soft bowl-texture monitor; reports `food_state`/`food_level`
  over WS for the `empty_bowl` feed mode and the UI overlay. Never filters
  detections.

## Feeder

One feeder process per camera with a `feeder:` block. It connects to the
detector's WebSocket and decides when the door may open:

- `ZoneState` smooths presence and identity votes over a sliding window,
  weighting votes by `cat_score`, and counts simultaneous in-zone cats.
- A pure `decide()` verdict is debounced through an explicit door state
  machine (`CLOSED→ARMING→OPEN→CLOSING`) so single glitch frames never
  chatter the door.
- Fail-closed: detector silence, WS disconnects, and multi-cat / identity
  changes close the door; `dangerous_confusions` can block opening when an
  allowed identity is visually confusable with a non-allowed one.
- Two feed modes: `empty_bowl` (reacts to the bowl monitor) or `scheduled`
  (fixed clock times). Feeds and door sessions land in a shared SQLite
  journal — inspect with `just journal-feed <cat> <days>`.

## Labeling and training

Full guide: [training/README.md](training/README.md). The short loop:

```bash
# 1. Cluster recorded crops for review (detector score is the only gate —
#    old identity predictions are never trusted at cold start):
REVIEW_LABELS=cat_a,cat_b just label-build --min-score 0.5 --mode time

# 2. Bulk-label whole clusters in the browser (http://localhost:8095):
just setup label && just label-review 8095

# 3. Check class balance, then train from human labels only:
just label-stats
just train-run --val-frac 0.2 --test-frac 0.1

# 4. Compare candidate vs current on the same reviewed crops:
just train-compare --candidate new=models/trained/<stamp>/cat_classifier.pt ...

# 5. Promote (exports OpenVINO IR, switches the `current` symlink) + restart:
just classifier-promote && just classifier-restart
```

Design points worth knowing before you touch it:

- **No crop JPEGs.** Review and training decode crops from recordings in
  memory; labels live in a separate `data/review/reviews.db` keyed by source
  event — `events.db` is never modified.
- **Episode-level splits.** Consecutive frames of one visit are
  near-duplicates; whole episodes go to train/val/test, so validation accuracy
  is honest. Replay crops are leak-checked against val/test too.
- **Replay memory** (`just train-replay-set`) keeps a compact, balanced `.npz`
  set of approved crops so weekly fine-tuning doesn't forget older cats after
  recordings are pruned.
- **Identity-preserving augmentation only** — no flips/crops that could erase
  the differences between look-alike cats.
- Cluster embeddings for review: `--embedding auto` prefers DINOv2 (optional
  `label` extra), then cached EfficientNet-B0, then handcrafted visual
  features. DINOv2 is offline-only — never used at runtime.
- Training runs log to MLflow (`data/mlflow`) when it's installed; it's a
  no-op otherwise. Browse via the `mlflow` service or `just mlflow-ui`.

## Operator commands

`just --list` shows everything, grouped. The main ones:

| Command | What it does |
|---|---|
| `just configure` | Regenerate all camera-derived files from `cameras.yaml` |
| `just dev` / `just up` / `just down` / `just ps` / `just logs <svc>` | Stack lifecycle |
| `just label-build` / `label-review` / `label-stats` / `label-reset` | Cold-start cluster labeling |
| `just train-run` / `train-compare` / `train-replay-set` | Classifier training loop |
| `just classifier-promote` / `classifier-rollback` / `classifier-restart` | Runtime model swap |
| `just journal-feed <cat> [days]` | Meal history from the feed journal |
| `just recordings-index-rebuild` | Recovery-only full rebuild of the segment index |
| `just check` | compileall + pytest |

## Layout

```
cat_detection/
├── README.md                       — this file (run + what-is-it)
├── NOTES.md                        — design rationale, decisions, quirks
├── REPO_CONTEXT.md                 — compact handoff: invariants + workflows
├── justfile                        — operator command surface
├── cameras.yaml.example            — per-camera config; copy → cameras.yaml
├── tools/
│   ├── configure.py                — renders all camera-derived files
│   ├── promote_classifier.py       — model version dirs + `current` symlink
│   └── feed_log.py                 — feed-journal CLI
├── docker-compose.yml              — base stack (mediamtx, pruner, indexer, mlflow, webui)
├── docker-compose.cameras.yml      — GENERATED — detector/feeder service per camera
├── docker-compose{,.cameras}.dev.yml — dev overlays (Vite HMR / watchfiles)
├── detector/                       — PyAV → YOLO → classifier → SQLite + WS
│   ├── classifier.py / unknown.py  — runtime identity + open-set gate
│   └── export_classifier.py        — .pt → OpenVINO IR (parity-gated)
├── feeder/                         — WS consumer → decision → door FSM → REST
├── indexer/                        — live recording_segments index writer
├── pruner/                         — detection-aware segment GC
├── review/                         — FastAPI bulk cluster-labeling app
├── training/                       — dataset extraction, clustering, training,
│                                     comparison, replay memory (own README)
├── webui/                          — Svelte 5 SPA
└── tests/                          — pytest suite (`just check`)
```

## Data on disk

Everything mutable lives under `data/` (gitignored — operational state, never
committed):

- `data/recordings/<camera_id>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4` — 30-second
  fMP4 segments written by mediamtx. Filename = segment start, parsed with
  `RECORDING_TZ` (default `UTC`) — keep it consistent everywhere.
- `data/events/events.db` — SQLite: one row per detected box plus the
  `recording_segments` index. Schema in [`detector/storage.py`](detector/storage.py).
- `data/review/` — cluster manifest + human labels (`reviews.db`).
- `data/replay/` — compact `.npz` replay memory for weekly fine-tuning.
- `data/feed_journal/journal.db` — feeds and door sessions.
- `data/mlflow/` — MLflow file store.
- `models/trained/<timestamp>/` — training outputs;
  `models/classifier/versions/<id>` + `current` symlink — promoted runtime
  models.

Recordings + `events.db` are the **batch/training contract**: anything
non-interactive (trainer, review tools, ad-hoc scripts) reads them directly.
`wall_ms` is the authoritative join key between events and recordings — never
raw video PTS. The mediamtx HTTP API and the detector's WS/`/events` endpoints
exist for the *live* browser UI only and are unsuitable for batch use.
