# cat_detection

## What this is

A self-hosted system for detecting and identifying four cats — Alisa, Chuzh, Ellie, and Felisis — from an RTSP security camera. YOLO + ByteTrack tracks each cat across frames with a stable ID; a fine-tuned EfficientNet-B0 classifier identifies which cat it is per track. The system runs continuously, drives an auto-feeder via REST API, retrains weekly from live crops, stores all metrics in Grafana, and optionally streams annotated video over MJPEG.

---

## Architecture

| Component | Role | Profile |
|---|---|---|
| **cat-live** | 24/7 live detection: YOLO → classifier → sliding window → door state machine; saves crops to `data/crops/unsorted/` | `live` |
| **Airflow** | Weekly retrain DAG: dedup crops → auto-label → train | `airflow` |
| **MLflow** | Tracks training experiments, logs metrics and model artifacts | `monitoring` |
| **VictoriaMetrics** | Stores time-series metrics pushed by live detection and retrain | `monitoring` |
| **Grafana** | Dashboards: detections per cat, confidence, model accuracy, crop counts | `monitoring` |

All services are defined in a single `docker-compose.yml` using [Compose profiles](https://docs.docker.com/compose/profiles/). `just up` starts everything.

---

## Quick start

### Prerequisites

- Docker + Docker Compose v2
- Python 3.10+ and [uv](https://github.com/astral-sh/uv) (for local development)
- RTSP camera

### Setup

```bash
cp .env.example .env
# Edit .env — at minimum set RTSP_URL and change passwords
just build
just up
just status
```

### First-time training

Before the weekly DAG can run, you need an initial labelled dataset:

1. **Let cat-live run** — it saves crops to `data/crops/unsorted/` automatically.
2. **Group + label** — cluster unsorted crops by time and rename to cat names:
   ```bash
   uv run scripts/pipeline.py group-crops  # cluster unsorted crops by time
   ```
   Open `data/crops/groups/` in Finder, rename group folders to cat names (e.g. `alisa`, `ellie`).
3. **Scatter + train**:
   ```bash
   uv run scripts/pipeline.py scatter-groups  # move named groups → per-cat folders
   uv run scripts/pipeline.py train
   ```

Run `just --list` to see all available commands.

---

## Commands

| Command | Description |
|---|---|
| `just up` | Start all Docker services |
| `just down` | Stop all Docker services |
| `just build` | Build Docker images |
| `just logs [service]` | Stream logs (all or one service) |
| `just ps` | Show running containers |
| `just status` | Running containers + UI URLs |
| `just retrain` | Run full retrain pipeline locally |
| `just predict` | Run classifier on an image or folder |
| `just stats` | DuckDB row counts + crops per cat |
| `just setup` | Install Python dependencies (`uv sync`) |

---

## How retrain works

`cat-live` continuously saves bbox crops to `data/crops/unsorted/` as cats are detected. The Airflow DAG `cat_retrain` runs weekly and chains these steps:

1. **Dedup crops** — remove near-duplicate crops via perceptual hash
2. **Auto-label** — run the classifier on unsorted crops; confident predictions (≥ 0.8) are moved to per-cat folders, the rest are deleted
3. **Train** — fine-tune EfficientNet-B0 on all labelled crops, save to `models/cat_classifier_best.pt`, log to MLflow, push metrics to VictoriaMetrics

The same pipeline is available locally as `just retrain [--auto-label-threshold 0.8]`.

---

## How live detection works

```
RTSP frames (camera rate)
  └─ capture thread ──────────────────────────────────────────── display FPS counter
  └─ inference loop (--inference-fps, default 5/s)
       └─ YOLO + ByteTrack  → per-track IDs
            └─ classifier gated on detection (per track, every --reclassify-every frames)
                 └─ per-track sliding window → door state machine → feeder API
```

1. **ByteTrack** (`model.track(persist=True)`) assigns a stable numeric `track_id` to each cat across frames. The classifier only runs when YOLO returns ≥1 detection with an assigned track.
2. **Per-track classification** — each track is classified on first sight, then re-classified every `--reclassify-every` inference frames (default 30) or when confidence drops below `--threshold`.
3. **Per-track sliding window** — each track keeps its own window of N confident predictions (`--window-size`, default 5). Window majority drives the door state machine.
4. **Door state machine** — evaluated every inference tick:
   - **Multi-cat** (>1 active track) → close door; counter `cat_detection_multi_cat_events_total` incremented, annotated frame saved to `data/multi_cat_snapshots/`.
   - **Blocked cat** (any active track classified as not in allowlist, e.g. `felisis`) → close door; counter `cat_detection_blocked_events_total{cat="..."}` incremented.
   - **Single allowed cat with window majority** → open door.
   - **No cats for ≥ `DOOR_CLOSE_TIMEOUT_SEC`** → close door.
   - Otherwise → no change (wait for more evidence).
5. **Idempotent** — the door is only called when the desired state differs from the last known state. Resync from `GET /api/status` every 60 seconds in the background.
6. **Model hot-reload** — every 60 minutes if `models/cat_classifier_best.pt` mtime changes, reloads without restarting.

---

## Feeder API integration

`live_detect.py` calls the feeder's REST API directly. Set `FEEDER_API_URL` to the feeder's base URL (e.g. `http://192.168.0.50:8000`). Leave it unset to run detection-only with no feeder control.

| Endpoint | When called |
|---|---|
| `POST /api/door/open` | Single allowed cat reaches sliding-window majority |
| `POST /api/door/close` | Multi-cat, blocked cat, or no cat for ≥ `DOOR_CLOSE_TIMEOUT_SEC` |
| `GET /api/status` | On startup (to sync initial state) and every 60 s (background resync) |

Calls use a 3 s timeout and retry once on connection error or 5xx. On failure the local state is not updated, so the next inference tick retries automatically.

---

## Configuration

Copy `.env.example` to `.env` and edit. All variables have defaults so `.env` is optional for local development.

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | `rtsp://camera:password@192.168.0.213:554/stream1` | Camera stream URL |
| `LIVE_THRESHOLD` | `0.6` | Minimum classifier confidence |
| `INFERENCE_FPS` | `5` | Inference loop rate (detector + tracker ticks/sec) |
| `RECLASSIFY_EVERY` | `30` | Re-run classifier on a track every N inference frames |
| `FEEDER_API_URL` | _(empty = disabled)_ | Base URL of the feeder REST API (e.g. `http://192.168.0.50:8000`) |
| `DOOR_CLOSE_TIMEOUT_SEC` | `30` | Seconds with no cat before closing the door |
| `FEEDER_ALLOWED_CATS` | _(empty = all except felisis)_ | Comma-separated cat allowlist for the feeder |
| `MULTI_CAT_SNAPSHOT_RETENTION_DAYS` | `30` | Delete multi-cat snapshot JPEGs older than N days |
| `CROP_SAVE_COOLDOWN` | `10` | Seconds between saved crops per track for known (confident) cats |
| `CROP_RETENTION_DAYS` | `60` | Delete live-detection crops older than N days |
| `WEB_PORT` | _(unset = off)_ | Set to a port number (e.g. `8082`) to enable the MJPEG web viewer |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Airflow web UI password |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana web UI password |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server (override for local scripts) |
| `VM_URL` | `http://victoriametrics:8428` | VictoriaMetrics push URL |
| `VM_RETENTION` | `365d` | VictoriaMetrics data retention period |

## Web viewer

When `WEB_PORT` is set to a positive integer (e.g. `WEB_PORT=8082`), `live_detect.py` starts an MJPEG server on that port:

| Endpoint | Description |
|---|---|
| `GET /` | HTML page with embedded live stream and stats |
| `GET /stream.mjpg` | MJPEG stream of annotated frames (no double inference) |
| `GET /health` | `{"ok": true}` |
| `GET /stats.json` | Current FPS, active tracks, last 10 detections, uptime |

The stream is decoupled via a `Queue(maxsize=1)` — frames are dropped rather than buffered when the viewer is slow.

**Optional WebRTC sidecar**: for sub-second latency add a [go2rtc](https://github.com/AlexxIT/go2rtc) container pointing at `http://cat-live:8082/stream.mjpg` as an RTSP/MJPEG source and expose the WebRTC endpoint. go2rtc handles the protocol conversion; `live_detect.py` stays unchanged.

---

## Project structure

```
scripts/
├── pipeline.py                   # CLI entry point — all pipeline commands
├── pipeline_db.py                # Shared DB schema, constants, utilities
├── deduplicate_frames.py         # Perceptual-hash dedup for frames and crops
├── auto_label.py                 # Classify unsorted crops → move to per-cat folders
├── group_crops.py                # Cluster unsorted crops into time-based groups
├── scatter_groups.py             # Move named group folders → per-cat label folders
├── train_classifier.py           # Fine-tune EfficientNet-B0 on labelled crops
├── predict_cat.py                # Run classifier on image(s)
├── live_detect.py                # 24/7 live detection with sliding window + crop saving
├── metrics.py                    # push_metric() → VictoriaMetrics
├── export_cat_crops.py           # Export CVAT-annotated crops for training
├── import_cvat_annotations.py    # Import CVAT COCO annotations into DuckDB
├── assign_labels_from_folders.py # Sync crop labels from folder layout to DuckDB
│
│   # Unused — these processed raw videos recorded by ffmpeg (removed):
├── scan_cat_detections.py        # (unused) YOLO scan of raw videos → detections table
├── build_cat_intervals.py        # (unused) Merge detections → time intervals
├── extract_interval_frames.py    # (unused) Extract JPEG frames from intervals
├── auto_crop_cats.py             # (unused) YOLO crop cats from frames → crops/unsorted/
└── build_videos_index.py         # (unused) Build videos_index.csv from raw_videos/
```
