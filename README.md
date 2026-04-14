# cat_detection

## What this is

A self-hosted system for detecting and identifying four cats — Alisa, Chuzh, Ellie, and Felisis — from an RTSP security camera. YOLO detects cat presence in the frame; a fine-tuned EfficientNet-B0 classifier identifies which cat it is. The system runs continuously, feeds detections to a webhook (e.g. an auto-feeder controller), retrains weekly from new footage, and stores all metrics in Grafana.

---

## Architecture

| Component | Role | Profile |
|---|---|---|
| **ffmpeg** | Records RTSP stream → segmented `.mkv` files in `data/raw_videos/` | _(always on)_ |
| **cat-live** | 24/7 live detection: YOLO → classifier → sliding window → webhook | `live` |
| **Airflow** | Weekly retrain DAG: sample videos → scan → crop → auto-label → train | `airflow` |
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

1. **Record footage** — ffmpeg is already writing to `data/raw_videos/`
2. **Scan + crop** — detect cats and extract crops:
   ```bash
   uv run scripts/pipeline.py prepare    # scan → intervals → frames → dedup
   uv run scripts/pipeline.py auto-crop  # YOLO crop from frames
   uv run scripts/pipeline.py group-crops  # cluster unsorted crops by time
   ```
3. **Label** — open `data/crops/groups/` in Finder, rename group folders to cat names (e.g. `alisa`, `ellie`)
4. **Scatter + train**:
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

The Airflow DAG `cat_retrain` runs weekly and chains these steps:

1. **Sample** — query DuckDB for videos with no detections yet, pick N uniformly by date
2. **Scan** — run YOLO on sampled videos, write detections to DuckDB
3. **Intervals** — merge nearby detections into time intervals
4. **Frames** — extract JPEG frames from intervals
5. **Dedup frames** — remove near-duplicate frames via perceptual hash
6. **Auto-crop** — run YOLO on frames, save bboxes as crops to `data/crops/unsorted/`
7. **Dedup crops** — remove near-duplicate crops
8. **Auto-label** — run the classifier on unsorted crops; confident predictions (≥ 0.8) are moved to per-cat folders, the rest are deleted
9. **Train** — fine-tune EfficientNet-B0 on all labelled crops, save to `models/cat_classifier_best.pt`, log to MLflow, push metrics to VictoriaMetrics

The same pipeline is available locally as `just retrain [--sample-videos N] [--auto-label-threshold 0.8]`.

---

## How live detection works

```
RTSP frame → YOLO (cat present?) → EfficientNet-B0 (which cat?) → sliding window → webhook
```

1. **YOLO** detects cat bounding boxes in the frame (class `cat`, COCO id 15)
2. **Classifier** runs on each crop; if confidence ≥ `--threshold`, the prediction is logged and metrics are pushed to VictoriaMetrics
3. **Sliding window** keeps the last N predictions (`--window-size`, default 5). When ≥ majority (`--window-majority`, default 4) predictions agree on the same cat and their mean confidence ≥ threshold, a webhook fires
4. **Cooldown** — after firing, the same cat won't trigger another webhook for `--cooldown` seconds (default 30). If a different cat reaches majority, the cooldown resets and a new webhook fires immediately
5. **Model hot-reload** — every 60 minutes the process checks if `models/cat_classifier_best.pt` was updated on disk; if so, it reloads without restarting

---

## Webhook format

```json
{
  "timestamp": "2025-04-13T14:32:01.123456+00:00",
  "cat_name": "alisa",
  "confidence": 0.9241,
  "model_version": "1744552800",
  "bbox": {"x": 412, "y": 180, "w": 134, "h": 98}
}
```

`confidence` is the mean confidence of the majority cat across the window. `model_version` is the Unix mtime of the model file. `bbox` coordinates are in pixels relative to the original frame.

---

## Configuration

Copy `.env.example` to `.env` and edit. All variables have defaults so `.env` is optional for local development.

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | `rtsp://camera:password@192.168.0.213:554/stream1` | Camera stream URL |
| `LIVE_INTERVAL` | `1.0` | Seconds between detection frames |
| `LIVE_THRESHOLD` | `0.6` | Minimum classifier confidence |
| `WEBHOOK_URL` | _(empty)_ | POST target for detections |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Airflow web UI password |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana web UI password |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server (override for local scripts) |
| `VM_URL` | `http://victoriametrics:8428` | VictoriaMetrics push URL |
| `PUID` / `PGID` | `1000` | ffmpeg container filesystem permissions |
| `VM_RETENTION` | `365d` | VictoriaMetrics data retention period |

---

## Project structure

```
scripts/
├── pipeline.py                   # CLI entry point — all pipeline commands
├── pipeline_db.py                # Shared DB schema, constants, utilities
├── scan_cat_detections.py        # YOLO scan of raw videos → detections table
├── build_cat_intervals.py        # Merge detections → time intervals
├── extract_interval_frames.py    # Extract JPEG frames from intervals
├── deduplicate_frames.py         # Perceptual-hash dedup for frames and crops
├── auto_crop_cats.py             # YOLO crop cats from frames → crops/unsorted/
├── auto_label.py                 # Classify unsorted crops → move to per-cat folders
├── group_crops.py                # Cluster unsorted crops into time-based groups
├── scatter_groups.py             # Move named group folders → per-cat label folders
├── train_classifier.py           # Fine-tune EfficientNet-B0 on labelled crops
├── predict_cat.py                # Run classifier on image(s)
├── live_detect.py                # 24/7 live detection with sliding window
├── metrics.py                    # push_metric() → VictoriaMetrics
├── export_cat_crops.py           # Export CVAT-annotated crops for training
├── import_cvat_annotations.py    # Import CVAT COCO annotations into DuckDB
├── assign_labels_from_folders.py # Sync crop labels from folder layout to DuckDB
└── build_videos_index.py         # Build videos_index.csv from raw_videos/
```
