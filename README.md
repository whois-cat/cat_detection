# cat_detection

Self-hosted system that learns to tell your cats apart using a security camera. It watches a live RTSP feed, identifies which cat is in frame, and sends a webhook, designed to work with autofeeder that need to know who's eating.

Works with any number of cats. You label them once, and the system retrains itself weekly on new footage.

## How it works

The system has three layers that run independently:

```
┌─────────────────────────────────────────────────────────────────┐
│  RECORDING (always on)                                          │
│                                                                 │
│  RTSP camera ──► ffmpeg ──► data/raw_videos/*.mkv               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  WEEKLY RETRAIN (Airflow DAG)                                   │
│                                                                 │
│  Sample N new videos                                            │
│    └► YOLO scan (find cat in frame)                             │
│         └► Build time intervals                                 │
│              └► Extract frames                                  │
│                   └► Deduplicate (perceptual hash)              │
│                        └► Auto-crop (YOLO bbox ──► cat cutouts) │
│                             └► Auto-label (model classifies)    │
│                                  └► Train (EfficientNet-B0)     │
│                                       └► Log to MLflow          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ updated model
┌──────────────────────────────▼──────────────────────────────────┐
│  LIVE DETECTION (24/7)                                          │
│                                                                 │
│  RTSP stream ──► YOLO ──► Crop ──► Classifier ──► Sliding      │
│  (1 fps)        detect    bbox     which cat?     window        │
│                                                     │           │
│                                                     ▼           │
│                                              Webhook POST       │
│                                              → auto-feeder      │
└─────────────────────────────────────────────────────────────────┘
```

**Recording** — ffmpeg captures the camera and writes 2-hour MKV segments to disk continuously.

**Weekly retrain** — An Airflow DAG picks N unprocessed videos, runs them through the pipeline, and retrains the classifier. The model labels new data itself (high-confidence predictions get auto-sorted into training folders), so it improves without manual work after the initial setup.

**Live detection** — Reads the camera at 1 fps. YOLO finds the cat, the classifier identifies it, and a sliding window smooths the predictions before sending a webhook. This prevents the feeder from getting conflicting signals frame-to-frame.

## Stack

| Tool | Purpose |
|---|---|
| **YOLOv8** | Detects cats in frames — pretrained on COCO, no training needed |
| **EfficientNet-B0** | Classifies which cat — fine-tuned on your labeled crops |
| **DuckDB** | Tracks pipeline state: which videos are processed, detections, frames |
| **Airflow** | Orchestrates the weekly retrain pipeline |
| **MLflow** | Logs training runs: accuracy, confusion matrix, model artifacts |
| **VictoriaMetrics** | Stores time-series metrics (detections, confidence, model accuracy) |
| **Grafana** | Dashboards: activity per cat, model health, detection confidence |
| **CVAT** | Web-based annotation tool for initial manual labeling (optional) |
| **Docker Compose** | Runs everything via profiles |
| **uv** | Python dependency management |

## Getting started

### Prerequisites

- Docker + Docker Compose v2
- Python 3.10+ and [uv](https://github.com/astral-sh/uv) (for local development)
- An RTSP camera (tested with Tapo C200)
- [just](https://github.com/casey/just) command runner
- At least 15 minutes of video footage with your cats

### 1. Clone and configure

```bash
git clone https://github.com/whois-cat/cat_detection.git
cd cat_detection
cp .env.example .env
```

Edit `.env` — set `RTSP_URL` to your camera and change default passwords.

### 2. Initial training

The system needs to learn your cats once. There are two paths:

#### Option A: Quick method (no CVAT)

Best when your cats are visually distinct.

```bash
# Install dependencies
just setup

# Put your .mkv video files into data/raw_videos/, then:
just retrain --sample-videos 20

# YOLO finds cats, crops them -> data/crops/unsorted/
# Group similar crops by time proximity:
uv run scripts/pipeline.py group-crops
```

Open `data/crops/groups/` in your file manager. Each `group_NNN` folder contains crops from one time cluster — usually the same cat. Look at the images and rename the folder to your cat's name (e.g. `group_001` → `mars`, `group_002` → `luna`). Skip groups where you can't tell or there's no cat.

You can use any naming scheme. One folder per cat. Groups that contain the cat's name as a substring will be matched (e.g. `luna_1`, `luna_2` both map to `luna/`).

```bash
uv run scripts/pipeline.py scatter-groups
uv run scripts/pipeline.py train
```

Aim for ~200 crops per cat. The training output shows a confusion matrix — if two cats are being confused, add more diverse crops of those two (different angles, lighting, distances).

#### Option B: CVAT annotation

Best when cats look similar and you need precise bounding boxes.

```bash
just setup

# Process videos:
uv run scripts/pipeline.py prepare

# Start CVAT:
just cvat-up
just cvat-create-user
# Open http://localhost:8080
```

In CVAT:
1. Create a project with one label per cat
2. Create a task, upload frames from `data/frames/` (via Connected file share or drag-and-drop)
3. For each frame with a cat: draw a rectangle around the cat, select the label
4. Export as **COCO 1.0** format, unzip into `data/cvat_exports/`

```bash
uv run scripts/pipeline.py import-annotations --export-json data/cvat_exports/your_task/instances_default.json
uv run scripts/pipeline.py export-crops
uv run scripts/pipeline.py train
```

Not every frame needs labeling. 300–500 frames with good variety (different cats, poses, lighting) is a solid start.

### 3. Deploy

```bash
just build
just up
just status
```

This starts all services. You'll see:

| UI | URL |
|---|---|
| Airflow | http://localhost:8081 |
| MLflow | http://localhost:5050 |
| Grafana | http://localhost:3000 |

Live detection starts immediately. Check it:

```bash
just logs cat-live
```

## How the retrain cycle works

The Airflow DAG `cat_retrain` runs every Monday and does the following:

1. **Sample** — picks N unprocessed videos from `data/raw_videos/`, distributed evenly across dates
2. **Scan** — YOLO scans each video at 5-second intervals, records timestamps where a cat appears
3. **Intervals** — merges nearby detections into continuous time ranges (with padding)
4. **Frames** — extracts 1 JPEG per second within each interval
5. **Deduplicate** — removes near-identical frames using perceptual hashing
6. **Auto-crop** — runs YOLO again on frames, saves tight crops of each cat
7. **Deduplicate crops** — removes near-identical crops
8. **Auto-label** — runs the current classifier on unlabeled crops. Crops predicted with ≥ 80% confidence are moved to the corresponding cat's training folder. Low-confidence crops are discarded
9. **Train** — retrains the classifier on all labeled crops, logs metrics to MLflow, pushes results to VictoriaMetrics

The same pipeline runs locally with `just retrain`.

## How live detection works

The `cat-live` container reads the RTSP stream at 1 frame per second:

1. **YOLO** checks if there's a cat in the frame
2. **Classifier** identifies which cat from the cropped bounding box
3. **Sliding window** (default: 5 frames) accumulates predictions. When 4 out of 5 agree on the same cat with sufficient confidence, that's a detection
4. **Cooldown** (default: 30 seconds) prevents duplicate webhooks. If a different cat appears, cooldown resets
5. **Model hot-reload** — every 60 minutes, checks if the model file was updated on disk and reloads it

## Webhook

When a cat is detected, a JSON POST is sent to `WEBHOOK_URL` (if configured):

```json
{
  "timestamp": "2026-04-13T14:32:01.123Z",
  "cat_name": "mars",
  "confidence": 0.92,
  "model_version": "1744552800",
  "bbox": {"x": 412, "y": 180, "w": 134, "h": 98}
}
```

The `confidence` is the mean across the sliding window. `model_version` is the file's Unix mtime. If `WEBHOOK_URL` is empty, detections are only logged to stdout and VictoriaMetrics.

## Configuration

All variables are in `.env`. Everything has defaults — `.env` is optional for local development.

| Variable | Default | What it does |
|---|---|---|
| `RTSP_URL` | — | Camera RTSP stream address |
| `WEBHOOK_URL` | *(empty)* | Where to POST detections. Leave empty to just log |
| `LIVE_INTERVAL` | `1.0` | Seconds between frames in live detection |
| `LIVE_THRESHOLD` | `0.6` | Minimum confidence to consider a detection |
| `RETRAIN_SAMPLE_VIDEOS` | `10` | How many new videos to process per retrain cycle |
| `AUTO_LABEL_THRESHOLD` | `0.8` | Minimum confidence for auto-labeling crops |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Airflow UI login |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana UI login |
| `VM_RETENTION` | `365d` | How long VictoriaMetrics keeps metrics |

## Commands

```
just up              Start all services
just down            Stop all services
just build           Build Docker images
just logs [service]  Stream logs
just status          Running containers + UI links

just retrain         Run full pipeline locally
just predict         Test classifier on an image
just stats           Show DB counts and crops per cat
just setup           Install Python deps (uv sync)

just cvat-up         Start CVAT for manual annotation
just cvat-down       Stop CVAT
```

## Project structure

```
cat_detection/
├── scripts/
│   ├── pipeline.py                # CLI — all commands
│   ├── pipeline_db.py             # DuckDB schema, shared utilities
│   ├── scan_cat_detections.py     # YOLO scan videos → detections
│   ├── build_cat_intervals.py     # Detections → time intervals
│   ├── extract_interval_frames.py # Intervals → JPEG frames
│   ├── deduplicate_frames.py      # Perceptual hash dedup (frames + crops)
│   ├── auto_crop_cats.py          # YOLO → crop cat from frame
│   ├── auto_label.py              # Classify unsorted crops → sort into folders
│   ├── group_crops.py             # Cluster crops by time
│   ├── scatter_groups.py          # Named folders → per-cat training dirs
│   ├── train_classifier.py        # Fine-tune EfficientNet-B0
│   ├── predict_cat.py             # Run classifier on images
│   ├── live_detect.py             # 24/7 detection with sliding window
│   ├── metrics.py                 # Push metrics to VictoriaMetrics
│   ├── import_cvat_annotations.py # Import COCO JSON from CVAT
│   ├── export_cat_crops.py        # Crop from CVAT annotations
│   └── build_videos_index.py      # Video metadata CSV
├── dags/
│   └── retrain_dag.py             # Airflow weekly DAG
├── grafana/provisioning/          # Auto-provisioned datasource + dashboard
├── models/                        # Trained classifier
├── data/
│   ├── raw_videos/                # MKV from camera
│   ├── frames/                    # Extracted JPEGs
│   ├── crops/                     # Per-cat training images
│   └── metadata/                  # DuckDB, indexes
├── docker-compose.yml             # All services (profiles)
├── Dockerfile.pipeline            # Shared image
├── .env.example                   # Configuration template
└── justfile                       # Command runner
```
