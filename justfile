# cat_detection — justfile
# usage: just <command>
# list all commands: just --list

# defaults
default_batch_size := "64"
default_max_side := "512"

# ─── Pipeline ──────────────────────────────────────────────

# Run full pipeline: scan → intervals → frames → deduplicate
prepare *args:
    uv run scripts/pipeline.py prepare {{args}}

# Scan videos for cat detections (YOLO)
scan *args:
    uv run scripts/pipeline.py scan {{args}}

# Build merged intervals from detections
intervals:
    uv run scripts/pipeline.py intervals

# Extract JPEG frames from intervals
frames:
    uv run scripts/pipeline.py frames

# Deduplicate similar frames (requires imagehash)
deduplicate *args:
    uv run scripts/pipeline.py deduplicate {{args}}

# Deduplicate crops
dedup-crops *args:
    uv run scripts/pipeline.py deduplicate --mode crops {{args}}

# Group unsorted crops by time clusters
group-crops *args:
    uv run scripts/pipeline.py group-crops {{args}}

# Move named groups to cat folders
scatter-groups *args:
    uv run scripts/pipeline.py scatter-groups {{args}}

# ─── Annotation ────────────────────────────────────────────

# Import CVAT COCO annotations: just import path/to/instances_default.json
import path:
    uv run scripts/pipeline.py import-annotations --export-json {{path}}

# Export annotated crops for classifier training
export-crops *args:
    uv run scripts/pipeline.py export-crops {{args}}

# Auto-crop cats from extracted frames via YOLO
auto-crop *args:
    uv run scripts/pipeline.py auto-crop {{args}}

# Assign cat labels to crops based on folder layout
assign-labels *args:
    uv run scripts/pipeline.py assign-labels {{args}}

# ─── Classifier ────────────────────────────────────────────

# Train cat classifier
train *args:
    uv run scripts/pipeline.py train {{args}}

# Predict cat identity on image(s)
predict *args:
    uv run scripts/pipeline.py predict {{args}}

# Live cat detection from camera
live *args:
    uv run scripts/pipeline.py live {{args}}

# ─── CVAT ──────────────────────────────────────────────────

# Start CVAT (web UI at http://localhost:8080)
cvat-up:
    cd cvat && docker compose -f docker-compose.yml -f ../docker-compose.cvat.yml up -d

# Stop CVAT
cvat-down:
    cd cvat && docker compose -f docker-compose.yml -f ../docker-compose.cvat.yml down

# Show CVAT logs
cvat-logs:
    cd cvat && docker compose -f docker-compose.yml -f ../docker-compose.cvat.yml logs -f --tail=50

# Create CVAT superuser (run once after first cvat-up)
cvat-create-user:
    cd cvat && docker compose exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# ─── Data ──────────────────────────────────────────────────

# Show crop stats per cat
crop-stats:
    uv run python3 -c "from scripts.pipeline_db import print_crop_stats; print_crop_stats()"

# Build videos index CSV from raw_videos/
videos-index:
    uv run scripts/build_videos_index.py

# Show pipeline stats from DuckDB
stats:
    uv run python3 -c "\
    import duckdb; \
    db = duckdb.connect('data/metadata/cat_pipeline.duckdb', read_only=True); \
    tables = ['videos','detections','intervals','frames','annotations']; \
    [print(f'{t}: {db.execute(f\"SELECT count(*) FROM {t}\").fetchone()[0]}') for t in tables]; \
    "

# Open DuckDB shell
db:
    uv run -p duckdb python3 -c "import duckdb; db = duckdb.connect('data/metadata/cat_pipeline.duckdb'); print('connected. use db.sql(\"...\")'); import code; code.interact(local={'db': db})"

# ─── Housekeeping ──────────────────────────────────────────

# Install/sync project dependencies
setup:
    uv sync
