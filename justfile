# cat_detection
# just --list for all commands

# === Server (Docker) ===

# Start all services
up:
    docker compose --profile airflow --profile live --profile monitoring up -d

# Stop all services
down:
    docker compose --profile airflow --profile live --profile monitoring down

# Logs (all or one: just logs cat-live)
logs *args:
    docker compose --profile airflow --profile live --profile monitoring logs -f --tail=50 {{args}}

# Running containers (short)
ps:
    @docker compose --profile airflow --profile live --profile monitoring ps

# Running services + UI links
status:
    @docker compose --profile airflow --profile live --profile monitoring ps
    @echo ""
    @echo "Airflow:  http://localhost:${PORT_AIRFLOW:-19081}"
    @echo "MLflow:   http://localhost:${PORT_MLFLOW:-19050}"
    @echo "Grafana:  http://localhost:${PORT_GRAFANA:-19300}"
    @echo "Web view: http://localhost:${WEB_PORT:-19082}  (requires WEB_PORT set)"

# Build Docker images
build:
    docker compose --profile airflow --profile live --profile monitoring build

# === Local ===

# Run live detection locally (no Docker)
live *args:
    uv run scripts/live_detect.py {{args}}

# Run full retrain pipeline locally
retrain *args:
    uv run scripts/pipeline.py retrain {{args}}

# Check model on image or folder
predict *args:
    uv run scripts/pipeline.py predict {{args}}

# DB row counts + crops per cat
stats:
    @uv run scripts/pipeline.py stats

# === Setup ===

# Install Python dependencies
setup:
    uv sync
