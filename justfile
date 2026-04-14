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
    @echo "Airflow:  http://localhost:8081"
    @echo "MLflow:   http://localhost:5000"
    @echo "Grafana:  http://localhost:3000"

# Build Docker images
build:
    docker compose --profile airflow --profile live --profile monitoring build

# === Local ===

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
