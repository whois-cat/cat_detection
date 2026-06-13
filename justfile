# live2 operator commands. Run from the repo root.

set dotenv-load := true

COMPOSE := "docker compose -f docker-compose.yml -f docker-compose.cameras.yml"
DEV_COMPOSE := COMPOSE + " -f docker-compose.dev.yml -f docker-compose.cameras.dev.yml"

default:
    @just --list

# Regenerate mediamtx, per-camera compose, nginx, and cameras.json from cameras.yaml.
configure:
    python3 tools/configure.py

# Start the production-shaped local stack.
up: configure
    {{COMPOSE}} up -d --build

# Start the development stack with Vite/watchfiles in the foreground.
dev: configure
    {{COMPOSE}} down --remove-orphans 2>/dev/null || true
    {{DEV_COMPOSE}} up --build

# Stop the stack.
down:
    {{COMPOSE}} down

# Show running services.
ps:
    {{COMPOSE}} ps

# Tail logs. Example: `just logs detector-grey`.
logs SERVICE="":
    {{COMPOSE}} logs -f --tail=200 {{SERVICE}}

# One-time host venv for review apps.
review-setup:
    python3 -m venv .venv-review
    .venv-review/bin/pip install -r review/requirements.txt

# Cold-start clustering manifest. SERVICE is a detector service, e.g. detector-grey.
# Override REVIEW_LABELS/RECORDING_TZ; pass --embedding efficientnet if weights are cached.
cluster-manifest SERVICE *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="${RECORDING_TZ:-UTC}" \
        -v "$PWD":/work -w /work {{SERVICE}} \
        python -m training.build_cluster_manifest \
            --db "${EVENTS_DB:-data/events/events.db}" \
            --recordings "${RECORDINGS_ROOT:-data/recordings}" \
            --out "${CLUSTER_MANIFEST:-data/review/clusters.json}" \
            --labels "${REVIEW_LABELS:-}" \
            {{ARGS}}

# Bulk-label clusters in the browser. Requires `just review-setup` once.
cluster-review PORT="8095":
    #!/usr/bin/env bash
    set -euo pipefail
    PY=.venv-review/bin/python
    [ -x "$PY" ] || PY=python3
    CLUSTER_MANIFEST="${CLUSTER_MANIFEST:-data/review/clusters.json}" \
    RECORDINGS_ROOT="${RECORDINGS_ROOT:-data/recordings}" \
    REVIEW_DB="${REVIEW_DB:-data/review/reviews.db}" \
    REVIEW_LABELS="${REVIEW_LABELS:-alisa,chuzh,ellie,felisis}" \
    RECORDING_TZ="${RECORDING_TZ:-UTC}" \
    "$PY" -m uvicorn review.cluster_app:app --host 0.0.0.0 --port {{PORT}}

# Train the identity classifier from reviewed labels. Args pass through.
train-classifier *ARGS:
    uv run python -m training.train_classifier \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        {{ARGS}}

# Build/update compact replay memory from human-reviewed crops.
build-replay-set *ARGS:
    uv run python -m training.build_replay_set \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        --out "${REPLAY_SET:-data/replay}" \
        {{ARGS}}

# Compare candidate classifiers on the same human-reviewed crops.
compare-classifiers *ARGS:
    uv run python -m training.compare_classifiers \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        {{ARGS}}

# Fast local checks for changed Python code.
check:
    uv run python -m compileall feeder detector training pruner review
    uv run pytest
