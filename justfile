# live2 operator commands. Run from the repo root.

set dotenv-load := true

COMPOSE := "docker compose -f docker-compose.yml -f docker-compose.cameras.yml"
DEV_COMPOSE := COMPOSE + " -f docker-compose.dev.yml -f docker-compose.cameras.dev.yml"
CLUSTER_SERVICE := env_var_or_default("CLUSTER_SERVICE", "detector-grey")

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

# One-time dependency check/warmup for review apps. Uses uv; no python3-venv needed.
review-setup:
    uv run --with-requirements review/requirements.txt python -c "import av, fastapi, numpy, PIL, uvicorn; print('review deps ok')"

# Cold-start clustering manifest. Uses one detector container as the Python runtime.
# Override CLUSTER_SERVICE only if detector-grey is not present.
# Override REVIEW_LABELS/RECORDING_TZ; pass --embedding efficientnet if weights are cached.
cluster-manifest *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="${RECORDING_TZ:-UTC}" \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.build_cluster_manifest \
            --db "${EVENTS_DB:-data/events/events.db}" \
            --recordings "${RECORDINGS_ROOT:-data/recordings}" \
            --out "${CLUSTER_MANIFEST:-data/review/clusters.json}" \
            --labels "${REVIEW_LABELS:-}" \
            {{ARGS}}

# Bulk-label clusters in the browser.
cluster-review PORT="8095":
    CLUSTER_MANIFEST="${CLUSTER_MANIFEST:-data/review/clusters.json}" \
    RECORDINGS_ROOT="${RECORDINGS_ROOT:-data/recordings}" \
    REVIEW_DB="${REVIEW_DB:-data/review/reviews.db}" \
    REVIEW_LABELS="${REVIEW_LABELS:-alisa,chuzh,ellie,felisis}" \
    RECORDING_TZ="${RECORDING_TZ:-UTC}" \
    uv run --with-requirements review/requirements.txt \
        python -m uvicorn review.cluster_app:app --host 0.0.0.0 --port {{PORT}}

# Show reviewed label counts and class balance without training.
label-stats *ARGS:
    uv run python -m training.label_stats \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        --labels "${REVIEW_LABELS:-alisa,chuzh,ellie,felisis}" \
        {{ARGS}}

# Rebuild detector events from recordings with offline YOLO. Useful when live
# detector events are polluted by static false positives.
rescan-recordings *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="${RECORDING_TZ:-UTC}" \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.rescan_recordings \
            --db "${EVENTS_DB:-data/events/events.db}" \
            --recordings "${RECORDINGS_ROOT:-data/recordings}" \
            {{ARGS}}

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
