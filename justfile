# live2 operator commands. Run from the repo root.

set dotenv-load := true

COMPOSE := "docker compose -f docker-compose.yml -f docker-compose.cameras.yml"
DEV_COMPOSE := COMPOSE + " -f docker-compose.dev.yml -f docker-compose.cameras.dev.yml"
CLUSTER_SERVICE := env_var_or_default("CLUSTER_SERVICE", "detector-grey")
TRAINING_RUN := "uv run --project training"
CLASSIFIER_RUN := TRAINING_RUN + " --extra classifier"

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

# Reset ONLY the human-review state: MOVE (never delete) reviews.db + clusters.json
# into data/review/_backup_<ts>/ so a fresh review pass starts clean. WARNING: this
# discards the active review labels/clusters from their working paths — but events.db
# and recordings are NEVER touched, and nothing is rm'd (restore by moving files back).
# Stop the review app first. Set CONFIRM=1 to skip the prompt.
review-reset:
    #!/usr/bin/env bash
    set -euo pipefail
    review_db="${REVIEW_DB:-data/review/reviews.db}"
    manifest="${CLUSTER_MANIFEST:-data/review/clusters.json}"
    events_db="${EVENTS_DB:-data/events/events.db}"
    # Hard safety: never let a misconfigured REVIEW_DB point at the events DB.
    if [ "$review_db" = "$events_db" ]; then
        echo "review-reset: refusing — REVIEW_DB resolves to EVENTS_DB ($events_db)." >&2
        exit 1
    fi
    # Collect existing targets: reviews.db (+ its WAL/SHM sidecars) and the manifest.
    targets=()
    for f in "$review_db" "$review_db-wal" "$review_db-shm" "$manifest"; do
        [ -e "$f" ] && targets+=("$f")
    done
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "review-reset: nothing to move (no reviews.db / clusters.json found)."
        exit 0
    fi
    # Co-locate the backup with the review DB's dir (data/review by default), so a
    # custom REVIEW_DB still backs up next to itself instead of into the repo.
    backup="$(dirname "$review_db")/_backup_$(date +%Y%m%d-%H%M%S)"
    echo "review-reset will MOVE (not delete) into ${backup}/:"
    for f in "${targets[@]}"; do echo "  - $f"; done
    echo "NEVER touched: events.db ($events_db) and recordings."
    if [ "${CONFIRM:-0}" != "1" ]; then
        read -r -p "Proceed? [y/N] " ans
        case "$ans" in [yY]|[yY][eE][sS]) ;; *) echo "aborted."; exit 1 ;; esac
    fi
    mkdir -p "$backup"
    for f in "${targets[@]}"; do mv -v "$f" "$backup"/; done
    echo "review-reset: done -> ${backup}/"

# Show reviewed label counts and class balance without training.
label-stats *ARGS:
    uv run python -m training.label_stats \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        --labels "${REVIEW_LABELS:-alisa,chuzh,ellie,felisis}" \
        {{ARGS}}

# One-time dependency check/warmup for training commands. Uses uv.
training-setup:
    {{CLASSIFIER_RUN}} python -c "import av, cv2, numpy, torch, torchvision; print('training deps ok')"

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
    {{CLASSIFIER_RUN}} python -m training.train_classifier \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        {{ARGS}}

# Build/update compact replay memory from human-reviewed crops.
build-replay-set *ARGS:
    {{CLASSIFIER_RUN}} python -m training.build_replay_set \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        --out "${REPLAY_SET:-data/replay}" \
        {{ARGS}}

# Compare candidate classifiers on the same human-reviewed crops.
compare-classifiers *ARGS:
    {{CLASSIFIER_RUN}} python -m training.compare_classifiers \
        --db "${EVENTS_DB:-data/events/events.db}" \
        --recordings "${RECORDINGS_ROOT:-data/recordings}" \
        --reviews-db "${REVIEW_DB:-data/review/reviews.db}" \
        {{ARGS}}

# Fast local checks for changed Python code.
check:
    uv run python -m compileall feeder detector training pruner review
    uv run pytest
