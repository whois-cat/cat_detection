# live2 operator commands. Run from the repo root.

set dotenv-load := true

COMPOSE := "docker compose -f docker-compose.yml -f docker-compose.cameras.yml"
DEV_COMPOSE := COMPOSE + " -f docker-compose.dev.yml -f docker-compose.cameras.dev.yml"
CLUSTER_SERVICE := env_var_or_default("CLUSTER_SERVICE", "detector-grey")
TRAINING_RUN := "uv run --project training"
CLASSIFIER_RUN := TRAINING_RUN + " --extra classifier"

# Shared path/label defaults (override via the matching env var). These were
# duplicated inline across recipes; centralized here and substituted as {{...}}.
events_db   := env_var_or_default("EVENTS_DB",        "data/events/events.db")
recordings  := env_var_or_default("RECORDINGS_ROOT",  "data/recordings")
review_db   := env_var_or_default("REVIEW_DB",        "data/review/reviews.db")
manifest    := env_var_or_default("CLUSTER_MANIFEST", "data/review/clusters.json")
labels      := env_var_or_default("REVIEW_LABELS",    "alisa,chuzh,ellie,felisis")
rec_tz      := env_var_or_default("RECORDING_TZ",     "UTC")
journal_db  := env_var_or_default("FEED_JOURNAL_DB",  "data/feed_journal/journal.db")
replay_set  := env_var_or_default("REPLAY_SET",       "data/replay")

default:
    @just --list

# ───────────────────────────── stack ─────────────────────────────

# Regenerate mediamtx, per-camera compose, nginx, and cameras.json from cameras.yaml.
[group('stack')]
configure:
    python3 tools/configure.py

# Start the production-shaped local stack.
[group('stack')]
up: configure
    {{COMPOSE}} up -d --build

# Start the development stack with Vite/watchfiles in the foreground.
[group('stack')]
dev: configure
    {{COMPOSE}} down --remove-orphans 2>/dev/null || true
    {{DEV_COMPOSE}} up --build

# Stop the stack.
[group('stack')]
down:
    {{COMPOSE}} down

# Show running services.
[group('stack')]
ps:
    {{COMPOSE}} ps

# Tail logs. Example: `just logs detector-grey`.
[group('stack')]
logs SERVICE="":
    {{COMPOSE}} logs -f --tail=200 {{SERVICE}}

# ───────────────────────────── setup ─────────────────────────────

# One-time dependency check/warmup. TARGET: label | train | all (default).
[group('dev')]
setup TARGET="all":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{TARGET}}" in
      label)
        uv run --with-requirements review/requirements.txt \
            python -c "import av, fastapi, numpy, PIL, uvicorn; print('label deps ok')"
        ;;
      train)
        {{CLASSIFIER_RUN}} \
            python -c "import av, cv2, numpy, torch, torchvision; print('train deps ok')"
        ;;
      all)
        uv run --with-requirements review/requirements.txt \
            python -c "import av, fastapi, numpy, PIL, uvicorn; print('label deps ok')"
        {{CLASSIFIER_RUN}} \
            python -c "import av, cv2, numpy, torch, torchvision; print('train deps ok')"
        ;;
      *)
        echo "setup: unknown target '{{TARGET}}' — use: label | train | all" >&2
        exit 1
        ;;
    esac

# ──────────────────────────── labeling ───────────────────────────

# Cold-start clustering manifest. Uses one detector container as the Python runtime.
# Override CLUSTER_SERVICE only if detector-grey is not present.
# Override REVIEW_LABELS/RECORDING_TZ; pass --embedding efficientnet if weights are cached.
[group('label')]
label-build *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="{{rec_tz}}" \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.build_cluster_manifest \
            --db "{{events_db}}" \
            --recordings "{{recordings}}" \
            --out "{{manifest}}" \
            --labels "${REVIEW_LABELS:-}" \
            {{ARGS}}

# Bulk-label clusters in the browser.
[group('label')]
label-review PORT="8095":
    CLUSTER_MANIFEST="{{manifest}}" \
    RECORDINGS_ROOT="{{recordings}}" \
    REVIEW_DB="{{review_db}}" \
    REVIEW_LABELS="{{labels}}" \
    RECORDING_TZ="{{rec_tz}}" \
    uv run --with-requirements review/requirements.txt \
        python -m uvicorn review.cluster_app:app --host 0.0.0.0 --port {{PORT}}

# Show reviewed label counts and class balance without training.
[group('label')]
label-stats *ARGS:
    uv run python -m training.label_stats \
        --reviews-db "{{review_db}}" \
        --labels "{{labels}}" \
        {{ARGS}}

# Reset ONLY the human-review state: MOVE (never delete) reviews.db + clusters.json
# into data/review/_backup_<ts>/ so a fresh review pass starts clean. WARNING: this
# discards the active review labels/clusters from their working paths — but events.db
# and recordings are NEVER touched, and nothing is rm'd (restore by moving files back).
# Stop the review app first. Set CONFIRM=1 to skip the prompt.
[group('label')]
label-reset:
    #!/usr/bin/env bash
    set -euo pipefail
    review_db="{{review_db}}"
    manifest="{{manifest}}"
    events_db="{{events_db}}"
    # Hard safety: never let a misconfigured REVIEW_DB point at the events DB.
    if [ "$review_db" = "$events_db" ]; then
        echo "label-reset: refusing — REVIEW_DB resolves to EVENTS_DB ($events_db)." >&2
        exit 1
    fi
    # Collect existing targets: reviews.db (+ its WAL/SHM sidecars) and the manifest.
    targets=()
    for f in "$review_db" "$review_db-wal" "$review_db-shm" "$manifest"; do
        [ -e "$f" ] && targets+=("$f")
    done
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "label-reset: nothing to move (no reviews.db / clusters.json found)."
        exit 0
    fi
    # Co-locate the backup with the review DB's dir (data/review by default), so a
    # custom REVIEW_DB still backs up next to itself instead of into the repo.
    backup="$(dirname "$review_db")/_backup_$(date +%Y%m%d-%H%M%S)"
    echo "label-reset will MOVE (not delete) into ${backup}/:"
    for f in "${targets[@]}"; do echo "  - $f"; done
    echo "NEVER touched: events.db ($events_db) and recordings."
    if [ "${CONFIRM:-0}" != "1" ]; then
        read -r -p "Proceed? [y/N] " ans
        case "$ans" in [yY]|[yY][eE][sS]) ;; *) echo "aborted."; exit 1 ;; esac
    fi
    mkdir -p "$backup"
    for f in "${targets[@]}"; do mv -v "$f" "$backup"/; done
    echo "label-reset: done -> ${backup}/"

# ──────────────────────────── training ───────────────────────────

# Rebuild detector events from recordings with offline YOLO. Useful when live
# detector events are polluted by static false positives.
[group('train')]
train-rescan *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="{{rec_tz}}" \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.rescan_recordings \
            --db "{{events_db}}" \
            --recordings "{{recordings}}" \
            {{ARGS}}

# Train the identity classifier from reviewed labels. Args pass through.
[group('train')]
train-run *ARGS:
    {{CLASSIFIER_RUN}} python -m training.train_classifier \
        --db "{{events_db}}" \
        --recordings "{{recordings}}" \
        --reviews-db "{{review_db}}" \
        {{ARGS}}

# Build/update compact replay memory from human-reviewed crops.
[group('train')]
train-replay-set *ARGS:
    {{CLASSIFIER_RUN}} python -m training.build_replay_set \
        --db "{{events_db}}" \
        --recordings "{{recordings}}" \
        --reviews-db "{{review_db}}" \
        --out "{{replay_set}}" \
        {{ARGS}}

# Compare candidate classifiers on the same human-reviewed crops.
[group('train')]
train-compare *ARGS:
    {{CLASSIFIER_RUN}} python -m training.compare_classifiers \
        --db "{{events_db}}" \
        --recordings "{{recordings}}" \
        --reviews-db "{{review_db}}" \
        {{ARGS}}

# ──────────────────────────── journal ────────────────────────────

# Show how a cat has been eating (door-open sessions) over the last N days.
# Example: `just journal-feed alisa 7`
[group('journal')]
journal-feed CAT DAYS="3":
    python3 tools/feed_log.py {{CAT}} --days {{DAYS}} --db {{journal_db}}

# ─────────────────────────────── dev ─────────────────────────────

# Fast local checks for changed Python code.
[group('dev')]
check:
    uv run python -m compileall feeder detector training pruner review
    uv run pytest
