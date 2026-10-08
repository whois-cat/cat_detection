# Operator commands. Run from the repo root. See README.md / PLAN.md.

set dotenv-load := true

COMPOSE := "docker compose"
# Container used as the Python runtime for labeling/training helpers that need
# torch + openvino + av (the cv-worker image has them).
CLUSTER_SERVICE := env_var_or_default("CLUSTER_SERVICE", "cv-worker")
TRAINING_RUN := "uv run --project training"
CLASSIFIER_RUN := TRAINING_RUN + " --extra classifier"

# Shared path/label defaults (override via the matching env var).
# events_db/recordings are the previous stack's data, still used for training.
events_db   := env_var_or_default("EVENTS_DB",        "data/events/events.db")
recordings  := env_var_or_default("RECORDINGS_ROOT",  "data/recordings")
review_db   := env_var_or_default("REVIEW_DB",        "data/review/reviews.db")
manifest    := env_var_or_default("CLUSTER_MANIFEST", "data/review/clusters.json")
# No hardcoded cat names: set REVIEW_LABELS=name1,name2,... for your cats.
# When empty, the review UI falls back to the labels baked into the manifest.
labels      := env_var_or_default("REVIEW_LABELS",    "")
rec_tz      := env_var_or_default("RECORDING_TZ",     "UTC")
journal_db  := env_var_or_default("FEED_JOURNAL_DB",  "data/decider/feed_journal/journal.db")
replay_set  := env_var_or_default("REPLAY_SET",       "data/replay")
streamhub_port := env_var_or_default("STREAMHUB_PORT", "8096")
# New-stack YOLO fine-tune pipeline (independent of the old events.db path).
yolo_recordings := env_var_or_default("YOLO_RECORDINGS", "data/streamhub/recordings")
yolo_dataset    := env_var_or_default("YOLO_DATASET",    "data/yolo_dataset")
config_yaml     := env_var_or_default("CONFIG_YAML",     "config.yaml")
YOLO_RUN       := TRAINING_RUN
YOLO_TRAIN_RUN := TRAINING_RUN + " --extra yolo"

default:
    @just --list

# ───────────────────────────── stack ─────────────────────────────
# streamhub, cv-worker, decider, pruner (docker-compose.yml, config.yaml).

# Build and start the stack. Data dirs are created first so they belong to
# you, not root (containers run as UID/GID, default 1000).
[group('stack')]
up:
    mkdir -p data/streamhub data/decider
    {{COMPOSE}} up -d --build

# Stop the stack.
[group('stack')]
down:
    {{COMPOSE}} down

# Show running services.
[group('stack')]
ps:
    {{COMPOSE}} ps

# Tail logs. Example: `just logs cv-worker`.
[group('stack')]
logs SERVICE="":
    {{COMPOSE}} logs -f --tail=200 {{SERVICE}}

# Per-camera ingest status (connection, fps, clock correction).
[group('stack')]
status:
    @curl -s http://127.0.0.1:{{streamhub_port}}/api/status | python3 -m json.tool

# Config, secrets, data and models stay. Lists only unless ARGS=--yes.
# Remove ignored junk: venvs, node_modules, caches, build output, old-stack leftovers.
[group('stack')]
clean *ARGS:
    python3 tools/clean.py junk {{ARGS}}

# State (feed journal, pins), config, models and the previous stack's training
# data stay. Lists only unless ARGS=--yes.
# `clean` + the new stack's history: recordings (with sidecars), dry-run journal.
[group('stack')]
clean-history *ARGS:
    python3 tools/clean.py history {{ARGS}}

# Webui dev server (hot reload) against a running streamhub.
[group('stack')]
webui-dev STREAMHUB=("http://127.0.0.1:" + streamhub_port):
    cd webui && npm install --no-audit --no-fund && STREAMHUB={{STREAMHUB}} npm run dev

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

# Cold-start clustering manifest. Uses the cv-worker container as the Python runtime.
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

# Time/episode review manifest: one feeding visit per camera becomes a review
# group. Override EPISODE_GAP_SEC or pass extra args after the recipe name.
[group('label')]
label-build-time *ARGS:
    {{COMPOSE}} run --rm --no-deps \
        -e RECORDING_TZ="{{rec_tz}}" \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.build_cluster_manifest \
            --db "{{events_db}}" \
            --recordings "{{recordings}}" \
            --out "{{manifest}}" \
            --labels "${REVIEW_LABELS:-}" \
            --mode time \
            --episode-gap-sec "${EPISODE_GAP_SEC:-30}" \
            {{ARGS}}

# Validate a review cluster manifest: hard cap, valid indices, no hidden fields.
[group('label')]
label-validate MAX="16":
    python3 -m training.validate_cluster_manifest \
        --manifest "{{manifest}}" --max-cluster-size {{MAX}}

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
        --events-db "{{events_db}}" \
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

# Rebuild the previous stack's detector events from its recordings with
# offline YOLO. Useful when those events are polluted by static false positives.
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

# Browse MLflow experiment runs from the local file store (./data/mlflow).
# Alternatively `just up` runs the `mlflow` container UI on $MLFLOW_PORT (5000).
[group('train')]
mlflow-ui PORT="5000":
    {{CLASSIFIER_RUN}} python -m mlflow ui \
        --backend-store-uri "data/mlflow" --port {{PORT}}

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

# Promote a trained checkpoint to the active runtime model volume
# (models/classifier/versions/<id> + switch the `current` symlink). Default (no
# SRC) selects the newest models/trained/*/cat_classifier.pt. Runs inside the
# cv-worker container (torch + openvino) to export the OpenVINO IR; writes to the
# host repo via the bind mount. Restart after: `just classifier-restart`.
[group('train')]
classifier-promote SRC="":
    {{COMPOSE}} run --rm --no-deps -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python tools/promote_classifier.py promote --src "{{SRC}}"

# Roll back the `current` symlink to the previous version (default) or VERSION=<id>.
# No export needed, so this runs on the host. Restart after: `just classifier-restart`.
[group('train')]
classifier-rollback VERSION="":
    python3 tools/promote_classifier.py rollback --version "{{VERSION}}"

# Restart cv-worker so it picks up a freshly promoted `current`. No image rebuild.
[group('train')]
classifier-restart:
    {{COMPOSE}} restart cv-worker

# ───────────────────────── yolo fine-tune ────────────────────────
# Prepare data for a single-class `cat` YOLO fine-tune from streamhub recordings.
# All commands are offline/manual and never touch the running stack or models.

# Collect full, un-annotated frames (saved as the detector sees them: ROI +
# rotation) into the review catalog. Resumable; sidecar boxes are hints only.
# Example: just yolo-collect --camera black,grey --from 2026-10-01 --to 2026-10-02 --tag shaved
[group('yolo')]
yolo-collect *ARGS:
    {{YOLO_RUN}} python -m training.streamhub_dataset \
        --recordings "{{yolo_recordings}}" \
        --config "{{config_yaml}}" \
        --out "{{yolo_dataset}}" \
        {{ARGS}}

# Review-queue composition and storage usage (read-only).
[group('yolo')]
yolo-queue *ARGS:
    {{YOLO_RUN}} python -m training.yolo_review \
        --catalog "{{yolo_dataset}}/catalog.sqlite3" --root "{{yolo_dataset}}" \
        status {{ARGS}}

# Export an unreviewed batch as a CVAT/COCO zip (model-suggested boxes included,
# marked as suggestions, never as truth). Example: just yolo-review-export out/batch.zip 200
[group('yolo')]
yolo-review-export OUT LIMIT="100" *ARGS:
    {{YOLO_RUN}} python -m training.yolo_review \
        --catalog "{{yolo_dataset}}/catalog.sqlite3" --root "{{yolo_dataset}}" \
        export --out "{{OUT}}" --limit {{LIMIT}} {{ARGS}}

# Import a verified CVAT COCO export back into the catalog (validates IDs, dims,
# boxes; empty frames need --confirm-empty). Example: just yolo-review-import cvat.zip
[group('yolo')]
yolo-review-import PACKAGE *ARGS:
    {{YOLO_RUN}} python -m training.yolo_review \
        --catalog "{{yolo_dataset}}/catalog.sqlite3" --root "{{yolo_dataset}}" \
        import "{{PACKAGE}}" {{ARGS}}

# Build an immutable, visit-group-split dataset version from reviewed samples.
[group('yolo')]
yolo-build-version *ARGS:
    {{YOLO_RUN}} python -m training.yolo_build_version \
        --catalog "{{yolo_dataset}}/catalog.sqlite3" --root "{{yolo_dataset}}" {{ARGS}}

# Fine-tune YOLO from a pretrained .pt on a dataset version. Writes a report to
# models/trained/<run>/ and never deploys. Example:
#   just yolo-train --dataset data/yolo_dataset/versions/<id> --weights yolov8n.pt
[group('yolo')]
yolo-train *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_train {{ARGS}}

# Evaluate a trained .pt or OpenVINO export on the held-out test split.
[group('yolo')]
yolo-evaluate *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_evaluate {{ARGS}}

# Export a trained .pt to OpenVINO with a parity gate against the source .pt.
[group('yolo')]
yolo-export *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_export {{ARGS}}

# Compare two run reports: same data/config? how long, what resources, quality delta.
[group('yolo')]
yolo-compare A B *ARGS:
    {{YOLO_RUN}} python -m training.compare_yolo_reports "{{A}}" "{{B}}" {{ARGS}}

# ──────────────────────────── journal ────────────────────────────

# Show how a cat has been eating (door-open sessions) over the last N days.
# Example: `just journal-feed <cat-name> 7`
[group('journal')]
journal-feed CAT DAYS="3":
    python3 tools/feed_log.py {{CAT}} --days {{DAYS}} --db {{journal_db}}

# Run all tests (new stack, then training/review).
[group('dev')]
check:
    cd streamhub && go vet ./... && go test ./...
    cd webui && npm test
    cd hubclient && uv run --quiet --python 3.12 --group dev pytest -q
    cd cv-worker && uv run --quiet --python 3.12 --group dev pytest -q
    cd decider && uv run --quiet --python 3.12 --group dev pytest -q
    uv run --extra test python -m compileall -q training review tools
    uv run --extra test pytest -q tests