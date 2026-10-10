# Operator commands. Run from the repo root. See README.md / PLAN.md.

set dotenv-load := true

COMPOSE := "docker compose"
# Container used as the Python runtime for labeling/training helpers that need
# torch + openvino + av (the cv-worker image has them).
CLUSTER_SERVICE := env_var_or_default("CLUSTER_SERVICE", "cv-worker")
TRAINING_RUN := "uv run --project training"
CLASSIFIER_RUN := TRAINING_RUN + " --extra classifier"

# Shared path/label defaults (override via the matching env var).
yolo_recordings := env_var_or_default("YOLO_RECORDINGS", "data/streamhub/recordings")
yolo_dataset    := env_var_or_default("YOLO_DATASET",    "data/yolo_dataset")
catalog         := yolo_dataset + "/catalog.sqlite3"
config_yaml     := env_var_or_default("CONFIG_YAML",     "config.yaml")
review_db   := env_var_or_default("REVIEW_DB",        "data/review/reviews.db")
manifest    := env_var_or_default("CLUSTER_MANIFEST", "data/review/clusters.json")
# No hardcoded cat names: set REVIEW_LABELS=name1,name2,... for your cats.
# When empty, the review UI falls back to the labels baked into the manifest.
labels      := env_var_or_default("REVIEW_LABELS",    "")
rec_tz      := env_var_or_default("RECORDING_TZ",     "UTC")
# Classifier crop padding; must equal cv-worker's CLASSIFIER_PAD_FRAC (docker-compose.yml).
cat_pad_frac := env_var_or_default("CLASSIFIER_PAD_FRAC", "0.05")
journal_db  := env_var_or_default("FEED_JOURNAL_DB",  "data/decider/feed_journal/journal.db")
streamhub_port := env_var_or_default("STREAMHUB_PORT", "8096")
YOLO_RUN       := TRAINING_RUN
YOLO_TRAIN_RUN := TRAINING_RUN + " --extra yolo"
YOLO_LABEL_RUN := TRAINING_RUN + " --extra label"
LABEL_COMPOSE  := COMPOSE + " -f docker-compose.label.yml"
label_studio_port := env_var_or_default("LABEL_STUDIO_PORT", "8080")

default:
    @just --list --unsorted

# ───────────────────────────── stack ─────────────────────────────
# streamhub, cv-worker, decider, pruner (docker-compose.yml, config.yaml).

# Data dirs are created first so they belong to you, not root (containers run
# as UID/GID, default 1000).
# Build and start the stack.
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
# `clean` + the stack's history: recordings (with sidecars), dry-run journal.
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

# ─────────────────────── box: where is the cat ───────────────────────
# `cat` YOLO fine-tune from streamhub recordings, in pipeline order:
#   box-collect -> box-label (in Label Studio) -> box-sync -> box-build
#   -> box-train -> box-eval -> box-export
# All offline/manual; never touches the running stack or deployed models.

# Frames are saved as the detector sees them (ROI + rotation). Resumable;
# sidecar boxes are hints only. Example:
#   just box-collect --camera black,grey --from 2026-10-01 --to 2026-10-02 --tag shaved
# Collect frames from recordings into the catalog.
[group('box')]
box-collect *ARGS:
    {{YOLO_RUN}} python -m training.streamhub_dataset \
        --recordings "{{yolo_recordings}}" \
        --config "{{config_yaml}}" \
        --out "{{yolo_dataset}}" \
        {{ARGS}}

# Review-queue composition and storage usage (read-only).
[group('box')]
box-queue *ARGS:
    {{YOLO_RUN}} python -m training.yolo_review \
        --catalog "{{catalog}}" --root "{{yolo_dataset}}" \
        status {{ARGS}}

# Hardest frames first (model unsure, several cats, ...), with the model's boxes
# pre-filled; --limit N pushes a batch. First run: open the URL, log in, put
# Account & Settings -> Access Token into .env as LABEL_STUDIO_API_KEY, run again.
# Start Label Studio and push unreviewed frames into it.
[group('box')]
box-label *ARGS:
    {{LABEL_COMPOSE}} up -d
    {{YOLO_LABEL_RUN}} python -m training.yolo_label_studio \
        --catalog "{{catalog}}" push {{ARGS}}
    @echo "Label at http://localhost:{{label_studio_port}} (Label All Tasks) — when done: just box-sync"

# Pull submitted boxes from Label Studio back into the catalog.
[group('box')]
box-sync:
    {{YOLO_LABEL_RUN}} python -m training.yolo_label_studio \
        --catalog "{{catalog}}" pull

# Stop Label Studio (projects and annotations persist in a docker volume).
[group('box')]
box-label-stop:
    {{LABEL_COMPOSE}} down

# Build an immutable, visit-group-split dataset version from verified frames.
[group('box')]
box-build *ARGS:
    {{YOLO_RUN}} python -m training.yolo_build_version \
        --catalog "{{catalog}}" --root "{{yolo_dataset}}" {{ARGS}}

# Writes a report to models/trained/<run>/ and never deploys. Example:
#   just box-train --dataset data/yolo_dataset/versions/<id> --weights yolov8n.pt
# Fine-tune YOLO on a dataset version.
[group('box')]
box-train *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_train {{ARGS}}

# Evaluate a trained .pt or OpenVINO export on the held-out test split.
[group('box')]
box-eval *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_evaluate {{ARGS}}

# Runs in the cv-worker image so the IR is built by the same ultralytics/OpenVINO
# that will serve it. Writes models/trained/<run>/weights/best_int8_openvino_model;
# to deploy, set YOLO_WEIGHTS=/opt/models/trained/<run>/weights/best_int8_openvino_model
# in .env and `just up` (remove it to roll back).
# Export a trained .pt to OpenVINO with a parity gate against the source .pt.
[group('box')]
box-export *ARGS:
    {{COMPOSE}} run --rm --no-deps --user "$(id -u):$(id -g)" \
        -e YOLO_CONFIG_DIR=/tmp/ultralytics -e HOME=/tmp \
        -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python -m training.yolo_export {{ARGS}}

# Compare two run reports: same data/config? how long, what resources, quality delta.
[group('box')]
box-compare A B *ARGS:
    {{YOLO_RUN}} python -m training.compare_yolo_reports "{{A}}" "{{B}}" {{ARGS}}

# ─────────────────────── cat: which cat is it ───────────────────────
# Identity classifier, from the boxes verified with box-label/box-sync:
#   cat-groups -> cat-label (in the browser) -> cat-train -> cat-compare
#   -> cat-promote -> cat-restart
# Crops are cut from the catalog frames with the runtime padding (cat_pad_frac).

# One feeding visit per camera becomes one group (EPISODE_GAP_SEC, default 30);
# `--mode embedding` groups by look instead. Rebuild after each box-sync.
# Group verified cat crops for bulk labelling.
[group('cat')]
cat-groups *ARGS:
    {{CLASSIFIER_RUN}} python -m training.build_cluster_manifest \
        --catalog "{{catalog}}" \
        --out "{{manifest}}" \
        --labels "{{labels}}" \
        --pad-frac {{cat_pad_frac}} \
        --mode time \
        --episode-gap-sec "${EPISODE_GAP_SEC:-30}" \
        {{ARGS}}

# Name the cat in each group in the browser (bulk; split mixed groups).
[group('cat')]
cat-label PORT="8095":
    CLUSTER_MANIFEST="{{manifest}}" \
    REVIEW_DB="{{review_db}}" \
    REVIEW_LABELS="{{labels}}" \
    RECORDING_TZ="{{rec_tz}}" \
    uv run --with-requirements review/requirements.txt \
        python -m uvicorn review.cluster_app:app --host 0.0.0.0 --port {{PORT}}

# Labelled crop counts and class balance, without training.
[group('cat')]
cat-stats *ARGS:
    uv run python -m training.label_stats \
        --reviews-db "{{review_db}}" \
        --labels "{{labels}}" \
        {{ARGS}}

# Train the identity classifier from the labelled crops. Args pass through.
[group('cat')]
cat-train *ARGS:
    {{CLASSIFIER_RUN}} python -m training.train_classifier \
        --catalog "{{catalog}}" \
        --reviews-db "{{review_db}}" \
        --pad-frac {{cat_pad_frac}} \
        {{ARGS}}

# Example:
#   just cat-compare --candidate current=models/classifier/current \
#                    --candidate new=models/trained/<run>/cat_classifier.pt --baseline current
# Compare candidate classifiers on the same labelled crops.
[group('cat')]
cat-compare *ARGS:
    {{CLASSIFIER_RUN}} python -m training.compare_classifiers \
        --catalog "{{catalog}}" \
        --reviews-db "{{review_db}}" \
        --pad-frac {{cat_pad_frac}} \
        {{ARGS}}

# Writes models/classifier/versions/<id> and switches the `current` symlink.
# Default (no SRC): the newest models/trained/*/cat_classifier.pt. Runs in the
# cv-worker container (torch + openvino) to export the OpenVINO IR.
# Promote a trained checkpoint to the runtime model (then: just cat-restart).
[group('cat')]
cat-promote SRC="":
    {{COMPOSE}} run --rm --no-deps -v "$PWD":/work -w /work {{CLUSTER_SERVICE}} \
        python tools/promote_classifier.py promote --src "{{SRC}}"

# No export needed, so this runs on the host.
# Roll back to the previous model or VERSION=<id> (then: just cat-restart).
[group('cat')]
cat-rollback VERSION="":
    python3 tools/promote_classifier.py rollback --version "{{VERSION}}"

# Restart cv-worker so it picks up a freshly promoted `current`. No image rebuild.
[group('cat')]
cat-restart:
    {{COMPOSE}} restart cv-worker

# MOVES (never deletes) reviews.db + clusters.json into data/review/_backup_<ts>/
# (restore by moving them back). The catalog is never touched. Stop cat-label
# first. Set CONFIRM=1 to skip the prompt.
# Start cat labelling from scratch.
[group('cat')]
cat-reset:
    #!/usr/bin/env bash
    set -euo pipefail
    review_db="{{review_db}}"
    manifest="{{manifest}}"
    targets=()
    for f in "$review_db" "$review_db-wal" "$review_db-shm" "$manifest"; do
        [ -e "$f" ] && targets+=("$f")
    done
    if [ "${#targets[@]}" -eq 0 ]; then
        echo "cat-reset: nothing to move (no reviews.db / clusters.json found)."
        exit 0
    fi
    # Back up next to the review DB, so a custom REVIEW_DB stays out of the repo.
    backup="$(dirname "$review_db")/_backup_$(date +%Y%m%d-%H%M%S)"
    echo "cat-reset will MOVE (not delete) into ${backup}/:"
    for f in "${targets[@]}"; do echo "  - $f"; done
    if [ "${CONFIRM:-0}" != "1" ]; then
        read -r -p "Proceed? [y/N] " ans
        case "$ans" in [yY]|[yY][eE][sS]) ;; *) echo "aborted."; exit 1 ;; esac
    fi
    mkdir -p "$backup"
    for f in "${targets[@]}"; do mv -v "$f" "$backup"/; done
    echo "cat-reset: done -> ${backup}/"

# Alternatively `just up` runs the `mlflow` container UI on $MLFLOW_PORT (5000).
# Browse MLflow experiment runs from the local file store (./data/mlflow).
[group('dev')]
mlflow-ui PORT="5000":
    {{CLASSIFIER_RUN}} python -m mlflow ui \
        --backend-store-uri "data/mlflow" --port {{PORT}}

# ──────────────────────────── journal ────────────────────────────

# Example: `just journal-feed <cat-name> 7`
# Show how a cat has been eating (door-open sessions) over the last N days.
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