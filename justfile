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
# No hardcoded cat names: set CAT_NAMES=name1,name2,... for your cats.
cat_names   := env_var_or_default("CAT_NAMES",        "")
# Classifier crop padding; must equal cv-worker's CLASSIFIER_PAD_FRAC (docker-compose.yml).
cat_pad_frac := env_var_or_default("CLASSIFIER_PAD_FRAC", "0.05")
journal_db  := env_var_or_default("FEED_JOURNAL_DB",  "data/decider/feed_journal/journal.db")
streamhub_port := env_var_or_default("STREAMHUB_PORT", "8096")
YOLO_RUN       := TRAINING_RUN
YOLO_TRAIN_RUN := TRAINING_RUN + " --extra yolo"
YOLO_LABEL_RUN := TRAINING_RUN + " --extra label"
LABEL_COMPOSE  := COMPOSE + " -f docker-compose.label.yml"
# One-off command in the cv-worker image (torch + openvino + ultralytics, the
# serving versions), as you, on the repo mounted at /work.
CV_WORKER_RUN  := COMPOSE + " run --rm --no-deps --user \"$(id -u):$(id -g)\" -e HOME=/tmp -e YOLO_CONFIG_DIR=/tmp/ultralytics -e YOLO_DATASET=" + yolo_dataset + " -v \"$PWD\":/work -w /work " + CLUSTER_SERVICE
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
    mkdir -p data/streamhub data/decider models/detector models/classifier
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
      label|train|all) ;;
      *) echo "setup: unknown target '{{TARGET}}' — use: label | train | all" >&2; exit 1 ;;
    esac
    if [ "{{TARGET}}" != train ]; then
        {{YOLO_LABEL_RUN}} python -c "import label_studio_sdk; print('label deps ok')"
    fi
    if [ "{{TARGET}}" != label ]; then
        {{CLASSIFIER_RUN}} \
            python -c "import av, cv2, numpy, torch, torchvision; print('train deps ok')"
    fi

# ─────────────── detector: where the cats are (and who) ───────────────
# One Label Studio pass labels both models: a box around every cat, and the
# cat's name on each box (CAT_NAMES). Boxes train the detector, named boxes the
# classifier. In pipeline order:
#   detector-collect -> detector-label (in Label Studio) -> detector-sync
#   -> detector-build -> detector-train -> detector-eval, then
#   `just deploy detector <run>`; classifier-train for the names.
# `detector-round` does collect + label in one go (e.g. weekly from cron).
# All offline/manual; never touches the running stack or deployed models.

# Frames are saved as the detector sees them (ROI + rotation). Resumable; the
# live models' boxes and names (sidecars) are only suggestions. Examples:
#   just detector-collect --days 7
#   just detector-collect --camera black,grey --from 2026-10-09T00:00-04:00 --to 2026-10-11T00:00-04:00
# Collect frames from recordings into the catalog (default: all cameras).
[group('detector')]
detector-collect *ARGS:
    {{YOLO_RUN}} python -m training.streamhub_dataset \
        --recordings "{{yolo_recordings}}" \
        --config "{{config_yaml}}" \
        --out "{{yolo_dataset}}" \
        {{ARGS}}

# Review-queue composition and storage usage (read-only).
[group('detector')]
detector-queue *ARGS:
    {{YOLO_RUN}} python -m training.yolo_review \
        --catalog "{{catalog}}" --root "{{yolo_dataset}}" \
        status {{ARGS}}

# Hardest frames first (models unsure of a box or a name, several cats, ...),
# with the models' boxes and names pre-filled. ARGS: --limit N (a batch),
# --reopen-unnamed (frames reviewed before names existed, to name their cats),
# --notify (Telegram). First run: open the URL, log in, put Account & Settings
# -> Access Token into .env as LABEL_STUDIO_API_KEY, run again.
# Start Label Studio and push frames to label into it.
[group('detector')]
detector-label *ARGS:
    {{LABEL_COMPOSE}} up -d
    {{YOLO_LABEL_RUN}} python -m training.yolo_label_studio \
        --catalog "{{catalog}}" push --names "{{cat_names}}" {{ARGS}}
    @echo "Label at http://localhost:{{label_studio_port}} (Label All Tasks) — when done: just detector-sync"

# Pull submitted boxes and cat names from Label Studio back into the catalog.
[group('detector')]
detector-sync:
    {{YOLO_LABEL_RUN}} python -m training.yolo_label_studio \
        --catalog "{{catalog}}" pull

# Collect the last DAYS days from all cameras, queue them in Label Studio and
# say so in Telegram (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID). For cron, e.g.
#   0 9 * * 1  cd /path/to/cat_detection && just detector-round 7
# Collect + label in one go.
[group('detector')]
detector-round DAYS="7":
    just detector-collect --days {{DAYS}}
    just detector-label --notify

# Stop Label Studio (projects and annotations persist in a docker volume).
[group('detector')]
detector-label-stop:
    {{LABEL_COMPOSE}} down

# Build an immutable, visit-group-split dataset version from verified frames.
[group('detector')]
detector-build *ARGS:
    {{YOLO_RUN}} python -m training.yolo_build_version \
        --catalog "{{catalog}}" --root "{{yolo_dataset}}" {{ARGS}}

# Writes a report to models/trained/<run>/ and never deploys. Example:
#   just detector-train --dataset data/yolo_dataset/versions/<id> --weights yolov8n.pt
# Fine-tune YOLO on a dataset version.
[group('detector')]
detector-train *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_train {{ARGS}}

# Evaluate a trained .pt or OpenVINO export on the held-out test split.
[group('detector')]
detector-eval *ARGS:
    {{YOLO_TRAIN_RUN}} python -m training.yolo_evaluate {{ARGS}}

# Compare two run reports: same data/config? how long, what resources, quality delta.
[group('detector')]
detector-compare A B *ARGS:
    {{YOLO_RUN}} python -m training.compare_yolo_reports "{{A}}" "{{B}}" {{ARGS}}

# ─────────────────── classifier: which cat it is ───────────────────
# Trained on the named boxes from detector-label/detector-sync (unknown and
# unnamed boxes are skipped):
#   classifier-stats -> classifier-train -> classifier-compare, then
#   `just deploy classifier <run>`
# Crops are cut from the catalog frames with the runtime padding (cat_pad_frac).

# How many named boxes each cat has, per camera.
[group('classifier')]
classifier-stats:
    {{YOLO_RUN}} python -m training.catalog_crops --catalog "{{catalog}}"

# Train the identity classifier from the named boxes. Args pass through.
[group('classifier')]
classifier-train *ARGS:
    {{CLASSIFIER_RUN}} python -m training.train_classifier \
        --catalog "{{catalog}}" \
        --pad-frac {{cat_pad_frac}} \
        {{ARGS}}

# Example:
#   just classifier-compare --candidate current=models/classifier/current \
#                           --candidate new=models/trained/<run>/cat_classifier.pt --baseline current
# Compare candidate classifiers on the same named boxes.
[group('classifier')]
classifier-compare *ARGS:
    {{CLASSIFIER_RUN}} python -m training.compare_classifiers \
        --catalog "{{catalog}}" \
        --pad-frac {{cat_pad_frac}} \
        {{ARGS}}

# Alternatively `just up` runs the `mlflow` container UI on $MLFLOW_PORT (5000).
# Browse MLflow experiment runs from the local file store (./data/mlflow).
[group('dev')]
mlflow-ui PORT="5000":
    {{CLASSIFIER_RUN}} python -m mlflow ui \
        --backend-store-uri "data/mlflow" --port {{PORT}}

# ─────────────────── model: what cv-worker runs ───────────────────
# Detector and classifier are deployed the same way: models/<kind>/versions/<run>
# plus a `current` symlink (tools/models.py). KIND is `detector` or `classifier`.

# Show which detector and classifier versions are running.
[group('model')]
models:
    {{TRAINING_RUN}} python tools/models.py status

# Exports the run with its model's quality gate in the cv-worker image (nothing
# switches if it fails). ARGS: a models/trained/<run> name (default the newest)
# and, detector only, --skip-gate to install an export that failed the gate
# after comparing it with the deployed model yourself.
# Example: just deploy detector yolo-20261010-011816
# Put a trained run into service and restart cv-worker.
[group('model')]
deploy KIND *ARGS:
    {{CV_WORKER_RUN}} python tools/models.py deploy {{KIND}} {{ARGS}}
    {{COMPOSE}} restart cv-worker

# Switch to the previous version (or VERSION) and restart cv-worker.
[group('model')]
rollback KIND VERSION="":
    {{TRAINING_RUN}} python tools/models.py rollback {{KIND}} {{VERSION}}
    {{COMPOSE}} restart cv-worker

# ──────────────────────────── journal ────────────────────────────

# Example: `just journal-feed <cat-name> 7`
# Show how a cat has been eating (door-open sessions) over the last N days.
[group('journal')]
journal-feed CAT DAYS="3":
    python3 tools/feed_log.py {{CAT}} --days {{DAYS}} --db {{journal_db}}

# Run all tests (new stack, then training).
[group('dev')]
check:
    cd streamhub && go vet ./... && go test ./...
    cd webui && npm test
    cd hubclient && uv run --quiet --python 3.12 --group dev pytest -q
    cd cv-worker && uv run --quiet --python 3.12 --group dev pytest -q
    cd decider && uv run --quiet --python 3.12 --group dev pytest -q
    uv run --extra test python -m compileall -q training tools
    uv run --extra test pytest -q tests