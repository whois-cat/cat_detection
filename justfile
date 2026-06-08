# live2 service commands. Run from the live2/ directory.
#
#   just configure      — regenerate the multi-camera config files from cameras.yaml
#   just devserver      — vite HMR + python watchfiles, source mounted live
#   just almostprod     — production-shaped build (nginx + baked images)
#   just down           — stop everything
#   just logs SERVICE   — tail logs for one service (or omit for all)
#   just status         — show running containers + open ports

set dotenv-load := true

PROD := "docker compose -f docker-compose.yml -f docker-compose.cameras.yml"
DEV  := PROD + " -f docker-compose.dev.yml -f docker-compose.cameras.dev.yml"

# Render mediamtx.yml + docker-compose.cameras.{yml,dev.yml} + nginx.conf
# + cameras.json from cameras.yaml. Required before the first `up`, and
# every time cameras.yaml changes.
configure:
    python tools/configure.py

# Dev stack: vite (HMR) + watchfiles-wrapped detector(s). mediamtx and
# pruner come along unchanged. Tears down any prior stack first so ports
# are free.
devserver: configure
    {{PROD}} down --remove-orphans 2>/dev/null || true
    {{DEV}} up --build
    @echo
    @echo "WebUI (vite dev):   http://localhost:${WEB_PORT:-8090}"
    @echo "mediamtx HTTP API:  http://localhost:9997"

# Production-shaped: nginx-served Svelte build, detectors without watchfiles.
# Not actually deployed anywhere; the no-HMR variant for sanity checks.
almostprod: configure
    {{PROD}} down --remove-orphans 2>/dev/null || true
    {{PROD}} up --build
    @echo
    @echo "WebUI (nginx):      http://localhost:${WEB_PORT:-8090}"

# Stop and remove everything (both stacks share the same project name so this
# covers either).
down:
    {{PROD}} down

# Tail logs. `just logs` for everything, `just logs detector-living` for one.
logs SERVICE="":
    {{PROD}} logs -f --tail=200 {{SERVICE}}

status:
    {{PROD}} ps

# Rebuild the prod webui image (useful after Svelte changes when running in
# almostprod mode, since prod bakes the bundle into the image).
rebuild-webui:
    {{PROD}} up -d --build webui

# Rebuild all detector images. Per-camera rebuilds: `docker compose ... up -d --build detector-<id>`.
rebuild-detectors:
    {{PROD}} up -d --build

# ── Crop label-review tooling ───────────────────────────────────────────────
# Stage A — build the metadata-only review manifest. Runs INSIDE the detector
# image (it carries openvino + the baked classifier IR) with the repo mounted at
# /work. Pass the detector service name + any build_review_manifest flags:
#   just review-manifest detector-grey --confuse alisa,felisis --min-score 0.3
# --model is auto-detected when the DB has a single model. Output is metadata
# only (data/review/manifest.jsonl) — no images. Overridable env:
#   EVENTS_DB, RECORDINGS_ROOT, CLASSIFIER_IR, REVIEW_MANIFEST.
review-manifest SERVICE *ARGS:
    {{PROD}} run --rm --no-deps -v "$PWD":/work -w /work {{SERVICE}} \
        python -m training.build_review_manifest \
            --db "${EVENTS_DB:-data/events/events.db}" \
            --recordings "${RECORDINGS_ROOT:-data/recordings}" \
            --classifier "${CLASSIFIER_IR:-/opt/models/cat_classifier_openvino}" \
            --out "${REVIEW_MANIFEST:-data/review/manifest.jsonl}" \
            {{ARGS}}

# One-time: create the Stage B venv on the host (no openvino/torch).
review-setup:
    python3 -m venv .venv-review
    .venv-review/bin/pip install -r review/requirements.txt

# Stage B — the review web app. Decodes crops on the fly (no image files) and
# writes corrections to a SEPARATE reviews.db; events.db is never modified.
# Run `just review-setup` once first. Highlight the confusable pair (not
# hardcoded) via env:  REVIEW_CONFUSE=alisa,felisis just review 8095
review PORT="8095":
    #!/usr/bin/env bash
    set -euo pipefail
    PY=.venv-review/bin/python
    [ -x "$PY" ] || PY=python3
    REVIEW_MANIFEST="${REVIEW_MANIFEST:-data/review/manifest.jsonl}" \
    RECORDINGS_ROOT="${RECORDINGS_ROOT:-data/recordings}" \
    REVIEW_DB="${REVIEW_DB:-data/review/reviews.db}" \
    REVIEW_CONFUSE="${REVIEW_CONFUSE:-}" \
    "$PY" -m uvicorn review.app:app --host 0.0.0.0 --port {{PORT}}
