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
