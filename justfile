# live2 service commands. Run from the live2/ directory.
#
#   just devserver      — vite HMR + python watchfiles, source mounted live
#   just almostprod     — production-shaped build (nginx + baked image)
#   just down           — stop everything
#   just logs SERVICE   — tail logs for one service (or omit for all)
#   just status         — show running containers + open ports

set dotenv-load := true

PROD     := "docker compose -f docker-compose.yml"
DEV      := "docker compose -f docker-compose.yml -f docker-compose.dev.yml"

# Start the dev stack: vite (HMR) + watchfiles-wrapped detector. mediamtx
# and pruner come along unchanged. Tears down any prior stack first so
# port 8090 etc. are free.
devserver:
    {{PROD}} down --remove-orphans 2>/dev/null || true
    {{DEV}} up --build
    @echo
    @echo "WebUI (vite dev):   http://localhost:${WEB_PORT:-8090}"
    @echo "mediamtx HTTP API:  http://localhost:9997"
    @echo "Detector control:   http://localhost:8092"

# Production-shaped: nginx-served Svelte build, detector without watchfiles.
# Not actually deployed anywhere; just the no-HMR variant for sanity checks.
almostprod:
    {{PROD}} down --remove-orphans 2>/dev/null || true
    {{PROD}} up --build
    @echo
    @echo "WebUI (nginx):      http://localhost:${WEB_PORT:-8090}"

# Stop and remove everything (both stacks share the same project name so this
# covers either).
down:
    {{PROD}} down

# Tail logs. `just logs` for everything, `just logs detector` for one.
logs SERVICE="":
    {{PROD}} logs -f --tail=200 {{SERVICE}}

status:
    {{PROD}} ps

# Rebuild the prod webui image (useful after Svelte changes when running in
# almostprod mode, since prod bakes the bundle into the image).
rebuild-webui:
    {{PROD}} up -d --build webui

# Same for detector.
rebuild-detector:
    {{PROD}} up -d --build detector
