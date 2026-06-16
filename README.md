# live2 — single-decode video + detection pipeline

A self-hosted cat-detection setup. Ingests **one or more cameras'** H.264
streams once via mediamtx, decodes each once inside a per-camera detector
container, and surfaces both **live** and **historical** detections in a
browser UI — without ever re-encoding video.

```
camera RTSP ──► mediamtx ─┬── WebRTC (WHEP)            ──► browser <video>
                          │
                          ├── RTSP republish            ──► detector (PyAV + YOLO)
                          │                                  ├─► WebSocket events
                          │                                  └─► SQLite (events.db)
                          │
                          └── fMP4 recording (30 s segs)──► data/recordings/<camera>/*.mp4
                                                            │
                                                            ├─► pruner (detection-aware
                                                            │   deletion of "boring" segs
                                                            │   older than KEEP_RECENT_HOURS)
                                                            │
                                                            └─► mediamtx playback server
                                                                /recordings/list, /get
                                                                  └─► browser history view
```

- Decode happens **once**, inside the detector — the model needs pixels.
- Live and recording paths are bitstream-copies of the camera's H.264.
  No re-encode anywhere.
- The browser switches between live (WebRTC) and history (native fMP4
  from mediamtx playback) based on the timeline playhead.

Read [NOTES.md](NOTES.md) for the deep design rationale; that's the "why
does it look like that" doc. This file is just the "how do I run it".

## Run

```bash
cp .env.example   .env             # WEB_PORT + pruner knobs
cp cameras.yaml.example cameras.yaml
# Edit cameras.yaml — at minimum: webrtc_host, and one RTSP URL per camera.
just dev                           # runs `just configure` then docker compose up
# or step-by-step:
#   just configure                 # render mediamtx.yml, compose overlays, nginx.conf
#   docker compose -f docker-compose.yml -f docker-compose.cameras.yml up -d --build
# Open http://localhost:8090  (WEB_PORT)
```

Each camera in `cameras.yaml` becomes:
- one mediamtx path (`paths.<id>`)
- one detector container (`detector-<id>`)
- one entry in the UI's camera picker
- a per-camera recordings directory (`data/recordings/<id>/`)

Generated files (`mediamtx/mediamtx.yml`, `docker-compose.cameras*.yml`,
`webui/nginx.conf`, `webui/public/cameras.json`) are **not committed**.
They are server-specific outputs of `just configure`; rerun it after editing
`cameras.yaml`.

### Camera credentials

Camera RTSP URLs carry passwords, so they never land in a committed file.
`just configure` writes the credentialed URLs to **`secrets/cameras.env`**
(gitignored, chmod 600) as `MTX_PATHS_<ID>_SOURCE=…` env vars, which
docker-compose loads into the mediamtx container (mediamtx does no `${VAR}`
substitution in YAML, only this env override). The committed
`mediamtx/mediamtx.yml` has no `source:` line. Camera ids must be
**hyphen-free** because they become mediamtx path keys addressed via
underscore-separated env vars. Rotate a camera password by editing
`cameras.yaml` and rerunning `just configure`.

## Services

| Service | Role | Ports |
|---|---|---|
| `mediamtx`       | RTSP ingest, WebRTC (WHEP) egress, RTSP republish, fMP4 recording, playback server | 8554, 8889, 9996, 9997 (host network) |
| `detector-<id>`  | PyAV decode → configurable detector (`blob`/`yolo`) → SQLite + WS. One per camera in `cameras.yaml`. | internal 8091, 8092 (proxied by webui) |
| `pruner`         | Detection-aware deletion of recording segments older than `KEEP_RECENT_HOURS` | — |
| `webui`          | Svelte 5 SPA (vite dev or nginx prod), proxies WS + mediamtx + detector(s) | `${WEB_PORT}` (default 8090) |

## Detectors

The detector is swappable via `DETECTOR_TYPE`:

- `blob` — bright-blob filter (threshold + connected components). Tiny,
  zero-dep, useful for end-to-end testing without a model (shine a
  flashlight at the camera).
- `yolo` — Ultralytics YOLO. Default weights are a **pre-quantised INT8
  OpenVINO IR** baked into the image at build time (`yolov8n` for COCO
  cat class), running ~3-4× faster than torch CPU. Swap via `YOLO_WEIGHTS`
  to use your own fine-tune.

To train your own model on the recorded data, see [`training/`](training/).

### Bowl Food Texture Monitor

Each camera can optionally define `food_region` in `cameras.yaml`. It uses the
same camera/UI-normalized coordinates as `ignore_regions` and
`decision_polygon`, and only reports `food_state` / `food_level` over WebSocket
(no events.db migration). Calibrate it by checking one definitely full bowl
frame and one definitely empty bowl frame, then set `food_empty_below` and
`food_full_above` between those observed texture fractions.

## Layout

```
live2/
├── README.md                       — this file (run + what-is-it)
├── NOTES.md                        — design rationale, decisions, quirks, deferred
├── justfile                        — configure / dev / prod / down / logs
├── cameras.yaml.example            — per-camera config; copy → cameras.yaml
├── tools/configure.py              — renders all multi-camera derived files
├── docker-compose.yml              — base stack (mediamtx, pruner, webui)
├── docker-compose.cameras.yml      — GENERATED/ignored — one detector service per camera
├── docker-compose.dev.yml          — webui dev overlay (vite HMR)
├── docker-compose.cameras.dev.yml  — GENERATED/ignored — per-detector watchfiles overlay
├── .env.example                    — pruner + WEB_PORT
├── mediamtx/mediamtx.yml           — GENERATED/ignored — multi-path server config
├── detector/                       — PyAV → swappable Detector → SQLite + WS
├── pruner/                         — detection-aware segment GC
├── webui/                          — Svelte 5 SPA
│   ├── nginx.conf                  — GENERATED/ignored — per-camera proxy rules
│   └── public/cameras.json         — GENERATED/ignored — UI camera picker source
└── training/                       — extract datasets from recordings + events.db
```

## Data on disk

Everything mutable lives under `data/`:

- `data/recordings/<camera_id>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4` —
  30-second fMP4 segments, mediamtx-written. Filename = segment start in
  local wall-clock.
- `data/events/events.db` — SQLite, one row per detected box. Schema in
  [`detector/storage.py`](detector/storage.py).

Both are the **batch / training contract** — anything reading data
non-interactively (the trainer, ad-hoc scripts, future analytics) reads
them directly. The mediamtx HTTP API and the detector's WS/`/events`
endpoint are explicitly for the *live* browser UI and unsuitable for
batch use.
