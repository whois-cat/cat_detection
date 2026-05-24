# live2 — single-decode video + detection pipeline

A self-hosted cat-detection setup whose entire job is: ingest one camera's
H.264 once, decode it once, run detection, and surface both **live** and
**historical** detections in a browser UI without ever re-encoding video.

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
cp .env.example .env
# Edit RTSP_URL and WEBRTC_HOST at minimum
just devserver        # vite HMR + watchfiles detector
# or: docker compose up -d --build
# Open http://localhost:8090  (WEB_PORT)
```

## Services

| Service | Role | Ports |
|---|---|---|
| `mediamtx` | RTSP ingest, WebRTC (WHEP) egress, RTSP republish, fMP4 recording, playback server | 8554, 8889, 9996, 9997 |
| `detector` | PyAV decode → configurable detector (`blob`/`yolo`) → SQLite + WS | 8091, 8092 |
| `pruner`   | Detection-aware deletion of recording segments older than `KEEP_RECENT_HOURS` | — |
| `webui`    | Svelte 5 SPA (vite dev or nginx prod), proxies WS + mediamtx + detector | `${WEB_PORT}` (default 8090) |

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

## Layout

```
live2/
├── README.md           — this file (run + what-is-it)
├── NOTES.md            — design rationale, decisions, quirks, deferred
├── justfile            — dev / prod / down / logs / rebuild-*
├── docker-compose.yml
├── docker-compose.dev.yml
├── .env.example
├── mediamtx/mediamtx.yml
├── detector/           — PyAV → swappable Detector → SQLite + WS
├── pruner/             — detection-aware segment GC
├── webui/              — Svelte 5 SPA
└── training/           — extract datasets from recordings + events.db
                          (separate package, modeller-facing)
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
