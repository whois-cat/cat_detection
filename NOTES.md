# live2 — design notes, decisions, quirks

This is the deeper context behind the `live2/` setup. The top-level
[README](README.md) is the "how do I run it" doc; this is the "why does it
look like that" doc.

---

## 1. Project goals

Verbatim from the user at the start of the work:

- **Minimize CPU/GPU load** — the camera already encodes H.264. We should
  keep that stream, not re-encode anywhere we don't have to.
- **Browser-side detection viewing requires metadata perfectly bound to
  frames.** The previous MJPEG-with-burned-in-rectangles approach
  achieves this trivially but is wasteful. The new pipeline must keep
  metadata→frame correspondence while *not* re-encoding.
- **Store interesting video to disk** — same un-re-encoded H.264, for a
  historical browser view *and* for retraining / model comparison.
- **Latency budget (two tiers):**
  1. Model → action (feeder control) — **most important**, sub-second.
  2. Browser WebUI — less important, "feels live" is enough.
- **Timeline-style history UI** — rewind, scrub, see what was detected
  when, see retained vs missing data.
- **Resilience preference:** restart the *whole pipeline* on any single
  failure rather than per-component reconnection logic, even at the cost
  of a few seconds of total outage.
- **Model is not the focus** — the harness is. The detector is swappable
  (`DETECTOR_TYPE=blob` for testing, `yolo` for real work). Expect the
  model to change repeatedly across iterations.

---

## 2. Architecture

```
camera RTSP ──► mediamtx ─┬── WebRTC (WHEP)             ──► browser <video> (liveVideo)
                          │
                          ├── RTSP republish             ──► detector (PyAV + YOLO/blob)
                          │                                  ├─► WS events (live overlay)
                          │                                  └─► SQLite (events.db)
                          │
                          └── fMP4 recording (30 s segs) ──► data/recordings/<camera>/*.mp4
                                                             │
                                                             ├─► pruner (detection-aware
                                                             │   deletion of segs older than
                                                             │   KEEP_RECENT_HOURS)
                                                             │
                                                             └─► mediamtx playback server
                                                                 /recordings/{list,get}
                                                                   └─► browser historyVideo
                                                                       (native fMP4)
```

- Camera produces H.264 once.
- Decode happens once (in the detector — the model needs pixels).
- Recording + WebRTC live paths are bitstream copies. No re-encode anywhere.
- mediamtx hides the fMP4 segment files behind a single virtual stream
  URL. The browser never sees individual segment paths.
- The browser switches between `liveVideo` (WebRTC) and `historyVideo`
  (native fMP4 from mediamtx playback) based on whether the playhead is
  within `LIVE_THRESHOLD_MS = 5000` of `nowMs`. Both share the same
  overlay `<canvas>` inside a stable 16:9 container.

### Sync model

Detection events carry `wall_ms` (detector wall-clock at decode), `pts`,
`tb_num`/`tb_den` (RTSP timebase), `media_t`, frame size, model name,
per-event `cat` label, and `track_id` (for trackers; populated for none
at the moment).

**Wall-clock is the cross-system join key.** PTS is canonical within a
single stream, but the detector and mediamtx's recorder are independent
consumers — their PTS values don't share an origin. mediamtx doesn't
publish a `(camera_pts ↔ segment_pts)` anchor. Wall-clock is what both
sides write down. Same reasoning the [`training/`](training/) extractor
uses to join events to recorded segments.

### Coordinate basis

**Camera frame is the source of truth.** The recorded H.264 is in camera
orientation (we don't re-encode). Anything pinned to camera coords stays
aligned with the recording forever, including across rotation-config
changes. So:

- `DETECT_ROI` / `ACTION_POLYGON` env vars are authored in **camera
  coords**.
- `ev.w`, `ev.h`, box `x/y/w/h` in events are camera coords.
- The browser shows the stream in its native (camera) orientation —
  sideways if the camera is mounted sideways. No CSS rotation.
- The detector may *internally* rotate the inference crop
  (`FRAME_ROTATE_DEG`) when the camera is mounted sideways and has no
  firmware rotate option — YOLO is not rotation-invariant. Detected
  boxes are un-rotated back to camera coords before emit. The rotation
  lives purely inside the inference step.

---

## 3. Tech decisions and rationale

### Streaming hub: **mediamtx**

One Go binary handling RTSP ingest, WebRTC egress (WHEP), RTSP republish
for the detector, fMP4 recording, and a playback server. Single upstream
connection to the camera → every consumer sees the same break or
discontinuity at the same moment when the camera blips. Two
non-default knobs:

- `recordSegmentDuration: 30s` — fine pruning granularity.
- `recordDeleteAfter: 720h` — hard 30-day cap. The detection-aware pruner
  is what does day-to-day GC; this is the "if pruner is broken" backstop.

### Live delivery: **WebRTC via WHEP**

WHEP = WebRTC-HTTP Egress Protocol — IETF-standard SDP exchange over
plain HTTP. mediamtx exposes it at `/<path>/whep`. Browser code is a
standard SDP dance; no per-vendor signalling.

### History delivery: **native fMP4 streaming** (not HLS)

mediamtx's `/recordings/get?…` returns a single chunked fragmented MP4
stream (no HTTP Range support), NOT an HLS playlist. We dropped hls.js
when we discovered this. `historyVideo.src = <playback URL>` plays
natively in any modern browser.

Each user-initiated seek triggers a fresh fetch starting **at** the
target wall-clock, with a 15-minute window. Browser plays from byte 0 of
the new stream — instant first frame, no "seek into unbuffered" failures.

Trade-offs: one HTTP request per seek (fine); a 15-min window then runs
out and the user must re-seek to continue (auto-extend is on the deferred
list).

### Recording: **30 s fMP4 segments + detection-aware pruner**

The `pruner` service runs every `PRUNER_INTERVAL_SEC` (default 1h):

- Reads detection wall_ms values from `events.db` (read-only WAL URI).
- For each segment file in `data/recordings/<camera>/`:
  - **Newer than `KEEP_RECENT_HOURS` (default 24)** → always kept whole.
    The "recent history is always there" guarantee.
  - **Older** → deleted unless some detection's wall_ms falls within
    `[segment_start − PRE_ROLL, segment_end + POST_ROLL]`.
- Safety net: if `events.db` is empty (likely config/data issue rather
  than literally no detections), the pruner skips that pass.
- mediamtx re-scans the recording dir periodically and `/recordings/list`
  reflects the new gaps.

### Detector pipeline shape

Two threads share a **single-slot frame buffer**:

- Decoder thread drains PyAV's RTSP buffer as fast as it can, overwriting
  the slot with each new frame.
- Detector thread takes whatever's in the slot, runs inference, repeats.
- Frames produced while the detector is busy are simply dropped — only
  the newest survives.

Result: **adaptive detection FPS**. The detector processes at
`min(camera_fps, 1/detector_latency)` — never on a stale frame. The
broadcast stats msg reports both `fps_in` and `fps_processed`.

### Detector module: **abstract `Detector` ABC**

[`detector/detectors.py`](detector/detectors.py) defines a tiny `Detector`
base with `detect(img_bgr) → list[box dict]`. Two concrete impls:

- `BrightBlobDetector` — threshold + connected components. For testing.
- `YoloDetector` — Ultralytics YOLO; loads the INT8 OpenVINO IR by
  default (pre-quantised at image build time).

Swapping the model is changing `DETECTOR_TYPE` and possibly `YOLO_WEIGHTS`.

### Storage: **SQLite (WAL mode)** for events

One row per detected box (multiple boxes in one frame = multiple rows
sharing `wall_ms`). Schema includes `camera_id`, `model`, `track_id`,
`source` so multi-camera and multi-model setups don't need a schema
change. WAL means the pruner and `/events` HTTP endpoint can read
concurrently with the detector's writes.

### Browser UI: **Svelte 5 with runes**

- `$state` / `$derived` / `$effect` everywhere.
- `mode` is derived from `playheadMs` relative to `nowMs`; the UI is one
  component, not separate Live/History tabs.
- Two `<video>` elements live in the same 16:9 container; only one is
  visible at a time (via `hidden={mode !== ...}`).
- Overlay canvas is sized to the source `videoWidth × videoHeight` and
  drawn in camera coords (matches event coords).

### Timeline: canvas + time-aligned buckets

- Canvas, not DOM (24 h of events would die in DOM).
- Density bars use time-aligned bucketing — bucket boundaries pinned to
  absolute wall-clock multiples. Without this, panning shifts bucket
  membership and bar heights wobble.
- Pinch-to-zoom (multi-pointer, midpoint anchored).
- Pan/zoom state in URL hash `#from=&to=` — survives reloads, shareable.
- Per-cat colour stacking. Single source of truth: `CAT_COLORS` in
  [`webui/src/Timeline.svelte`](webui/src/Timeline.svelte).

### Cat colour palette

| cat     | colour                  |
|---------|-------------------------|
| felisis | `#555555` (dark gray)   |
| alisa   | `#a04935` (brown-red)   |
| ellie   | `#cccccc` (light gray)  |
| chuzh   | `#ffffff` (white)       |

### Gap overlay

When the history-mode playhead falls outside any actual recording range
(pruned, or before recording started), the video is replaced by a
"no recording at this time" panel. Only triggered after `/recordings/list`
has returned at least once — no false positives during initial load.

---

### Multi-camera

`cameras.yaml` at the project root is the single source of truth. Each
entry produces:

- one mediamtx `paths:` block (same `id` as the camera).
- one `detector-<id>` service in `docker-compose.cameras.yml`.
- one nginx proxy stanza in `webui/nginx.conf`.
- one entry in `webui/public/cameras.json`, which the UI fetches at
  startup to populate the camera picker.

`tools/configure.py` renders all five derived files. `just configure`
runs it; `just dev` / `just up` run configure first
automatically. Generated files are committed so diffs are inspectable.

Networking topology:
- mediamtx stays on the host network (WebRTC ICE needs it).
- Per-camera detectors run on the default bridge network. They reach
  mediamtx as `host.docker.internal:8554` and pull
  `rtsp://host.docker.internal:8554/<camera_id>`.
- webui (also on the bridge) talks to each detector by container DNS
  name (`detector-<id>:8091`). nginx fans out `/detector/<id>/*` to the
  matching upstream. Dev (`vite`) does the same routing via its `proxy`
  table, built from `public/cameras.json` at start.
- A single shared `events.db` receives writes from all detectors. Rows
  are tagged with `camera_id`; the UI filters by selected camera.

URL hash: `#camera=<id>` is added to the existing `#from=&to=` pan/zoom
state. Reloading or sharing a URL restores both the timeline view and
the selected camera.

UI behaviour on camera switch (`switchCamera()` in App.svelte):
- Close the WS for the previous camera; open a new WS for the new one.
- Tear down WebRTC PeerConnection and recreate against the new WHEP URL.
- Reset `allEvents`, `recentLive`, `hlsRanges`, `stats`, `detectRoi`,
  `actionPolygon` — they're per-camera state.
- Reload model picker (each camera may run its own DETECTOR_TYPE /
  YOLO_WEIGHTS).
- Refresh recordings list for the new path.

---

## 4. Component reference

| Component | Role | Key files |
|---|---|---|
| `mediamtx`        | RTSP ingest, WHEP egress, RTSP republish, fMP4 recording, playback | `mediamtx/mediamtx.yml` (generated) |
| `detector-<id>`   | PyAV decode → swappable Detector → SQLite + WS. One per camera. | `detector/{main.py,detectors.py,storage.py}` |
| `pruner`          | Detection-aware segment GC | `pruner/pruner.py` |
| `webui`           | Svelte 5 SPA + reverse proxies to detectors & mediamtx | `webui/src/{App,Timeline}.svelte`, `webui/nginx.conf` (generated) |
| `tools/configure.py` | Renders all multi-camera derived files from `cameras.yaml` | `tools/configure.py` |
| `training/`       | Extract datasets (classifier crops, YOLO labels) from recordings + events.db | `training/{README,db,segments,sources,extract_*}.py` |

### Ports

- `8554` — mediamtx RTSP (detectors pull from `host.docker.internal:8554/<id>`)
- `8889` — mediamtx WebRTC (WHEP)
- `8091` — per-detector WebSocket + `/events` REST (internal; webui proxies)
- `8092` — per-detector control endpoint (internal; webui proxies)
- `9996` — mediamtx playback server (`/list`, `/get`)
- `9997` — mediamtx HTTP API
- `${WEB_PORT}` — webui (nginx in prod, vite in dev; default 8090). Only
  externally-published port besides mediamtx's host-network ones.

### Detector event format

Wire (WebSocket / `/events`):

```json
{
  "wall_ms": 1715961023456,
  "pts":     5130,
  "tb_num":  1, "tb_den": 90000,
  "media_t": 0.057,
  "w": 2304, "h": 1296,
  "rotate_deg": 90,
  "cat": "alisa",
  "cat_score": 0.91,
  "camera_id": "default",
  "model": "yolov8n",
  "boxes": [{"x": 412, "y": 220, "w": 80, "h": 60,
             "score": 0.83, "in_action": true}]
}
```

- **One event per detected box.** Multiple boxes in one frame produce
  multiple events with identical `wall_ms`/`pts`.
- **`cat_score`**: identity-classifier confidence for `cat`, top-level
  (mirrors `cat`). Present only for `yolo_cat`; `null` for blob/yolo (no
  per-cat identity). The feeder weights identity votes by this value and
  drops frames below its `CLASSIFIER_MIN_CONF`, so it MUST travel with the
  event — it is also persisted to the `cat_score` column for training.
- **`rotate_deg`**: the `FRAME_ROTATE_DEG` applied to the inference input for
  this event, persisted (nullable column) so training/review can re-rotate
  each crop by its OWN recorded value — data captured under different (or
  later-changed) camera rotations then mixes correctly. Boxes and `w`/`h`
  stay camera-orientation. Old rows are `NULL`; consumers fall back to a
  configured default (`--default-rotate-deg`) and log one warning.
- **Camera coordinates everywhere.** See the coordinate-basis discussion
  in §2.
- **`in_action`**: box-center inside `ACTION_POLYGON`. Action-trigger
  consumers should gate on this.
- **Clear events**: on detection→no-detection transition the detector
  emits one event with `boxes: []` (not persisted) so the browser overlay
  snaps off instantly.

### Stats broadcast

On the same WebSocket, identified by `kind`:

```json
{
  "kind": "stats",
  "camera_id": "default",
  "model": "yolov8n",
  "fps_in": 14.92,
  "fps_processed": 5.10,
  "active_tracks": 0,
  "wall_ms": 1715961023000,
  "detect_roi": [0, 0, 1, 1],
  "action_polygon": [[0,0],[1,0],[1,1],[0,1]]
}
```

Drives the FPS line in the live UI and feeds the ROI overlays.

### Detector control channel

`POST /delay {ms: int}` on port 8092 (proxied via webui as
`/detector-control/delay`). Injects an artificial post-detection delay,
useful for testing UI behaviour under a deliberately slow model.

---

## 5. Configuration

**Per-camera config**: `cameras.yaml` (template in `cameras.yaml.example`).
Each camera's knobs (RTSP URL, detector type, ROIs, rotation, …) live
here. After editing, run `just configure`.

**Per-camera knobs** (in `cameras.yaml`):

| Field | Default | Purpose |
|---|---|---|
| `id` | — | Slug used as mediamtx path, recordings dir, CAMERA_ID, and URL prefix |
| `rtsp` | — | Camera RTSP URL with credentials |
| `label` | title-cased id | Display name in the UI picker |
| `detector_type` | `blob` | `blob` or `yolo` |
| `yolo_weights` | `/opt/models/yolov8n_int8_openvino_model/` | Pre-quantised INT8 by default |
| `yolo_conf` | 0.25 | YOLO confidence threshold |
| `blob_bright_threshold` | 240 | Blob detector grayscale threshold |
| `blob_min_area` | 500 | Minimum blob area (px) |
| `detect_roi` | `0,0,1,1` | Inference crop (camera-frame fractions) |
| `action_polygon` | `0,0,1,1` | Box-center gate (rect or N-vertex polygon, camera coords) |
| `rotate_deg` | 0 | Internal-to-model rotation, 0/90/180/270 |
| `artificial_delay_ms` | 0 | Default detector-emit delay (live-tunable via UI) |

**Global knobs** (in `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `WEB_PORT` | 8090 | Browser-facing port |
| `PRUNER_PRE_ROLL_SEC` | 30 | Keep this much before each detection |
| `PRUNER_POST_ROLL_SEC` | 30 | Keep this much after each detection |
| `PRUNER_KEEP_RECENT_HOURS` | 24 | Never prune segments newer than this |
| `PRUNER_INTERVAL_SEC` | 3600 | Pruner cadence |
| `PRUNER_DRY_RUN` | 0 | Set to 1 to log without deleting |

`cameras.yaml` also has a top-level `webrtc_host` (the public host the
browser dials for WebRTC ICE — usually the docker host's LAN IP).

---

## 6. Dev / prod workflow

### `just dev`

Vite HMR + `watchfiles`-wrapped detector. Svelte and Python edits reload
without container restarts. mediamtx + pruner unchanged from production.
Image rebuilds (`requirements.txt`, Dockerfile, the OpenVINO export step)
do require `--build` (which `devserver` passes).

Dev overlay ([`docker-compose.dev.yml`](docker-compose.dev.yml)) mounts
`./detector` at `/app`, which would mask the baked-in OpenVINO model dir
— so the model lives at `/opt/models/`, OUTSIDE the bind mount.

### `just up`

Production-shaped: nginx serves baked Svelte assets; detector runs
straight `python -u main.py`. Same backend proxies. Used for sanity
checks before a release; not actually deployed anywhere.

### `just down` / `just logs [SERVICE]` / `just ps` (rebuild via `just up`, which passes `--build`)

---

## 7. Quirks and historical notes

Leaving the trail so we don't fall in the same hole twice. Some are
resolved by later architecture decisions (noted).

1. **WebRTC video frames decoded but not rendered**: pure CSS bug — the
   overlay canvas had `background:#000` covering the video. Fixed by
   `background: transparent`.
2. **`playoutDelayHint` is Chromium-only in practice**: Firefox accepts
   the assignment but it's a silent no-op. Adaptive sync hint was
   eventually dropped; dashed-by-age boxes work on both browsers.
3. **`mediaTime` ≠ capture time** in `requestVideoFrameCallback`. It's a
   playback timeline, not wall-clock at frame capture.
4. **Docker Compose port-list merge accumulates**: prod has `8090:80`;
   dev would append `8090:5173` → Docker tries to bind 8090 twice →
   "port already allocated" with nothing actually listening. Solved with
   `ports: !override` in the dev compose.
5. **Two `<video>` elements + `[hidden]` + scoped `display: block`**:
   Svelte's scoped `video { display: block }` outranked the UA's
   `[hidden] { display: none }`, so the hidden video stayed visible
   covering the live one. Fixed with explicit
   `video[hidden] { display: none }`.
6. **`inline-block` `.wrap` shifted when video changed**: intrinsic-size
   reflow as siblings appeared. Fixed by making `.wrap` a fixed 16:9 box
   and absolutely positioning videos + canvas inside.
7. **Pixel-aligned timeline buckets wobble during pan**: events hop
   between adjacent columns as the view shifts → visible bar-height
   jitter. Fixed by aligning bucket boundaries to absolute time.
8. **Empty-frame events were dropped silently**: detector only emitted
   on positive boxes, so the last positive event aged out through dashes
   when the object disappeared. Now emits a single "clear" event
   (`boxes: []`) on the transition, not persisted.
9. **`mode === 'live'` after follow drift**: `playheadMs` was set once
   and not updated as `nowMs` advanced; mode flipped to `history` after
   ~5 s. Fixed with an explicit `follow` flag that the tick advances and
   that any user pan/zoom breaks.
10. **YAML linter doesn't know `!override` / `!reset`**: Compose v2
    supports those tags but the editor's schema doesn't. Warnings safe to
    ignore.
11. **mediamtx YAML doesn't do `${VAR}` substitution**: only the
    `MTX_<UPPERCASE>` env-var override mechanism. Got
    `invalid source: '${RTSP_URL}'` until we set
    `MTX_PATHS_CAMERA_SOURCE` directly on the container. This is now how camera
    credentials stay out of git: `configure.py` writes the credentialed RTSP
    URLs to the gitignored `secrets/cameras.env` as `MTX_PATHS_<ID>_SOURCE`,
    docker-compose loads it via `env_file`, and the committed `mediamtx.yml`
    paths have no `source:` line. mediamtx derives the path key from the env
    var name by splitting on underscores (`MTX_PATHS_<KEY>_SOURCE`), so a
    hyphen in a path name can't be addressed — camera ids must be hyphen-free
    (enforced in `configure.py`).
12. **mediamtx `/get` returns chunked fMP4, not an HLS playlist**.
    hls.js logged `manifestParsingError`. Dropped hls.js, switched to
    native `<video src>` playback.
13. **`accept-ranges: none`** on mediamtx playback: can't seek within a
    downloaded fMP4 past the buffered range. Solution: each user seek
    loads a fresh fMP4 starting at the target — no lead-in, no "seek
    into not-yet-buffered" failures.
14. **Reload loop on playback**: an `$effect` reading `playheadMs` was
    calling `scheduleHistorySeek` on every `timeupdate` (~5×/sec),
    causing a full media reload every 200 ms (~50 MB transfers). Fixed
    by making user-initiated seeks call `scheduleHistorySeek` directly
    and stripping it from the effect.
15. **Src reload doesn't auto-resume**: setting `historyVideo.src = newUrl`
    fires `pause`. Need explicit `canplay → play()`, gated by a
    `userWantsPlaying` flag so explicit user pauses are respected.
16. **mediamtx `/list` may return duration as a number or Go-string**:
    `durationToMs` handles seconds / nanoseconds / `"1h2m3s"` defensively.
17. **OpenVINO model dir masked by dev bind-mount**: dev compose mounts
    `./detector → /app`; the OpenVINO export originally lived at
    `/app/yolov8n_int8_openvino_model/` and disappeared at runtime.
    Moved to `/opt/models/` and updated `YOLO_WEIGHTS` default.
18. **Sideways camera tanked YOLO recall**: CNNs aren't rotation-invariant.
    Added `FRAME_ROTATE_DEG` that rotates only the inference crop and
    un-rotates detected boxes back to camera coords. UI unchanged
    (camera-orientation is the source of truth).
19. **Classifier preprocessing drifted from training by a row/column**:
    `detector/classifier.py::_preprocess` reimplements the torchvision
    Resize(256)+CenterCrop(224) pipeline by hand. It used floor (`//2`) for the
    center-crop offset, but torchvision uses `round((dim-224)/2)` — so for an
    odd size difference the crop was shifted one pixel off from training. (Resize
    must also use `int(256*long/short)` truncation, matching torchvision, not
    `round`.) Caught by the parity gate added to `export_classifier.py`: it
    compares `_preprocess` against the torchvision transform and the exported
    OpenVINO logits against torch on real/synthetic crops, failing the build if
    `max|Δ| >= 1e-3`. Pass `--crops <dir>` to validate on real cat crops.

---

## 8. Deferred / TODO

### Auto-extend history playback window

When `historyVideo.currentTime` approaches `PLAYBACK_WINDOW_SEC` (15
min), load the next window starting at the current wall-clock. Otherwise
continuous historical playback stalls after 15 minutes.

### Auto-skip across gaps during playback

Right now playback into a deleted region pauses at the end of the last
segment. Could detect `ended` + range check and auto-jump to the next
range. Gap overlay shipped instead; auto-jump can wait.

### Track-aware identification

`track_id` is in the schema but nothing populates it. Ultralytics has
ByteTrack built into `YOLO.track(...)`; flipping `detect` → `track` would
add stable IDs. Required for the parent project's sliding-window
majority-vote stability policy, and for cheap per-track classifier
reclassification (every Nth frame, not every frame).

### Per-cat classifier on top of YOLO

Today `YoloDetector` returns the COCO class name (`cat`). For per-cat
identity, run an EfficientNet-style classifier on each cat crop and
override `b["cat"]` with the classifier's prediction. The parent project
already has a fine-tuned 4-cat classifier; wiring is straightforward
inside [`detectors.py`](detector/detectors.py).

### Labeling UI / human correction feedback loop

`storage.py` already has `source` column (default `'detector'`). Manual
labels would write rows with a different source so detector-vs-human
agreement is queryable. Frontend tooling not yet built; CVAT in the
parent project is the interim path.

### Action consumer (feeder, etc.)

`in_action` flag is computed and broadcast per box; nothing acts on it
yet. The downstream consumer (feeder relay, MQTT, webhook, …) is
deliberately out of scope until it's needed.

### Health checks / restart-everything policy

The user's stated resilience preference: restart the whole pipeline if
anything dies. Currently everything is `restart: unless-stopped` per
service. Wiring health-check chains with `depends_on:
condition: service_healthy` would honour the preference more literally.

### PTS-preserving recorder (replace mediamtx for storage?)

Validated in [`training/validate_sync.py`](training/validate_sync.py): mediamtx
rebases PTS in BOTH the segment files AND the playback API. Each segment's
internal PTS starts at 0; playback also starts near 0 per `/get` request.
The detector's `event.pts` (camera-stream RTP epoch) survives nowhere on
the storage side. As a result, **wall-clock is forced to be the only
cross-system join key** (extractor in [`training/segments.py`](training/segments.py)
documents this), which costs us frame-perfect identity — we get
"sub-frame in practice" instead.

If frame-perfect matching becomes load-bearing (e.g. for very precise
overlay alignment, or for downstream pipelines that want PTS to be the
join key everywhere), the right move is to **write our own multiplexer**
that ingests the same RTSP republish mediamtx already exposes, segments
into fMP4 ourselves, and writes the original camera PTS through unchanged.
PyAV exposes the necessary primitives — open the RTSP source, copy
packets without re-encoding, write to an fMP4 output container with
explicit time-base passthrough, and rotate output files every N seconds
similar to mediamtx's segmenter. We keep mediamtx for the WebRTC live
path; only the recording-and-playback piece changes.

Scope when we do this:
- New service: `recorder/` Python container (PyAV `av.open` source +
  `av.open` output, packet remux, time-base passthrough).
- Filename convention stays the same (wall-clock at segment start) so
  the existing pruner and extractor don't need to change.
- A small replacement for mediamtx's `/get` playback (or a different
  approach: serve segments directly with HTTP range support, since our
  files would be proper seekable fMP4 with single PTS epoch per stream).
- Validate: re-run `training/validate_sync.py`; Methods A and B should
  now agree, and `event.pts` should fall inside the segment's PTS range.

Defer until either (a) we hit a concrete problem with the wall-clock
approach (training data showing visible misalignment with overlays, or a
downstream consumer needing PTS-keyed access), or (b) we decide to drop
mediamtx for unrelated reasons.

### Storage budget measurement

With 30 s segments + 30-day cap + detection-aware pruner, we should be
well under any realistic disk budget. Worth measuring after a week of
running.

---

## 9. Resume checklist

1. Verify `docker-compose.yml` is still mediamtx + detector + pruner +
   webui.
2. Inspect `data/events/events.db` — is the detector producing rows?
   Are the cat labels sensible?
3. Try `training/extract_classifier.py` against a few hours of recordings
   to confirm the joining still works end-to-end.
4. Decide whether to wire ByteTrack and the per-cat classifier (see §8)
   for the next iteration, or sit on the COCO-class detector while
   focusing on harness / UI features.
