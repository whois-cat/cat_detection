"""
Pulls H.264 from mediamtx over RTSP, decodes with PyAV, runs a configurable
detector, persists events to SQLite and broadcasts them on a WebSocket.

Events sent over WebSocket have the shape:
  {
    "wall_ms":  int,
    "pts":      int,
    "tb_num":   int, "tb_den": int,
    "media_t":  float,
    "w": int, "h": int,
    "rotate_deg": int,           # FRAME_ROTATE_DEG applied to the inference input
    "cat":       str | null,
    "cat_score": float | null,   # classifier identity confidence; null for blob/yolo
    "camera_id": str,
    "model":    str,
    "food_level": float | null,   # runtime-only WS field when FOOD_REGION is set
    "food_state": str,            # "empty" | "has_food" | "unknown"
    "boxes":    [{"x":int, "y":int, "w":int, "h":int, "score":float, "in_action":bool}, ...]
  }

Also broadcast on the same WS, identified by `kind` field:
  {"kind": "stats", "fps_in": float, "fps_processed": float,
   "active_tracks": int, "camera_id": str, "model": str,
   "detect_roi": [x0,y0,x1,y1],
   "action_polygon": [[x,y], ...],
   "decision_polygon": [[x,y], ...],  # alias of action_polygon
   "food_region": [[x,y], ...] | null,
   "ignore_regions": [{"name": str, "points": [[x,y], ...]}]}

Storage: SQLite at /data/events/events.db (one row per detection box).
"""
import asyncio
import json
import os
import queue
import threading
import time

import av
import numpy as np
from aiohttp import web, WSMsgType

from bowl_monitor import BowlMonitor
from detectors import build_detector
from storage import init_db, insert_event, query_events, list_models


# ---- config ----

RTSP_SOURCE   = os.environ["RTSP_SOURCE"]
CAMERA_ID     = os.environ.get("CAMERA_ID", "default")
DETECTOR_TYPE = os.environ.get("DETECTOR_TYPE", "blob")


def _parse_coord_values(raw: str) -> list[float]:
    raw = raw.strip()
    if raw.startswith("["):
        data = json.loads(raw)
        if data and all(isinstance(v, (list, tuple)) for v in data):
            return [float(x) for pair in data for x in pair]
        return [float(v) for v in data]
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_rect(env_name: str) -> tuple[float, float, float, float]:
    raw = os.environ.get(env_name, "0,0,1,1")
    parts = tuple(_parse_coord_values(raw))
    assert len(parts) == 4, f"{env_name} must be x0,y0,x1,y1"
    return parts  # type: ignore[return-value]


def _parse_polygon(env_name: str) -> list[tuple[float, float]]:
    """Parse a polygon from a flat comma-separated list of fractional coords.
    4 values = axis-aligned rect (expanded to 4 vertices); >4 = explicit polygon."""
    raw = os.environ.get(env_name, "0,0,1,1")
    vals = _parse_coord_values(raw)
    return _polygon_from_values(env_name, vals)


def _parse_optional_polygon(env_name: str) -> list[tuple[float, float]] | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    return _polygon_from_values(env_name, _parse_coord_values(raw))


def _polygon_from_values(env_name: str, vals: list[float]) -> list[tuple[float, float]]:
    assert len(vals) >= 4 and len(vals) % 2 == 0, \
        f"{env_name} must be an even number of fractional coords (>=4)"
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray casting. Edges are treated as half-open to dodge double-counts at vertices."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xc:
                inside = not inside
        j = i
    return inside


def _is_soft_food_region_name(name: str) -> bool:
    tokens = str(name).lower().replace("-", "_").replace(" ", "_").split("_")
    return "bowl" in tokens or "food" in tokens


# Frame rotation. 0/90/180/270 (clockwise degrees). Camera mounted sideways
# with no firmware rotate option → we rotate the model's input only. Camera
# frame stays the source of truth everywhere else: env-var ROIs are camera
# coords, ev.w/ev.h are camera dims, recordings on disk are camera-orientation
# H.264, the UI shows the stream as-is (no CSS rotation). The rotation lives
# *purely* inside the inference step and detected boxes are un-rotated back
# to camera coords before emit. This keeps history alignment stable across
# rotation-config changes — old events still line up with old recordings.
FRAME_ROTATE_DEG = int(os.environ.get("FRAME_ROTATE_DEG", "0"))
assert FRAME_ROTATE_DEG in (0, 90, 180, 270), "FRAME_ROTATE_DEG must be 0/90/180/270"
# np.rot90 turns counter-clockwise; we want clockwise → k = -deg/90 mod 4.
_ROT90_K = (-FRAME_ROTATE_DEG // 90) % 4

DETECT_ROI     = _parse_rect("DETECT_ROI")          # camera-frame fractions
ACTION_POLYGON = _parse_polygon("ACTION_POLYGON")   # camera-frame fractions
FOOD_REGION    = _parse_optional_polygon("FOOD_REGION")  # camera-frame fractions
DETECT_ROI_IS_FULL = DETECT_ROI == (0.0, 0.0, 1.0, 1.0)
ACTION_ROI_IS_FULL = ACTION_POLYGON == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
FOOD_TILES = int(os.environ.get("FOOD_TILES", "8"))
FOOD_TEX_THRESH = float(os.environ.get("FOOD_TEX_THRESH", "12.0"))
FOOD_EMPTY_BELOW = float(os.environ.get("FOOD_EMPTY_BELOW", "0.20"))
FOOD_FULL_ABOVE = float(os.environ.get("FOOD_FULL_ABOVE", "0.40"))
FOOD_CHECK_INTERVAL_SEC = float(os.environ.get("FOOD_CHECK_INTERVAL_SEC", "5"))
FOOD_MEDIAN_WINDOW = int(os.environ.get("FOOD_MEDIAN_WINDOW", "10"))
FOOD_MONITOR = (
    BowlMonitor(
        FOOD_REGION,
        empty_below=FOOD_EMPTY_BELOW,
        full_above=FOOD_FULL_ABOVE,
        window=FOOD_MEDIAN_WINDOW,
        tiles=FOOD_TILES,
        tex_thresh=FOOD_TEX_THRESH,
    )
    if FOOD_REGION is not None
    else None
)
_last_food_check_monotonic = 0.0
_food_state = "unknown"
_food_level: float | None = None


def _parse_ignore_regions() -> list[dict]:
    raw = os.environ.get("IGNORE_REGIONS", "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    regions = []
    for i, region in enumerate(data):
        name = f"ignore-{i}"
        coords = region
        if isinstance(region, dict):
            name = str(region.get("name") or name)
            coords = region.get("rect", region.get("points", region.get("polygon")))
        if _is_soft_food_region_name(name):
            continue
        if isinstance(coords, str):
            vals = [float(v.strip()) for v in coords.split(",") if v.strip()]
        elif coords and all(isinstance(v, (list, tuple)) for v in coords):
            vals = [float(x) for pair in coords for x in pair]
        else:
            vals = [float(v) for v in coords]
        if len(vals) == 4:
            x0, y0, x1, y1 = vals
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            points = [(vals[j], vals[j + 1]) for j in range(0, len(vals), 2)]
        else:
            raise ValueError(f"IGNORE_REGIONS[{i}] must be a rect or polygon")
        for x, y in points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"IGNORE_REGIONS[{i}] point out of [0,1]: {(x, y)!r}")
        regions.append({"name": name, "points": points})
    return regions


IGNORE_REGIONS = _parse_ignore_regions()
IGNORE_REGION_MIN_COVERAGE = max(
    0.0,
    min(1.0, float(os.environ.get("IGNORE_REGION_MIN_COVERAGE", "0.8"))),
)


def _maybe_update_food_monitor(
    frame_bgr: np.ndarray,
    boxes: list[dict],
    frame_w: int,
    frame_h: int,
) -> tuple[str, float | None]:
    global _last_food_check_monotonic, _food_state, _food_level
    if FOOD_MONITOR is None or FOOD_REGION is None:
        return "unknown", None

    now = time.monotonic()
    if now - _last_food_check_monotonic < FOOD_CHECK_INTERVAL_SEC:
        return _food_state, _food_level
    _last_food_check_monotonic = now

    cat_in_region = False
    for b in boxes:
        cx = (b["x"] + b["w"] * 0.5) / max(1, frame_w)
        cy = (b["y"] + b["h"] * 0.5) / max(1, frame_h)
        if _point_in_polygon(cx, cy, FOOD_REGION):
            cat_in_region = True
            break

    frame_gray = (
        0.114 * frame_bgr[..., 0]
        + 0.587 * frame_bgr[..., 1]
        + 0.299 * frame_bgr[..., 2]
    ).astype(np.float32)
    _food_state, fraction = FOOD_MONITOR.update(frame_gray, cat_in_region)
    if fraction is not None:
        _food_level = fraction
    return _food_state, _food_level


def _attach_food_fields(ev: dict, state: str, level: float | None) -> None:
    if FOOD_MONITOR is None:
        return
    ev["food_state"] = state
    ev["food_level"] = None if level is None else round(float(level), 3)


def _with_food_fields(ev: dict, state: str, level: float | None) -> dict:
    _attach_food_fields(ev, state, level)
    return ev


def _box_region_coverage(
    box: dict,
    frame_w: int,
    frame_h: int,
    region: list[tuple[float, float]],
    *,
    samples: int = 7,
) -> float:
    if frame_w <= 0 or frame_h <= 0 or box["w"] <= 0 or box["h"] <= 0:
        return 0.0
    samples = max(2, samples)
    inside = 0
    total = samples * samples
    for iy in range(samples):
        py = (box["y"] + (iy + 0.5) * box["h"] / samples) / frame_h
        for ix in range(samples):
            px = (box["x"] + (ix + 0.5) * box["w"] / samples) / frame_w
            if _point_in_polygon(px, py, region):
                inside += 1
    return inside / total


def _box_in_ignore_region(box: dict, frame_w: int, frame_h: int) -> bool:
    return any(
        _box_region_coverage(box, frame_w, frame_h, region["points"])
        >= IGNORE_REGION_MIN_COVERAGE
        for region in IGNORE_REGIONS
    )


def _unrotate_box_to_camera(rx: int, ry: int, rw: int, rh: int,
                            rot_W: int, rot_H: int) -> tuple[int, int, int, int]:
    """Map a pixel-space box from the rotated inference image back to the
    pre-rotation (camera-orientation) crop. The image was rotated
    FRAME_ROTATE_DEG CW for inference, so the inverse rotates a box CCW."""
    if FRAME_ROTATE_DEG == 0:
        return (rx, ry, rw, rh)
    if FRAME_ROTATE_DEG == 90:
        return (ry, rot_W - rx - rw, rh, rw)
    if FRAME_ROTATE_DEG == 180:
        return (rot_W - rx - rw, rot_H - ry - rh, rw, rh)
    # 270 CW
    return (rot_H - ry - rh, rx, rh, rw)

EVENTS_DB     = os.environ.get("EVENTS_DB", "/data/events/events.db")
WS_PORT       = int(os.environ.get("WS_PORT", "8091"))
CONTROL_PORT  = int(os.environ.get("CONTROL_PORT", "8092"))
STATS_INTERVAL_SEC = float(os.environ.get("STATS_INTERVAL_SEC", "1.0"))

# Dynamic artificial delay (ms), tunable at runtime via the control endpoint.
artificial_delay_ms = int(os.environ.get("ARTIFICIAL_DELAY_MS", "0"))


# ---- queues / shared state ----

event_q:   "queue.Queue[dict]"       = queue.Queue(maxsize=200)
pending_q: "queue.Queue[tuple[int, dict]]" = queue.Queue(maxsize=400)

# FPS counters — incremented by the decoder/detector threads, read+reset by the
# stats broadcaster. Single writer per counter; plain ints are fine in CPython.
_fps_in_count = 0          # frames decoded from RTSP
_fps_proc_count = 0        # frames the detector actually processed

# Latest-frame slot: decoder always overwrites; detector takes whatever's there
# and runs. Frames produced while the detector is busy are dropped (only the
# newest survives) — this gives correct adaptive detection fps regardless of
# detector wall-time, instead of processing a stale "every Nth" decoded frame.
_latest_lock  = threading.Lock()
_latest_cv    = threading.Condition(_latest_lock)
_latest_frame: "tuple[av.VideoFrame, int, int] | None" = None   # (frame, tb_num, tb_den)

# Initialised in main() so we can pass model_name into events.
detector = None  # type: ignore


# ---- pipeline ----

def delay_worker():
    """Holds events for artificial_delay_ms before forwarding them downstream."""
    while True:
        release_ms, ev = pending_q.get()
        wait = release_ms - int(time.time() * 1000)
        if wait > 0:
            time.sleep(wait / 1000)
        try:
            event_q.put_nowait(ev)
        except queue.Full:
            pass


def decoder_loop():
    """Pull RTSP frames as fast as the camera produces them; publish only the
    latest into the shared slot. Reconnect forever.

    This thread MUST keep draining libav's internal buffer — otherwise frames
    accumulate and PTS drifts further and further behind wall-clock. We do not
    do any image conversion or detection work here; the detector thread handles
    that (and lazily calls to_ndarray on whatever frame is currently in the slot)."""
    global _fps_in_count, _latest_frame
    while True:
        try:
            container = av.open(
                RTSP_SOURCE,
                options={"rtsp_transport": "tcp", "stimeout": "5000000"},
            )
            stream = container.streams.video[0]
            tb = stream.time_base
            tb_num, tb_den = tb.numerator, tb.denominator
            print(f"[decoder] connected camera={CAMERA_ID} model={detector.model_name} "
                  f"time_base={tb_num}/{tb_den}", flush=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                _fps_in_count += 1
                with _latest_cv:
                    _latest_frame = (frame, tb_num, tb_den)
                    _latest_cv.notify()
        except Exception as e:
            print(f"[decoder] error: {e!r}; reconnecting in 2s", flush=True)
            time.sleep(2)


def detector_loop():
    """Take the most-recent frame from the shared slot, run detection, enqueue
    events. Blocks until a frame is available, then processes it immediately —
    no fixed cadence. Detection rate (fps_proc) is therefore the natural
    `min(camera_fps, 1/detector_latency)`."""
    global _fps_proc_count, _latest_frame
    prev_had_detections = False
    while True:
        with _latest_cv:
            while _latest_frame is None:
                _latest_cv.wait()
            frame, tb_num, tb_den = _latest_frame
            _latest_frame = None

        try:
            img_cam = frame.to_ndarray(format="bgr24")
        except Exception as e:
            print(f"[detector] frame decode failed: {e!r}", flush=True)
            continue
        cam_H, cam_W = img_cam.shape[:2]
        _fps_proc_count += 1

        # Crop in camera coords (DETECT_ROI is camera-frame fractions), then
        # rotate ONLY the inference input. Boxes come back in rotated-crop
        # pixel coords; we un-rotate to crop-local camera coords and then
        # offset by the crop origin to get full-frame camera coords.
        if DETECT_ROI_IS_FULL:
            x0 = y0 = 0
            crop_cam = img_cam
        else:
            x0 = int(DETECT_ROI[0] * cam_W); y0 = int(DETECT_ROI[1] * cam_H)
            x1 = int(DETECT_ROI[2] * cam_W); y1 = int(DETECT_ROI[3] * cam_H)
            crop_cam = img_cam[y0:y1, x0:x1]

        if not crop_cam.size:
            local_boxes = []
            rot_W = rot_H = 0
        else:
            if _ROT90_K:
                img_inf = np.ascontiguousarray(np.rot90(crop_cam, k=_ROT90_K))
            else:
                img_inf = crop_cam
            rot_H, rot_W = img_inf.shape[:2]
            local_boxes = detector.detect(img_inf)

        boxes = []
        for b in local_boxes:
            ux, uy, uw, uh = _unrotate_box_to_camera(
                b["x"], b["y"], b["w"], b["h"], rot_W, rot_H,
            )
            boxes.append({**b, "x": ux + x0, "y": uy + y0, "w": uw, "h": uh})
        if IGNORE_REGIONS:
            boxes = [b for b in boxes if not _box_in_ignore_region(b, cam_W, cam_H)]

        wall_ms = int(time.time() * 1000)
        pts = int(frame.pts)
        media_t = pts * tb_num / tb_den
        base = {
            "wall_ms": wall_ms,
            "pts": pts,
            "tb_num": tb_num, "tb_den": tb_den,
            "media_t": float(media_t),
            "w": cam_W, "h": cam_H,    # camera-frame dims (source of truth)
            # Rotation applied to the inference input for THIS event. Persisted so
            # training/review can re-rotate each crop by its own recorded value —
            # data captured under different (or later-changed) rotate_deg mixes
            # correctly. Boxes/dims above stay camera-orientation.
            "rotate_deg": FRAME_ROTATE_DEG,
            "camera_id": CAMERA_ID,
            "model": detector.model_name,
        }
        food_state, food_level = _maybe_update_food_monitor(img_cam, boxes, cam_W, cam_H)

        if not boxes:
            if prev_had_detections:
                clear_ev = {**base, "cat": None, "boxes": []}
                _attach_food_fields(clear_ev, food_state, food_level)
                try:
                    pending_q.put_nowait((wall_ms + artificial_delay_ms, clear_ev))
                except queue.Full:
                    pass
            prev_had_detections = False
            continue

        prev_had_detections = True

        for b in boxes:
            # ACTION gate operates in camera coords on camera box centers.
            cx = (b["x"] + b["w"] * 0.5) / cam_W
            cy = (b["y"] + b["h"] * 0.5) / cam_H
            in_action = ACTION_ROI_IS_FULL or _point_in_polygon(cx, cy, ACTION_POLYGON)
            ev = {
                **base,
                "cat": b.get("cat"),
                "cat_score": b.get("cat_score"),  # classifier identity confidence (None for blob/yolo)
                "boxes": [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                           "score": b["score"], "in_action": in_action}],
            }
            _attach_food_fields(ev, food_state, food_level)
            try:
                pending_q.put_nowait((wall_ms + artificial_delay_ms, ev))
            except queue.Full:
                pass


# ---- aiohttp ----

clients: "set[web.WebSocketResponse]" = set()
db_conn = None  # initialised in main()


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)
    clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
    finally:
        clients.discard(ws)
    return ws


async def events_handler(request: web.Request) -> web.Response:
    """Query events by wall-clock range. Filters:
       ?from=<ms>&to=<ms>  required
       &camera=<id>        optional, defaults to this detector's CAMERA_ID
       &model=<name>       optional"""
    try:
        t_from = int(request.query["from"])
        t_to   = int(request.query["to"])
    except (KeyError, ValueError):
        return web.json_response({"error": "from/to required (epoch ms)"}, status=400)
    camera = request.query.get("camera", CAMERA_ID)
    model  = request.query.get("model")
    rows = query_events(db_conn, camera_id=camera, t_from=t_from, t_to=t_to, model=model)
    return web.json_response(rows)


async def models_handler(request: web.Request) -> web.Response:
    """List models that have any events for a given camera."""
    camera = request.query.get("camera", CAMERA_ID)
    return web.json_response({"models": list_models(db_conn, camera_id=camera)})


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "clients": len(clients)})


async def config_handler(_request: web.Request) -> web.Response:
    """Runtime config the UI needs at startup, before any WS message arrives.
    The UI uses `camera_id` to construct WHEP + recording playback URLs that
    match mediamtx's path name (single source of truth via $CAMERA_ID)."""
    return web.json_response({
        "camera_id": CAMERA_ID,
        "model":     detector.model_name if detector else None,
    })


# ---- control server ----

async def get_delay(_request: web.Request) -> web.Response:
    return web.json_response({"ms": artificial_delay_ms})


async def set_delay(request: web.Request) -> web.Response:
    global artificial_delay_ms
    try:
        body = await request.json()
        ms = int(body["ms"])
    except (KeyError, ValueError, TypeError):
        return web.json_response({"error": "expected {\"ms\": int}"}, status=400)
    artificial_delay_ms = max(0, min(10_000, ms))
    print(f"[control] artificial_delay_ms = {artificial_delay_ms}", flush=True)
    return web.json_response({"ms": artificial_delay_ms})


# ---- fanout: persist + broadcast ----

async def broadcast_str(line: str) -> None:
    dead = []
    for ws in clients:
        if ws.closed:
            dead.append(ws); continue
        try:
            await ws.send_str(line)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def fanout_task():
    """Pulls from event_q, writes one DB row per box (skipping clear events),
    and broadcasts on WS."""
    loop = asyncio.get_running_loop()
    while True:
        ev = await loop.run_in_executor(None, event_q.get)
        # Persist only events with boxes (clears are transient signals).
        if ev.get("boxes"):
            for b in ev["boxes"]:
                insert_event(
                    db_conn,
                    camera_id=ev["camera_id"], model=ev["model"], wall_ms=ev["wall_ms"],
                    pts=ev.get("pts"), tb_num=ev.get("tb_num"), tb_den=ev.get("tb_den"),
                    media_t=ev.get("media_t"),
                    frame_w=ev["w"], frame_h=ev["h"], rotate_deg=ev.get("rotate_deg"),
                    cat=ev.get("cat"), cat_score=ev.get("cat_score"),
                    box_x=b["x"], box_y=b["y"], box_w=b["w"], box_h=b["h"],
                    score=b.get("score", 0.0),
                    track_id=ev.get("track_id"),
                )
        # Broadcast everything (clears included).
        await broadcast_str(json.dumps(ev, separators=(",", ":")))


async def stats_task():
    """Periodic stats broadcast: input fps + processed fps + model + camera.

    Browser displays this in the header so the user can see live throughput
    even during quiet (no-detection) periods."""
    global _fps_in_count, _fps_proc_count
    last_t = time.time()
    while True:
        await asyncio.sleep(STATS_INTERVAL_SEC)
        now = time.time()
        elapsed = max(1e-6, now - last_t)
        fps_in = _fps_in_count / elapsed
        fps_proc = _fps_proc_count / elapsed
        _fps_in_count = 0
        _fps_proc_count = 0
        last_t = now
        msg = {
            "kind": "stats",
            "camera_id": CAMERA_ID,
            "model": detector.model_name if detector else "(unknown)",
            "fps_in": round(fps_in, 2),
            "fps_processed": round(fps_proc, 2),
            "active_tracks": 0,    # placeholder; tracker integration TBD
            "wall_ms": int(now * 1000),
            # detect_roi: axis-aligned rect [x0,y0,x1,y1] in [0..1].
            # action_polygon: list of [x,y] vertices in [0..1] (closed polygon).
            # UI renders overlays when either is not the full frame.
            # Camera-frame coords (source of truth). UI shows the video in
            # its native (camera) orientation, so these draw directly.
            "detect_roi":     list(DETECT_ROI),
            "action_polygon": [[x, y] for x, y in ACTION_POLYGON],
            "decision_polygon": [[x, y] for x, y in ACTION_POLYGON],
            "food_region": (
                [[x, y] for x, y in FOOD_REGION]
                if FOOD_REGION is not None
                else None
            ),
            "ignore_regions": [
                {
                    "name": region["name"],
                    "points": [[x, y] for x, y in region["points"]],
                }
                for region in IGNORE_REGIONS
            ],
            "ignore_region_min_coverage": IGNORE_REGION_MIN_COVERAGE,
        }
        _attach_food_fields(msg, _food_state, _food_level)
        await broadcast_str(json.dumps(msg, separators=(",", ":")))


# ---- main ----

async def main():
    global detector, db_conn
    from pathlib import Path
    db_conn = init_db(Path(EVENTS_DB))
    detector = build_detector(DETECTOR_TYPE)
    print(f"[detector] camera={CAMERA_ID} type={DETECTOR_TYPE} "
          f"model={detector.model_name} db={EVENTS_DB}", flush=True)

    threading.Thread(target=decoder_loop, daemon=True).start()
    threading.Thread(target=detector_loop, daemon=True).start()
    threading.Thread(target=delay_worker, daemon=True).start()

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/events", events_handler)
    app.router.add_get("/models", models_handler)
    app.router.add_get("/health", health)
    app.router.add_get("/config", config_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WS_PORT)
    await site.start()
    print(f"[server] listening on :{WS_PORT}", flush=True)

    control_app = web.Application()
    control_app.router.add_get("/delay", get_delay)
    control_app.router.add_post("/delay", set_delay)
    control_runner = web.AppRunner(control_app)
    await control_runner.setup()
    control_site = web.TCPSite(control_runner, "0.0.0.0", CONTROL_PORT)
    await control_site.start()
    print(f"[control] listening on :{CONTROL_PORT}", flush=True)

    asyncio.create_task(fanout_task())
    asyncio.create_task(stats_task())
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
