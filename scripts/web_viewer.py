"""MJPEG web viewer for live_detect.py.

Called from live_detect via --web flag. Runs a FastAPI/uvicorn server in a
daemon thread and exposes annotated frames over MJPEG without re-running inference.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

_frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
_stats: dict = {}
_stats_lock = threading.Lock()
_recent_detections: list[dict] = []
_detections_lock = threading.Lock()
_MAX_RECENT = 20

_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Cat Detection Live</title>
<style>
  body{margin:0;background:#111;display:flex;flex-direction:column;align-items:center;
       padding:20px;font-family:monospace;color:#eee}
  h1{font-size:18px;margin-bottom:12px}
  img{max-width:960px;width:100%;border:1px solid #333}
  pre{margin-top:12px;font-size:13px;background:#1a1a1a;padding:12px;border-radius:4px;min-width:320px}
</style>
</head>
<body>
<h1>Cat Detection</h1>
<img src="/stream.mjpg" />
<pre id="stats">Loading…</pre>
<script>
setInterval(function(){
  fetch('/stats.json').then(function(r){return r.json();}).then(function(d){
    document.getElementById('stats').textContent = JSON.stringify(d, null, 2);
  }).catch(function(){});
}, 1000);
</script>
</body>
</html>"""


def push_frame(frame_bgr) -> None:
    """Encode frame as JPEG and drop into the queue (non-blocking, drops if full)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return
    try:
        _frame_queue.put_nowait(buf.tobytes())
    except queue.Full:
        pass


def update_stats(**kwargs) -> None:
    with _stats_lock:
        _stats.update(kwargs)


def record_detection(track_id: int, cat_name: str, confidence: float) -> None:
    entry = {
        "track_id": track_id,
        "cat_name": cat_name,
        "confidence": round(confidence, 4),
        "ts": time.time(),
    }
    with _detections_lock:
        _recent_detections.append(entry)
        if len(_recent_detections) > _MAX_RECENT:
            _recent_detections.pop(0)


def create_app(start_time: float):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _INDEX_HTML

    @app.get("/stream.mjpg")
    async def stream():
        async def _gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                try:
                    data = _frame_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                yield boundary + data + b"\r\n"

        return StreamingResponse(
            _gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True})

    @app.get("/stats.json")
    async def stats_endpoint():
        with _stats_lock:
            data = dict(_stats)
        with _detections_lock:
            data["last_detections"] = list(_recent_detections[-10:])
        data["uptime_s"] = round(time.time() - start_time, 1)
        return JSONResponse(data)

    return app


def start_server(port: int, start_time: float) -> None:
    """Block the calling thread running uvicorn. Intended to run in a daemon thread."""
    import uvicorn

    uvicorn.run(create_app(start_time), host="0.0.0.0", port=port, log_level="warning")
