"""Harness: receives every frame of the served cameras from streamhub, decodes
them all (P-frames depend on earlier frames), and runs the model on the newest
decoded frame per camera whenever it's free — frames arriving while the model
is busy are dropped, so nothing queues up. Cameras are served round robin.

max_fps caps CPU use: each camera is inferred at most that many times per
second, so when a pass over all cameras finishes early the loop sleeps.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import av
import numpy as np

from hubclient import HubConnection

from .geometry import prepare
from .models import Model

log = logging.getLogger("cv_worker")


@dataclass
class _Camera:
    decoder: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    frame: av.VideoFrame | None = None  # newest decoded, not yet inferred
    pts: int = 0
    served_at: float = 0.0  # when the model last ran on this camera


class Harness:
    def __init__(self, model: Model, max_fps: float = 0) -> None:
        self.model = model
        # Minimum seconds between two inferences of one camera; 0 = no cap.
        self._min_interval = 1 / max_fps if max_fps > 0 else 0.0
        self._cams: dict[str, _Camera] = {}
        self._cond = threading.Condition()
        self._closed = False

    # ---- one hub session ----

    def session(self, conn: HubConnection) -> None:
        """Run until the connection fails."""
        with self._cond:
            self._cams.clear()
            self._closed = False
        reader = threading.Thread(target=self._read, args=(conn,), name="hub-reader", daemon=True)
        reader.start()
        try:
            self._infer_loop(conn)
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _read(self, conn: HubConnection) -> None:
        try:
            while True:
                msg = conn.recv()
                typ = msg.get("type")
                if typ == "frame":
                    self._on_frame(msg)
                elif typ == "stream":
                    self._on_stream(msg)
        except (OSError, ConnectionError) as e:
            log.info("hub reader stopped: %s", e)
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def _on_stream(self, msg: dict[str, Any]) -> None:
        cam = msg["camera"]
        decoder = av.CodecContext.create("h264", "r")
        with self._cond:
            c = self._cams.setdefault(cam, _Camera())
            c.decoder, c.config, c.frame = decoder, msg.get("config") or {}, None
        log.info("camera %s: %sx%s, config %s", cam, msg.get("width"), msg.get("height"), c.config)

    def _on_frame(self, msg: dict[str, Any]) -> None:
        cam = msg["camera"]
        c = self._cams.get(cam)
        if c is None or c.decoder is None:
            return
        try:
            frames = c.decoder.decode(av.Packet(msg["data"]))
        except av.error.FFmpegError as e:
            log.warning("camera %s: decode error at pts %s: %s", cam, msg["pts"], e)
            return
        if frames:
            # No B-frames: each packet yields its own picture.
            with self._cond:
                c.frame, c.pts = frames[-1], msg["pts"]
                self._cond.notify_all()

    # ---- inference ----

    def _next(self) -> tuple[str, _Camera, av.VideoFrame, int] | None:
        with self._cond:
            while True:
                if self._closed:
                    return None
                now = time.monotonic()
                fresh = [(c.served_at, cam) for cam, c in self._cams.items() if c.frame is not None]
                due = [(at, cam) for at, cam in fresh if now - at >= self._min_interval]
                if due:
                    _, cam = min(due)
                    c = self._cams[cam]
                    frame, pts, c.frame = c.frame, c.pts, None
                    c.served_at = now
                    return cam, c, frame, pts
                # Nothing fresh: wait for a frame. Fresh but capped: wait until
                # the earliest camera is due (a newer frame may replace it meanwhile).
                timeout = min(at for at, _ in fresh) + self._min_interval - now if fresh else None
                self._cond.wait(timeout)

    def _infer_loop(self, conn: HubConnection) -> None:
        while (item := self._next()) is not None:
            cam, c, frame, pts = item
            t0 = time.perf_counter()
            dets = self.infer(frame.to_ndarray(format="bgr24"), c.config)
            infer_ms = (time.perf_counter() - t0) * 1000
            conn.send({"type": "result", "camera": cam, "pts": pts, "infer_ms": round(infer_ms, 2), "dets": dets})

    def infer(self, img_bgr: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Run the model on a camera frame; boxes as fractions of the frame."""
        inp, to_camera = prepare(img_bgr, config)
        out = []
        for d in self.model.infer(np.ascontiguousarray(inp)):
            det: dict[str, Any] = {"box": [round(v, 5) for v in to_camera(d.box)], "score": round(d.score, 4)}
            if d.cats:
                det["cats"] = {k: round(v, 5) for k, v in d.cats.items()}
            out.append(det)
        return out
