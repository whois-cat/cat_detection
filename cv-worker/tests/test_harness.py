import socket
import struct
import threading
import time
from pathlib import Path

import msgpack

import hubclient
from cv_worker.harness import Harness
from cv_worker.models import Det

TINY = Path(__file__).parents[2] / "streamhub/testdata/tiny.h264"  # 64x64, 45 frames, IDR every 10


def access_units(data: bytes) -> list[bytes]:
    """Split Annex-B into access units at access unit delimiters (type 9)."""
    nals = [n.rstrip(b"\x00") for n in data.split(b"\x00\x00\x01") if n.rstrip(b"\x00")]
    aus: list[list[bytes]] = []
    for n in nals:
        if n[0] & 0x1F == 9:
            aus.append([])
        else:
            aus[-1].append(n)
    return [b"".join(b"\x00\x00\x00\x01" + n for n in au) for au in aus]


class SlowModel:
    name, version = "stub", "1"

    def __init__(self):
        self.shapes = []

    def infer(self, img):
        self.shapes.append(img.shape)
        time.sleep(0.05)  # slower than frames arrive
        return [Det(box=(0, 0, 16, 32), score=0.9, cats={"a": 0.7, "b": 0.3})]


def send(conn, msg):
    body = msgpack.packb(msg, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(body)) + body)


def run_session(model, frames_per_camera, cameras=("grey",), frame_gap=0.01, **harness_kw):
    """Stream frames of the tiny clip to a Harness through a fake hub; return results."""
    aus = access_units(TINY.read_bytes())[:frames_per_camera]
    server = socket.create_server(("127.0.0.1", 0))
    results = []

    def hub():
        conn, _ = server.accept()
        f = conn.makefile("rb")

        def read():
            (n,) = struct.unpack(">I", f.read(4))
            return msgpack.unpackb(f.read(n), raw=False)

        assert read()["type"] == "hello"
        for cam in cameras:
            send(conn, {"type": "stream", "camera": cam, "width": 64, "height": 64, "config": {}})
        for i, au in enumerate(aus):
            for cam in cameras:
                send(conn, {"type": "frame", "camera": cam, "pts": 1000 + i, "key": i % 10 == 0, "data": au})
            time.sleep(frame_gap)
        conn.settimeout(1.5)
        try:
            while True:
                m = read()
                if m["type"] == "result":
                    results.append(m)
        except (OSError, struct.error):
            pass
        conn.close()

    t = threading.Thread(target=hub)
    t.start()
    started = time.monotonic()
    with hubclient.HubConnection(f"127.0.0.1:{server.getsockname()[1]}", {"role": "cv"}) as conn:
        Harness(model, **harness_kw).session(conn)
    t.join()
    return results, time.monotonic() - started


class FastModel:
    name, version = "stub", "1"

    def infer(self, img):
        return []


def test_max_fps_caps_each_camera():
    # 45 frames per camera over ~2.3 s, model instant: uncapped it would infer
    # almost every frame; capped at 4/s it may do ~4 per second per camera.
    results, _ = run_session(FastModel(), 45, cameras=("grey", "beige"), frame_gap=0.05, max_fps=4)
    for cam in ("grey", "beige"):
        pts = [r["pts"] for r in results if r["camera"] == cam]
        assert 5 <= len(pts) <= 12, (cam, len(pts))
        assert pts[-1] == 1044, "the newest frame must still be inferred after the cap"


def test_session_drops_stale_frames_and_maps_boxes():
    aus = access_units(TINY.read_bytes())
    assert len(aus) == 45
    server = socket.create_server(("127.0.0.1", 0))
    results = []

    def hub():
        conn, _ = server.accept()
        f = conn.makefile("rb")

        def read():
            (n,) = struct.unpack(">I", f.read(4))
            return msgpack.unpackb(f.read(n), raw=False)

        assert read()["type"] == "hello"
        send(conn, {"type": "stream", "camera": "grey", "width": 64, "height": 64, "config": {"rotate_deg": 90}})
        for i, au in enumerate(aus):
            send(conn, {"type": "frame", "camera": "grey", "pts": 1000 + i, "key": i % 10 == 0, "data": au})
            time.sleep(0.01)
        deadline = time.monotonic() + 3
        conn.settimeout(0.5)
        while time.monotonic() < deadline:
            try:
                m = read()
            except (OSError, struct.error):
                break
            if m["type"] == "result":
                results.append(m)
                if m["pts"] == 1044:
                    break
        conn.close()

    t = threading.Thread(target=hub)
    t.start()
    model = SlowModel()
    with hubclient.HubConnection(f"127.0.0.1:{server.getsockname()[1]}", {"role": "cv"}) as conn:
        Harness(model).session(conn)
    t.join()

    pts = [r["pts"] for r in results]
    assert pts and pts == sorted(pts) and len(set(pts)) == len(pts)
    assert all(1000 <= p <= 1044 for p in pts)
    assert len(pts) < 45 / 2, "stale frames were not dropped"
    assert pts[-1] == 1044, "newest frame was not inferred"
    assert model.shapes[0] == (64, 64, 3)
    # Rotating clockwise moves the camera's bottom-left corner to the input's
    # top-left, so a 16x32 box there is x 0..0.5, y 0.75..1 of the camera frame.
    assert results[0]["dets"] == [{"box": [0.0, 0.75, 0.5, 0.25], "score": 0.9, "cats": {"a": 0.7, "b": 0.3}}]
    assert results[0]["infer_ms"] >= 50
