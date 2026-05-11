#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "httpx>=0.27.0",
#   "numpy>=2.2.0",
#   "opencv-python>=4.10.0",
#   "pillow>=11.1.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "ultralytics>=8.3.0",
# ]
# ///

from __future__ import annotations

import os
import signal
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click

from pipeline_db import DEFAULT_MODEL_PATH, connect_db, make_uid
from metrics import push_metric


COCO_CAT_CLASS_ID = 15
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_CLASSIFIER = DEFAULT_MODEL_PATH
DEFAULT_THRESHOLD = float(os.environ.get("LIVE_THRESHOLD", "0.6"))
DEFAULT_YOLO_CONFIDENCE = 0.3
DEFAULT_VM_URL = os.environ.get("VM_URL", "http://victoriametrics:8428")
DEFAULT_FEEDER_API_URL: str | None = os.environ.get("FEEDER_API_URL") or None
DEFAULT_DOOR_CLOSE_TIMEOUT = int(os.environ.get("DOOR_CLOSE_TIMEOUT_SEC", "30"))
DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_MAJORITY = 4
DEFAULT_INFERENCE_FPS = int(os.environ.get("INFERENCE_FPS", "5"))
DEFAULT_RECLASSIFY_EVERY = int(os.environ.get("RECLASSIFY_EVERY", "30"))
DEFAULT_MAX_MISSING_FRAMES = 15
DEFAULT_WEB_PORT = int(os.environ.get("WEB_PORT") or "0")
DEFAULT_SNAPSHOT_RETENTION_DAYS = int(os.environ.get("MULTI_CAT_SNAPSHOT_RETENTION_DAYS", "30"))
DEFAULT_ALLOWED_CATS: str | None = os.environ.get("FEEDER_ALLOWED_CATS") or None
DEFAULT_CROP_SAVE_COOLDOWN = int(os.environ.get("CROP_SAVE_COOLDOWN", "10"))
DEFAULT_CROP_RETENTION_DAYS = int(os.environ.get("CROP_RETENTION_DAYS", "60"))
MODEL_RELOAD_INTERVAL_SECONDS = 3600
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEEDER_BLOCKED_DEFAULT = {"felisis"}


@dataclass
class TrackState:
    window: deque
    missing_frames: int = 0
    frames_since_classify: int = 0
    last_label: str | None = None
    last_conf: float = 0.0


class FpsCounter:
    def __init__(self, window_secs: float = 1.0):
        self._times: deque = deque()
        self._window = window_secs

    def tick(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class FeederClient:
    """Thin synchronous client for the feeder REST API."""

    def __init__(self, base_url: str, vm_url: str) -> None:
        import httpx

        self._base = base_url.rstrip("/")
        self._vm_url = vm_url
        self._state = "closed"
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=3.0)

    def _call(self, path: str) -> bool:
        """POST to path with one retry on connection error or 5xx."""
        for attempt in range(2):
            try:
                resp = self._http.post(f"{self._base}{path}")
                if resp.is_success:
                    return True
                if 500 <= resp.status_code < 600 and attempt == 0:
                    continue
                click.echo(f"feeder: POST {path} → {resp.status_code}")
                return False
            except Exception as exc:
                if attempt == 0:
                    continue
                click.echo(f"feeder: POST {path} error: {exc}")
                return False
        return False

    def force_closed(self) -> None:
        """Force door closed on startup to guarantee a known starting state."""
        click.echo("[door] startup: forcing closed state")
        ok = self._call("/api/door/close")
        push_metric("cat_detection_door_close_calls_total", 1.0, {"result": "ok" if ok else "error"}, self._vm_url)
        if ok:
            push_metric("cat_detection_door_state", 0.0, {}, self._vm_url)
        else:
            click.echo("[door] startup: force-close failed, assuming closed")
        # Local state stays "closed" regardless — it's now authoritative.

    def set_door(self, desired: str, reason: str) -> None:
        """Call open/close endpoint; push metrics; update local state only on success."""
        prev = self.state
        path = "/api/door/open" if desired == "open" else "/api/door/close"
        metric = (
            "cat_detection_door_open_calls_total"
            if desired == "open"
            else "cat_detection_door_close_calls_total"
        )
        ok = self._call(path)
        push_metric(metric, 1.0, {"result": "ok" if ok else "error"}, self._vm_url)
        if ok:
            with self._lock:
                self._state = desired
            push_metric(
                "cat_detection_door_state",
                1.0 if desired == "open" else 0.0,
                {},
                self._vm_url,
            )
            click.echo(f"[door] state: {prev} → {desired} (reason: {reason})")
        else:
            click.echo(f"[door] {desired} failed — will retry next tick")

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def close(self) -> None:
        self._http.close()


def select_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_classifier(model_path: Path, device):
    import torch
    from torchvision import models

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]
    classifier = models.efficientnet_b0(weights=None)
    in_features = classifier.classifier[1].in_features
    classifier.classifier[1] = torch.nn.Linear(in_features, num_classes)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(device)
    classifier.eval()
    return classifier, class_names


def build_classifier_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def resolve_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def is_cat_allowed(cat: str, allowed_cats: set[str] | None) -> bool:
    if allowed_cats is not None:
        return cat.lower() in allowed_cats
    return cat.lower() not in FEEDER_BLOCKED_DEFAULT


def classify_crop(crop_rgb, classifier, transform, device, class_names: list[str]) -> tuple[str, float]:
    import torch
    from PIL import Image

    pil_crop = Image.fromarray(crop_rgb)
    tensor = transform(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1).squeeze(0)
    idx = int(probs.argmax().item())
    return class_names[idx], float(probs[idx].item())


def cleanup_old_snapshots(snapshot_root: Path, retention_days: int) -> None:
    if not snapshot_root.exists():
        return
    cutoff = time.time() - retention_days * 86400
    for date_dir in snapshot_root.iterdir():
        if not date_dir.is_dir():
            continue
        for jpg in date_dir.glob("*.jpg"):
            if jpg.stat().st_mtime < cutoff:
                jpg.unlink(missing_ok=True)
        try:
            date_dir.rmdir()
        except OSError:
            pass


def cleanup_old_crops(crop_dir: Path, retention_days: int) -> None:
    if not crop_dir.exists():
        return
    cutoff = time.time() - retention_days * 86400
    for jpg in crop_dir.glob("*.jpg"):
        if jpg.stat().st_mtime < cutoff:
            jpg.unlink(missing_ok=True)


def save_multi_cat_snapshot(frame_bgr, cats: list[str], snapshot_root: Path) -> None:
    import cv2

    now = datetime.now()
    dest_dir = snapshot_root / now.strftime("%Y-%m-%d")
    dest_dir.mkdir(parents=True, exist_ok=True)
    cats_str = "_".join(sorted(c for c in cats if c))
    dest = dest_dir / f"{now.strftime('%H%M%S')}_{cats_str}.jpg"
    cv2.imwrite(str(dest), frame_bgr)


def draw_overlay(
    frame,
    display_fps: float,
    inference_fps: float,
    n_tracks: int,
    is_multi_cat: bool,
    blocked_labels: list[str],
) -> None:
    import cv2

    lines = [
        f"Display FPS:  {display_fps:5.1f}",
        f"Inference FPS:{inference_fps:5.1f}",
        f"Tracks:       {n_tracks:5d}",
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    if is_multi_cat:
        cv2.putText(frame, "MULTI-CAT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, "MULTI-CAT", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y += 28
    for bl in blocked_labels:
        cv2.putText(frame, f"BLOCKED:{bl}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"BLOCKED:{bl}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        y += 24


@click.command()
@click.option("--source", required=True, help="RTSP URL or camera device index.")
@click.option(
    "--model",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=DEFAULT_CLASSIFIER,
    show_default=True,
    help="Path to the trained classifier checkpoint.",
)
@click.option("--yolo", type=str, default=DEFAULT_YOLO_MODEL, show_default=True, help="YOLO model name or path.")
@click.option(
    "--inference-fps",
    type=click.IntRange(min=1),
    default=DEFAULT_INFERENCE_FPS,
    show_default=True,
    help="Inference loop rate (detector + tracker + classifier ticks/sec). Env: INFERENCE_FPS",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum classifier confidence to accept a prediction.",
)
@click.option(
    "--feeder-api-url",
    type=str,
    default=DEFAULT_FEEDER_API_URL,
    help="Base URL of the feeder REST API (e.g. http://192.168.0.50:8000). Env: FEEDER_API_URL",
)
@click.option(
    "--door-close-timeout",
    type=click.IntRange(min=1),
    default=DEFAULT_DOOR_CLOSE_TIMEOUT,
    show_default=True,
    help="Seconds with no cat in frame before closing the door. Env: DOOR_CLOSE_TIMEOUT_SEC",
)
@click.option("--vm-url", type=str, default=DEFAULT_VM_URL, show_default=True, help="VictoriaMetrics push URL.")
@click.option("--window-size", type=click.IntRange(min=1), default=DEFAULT_WINDOW_SIZE, show_default=True)
@click.option("--window-majority", type=click.IntRange(min=1), default=DEFAULT_WINDOW_MAJORITY, show_default=True)
@click.option(
    "--reclassify-every",
    type=click.IntRange(min=1),
    default=DEFAULT_RECLASSIFY_EVERY,
    show_default=True,
    help="Re-run classifier on a track every N inference frames. Env: RECLASSIFY_EVERY",
)
@click.option(
    "--max-missing-frames",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_MISSING_FRAMES,
    show_default=True,
    help="Drop a track after this many consecutive inference frames with no detection.",
)
@click.option(
    "--allowed-cats",
    type=str,
    default=DEFAULT_ALLOWED_CATS,
    help="Comma-separated cat allowlist for the feeder. Default: all except felisis. Env: FEEDER_ALLOWED_CATS",
)
@click.option(
    "--snapshot-dir",
    type=click.Path(path_type=Path),
    default=Path("data/multi_cat_snapshots"),
    show_default=True,
    help="Root directory for multi-cat snapshots.",
)
@click.option(
    "--snapshot-retention-days",
    type=click.IntRange(min=1),
    default=DEFAULT_SNAPSHOT_RETENTION_DAYS,
    show_default=True,
    help="Delete multi-cat snapshots older than this many days. Env: MULTI_CAT_SNAPSHOT_RETENTION_DAYS",
)
@click.option(
    "--crop-dir",
    type=click.Path(path_type=Path),
    default=Path("data/crops/unsorted"),
    show_default=True,
    help="Directory where live detection crops are saved for retraining.",
)
@click.option(
    "--crop-save-cooldown",
    type=click.IntRange(min=0),
    default=DEFAULT_CROP_SAVE_COOLDOWN,
    show_default=True,
    help="Seconds between saved crops for the same track (known cats). Env: CROP_SAVE_COOLDOWN",
)
@click.option(
    "--crop-retention-days",
    type=click.IntRange(min=1),
    default=DEFAULT_CROP_RETENTION_DAYS,
    show_default=True,
    help="Delete saved crops older than this many days. Env: CROP_RETENTION_DAYS",
)
@click.option("--show", is_flag=True, help="Show video window with bboxes (updates at inference rate).")
@click.option("--save-log", is_flag=True, help="Append detections to DuckDB.")
@click.option(
    "--web-port",
    type=int,
    default=DEFAULT_WEB_PORT,
    show_default=True,
    help="Port for the MJPEG web viewer (0 or unset = disabled). Env: WEB_PORT",
)
def main(
    source,
    model,
    yolo,
    inference_fps,
    threshold,
    feeder_api_url,
    door_close_timeout,
    vm_url,
    window_size,
    window_majority,
    reclassify_every,
    max_missing_frames,
    allowed_cats,
    snapshot_dir,
    snapshot_retention_days,
    crop_dir,
    crop_save_cooldown,
    crop_retention_days,
    show,
    save_log,
    web_port,
):
    """Live cat detection from camera with ByteTrack multi-cat tracking."""
    run_live_detect(
        source=source,
        model_path=model,
        yolo_model=yolo,
        inference_fps=inference_fps,
        threshold=threshold,
        feeder_api_url=feeder_api_url,
        door_close_timeout=door_close_timeout,
        vm_url=vm_url,
        window_size=window_size,
        window_majority=window_majority,
        reclassify_every=reclassify_every,
        max_missing_frames=max_missing_frames,
        allowed_cats=allowed_cats,
        snapshot_dir=snapshot_dir,
        snapshot_retention_days=snapshot_retention_days,
        crop_dir=crop_dir,
        crop_save_cooldown=crop_save_cooldown,
        crop_retention_days=crop_retention_days,
        show=show,
        save_log=save_log,
        web_port=web_port,
    )


def run_live_detect(
    source: str,
    model_path: Path = DEFAULT_CLASSIFIER,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    inference_fps: int = DEFAULT_INFERENCE_FPS,
    threshold: float = DEFAULT_THRESHOLD,
    feeder_api_url: str | None = None,
    door_close_timeout: int = DEFAULT_DOOR_CLOSE_TIMEOUT,
    vm_url: str = DEFAULT_VM_URL,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_majority: int = DEFAULT_WINDOW_MAJORITY,
    reclassify_every: int = DEFAULT_RECLASSIFY_EVERY,
    max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
    allowed_cats: str | None = None,
    snapshot_dir: Path = Path("data/multi_cat_snapshots"),
    snapshot_retention_days: int = DEFAULT_SNAPSHOT_RETENTION_DAYS,
    crop_dir: Path = Path("data/crops/unsorted"),
    crop_save_cooldown: int = DEFAULT_CROP_SAVE_COOLDOWN,
    crop_retention_days: int = DEFAULT_CROP_RETENTION_DAYS,
    show: bool = False,
    save_log: bool = False,
    web_port: int = DEFAULT_WEB_PORT,
) -> None:
    import cv2
    from ultralytics import YOLO

    start_time = time.time()
    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        _shutdown = True
        click.echo("live: shutdown signal received")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    allowed_set: set[str] | None = (
        {c.strip().lower() for c in allowed_cats.split(",") if c.strip()}
        if allowed_cats
        else None
    )

    device = select_device()
    click.echo(f"live: device={device}")

    detector = YOLO(yolo_model)
    detector.to(str(device))

    model_path = Path(model_path)
    classifier, class_names = load_classifier(model_path, device)
    transform = build_classifier_transform()
    click.echo(f"live: classes={class_names}")

    model_mtime = model_path.stat().st_mtime if model_path.exists() else 0.0
    last_reload_wall = time.monotonic()

    snapshot_dir = Path(snapshot_dir)
    cleanup_old_snapshots(snapshot_dir, snapshot_retention_days)

    crop_dir = Path(crop_dir)
    cleanup_old_crops(crop_dir, crop_retention_days)

    resolved_source = resolve_source(source)
    capture = cv2.VideoCapture(resolved_source)
    if not capture.isOpened():
        raise click.ClickException(f"failed to open source: {source}")

    connection = connect_db() if save_log else None

    click.echo(
        f"live: source={source} inference_fps={inference_fps} threshold={threshold} "
        f"window={window_size}/{window_majority} "
        f"reclassify_every={reclassify_every} max_missing={max_missing_frames} "
        f"feeder={'on' if feeder_api_url else 'off'} web={'on' if web_port > 0 else 'off'}"
    )

    # Feeder client
    feeder: FeederClient | None = None
    if feeder_api_url:
        feeder = FeederClient(feeder_api_url, vm_url)
        feeder.force_closed()
        click.echo(f"live: feeder at {feeder_api_url} initial_state={feeder.state}")
    else:
        click.echo("live: feeder integration disabled")

    # Web server
    if web_port > 0:
        import web_viewer as wv

        threading.Thread(
            target=wv.start_server, args=(web_port, start_time), daemon=True
        ).start()
        click.echo(f"live: web viewer at http://0.0.0.0:{web_port}/")

    # Capture thread
    _latest: dict = {"frame": None}
    _frame_lock = threading.Lock()
    _capture_done = threading.Event()
    display_fps = FpsCounter()

    def _capture_worker():
        while not _shutdown and not _capture_done.is_set():
            grabbed, frame_bgr = capture.read()
            if not grabbed:
                _capture_done.set()
                break
            with _frame_lock:
                _latest["frame"] = frame_bgr
            display_fps.tick()

    threading.Thread(target=_capture_worker, daemon=True).start()

    track_states: dict[int, TrackState] = {}
    crop_saved_at: dict[int, float] = {}
    last_active_time = time.monotonic()
    _last_noop_desired: str | None = None  # suppress repeated no-op log lines

    inference_fps_counter = FpsCounter()
    inference_period = 1.0 / inference_fps

    try:
        while not _shutdown and not _capture_done.is_set():
            loop_start = time.monotonic()

            with _frame_lock:
                frame_bgr = _latest["frame"]

            if frame_bgr is None:
                time.sleep(0.01)
                continue

            # Model hot-reload every 60 minutes if mtime changed
            if loop_start - last_reload_wall >= MODEL_RELOAD_INTERVAL_SECONDS and model_path.exists():
                new_mtime = model_path.stat().st_mtime
                if new_mtime != model_mtime:
                    click.echo(f"live: reloading model from {model_path}")
                    classifier, class_names = load_classifier(model_path, device)
                    transform = build_classifier_transform()
                    model_mtime = new_mtime
                    click.echo(f"live: reloaded classes={class_names}")
                last_reload_wall = loop_start

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated = frame_bgr.copy()
            timestamp = datetime.now(timezone.utc)

            results = detector.track(
                source=frame_rgb,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=DEFAULT_YOLO_CONFIDENCE,
                classes=[COCO_CAT_CLASS_ID],
                device=str(device),
            )

            result = results[0]
            boxes = result.boxes

            active_ids: set[int] = set()
            blocked_in_frame: list[str] = []
            is_multi_cat = False
            frame_has_blocked = False
            frame_allowed_majority: str | None = None
            frame_allowed_majority_conf: float = 0.0

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.tolist()
                id_tensor = boxes.id
                track_ids = id_tensor.int().tolist() if id_tensor is not None else [None] * len(xyxy)
                active_ids = {int(tid) for tid in track_ids if tid is not None}

                frame_height, frame_width = frame_rgb.shape[:2]

                for coords, track_id in zip(xyxy, track_ids):
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = (int(round(v)) for v in coords)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_width, x2), min(frame_height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    state = track_states.setdefault(
                        track_id, TrackState(window=deque(maxlen=window_size))
                    )
                    state.missing_frames = 0

                    needs_classify = (
                        state.last_label is None
                        or state.frames_since_classify >= reclassify_every
                        or state.last_conf < threshold
                    )
                    if needs_classify:
                        crop_rgb = frame_rgb[y1:y2, x1:x2]
                        state.last_label, state.last_conf = classify_crop(
                            crop_rgb, classifier, transform, device, class_names
                        )
                        state.frames_since_classify = 0
                    else:
                        state.frames_since_classify += 1

                    label = state.last_label
                    conf = state.last_conf
                    is_known = conf >= threshold
                    label_text = label if is_known else "unknown"

                    color = (0, 255, 0) if is_known else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"#{track_id} {label_text} {conf:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # Save crop for retraining: unknowns always; known cats with cooldown
                    now_mono_crop = time.monotonic()
                    last_crop = crop_saved_at.get(track_id, 0.0)
                    if not is_known or (now_mono_crop - last_crop >= crop_save_cooldown):
                        crop_bgr = frame_bgr[y1:y2, x1:x2]
                        if crop_bgr.size > 0:
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            cv2.imwrite(str(crop_dir / f"{track_id}_{ts_str}_{label_text}.jpg"), crop_bgr)
                            if is_known:
                                crop_saved_at[track_id] = now_mono_crop

                    if not is_known:
                        continue

                    push_metric("cat_detected", 1.0, {"cat_name": label}, vm_url)
                    push_metric("cat_confidence", conf, {"cat_name": label}, vm_url)

                    if not is_cat_allowed(label, allowed_set):
                        if label not in blocked_in_frame:
                            blocked_in_frame.append(label)
                        frame_has_blocked = True

                    if save_log and connection is not None:
                        det_uid = make_uid("live", timestamp.isoformat(), label, round(conf, 4), source)
                        connection.execute(
                            """
                            INSERT INTO live_detections (detection_uid, timestamp, cat_name, confidence, source)
                            VALUES (?, ?, ?, ?, ?) ON CONFLICT (detection_uid) DO NOTHING
                            """,
                            [det_uid, timestamp, label, conf, source],
                        )

                    state.window.append((label, conf))

                    if len(state.window) < window_size:
                        continue

                    counts = Counter(n for n, _ in state.window)
                    majority_cat, majority_count = counts.most_common(1)[0]
                    if majority_count < window_majority:
                        continue

                    entries = [(c, n) for n, c in state.window if n == majority_cat]
                    mean_conf = sum(c for c, _ in entries) / len(entries)
                    if mean_conf < threshold:
                        continue

                    if web_port > 0:
                        wv.record_detection(track_id, majority_cat, mean_conf)

                    # Log majority events; door decisions are made after the per-track loop
                    if len(active_ids) > 1:
                        if not is_multi_cat:
                            is_multi_cat = True
                            cats = sorted(
                                s.last_label
                                for s in track_states.values()
                                if s.last_label and s.missing_frames == 0
                            )
                            push_metric("cat_detection_multi_cat_events_total", 1.0, {}, vm_url)
                            click.echo(f"live: multi_cat cats={cats}")
                            save_multi_cat_snapshot(annotated, cats, snapshot_dir)
                    elif not is_cat_allowed(majority_cat, allowed_set):
                        push_metric("cat_detection_blocked_events_total", 1.0, {"cat": majority_cat}, vm_url)
                        click.echo(f"live: blocked cat={majority_cat}")
                    else:
                        frame_allowed_majority = majority_cat
                        frame_allowed_majority_conf = mean_conf
                        click.echo(f"live: cat={majority_cat} conf={mean_conf:.4f}")

            # Track when we last had any detection
            if active_ids:
                last_active_time = loop_start

            # Age out missing tracks
            for tid in list(track_states.keys()):
                if tid not in active_ids:
                    track_states[tid].missing_frames += 1
                    if track_states[tid].missing_frames > max_missing_frames:
                        del track_states[tid]

            # ── Door state machine ──────────────────────────────────────────────
            if feeder is not None:
                desired: str | None = None
                reason = ""
                if len(active_ids) > 1:
                    desired = "closed"
                    reason = "multi_cat"
                elif frame_has_blocked:
                    desired = "closed"
                    reason = f"blocked:{','.join(blocked_in_frame)}"
                elif frame_allowed_majority is not None:
                    desired = "open"
                    reason = f"{frame_allowed_majority} conf={frame_allowed_majority_conf:.2f}"
                elif not active_ids and (loop_start - last_active_time) >= door_close_timeout:
                    desired = "closed"
                    reason = f"no_tracks for {loop_start - last_active_time:.1f}s"

                if desired is not None:
                    if desired != feeder.state:
                        feeder.set_door(desired, reason)
                    elif desired != _last_noop_desired:
                        click.echo(f"[door] no-op: already {desired}")
                    _last_noop_desired = desired
                else:
                    _last_noop_desired = None

                push_metric(
                    "cat_detection_door_state",
                    1.0 if feeder.state == "open" else 0.0,
                    {},
                    vm_url,
                )

            inference_fps_counter.tick()
            d_fps = display_fps.fps
            i_fps = inference_fps_counter.fps
            n_tracks = len(active_ids)

            draw_overlay(annotated, d_fps, i_fps, n_tracks, is_multi_cat, blocked_in_frame)

            push_metric("cat_detection_display_fps", d_fps, {}, vm_url)
            push_metric("cat_detection_inference_fps", i_fps, {}, vm_url)

            if web_port > 0:
                wv.push_frame(annotated)
                wv.update_stats(
                    display_fps=round(d_fps, 1),
                    inference_fps=round(i_fps, 1),
                    active_tracks=n_tracks,
                )

            if show:
                cv2.imshow("live_detect", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.monotonic() - loop_start
            sleep_time = inference_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        click.echo("live: shutting down")
        _capture_done.set()
        capture.release()
        if feeder is not None:
            feeder.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
