#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "numpy>=2.2.0",
#   "opencv-python>=4.10.0",
#   "pillow>=11.1.0",
#   "requests>=2.32.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
#   "ultralytics>=8.3.0",
# ]
# ///

from __future__ import annotations

import os
import signal
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import click

from pipeline_db import DEFAULT_MODEL_PATH, connect_db, make_uid
from metrics import push_metric


COCO_CAT_CLASS_ID = 15
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_CLASSIFIER = DEFAULT_MODEL_PATH
DEFAULT_INTERVAL = float(os.environ.get("LIVE_INTERVAL", "1.0"))
DEFAULT_THRESHOLD = float(os.environ.get("LIVE_THRESHOLD", "0.6"))
DEFAULT_YOLO_CONFIDENCE = 0.3
DEFAULT_VM_URL = os.environ.get("VM_URL", "http://victoriametrics:8428")
DEFAULT_WEBHOOK: str | None = os.environ.get("WEBHOOK_URL") or None
DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_MAJORITY = 4
DEFAULT_COOLDOWN = 30
MODEL_RELOAD_INTERVAL_SECONDS = 3600
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    if source.isdigit():
        return int(source)
    return source


def post_webhook(url: str, payload: dict) -> None:
    import requests

    try:
        requests.post(url, json=payload, timeout=5)
    except requests.RequestException as exc:
        click.echo(f"live: webhook error: {exc}")


@click.command()
@click.option("--source", required=True, help="RTSP URL or camera device index.")
@click.option(
    "--model",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=DEFAULT_CLASSIFIER,
    show_default=True,
    help="Path to the trained classifier checkpoint.",
)
@click.option(
    "--yolo",
    type=str,
    default=DEFAULT_YOLO_MODEL,
    show_default=True,
    help="YOLO model name or path.",
)
@click.option(
    "--interval",
    type=click.FloatRange(min=0.0),
    default=DEFAULT_INTERVAL,
    show_default=True,
    help="Seconds between detection checks.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum classifier confidence to accept a prediction.",
)
@click.option("--webhook", type=str, default=DEFAULT_WEBHOOK, help="POST JSON to this URL on match (default: $WEBHOOK_URL).")
@click.option(
    "--vm-url",
    type=str,
    default=DEFAULT_VM_URL,
    show_default=True,
    help="VictoriaMetrics push URL.",
)
@click.option(
    "--window-size",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW_SIZE,
    show_default=True,
    help="Number of recent predictions kept in the sliding window.",
)
@click.option(
    "--window-majority",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW_MAJORITY,
    show_default=True,
    help="Minimum predictions for the same cat to trigger a webhook.",
)
@click.option(
    "--cooldown",
    type=click.IntRange(min=0),
    default=DEFAULT_COOLDOWN,
    show_default=True,
    help="Seconds before re-sending a webhook for the same cat.",
)
@click.option("--show", is_flag=True, help="Show video window with bboxes.")
@click.option("--save-log", is_flag=True, help="Append detections to DuckDB.")
def main(
    source: str,
    model: Path,
    yolo: str,
    interval: float,
    threshold: float,
    webhook: str | None,
    vm_url: str,
    window_size: int,
    window_majority: int,
    cooldown: int,
    show: bool,
    save_log: bool,
) -> None:
    """Live cat detection from camera."""

    run_live_detect(
        source=source,
        model_path=model,
        yolo_model=yolo,
        interval=interval,
        threshold=threshold,
        webhook=webhook,
        vm_url=vm_url,
        window_size=window_size,
        window_majority=window_majority,
        cooldown=cooldown,
        show=show,
        save_log=save_log,
    )


def run_live_detect(
    source: str,
    model_path: Path = DEFAULT_CLASSIFIER,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    interval: float = DEFAULT_INTERVAL,
    threshold: float = DEFAULT_THRESHOLD,
    webhook: str | None = None,
    vm_url: str = DEFAULT_VM_URL,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_majority: int = DEFAULT_WINDOW_MAJORITY,
    cooldown: int = DEFAULT_COOLDOWN,
    show: bool = False,
    save_log: bool = False,
) -> None:
    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO

    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        _shutdown = True
        click.echo("live: shutdown signal received")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

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

    resolved_source = resolve_source(source)
    capture = cv2.VideoCapture(resolved_source)
    if not capture.isOpened():
        raise click.ClickException(f"failed to open source: {source}")

    connection = connect_db() if save_log else None

    click.echo(
        f"live: source={source} interval={interval}s threshold={threshold} "
        f"window={window_size}/{window_majority} cooldown={cooldown}s "
        f"webhook={'on' if webhook else 'off'} vm_url={vm_url} show={show} save_log={save_log}"
    )

    # Sliding window state: (cat_name, confidence, x1, y1, x2, y2)
    window: deque[tuple[str, float, int, int, int, int]] = deque(maxlen=window_size)
    cooldown_until: dict[str, float] = {}
    last_fired_cat: str | None = None

    last_check = 0.0

    try:
        while not _shutdown:
            grabbed, frame_bgr = capture.read()
            if not grabbed:
                click.echo("live: failed to read frame, stopping")
                break

            # Model hot-reload every 60 minutes if mtime changed
            now_wall = time.monotonic()
            if now_wall - last_reload_wall >= MODEL_RELOAD_INTERVAL_SECONDS and model_path.exists():
                new_mtime = model_path.stat().st_mtime
                if new_mtime != model_mtime:
                    click.echo(f"live: reloading model from {model_path}")
                    classifier, class_names = load_classifier(model_path, device)
                    transform = build_classifier_transform()
                    model_mtime = new_mtime
                    click.echo(f"live: reloaded classes={class_names}")
                last_reload_wall = now_wall

            now = time.perf_counter()
            if now - last_check < interval:
                if show:
                    cv2.imshow("live_detect", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue
            last_check = now

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = detector.predict(
                source=frame_rgb,
                verbose=False,
                conf=DEFAULT_YOLO_CONFIDENCE,
                classes=[COCO_CAT_CLASS_ID],
                device=str(device),
            )

            result = results[0]
            boxes = result.boxes
            timestamp = datetime.now(timezone.utc)

            if boxes is None or len(boxes) == 0:
                if show:
                    cv2.imshow("live_detect", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            xyxy = boxes.xyxy.tolist()
            frame_height, frame_width = frame_rgb.shape[:2]

            for coords in xyxy:
                x1, y1, x2, y2 = (int(round(v)) for v in coords)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_width, x2)
                y2 = min(frame_height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop_rgb = frame_rgb[y1:y2, x1:x2]
                pil_crop = Image.fromarray(crop_rgb)
                input_tensor = transform(pil_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0)

                top_index = int(probabilities.argmax().item())
                top_confidence = float(probabilities[top_index].item())
                top_name = class_names[top_index]

                label_text = top_name if top_confidence >= threshold else "unknown"

                if show:
                    color = (0, 255, 0) if top_confidence >= threshold else (0, 165, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_bgr,
                        f"{label_text} {top_confidence:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                if top_confidence < threshold:
                    continue

                click.echo(
                    f"live: {timestamp.isoformat()} {top_name} "
                    f"confidence={top_confidence:.4f}"
                )

                # Per-frame metrics — unchanged
                push_metric("cat_detected", 1.0, {"cat_name": top_name}, vm_url)
                push_metric("cat_confidence", top_confidence, {"cat_name": top_name}, vm_url)

                # Sliding window
                window.append((top_name, top_confidence, x1, y1, x2, y2))

                if len(window) == window_size:
                    counts = Counter(name for name, *_ in window)
                    majority_cat, majority_count = counts.most_common(1)[0]
                    if majority_count >= window_majority:
                        cat_entries = [
                            (conf, bx1, by1, bx2, by2)
                            for name, conf, bx1, by1, bx2, by2 in window
                            if name == majority_cat
                        ]
                        mean_conf = sum(c for c, *_ in cat_entries) / len(cat_entries)
                        if mean_conf >= threshold:
                            now_mono = time.monotonic()
                            if majority_cat != last_fired_cat:
                                cooldown_until[majority_cat] = 0.0
                            if now_mono >= cooldown_until.get(majority_cat, 0.0):
                                _, lx1, ly1, lx2, ly2 = cat_entries[-1]
                                model_version = model_path.stat().st_mtime if model_path.exists() else 0
                                payload = {
                                    "timestamp": timestamp.isoformat(),
                                    "cat_name": majority_cat,
                                    "confidence": round(mean_conf, 4),
                                    "model_version": str(int(model_version)),
                                    "bbox": {"x": lx1, "y": ly1, "w": lx2 - lx1, "h": ly2 - ly1},
                                }
                                if webhook:
                                    post_webhook(webhook, payload)
                                cooldown_until[majority_cat] = now_mono + cooldown
                                last_fired_cat = majority_cat

                if connection is not None:
                    detection_uid = make_uid(
                        "live",
                        timestamp.isoformat(),
                        top_name,
                        round(top_confidence, 4),
                        source,
                    )
                    connection.execute(
                        """
                        INSERT INTO live_detections (
                            detection_uid, timestamp, cat_name, confidence, source
                        )
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (detection_uid) DO NOTHING
                        """,
                        [detection_uid, timestamp, top_name, top_confidence, source],
                    )

            if show:
                cv2.imshow("live_detect", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        click.echo("live: shutting down")
        capture.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
