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

import time
from datetime import datetime, timezone
from pathlib import Path

import click

from pipeline_db import PROJECT_ROOT, connect_db, make_uid


COCO_CAT_CLASS_ID = 15
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_CLASSIFIER = PROJECT_ROOT / "data" / "models" / "cat_classifier_best.pt"
DEFAULT_INTERVAL = 1.0
DEFAULT_THRESHOLD = 0.6
DEFAULT_YOLO_CONFIDENCE = 0.3
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
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
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
@click.option("--webhook", type=str, default=None, help="POST JSON to this URL on match.")
@click.option("--show", is_flag=True, help="Show video window with bboxes.")
@click.option("--save-log", is_flag=True, help="Append detections to DuckDB.")
def main(
    source: str,
    model: Path,
    yolo: str,
    interval: float,
    threshold: float,
    webhook: str | None,
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
    show: bool = False,
    save_log: bool = False,
) -> None:
    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO

    device = select_device()
    click.echo(f"live: device={device}")

    detector = YOLO(yolo_model)
    detector.to(str(device))

    classifier, class_names = load_classifier(model_path, device)
    transform = build_classifier_transform()
    click.echo(f"live: classes={class_names}")

    resolved_source = resolve_source(source)
    capture = cv2.VideoCapture(resolved_source)
    if not capture.isOpened():
        raise click.ClickException(f"failed to open source: {source}")

    connection = connect_db() if save_log else None

    click.echo(
        f"live: source={source} interval={interval}s threshold={threshold} "
        f"webhook={'on' if webhook else 'off'} show={show} save_log={save_log}"
    )

    last_check = 0.0

    try:
        while True:
            grabbed, frame_bgr = capture.read()
            if not grabbed:
                click.echo("live: failed to read frame, stopping")
                break

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

                payload = {
                    "timestamp": timestamp.isoformat(),
                    "cat_name": top_name,
                    "confidence": top_confidence,
                    "source": source,
                }

                if webhook:
                    post_webhook(webhook, payload)

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
    except KeyboardInterrupt:
        click.echo("live: interrupted")
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
