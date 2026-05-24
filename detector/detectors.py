"""Detector implementations.

Each `Detector.detect(img_bgr)` returns a list of detection dicts:
    {"x": int, "y": int, "w": int, "h": int, "score": float, "cat": str | None}

To add a new detector:
  1. Subclass `Detector`, set `model_name`, implement `detect`.
  2. Register it in `build_detector()`.
  3. Set `DETECTOR_TYPE=<key>` env var to select it.

`cat` is the per-detection identity label. For models that can't tell cats
apart (out-of-the-box YOLO), it'll be the class name ('cat') or None. The
sticky-session pseudo-classifier in `BrightBlobDetector` is a dev placeholder.
"""
from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        ...


class BrightBlobDetector(Detector):
    """Dummy detector: thresholds bright pixels, returns connected components.
    Useful for development with a flashlight without bothering real cats.

    Includes a sticky-session pseudo-classifier — each detection session (from
    first hit to next gap) is randomly assigned ONE cat label, so the timeline
    density looks like one cat per visit. Reset on no-detection."""

    model_name = "blob-dummy"
    KNOWN_CATS = ["alisa", "chuzh", "ellie", "felisis"]

    def __init__(self, threshold: int = 240, min_area: int = 500):
        import cv2
        self._cv2 = cv2
        self.threshold = threshold
        self.min_area = min_area
        self._session_cat: str | None = None

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        cv2 = self._cv2
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                            "score": float(area), "cat": None})

        # Sticky session-cat: assign all blobs in a session to one random cat.
        if out:
            if self._session_cat is None:
                self._session_cat = random.choice(self.KNOWN_CATS)
            for b in out:
                b["cat"] = self._session_cat
        else:
            self._session_cat = None
        return out


class YoloDetector(Detector):
    """Ultralytics YOLO. Default COCO weights detect the generic 'cat' class.
    For per-cat identification you'll need a downstream classifier or a
    fine-tuned model — until then `cat` is just the COCO class name."""

    # COCO class IDs we keep. 15=cat, 16=dog (in case the user wants either).
    KEEP_CLS_IDS = (15,)

    def __init__(self, weights: str = "/opt/models/yolov8n_int8_openvino_model/", conf: float = 0.25):
        from ultralytics import YOLO
        # `task='detect'` silences "Unable to guess model task" for OpenVINO dirs
        # (which carry no task metadata in their YAML).
        self.model = YOLO(weights, task="detect")
        self.conf = conf
        # Clean stem from either a .pt file or an OpenVINO model directory.
        stem = os.path.basename(os.path.normpath(weights))
        stem = os.path.splitext(stem)[0]
        self.model_name = stem

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        results = self.model(
            img_bgr,
            classes=list(self.KEEP_CLS_IDS),
            conf=self.conf,
            verbose=False,
        )
        out: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                out.append({
                    "x": int(x1), "y": int(y1),
                    "w": int(x2 - x1), "h": int(y2 - y1),
                    "score": float(box.conf[0]),
                    "cat": cls_name,
                })
        return out


def build_detector(detector_type: str) -> Detector:
    if detector_type == "blob":
        return BrightBlobDetector(
            threshold=int(os.environ.get("BLOB_BRIGHT_THRESHOLD", "240")),
            min_area=int(os.environ.get("BLOB_MIN_AREA", "500")),
        )
    if detector_type == "yolo":
        return YoloDetector(
            weights=os.environ.get("YOLO_WEIGHTS", "yolov8n.pt"),
            conf=float(os.environ.get("YOLO_CONF", "0.25")),
        )
    raise ValueError(f"unknown DETECTOR_TYPE: {detector_type!r} "
                     f"(expected one of: blob, yolo)")
