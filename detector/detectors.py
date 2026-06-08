"""Detector implementations.

Each `Detector.detect(img_bgr)` returns a list of detection dicts:
    {"x": int, "y": int, "w": int, "h": int, "score": float,
     "cat": str | None, "cat_score": float | None}

To add a new detector:
  1. Subclass `Detector`, set `model_name`, implement `detect`.
  2. Register it in `build_detector()`.
  3. Set `DETECTOR_TYPE=<key>` env var to select it.

`cat` is the per-detection identity label:
  - blob/yolo: class name ('cat') or None — no per-cat identity.
  - yolo_cat:  EfficientNet-B0 classifier name, or "unknown" when confidence
               is below CLASSIFIER_MIN_CONF.
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


class YoloCatDetector(Detector):
    """YOLO + per-box EfficientNet-B0 identity classifier.

    Crops each detected cat from the same frame YOLO saw (img_bgr, inference
    coords) and classifies it. Sets b["cat"] to the classifier name, or
    "unknown" when confidence < CLASSIFIER_MIN_CONF.
    """

    KEEP_CLS_IDS = (15,)

    def __init__(
        self,
        weights: str = "/opt/models/yolov8n_int8_openvino_model/",
        conf: float = 0.25,
        classifier_dir: str = "/opt/models/cat_classifier_openvino/",
        min_conf: float = 0.5,
        pad_frac: float = 0.15,
    ) -> None:
        import cv2
        from classifier import CatClassifier
        from ultralytics import YOLO

        self._cv2 = cv2
        self.model = YOLO(weights, task="detect")
        self.conf = conf
        stem = os.path.splitext(os.path.basename(os.path.normpath(weights)))[0]
        self.model_name = f"{stem}+cat"
        self._classifier = CatClassifier(classifier_dir)
        self._min_conf = min_conf
        # Context padding around the box BEFORE classification. MUST match the
        # training crop padding (training._pad_crop / build_review_manifest
        # --pad-frac / train_classifier --pad-frac) or the classifier sees a
        # different framing at serve time than it was trained/reviewed on.
        self._pad_frac = pad_frac

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        cv2 = self._cv2
        results = self.model(
            img_bgr,
            classes=list(self.KEEP_CLS_IDS),
            conf=self.conf,
            verbose=False,
        )
        h, w = img_bgr.shape[:2]
        out: list[dict] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                # Expand the box by pad_frac (same clamp as training._pad_crop)
                # so the classifier sees the SAME framing it was trained on, then
                # crop from the inference-coord frame YOLO saw (img_bgr). The
                # emitted box stays the tight detection box — only the classifier
                # input is padded.
                pad = int(self._pad_frac * max(x2 - x1, y2 - y1))
                cx0, cy0 = max(0, x1 - pad), max(0, y1 - pad)
                cx1, cy1 = min(w, x2 + pad), min(h, y2 + pad)
                crop_bgr = img_bgr[cy0:cy1, cx0:cx1]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                cat_name, cat_score = self._classifier.classify(crop_rgb)
                if cat_score < self._min_conf:
                    cat_name = "unknown"
                out.append({
                    "x": x1, "y": y1,
                    "w": x2 - x1, "h": y2 - y1,
                    "score": float(box.conf[0]),
                    "cat": cat_name,
                    "cat_score": float(cat_score),
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
    if detector_type == "yolo_cat":
        return YoloCatDetector(
            weights=os.environ.get("YOLO_WEIGHTS", "/opt/models/yolov8n_int8_openvino_model/"),
            conf=float(os.environ.get("YOLO_CONF", "0.25")),
            classifier_dir=os.environ.get(
                "CLASSIFIER_WEIGHTS", "/opt/models/cat_classifier_openvino/"
            ),
            min_conf=float(os.environ.get("CLASSIFIER_MIN_CONF", "0.5")),
            pad_frac=float(os.environ.get("CLASSIFIER_PAD_FRAC", "0.15")),
        )
    raise ValueError(
        f"unknown DETECTOR_TYPE: {detector_type!r} "
        f"(expected one of: blob, yolo, yolo_cat)"
    )
