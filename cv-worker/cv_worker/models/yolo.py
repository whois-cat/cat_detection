"""YOLO cat detector with an optional per-cat identity classifier."""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import Det

COCO_CAT = 15


def identity_crop_box(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int,
                      pad_frac: float) -> tuple[int, int, int, int]:
    """Expand a box by pad_frac * max(w, h) on every side, clamped to the frame.
    Same geometry as training's crop padding, so the classifier sees the framing
    it was trained on."""
    pad = int(pad_frac * max(x2 - x1, y2 - y1))
    return max(0, x1 - pad), max(0, y1 - pad), min(frame_w, x2 + pad), min(frame_h, y2 + pad)


class YoloModel:
    def __init__(self, weights: str, conf: float = 0.25, classifier_dir: str | None = None,
                 pad_frac: float = 0.05) -> None:
        from ultralytics import YOLO

        # task='detect' because OpenVINO export dirs carry no task metadata.
        self._yolo = YOLO(weights, task="detect")
        self.conf = conf
        self.pad_frac = pad_frac
        stem = os.path.splitext(os.path.basename(os.path.normpath(weights)))[0]
        self._classifier = None
        self.name, self.version = stem, "0"
        if classifier_dir:
            from .classifier import CatClassifier
            self._classifier = CatClassifier(classifier_dir)
            self.name, self.version = f"{stem}+cat", self._classifier.version

    def infer(self, img_bgr: np.ndarray) -> list[Det]:
        h, w = img_bgr.shape[:2]
        boxes = []
        for r in self._yolo(img_bgr, classes=[COCO_CAT], conf=self.conf, verbose=False):
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].cpu().numpy())
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        cats: list[dict[str, float]] = [{} for _ in boxes]
        if self._classifier and boxes:
            crops = []
            for x1, y1, x2, y2, _ in boxes:
                cx0, cy0, cx1, cy1 = identity_crop_box(x1, y1, x2, y2, w, h, self.pad_frac)
                crops.append(cv2.cvtColor(img_bgr[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
            cats = self._classifier.probs(crops)
        return [Det(box=(x1, y1, x2 - x1, y2 - y1), score=s, cats=c)
                for (x1, y1, x2, y2, s), c in zip(boxes, cats)]
