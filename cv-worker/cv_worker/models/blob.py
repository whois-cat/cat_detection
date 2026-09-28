"""Development model: bright blobs (e.g. a flashlight) are "cats". No identity."""
from __future__ import annotations

import cv2
import numpy as np

from . import Det


class BlobModel:
    name = "blob"
    version = "1"

    def __init__(self, threshold: int = 240, min_area: int = 500) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def infer(self, img_bgr: np.ndarray, config: dict | None = None) -> list[Det]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                out.append(Det(box=(x, y, w, h), score=min(1.0, area / (10 * self.min_area))))
        return out
