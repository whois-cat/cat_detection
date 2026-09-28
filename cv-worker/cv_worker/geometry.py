"""Model-input preparation and box mapping.

The camera frame is the source of truth: boxes leave the worker as fractions
of the full camera frame in camera orientation. Rotation (for cameras mounted
sideways; detectors aren't rotation-invariant) and the detection area are
applied only to the model's input.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Box = tuple[float, float, float, float]  # x, y, w, h


def prepare(img: np.ndarray, config: dict[str, Any] | None = None) -> tuple[np.ndarray, Callable[[Box], Box]]:
    """Crop the detection area (`detect_roi`: [x0, y0, x1, y1] fractions) and
    rotate it clockwise by `rotate_deg` (0/90/180/270).

    Returns the model input and a function mapping a pixel box in that input to
    a box in fractions of the full camera frame."""
    config = config or {}
    rotate = int(config.get("rotate_deg") or 0)
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate_deg must be 0/90/180/270, got {rotate}")
    rx0, ry0, rx1, ry1 = config.get("detect_roi") or (0, 0, 1, 1)
    H, W = img.shape[:2]
    x0, y0 = int(rx0 * W), int(ry0 * H)
    x1, y1 = max(x0 + 1, int(rx1 * W)), max(y0 + 1, int(ry1 * H))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    # np.rot90 turns counter-clockwise for positive k.
    inp = np.rot90(crop, (-rotate // 90) % 4)

    def unrotate(xr: float, yr: float) -> tuple[float, float]:
        if rotate == 90:
            return yr, ch - xr
        if rotate == 180:
            return cw - xr, ch - yr
        if rotate == 270:
            return cw - yr, xr
        return xr, yr

    def to_camera(box: Box) -> Box:
        bx, by, bw, bh = box
        (ax, ay), (bx2, by2) = unrotate(bx, by), unrotate(bx + bw, by + bh)
        left, right = sorted((ax, bx2))
        top, bottom = sorted((ay, by2))
        left, right = max(0.0, (left + x0) / W), min(1.0, (right + x0) / W)
        top, bottom = max(0.0, (top + y0) / H), min(1.0, (bottom + y0) / H)
        return left, top, max(0.0, right - left), max(0.0, bottom - top)

    return inp, to_camera
