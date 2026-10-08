"""Camera -> model-input frame geometry, shared offline.

The runtime (``cv-worker/cv_worker/geometry.py``) feeds YOLO a *detection area*
crop (``detect_roi``) rotated clockwise by ``rotate_deg``, and reports boxes back
in fractions of the full camera frame. For a fine-tune to be valid, the offline
images and labels must live in the SAME model-input space the runtime uses.

This module is the single offline source of truth for that transform. It is
numpy-only (no av/torch) so the collector, the dataset-version builder and the
classifier crop path can all import it without pulling heavy dependencies.

Conventions match the runtime exactly:

- ``detect_roi`` is ``[x0, y0, x1, y1]`` in fractions of the camera frame; the
  pixel crop uses the same ``int()`` truncation and ``x1 = max(x0+1, ...)`` guard
  as ``cv_worker.geometry.prepare``.
- Rotation is clockwise by 0/90/180/270. ``np.rot90`` turns counter-clockwise,
  so the matching factor is ``k = (-rotate_deg // 90) % 4`` (same as the runtime
  and ``training.sources.rotate_crop``).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]  # x, y, w, h (fractions)


def rot90_factor(rotate_deg: int | None) -> int:
    """np.rot90 ``k`` for a clockwise rotation of ``rotate_deg`` (0/90/180/270)."""
    rotate = int(rotate_deg or 0)
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate_deg must be 0/90/180/270, got {rotate}")
    return (-rotate // 90) % 4


def rotate_image(img: np.ndarray, rotate_deg: int | None) -> np.ndarray:
    """Rotate an image clockwise by ``rotate_deg``; 0/None is a no-op."""
    k = rot90_factor(rotate_deg)
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def roi_pixels(
    width: int, height: int, detect_roi: Sequence[float] | None
) -> tuple[int, int, int, int]:
    """Pixel crop box ``(x0, y0, x1, y1)`` for ``detect_roi``, matching runtime."""
    rx0, ry0, rx1, ry1 = detect_roi or (0.0, 0.0, 1.0, 1.0)
    x0, y0 = int(rx0 * width), int(ry0 * height)
    x1 = min(width, max(x0 + 1, int(rx1 * width)))
    y1 = min(height, max(y0 + 1, int(ry1 * height)))
    return x0, y0, x1, y1


def apply_frame_geometry(
    img: np.ndarray, rotate_deg: int | None, detect_roi: Sequence[float] | None = None
) -> np.ndarray:
    """Return the model-input image: ``detect_roi`` crop then clockwise rotation.

    Identical framing to what ``cv-worker`` feeds the detector, so an offline
    frame trains/evaluates the model under the same geometry it runs under.
    """
    height, width = img.shape[:2]
    x0, y0, x1, y1 = roi_pixels(width, height, detect_roi)
    crop = img[y0:y1, x0:x1]
    return rotate_image(crop, rotate_deg)


def camera_box_to_model(
    box: Box,
    width: int,
    height: int,
    rotate_deg: int | None,
    detect_roi: Sequence[float] | None = None,
) -> Box | None:
    """Map a camera-frame fractional box into model-input fractional coordinates.

    ``box`` is ``(x, y, w, h)`` in fractions of the full camera frame (the sidecar
    convention). Returns ``(x, y, w, h)`` in fractions of the model input, or
    ``None`` when the box does not overlap the detection area. The box is clipped
    to the ROI; a degenerate (zero-area) overlap returns ``None``.
    """
    rotate = int(rotate_deg or 0)
    x0, y0, x1, y1 = roi_pixels(width, height, detect_roi)
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        return None

    bx, by, bw, bh = box
    # Camera-fraction corners -> crop-pixel corners, clipped to the ROI.
    left = min(max(bx * width - x0, 0.0), crop_w)
    right = min(max((bx + bw) * width - x0, 0.0), crop_w)
    top = min(max(by * height - y0, 0.0), crop_h)
    bottom = min(max((by + bh) * height - y0, 0.0), crop_h)
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None

    def crop_to_input(xc: float, yc: float) -> tuple[float, float]:
        if rotate == 90:
            return crop_h - yc, xc
        if rotate == 180:
            return crop_w - xc, crop_h - yc
        if rotate == 270:
            return yc, crop_w - xc
        return xc, yc

    input_w, input_h = (crop_h, crop_w) if rotate in (90, 270) else (crop_w, crop_h)
    corners = [
        crop_to_input(left, top),
        crop_to_input(right, top),
        crop_to_input(right, bottom),
        crop_to_input(left, bottom),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    mx0, mx1 = min(xs) / input_w, max(xs) / input_w
    my0, my1 = min(ys) / input_h, max(ys) / input_h
    mx0, my0 = max(0.0, mx0), max(0.0, my0)
    mx1, my1 = min(1.0, mx1), min(1.0, my1)
    if mx1 - mx0 <= 0.0 or my1 - my0 <= 0.0:
        return None
    return mx0, my0, mx1 - mx0, my1 - my0
