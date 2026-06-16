"""Lightweight food-level monitor for a fixed bowl region.

This is intentionally not ML. It estimates how much of a camera-normalized ROI
looks textured by splitting it into tiles and measuring mean gradient magnitude.
"""
from __future__ import annotations

from collections import deque
from statistics import median

import numpy as np


def _polygon_from_roi(roi) -> list[tuple[float, float]]:
    if roi is None:
        return []
    if isinstance(roi, (list, tuple)) and len(roi) == 4 and all(
        isinstance(v, (int, float)) for v in roi
    ):
        x0, y0, x1, y1 = (float(v) for v in roi)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if isinstance(roi, (list, tuple)) and roi and all(
        isinstance(v, (list, tuple)) for v in roi
    ):
        return [(float(x), float(y)) for x, y in roi]
    if isinstance(roi, (list, tuple)) and len(roi) >= 6 and len(roi) % 2 == 0:
        vals = [float(v) for v in roi]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    raise ValueError("food roi must be rect x0,y0,x1,y1 or a polygon")


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xc:
                inside = not inside
        j = i
    return inside


def food_fill_fraction(
    frame_gray: np.ndarray,
    roi,
    tiles: int,
    tex_thresh: float,
) -> float | None:
    """Return fraction of textured tiles in the food ROI.

    `roi` is in normalized camera/UI coordinates. Rects and polygons are both
    accepted. For polygons, only tiles whose center falls inside the polygon
    participate in the fraction.
    """
    if frame_gray is None or frame_gray.size == 0:
        return None
    if frame_gray.ndim == 3:
        frame_gray = frame_gray.mean(axis=2)

    poly = _polygon_from_roi(roi)
    if len(poly) < 3:
        return None

    tiles = max(1, int(tiles))
    h, w = frame_gray.shape[:2]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = max(0, min(w, int(np.floor(min(xs) * w))))
    y0 = max(0, min(h, int(np.floor(min(ys) * h))))
    x1 = max(0, min(w, int(np.ceil(max(xs) * w))))
    y1 = max(0, min(h, int(np.ceil(max(ys) * h))))
    if x1 <= x0 or y1 <= y0:
        return None

    textured = 0
    valid = 0
    for ty in range(tiles):
        ya = y0 + (y1 - y0) * ty // tiles
        yb = y0 + (y1 - y0) * (ty + 1) // tiles
        for tx in range(tiles):
            xa = x0 + (x1 - x0) * tx // tiles
            xb = x0 + (x1 - x0) * (tx + 1) // tiles
            if xb - xa < 2 or yb - ya < 2:
                continue
            cx = ((xa + xb) * 0.5) / max(1, w)
            cy = ((ya + yb) * 0.5) / max(1, h)
            if not _point_in_polygon(cx, cy, poly):
                continue
            tile = frame_gray[ya:yb, xa:xb].astype(np.float32, copy=False)
            gy, gx = np.gradient(tile)
            texture = np.sqrt(gx * gx + gy * gy).mean()
            valid += 1
            if float(texture) > tex_thresh:
                textured += 1

    if valid == 0:
        return None
    return textured / valid


class BowlMonitor:
    def __init__(
        self,
        roi,
        empty_below: float,
        full_above: float,
        window: int,
        *,
        tiles: int = 8,
        tex_thresh: float = 12.0,
    ) -> None:
        self.roi = roi
        self.empty_below = float(empty_below)
        self.full_above = float(full_above)
        self.window = max(1, int(window))
        self.tiles = max(1, int(tiles))
        self.tex_thresh = float(tex_thresh)
        self.state = "unknown"
        self._recent: deque[float] = deque(maxlen=self.window)

    def update(self, frame_gray: np.ndarray, cat_in_region: bool) -> tuple[str, float | None]:
        if cat_in_region:
            return self.state, None

        fraction = food_fill_fraction(
            frame_gray,
            self.roi,
            self.tiles,
            self.tex_thresh,
        )
        if fraction is None:
            return self.state, None

        self._recent.append(float(fraction))
        smoothed = float(median(self._recent))
        if smoothed <= self.empty_below:
            self.state = "empty"
        elif smoothed >= self.full_above:
            self.state = "has_food"
        return self.state, smoothed
