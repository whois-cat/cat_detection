"""Lightweight food-level monitor for a fixed bowl region.

This is intentionally not ML. The primary signal is RELATIVE darkness: how much
darker the bowl is than the ring of frame around it (food is dark, an empty tray
is glossy/bright). Using a contrast against the local surroundings — rather than
absolute brightness — keeps the measure stable across cameras and day/night IR.
Medians are used everywhere so a glare spot or a paw barely moves the number.

Texture (`food_fill_fraction`) is kept as an optional secondary signal, blended
in only when `texture_weight > 0`.
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


def _bbox_px(poly: list[tuple[float, float]], w: int, h: int) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = max(0, min(w, int(np.floor(min(xs) * w))))
    y0 = max(0, min(h, int(np.floor(min(ys) * h))))
    x1 = max(0, min(w, int(np.ceil(max(xs) * w))))
    y1 = max(0, min(h, int(np.ceil(max(ys) * h))))
    return x0, y0, x1, y1


def _polygon_mask_px(
    poly: list[tuple[float, float]],
    w: int,
    h: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray | None:
    """Boolean mask over the bbox [y0:y1, x0:x1] for pixels whose center lies
    inside `poly`. `poly` is normalized; the grid is in absolute pixels.

    Vectorized even/odd ray-cast: each polygon edge toggles the inside flag for
    pixels whose horizontal ray to the left crosses it (parity via XOR)."""
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    px = [(x * w, y * h) for x, y in poly]
    ys = (np.arange(y0, y1) + 0.5)[:, None]   # (bh, 1)
    xs = (np.arange(x0, x1) + 0.5)[None, :]   # (1, bw)
    inside = np.zeros((bh, bw), dtype=bool)
    n = len(px)
    j = n - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            xi, yi = px[i]
            xj, yj = px[j]
            cond_y = (yi > ys) != (yj > ys)                  # (bh, 1)
            xint = (xj - xi) * (ys - yi) / (yj - yi) + xi     # (bh, 1)
            inside ^= cond_y & (xs < xint)
            j = i
    return inside


def bowl_contrast(frame_gray: np.ndarray, roi, margin_frac: float) -> float | None:
    """Relative darkness of the bowl vs. its surrounding ring.

    Returns (ref_val - bowl_val), where both are MEDIAN brightness — of the
    pixels inside the ROI polygon, and of a ring of width
    `margin_frac * min(bbox_w, bbox_h)` around the ROI bbox, respectively.
    Positive when the bowl is darker than its surroundings (= food present).
    numpy-only; medians keep glare and a stray paw from dominating.
    """
    if frame_gray is None or frame_gray.size == 0:
        return None
    if frame_gray.ndim == 3:
        frame_gray = frame_gray.mean(axis=2)

    poly = _polygon_from_roi(roi)
    if len(poly) < 3:
        return None

    h, w = frame_gray.shape[:2]
    x0, y0, x1, y1 = _bbox_px(poly, w, h)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = _polygon_mask_px(poly, w, h, x0, y0, x1, y1)
    if mask is None or not mask.any():
        return None
    bowl_region = frame_gray[y0:y1, x0:x1]
    bowl_val = float(np.median(bowl_region[mask]))

    bw = x1 - x0
    bh = y1 - y0
    margin = int(round(max(0.0, margin_frac) * min(bw, bh)))
    if margin <= 0:
        return None
    ex0 = max(0, x0 - margin)
    ey0 = max(0, y0 - margin)
    ex1 = min(w, x1 + margin)
    ey1 = min(h, y1 + margin)
    ring_region = frame_gray[ey0:ey1, ex0:ex1]
    ring_mask = np.ones(ring_region.shape, dtype=bool)
    # Punch out the original bbox; the ring is the expanded box minus it.
    ring_mask[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    ring_pixels = ring_region[ring_mask]
    if ring_pixels.size == 0:
        return None
    ref_val = float(np.median(ring_pixels))
    return ref_val - bowl_val


def _luma(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3:
        return bgr.astype(np.float32)
    return (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1]
            + 0.299 * bgr[..., 2]).astype(np.float32)


def bowl_contrast_bgr(frame_bgr: np.ndarray, roi, margin_frac: float) -> float | None:
    """ROI-only equivalent of bowl_contrast: converts ONLY the expanded ROI
    rectangle to grayscale, never the whole frame. Numerically identical to
    bowl_contrast(luma(frame_bgr), ...) but allocates O(ROI) instead of O(frame).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    poly = _polygon_from_roi(roi)
    if len(poly) < 3:
        return None
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _bbox_px(poly, w, h)
    if x1 <= x0 or y1 <= y0:
        return None
    bw = x1 - x0
    bh = y1 - y0
    margin = int(round(max(0.0, margin_frac) * min(bw, bh)))
    if margin <= 0:
        return None
    ex0 = max(0, x0 - margin)
    ey0 = max(0, y0 - margin)
    ex1 = min(w, x1 + margin)
    ey1 = min(h, y1 + margin)
    sub_gray = _luma(frame_bgr[ey0:ey1, ex0:ex1])   # grayscale of ROI only

    mask = _polygon_mask_px(poly, w, h, x0, y0, x1, y1)
    if mask is None or not mask.any():
        return None
    bowl_local = sub_gray[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
    bowl_val = float(np.median(bowl_local[mask]))

    ring_mask = np.ones(sub_gray.shape, dtype=bool)
    ring_mask[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    ring_pixels = sub_gray[ring_mask]
    if ring_pixels.size == 0:
        return None
    ref_val = float(np.median(ring_pixels))
    return ref_val - bowl_val


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
        empty_level: float | None,
        full_level: float | None,
        empty_below: float,
        full_above: float,
        window: int,
        *,
        margin_frac: float = 0.35,
        texture_weight: float = 0.0,
        tiles: int = 8,
        tex_thresh: float = 12.0,
    ) -> None:
        self.roi = roi
        # Per-camera calibration anchors for raw contrast. When either is unset
        # the monitor runs in calibration mode (state stays "unknown").
        self.empty_level = None if empty_level is None else float(empty_level)
        self.full_level = None if full_level is None else float(full_level)
        self.empty_below = float(empty_below)
        self.full_above = float(full_above)
        self.window = max(1, int(window))
        self.margin_frac = float(margin_frac)
        self.texture_weight = float(texture_weight)
        self.tiles = max(1, int(tiles))
        self.tex_thresh = float(tex_thresh)
        self.state = "unknown"
        self._recent: deque[float] = deque(maxlen=self.window)

    def update(
        self, frame_bgr: np.ndarray, cat_in_region: bool
    ) -> tuple[str, float | None, float | None]:
        # Takes the BGR frame and converts ONLY the ROI to grayscale (no
        # full-frame float32). Don't measure through a cat sitting over the bowl.
        if cat_in_region:
            return self.state, None, None

        raw = bowl_contrast_bgr(frame_bgr, self.roi, self.margin_frac)
        if raw is None:
            return self.state, None, None

        if self.empty_level is None or self.full_level is None:
            # Not calibrated yet: surface raw so the operator can record the
            # empty/full reference numbers for this camera.
            return "unknown", None, raw

        span = self.full_level - self.empty_level
        if span == 0:
            return self.state, None, raw
        fill_b = (raw - self.empty_level) / span
        fill_b = min(1.0, max(0.0, fill_b))

        if self.texture_weight > 0:
            # Texture is an optional secondary signal; only here do we pay for a
            # full-frame grayscale. TODO: remap the ROI to a sub-frame to keep
            # this ROI-only as well when texture_weight is enabled.
            tex = food_fill_fraction(_luma(frame_bgr), self.roi, self.tiles, self.tex_thresh)
            w = self.texture_weight
            fill = fill_b if tex is None else (1.0 - w) * fill_b + w * float(tex)
        else:
            fill = fill_b

        self._recent.append(float(fill))
        smoothed = float(median(self._recent))
        if smoothed <= self.empty_below:
            self.state = "empty"
        elif smoothed >= self.full_above:
            self.state = "has_food"
        return self.state, smoothed, raw
