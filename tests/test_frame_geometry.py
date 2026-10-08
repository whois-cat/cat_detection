"""Image rotation and box projection must agree with each other and the runtime.

The strong check is end-to-end: paint a block at a known camera-frame box, apply
the saved-image geometry, and confirm the projected box lands on the same pixels
in the model-input image.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.frame_geometry import (
    apply_frame_geometry,
    camera_box_to_model,
    rotate_image,
)

W, H = 60, 40  # camera frame (width, height)


def _nonzero_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(image)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _painted_frame(px0: int, py0: int, px1: int, py1: int) -> tuple[np.ndarray, tuple]:
    image = np.zeros((H, W), dtype=np.uint8)
    image[py0:py1, px0:px1] = 255
    box = (px0 / W, py0 / H, (px1 - px0) / W, (py1 - py0) / H)
    return image, box


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_projected_box_matches_rotated_pixels(rotate: int) -> None:
    image, box = _painted_frame(12, 8, 30, 24)
    model_image = apply_frame_geometry(image, rotate, None)
    projected = camera_box_to_model(box, W, H, rotate, None)
    assert projected is not None

    ih, iw = model_image.shape[:2]
    mx, my, mw, mh = projected
    expected = (mx * iw, my * ih, (mx + mw) * iw, (my + mh) * ih)
    actual = _nonzero_bbox(model_image)
    for exp, act in zip(expected, actual):
        assert abs(exp - act) <= 1, (rotate, expected, actual)


def test_rotate_90_swaps_dimensions() -> None:
    image = np.zeros((H, W, 3), dtype=np.uint8)
    rotated = rotate_image(image, 90)
    assert rotated.shape[:2] == (W, H)


def test_identity_box_when_no_geometry() -> None:
    _, box = _painted_frame(12, 8, 30, 24)
    assert camera_box_to_model(box, W, H, 0, None) == pytest.approx(box)


def test_roi_crop_then_rotate_stays_consistent() -> None:
    roi = [0.25, 0.0, 1.0, 1.0]  # drop the left quarter
    image, box = _painted_frame(20, 8, 44, 24)  # fully inside the ROI (x >= 15)
    model_image = apply_frame_geometry(image, 90, roi)
    projected = camera_box_to_model(box, W, H, 90, roi)
    assert projected is not None

    ih, iw = model_image.shape[:2]
    mx, my, mw, mh = projected
    expected = (mx * iw, my * ih, (mx + mw) * iw, (my + mh) * ih)
    actual = _nonzero_bbox(model_image)
    for exp, act in zip(expected, actual):
        assert abs(exp - act) <= 1, (expected, actual)


def test_box_outside_roi_is_dropped() -> None:
    roi = [0.5, 0.0, 1.0, 1.0]
    box = (0.0, 0.0, 0.4, 1.0)  # entirely left of the ROI
    assert camera_box_to_model(box, W, H, 0, roi) is None
