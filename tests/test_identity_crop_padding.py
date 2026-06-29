"""Identity-crop padding: lower padding shrinks the crop (less background) while
always keeping the full detection box (body + tail), and runtime + training crop
geometry stay identical. No heavy deps (cv2/YOLO/OpenVINO) needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from detectors import identity_crop_box
from training.db import Box
from training.sources import _local_box, _pad_crop

# A box well inside a large frame so padding isn't clamped by the edges.
FRAME_W, FRAME_H = 1000, 1000
BX, BY, BW, BH = 400, 400, 120, 80  # x, y, w, h


def _runtime_dims(pad_frac):
    cx0, cy0, cx1, cy1 = identity_crop_box(
        BX, BY, BX + BW, BY + BH, FRAME_W, FRAME_H, pad_frac)
    return (cx1 - cx0, cy1 - cy0)


# ---- runtime helper --------------------------------------------------------

@pytest.mark.parametrize("pad_frac", [0.15, 0.05, 0.02, 0.0])
def test_runtime_never_cuts_the_box(pad_frac):
    cx0, cy0, cx1, cy1 = identity_crop_box(
        BX, BY, BX + BW, BY + BH, FRAME_W, FRAME_H, pad_frac)
    # The detection box is fully inside the padded crop — never cut.
    assert cx0 <= BX and cy0 <= BY
    assert cx1 >= BX + BW and cy1 >= BY + BH


def test_runtime_zero_padding_is_exactly_the_box():
    assert _runtime_dims(0.0) == (BW, BH)


def test_runtime_lower_padding_shrinks_crop_area():
    areas = [w * h for (w, h) in (_runtime_dims(p) for p in (0.15, 0.05, 0.02, 0.0))]
    # Strictly decreasing as padding drops, bottoming out at the bare box.
    assert areas == sorted(areas, reverse=True)
    assert len(set(areas)) == len(areas)
    assert areas[-1] == BW * BH


def test_runtime_clamps_at_frame_edges_without_cutting_box():
    # Box in the top-left corner: padding clamps to 0 but the box stays whole.
    cx0, cy0, cx1, cy1 = identity_crop_box(0, 0, 100, 100, FRAME_W, FRAME_H, 0.15)
    assert (cx0, cy0) == (0, 0)
    assert cx1 >= 100 and cy1 >= 100


# ---- training crop ---------------------------------------------------------

def _box():
    return Box(x=BX, y=BY, w=BW, h=BH, cat="cat_a",
               score=1.0, track_id=None, rowid=1, cat_score=None)


def _train_crop_area(pad_frac):
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    crop, local = _pad_crop(img, _box(), pad_frac)
    return crop.shape[1] * crop.shape[0], local


def test_training_lower_padding_shrinks_crop_area():
    areas = [_train_crop_area(p)[0] for p in (0.15, 0.05, 0.02, 0.0)]
    assert areas == sorted(areas, reverse=True)
    assert len(set(areas)) == len(areas)


def test_training_keeps_full_box_inside_crop():
    # The crop-local box keeps the full w/h and sits entirely within the crop —
    # the body + tail are never trimmed, whatever the padding.
    for pad_frac in (0.15, 0.05, 0.02, 0.0):
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        crop, local = _pad_crop(img, _box(), pad_frac)
        assert (local.w, local.h) == (BW, BH)
        assert local.x >= 0 and local.y >= 0
        assert local.x + local.w <= crop.shape[1]
        assert local.y + local.h <= crop.shape[0]


# ---- runtime/training consistency ------------------------------------------

@pytest.mark.parametrize("pad_frac", [0.15, 0.05, 0.02, 0.0])
def test_runtime_and_training_crop_dims_match(pad_frac):
    rt_w, rt_h = _runtime_dims(pad_frac)
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    crop, _ = _pad_crop(img, _box(), pad_frac)
    assert (rt_w, rt_h) == (crop.shape[1], crop.shape[0])
    # _local_box (pixel-free geometry) agrees too.
    lb = _local_box(_box(), pad_frac, FRAME_W, FRAME_H)
    assert (lb.w, lb.h) == (BW, BH)
