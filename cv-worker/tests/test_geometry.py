import numpy as np
import pytest

from cv_worker.geometry import prepare


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("roi", [None, (0.1, 0.0, 0.9, 0.8)])
def test_box_maps_back_to_camera(rotate, roi):
    img = np.zeros((60, 100), np.uint8)  # h=60, w=100
    img[5:15, 20:50] = 1                  # x 20..50, y 5..15
    cfg = {"rotate_deg": rotate}
    if roi:
        cfg["detect_roi"] = roi
    inp, to_camera = prepare(img, cfg)
    if not roi:
        assert inp.shape == ((60, 100) if rotate in (0, 180) else (100, 60))
    x, y, w, h = to_camera(bbox(inp))
    assert (x, y, w, h) == pytest.approx((0.2, 5 / 60, 0.3, 10 / 60))


def test_clockwise_rotation():
    img = np.zeros((2, 3), np.uint8)
    img[0, 0] = 1  # top-left
    inp, _ = prepare(img, {"rotate_deg": 90})
    assert inp.shape == (3, 2) and inp[0, 1] == 1  # rotated CW: top-left goes top-right


def test_bad_rotation():
    with pytest.raises(ValueError):
        prepare(np.zeros((2, 2)), {"rotate_deg": 45})
