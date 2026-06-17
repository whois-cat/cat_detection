"""Food monitor must read only the ROI: bowl_contrast_bgr converts just the ROI
region to grayscale yet matches the full-frame grayscale path exactly."""
import numpy as np

from bowl_monitor import BowlMonitor, _luma, bowl_contrast, bowl_contrast_bgr


def _frame():
    rng = np.random.default_rng(0)
    f = rng.integers(80, 200, size=(240, 320, 3), dtype=np.uint8)
    # Darken the bowl region (food is dark) so contrast is clearly positive.
    f[100:160, 200:280] = 20
    return f


ROI = [0.625, 0.4166, 0.875, 0.6666]  # ~ x:200-280, y:100-160 of 320x240


def test_roi_contrast_matches_full_frame_path():
    f = _frame()
    full = bowl_contrast(_luma(f), ROI, margin_frac=0.35)
    roi_only = bowl_contrast_bgr(f, ROI, margin_frac=0.35)
    assert full is not None and roi_only is not None
    assert abs(full - roi_only) < 1e-6
    assert roi_only > 0   # bowl darker than ring


def test_monitor_update_uses_bgr_and_calibrates():
    f = _frame()
    # Uncalibrated → state unknown, raw exposed for calibration.
    mon = BowlMonitor(ROI, empty_level=None, full_level=None,
                      empty_below=0.3, full_above=0.55, window=3)
    state, fill, raw = mon.update(f, cat_in_region=False)
    assert state == "unknown" and fill is None and raw is not None
    # Cat in region → no measurement.
    assert mon.update(f, cat_in_region=True) == ("unknown", None, None)


def test_monitor_calibrated_reports_fill():
    f = _frame()
    raw = bowl_contrast_bgr(f, ROI, margin_frac=0.35)
    mon = BowlMonitor(ROI, empty_level=0.0, full_level=raw,
                      empty_below=0.3, full_above=0.55, window=3)
    state, fill, _ = mon.update(f, cat_in_region=False)
    assert fill is not None and 0.0 <= fill <= 1.0
    assert state == "has_food"   # full bowl → fill ~1
