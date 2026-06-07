"""ZoneState sliding-window aggregator tests.

Covers the two smoothing guarantees the door FSM relies on: single-frame box
doubling never reports multi_cat, and the weighted identity vote ignores
low-confidence and "unknown" labels.
"""
from zone_state import ZoneState


def _zone():
    return ZoneState(window_sec=5, door_close_timeout_sec=30, classifier_min_conf=0.5)


def test_single_frame_double_box_is_not_multi():
    z = _zone()
    # Two boxes in ONE frame (same wall_t) — a YOLO double-detection glitch.
    z.update(0.0, "alisa", 0.9, True)
    z.update(0.0, "alisa", 0.9, True)
    assert z.snapshot(0.0).n_cats == 1


def test_two_frames_with_two_boxes_is_multi():
    z = _zone()
    for t in (0.0, 1.0):
        z.update(t, "alisa", 0.9, True)
        z.update(t, "ellie", 0.9, True)
    assert z.snapshot(1.0).n_cats == 2


def test_identity_vote_excludes_unknown_and_low_conf():
    z = _zone()
    z.update(0.0, "alisa", 0.9, True)     # counts
    z.update(0.0, "unknown", 0.99, True)  # excluded (unknown)
    z.update(1.0, "ellie", 0.3, True)     # excluded (< min_conf)
    z.update(1.0, "alisa", 0.8, True)     # counts
    assert z.snapshot(1.0).identity == "alisa"


def test_weighted_vote_beats_frequent_low_weight_label():
    z = _zone()
    # ellie appears more often but only just above threshold; alisa is confident.
    z.update(0.0, "ellie", 0.55, True)
    z.update(1.0, "ellie", 0.55, True)
    z.update(2.0, "alisa", 0.95, True)
    z.update(3.0, "alisa", 0.95, True)
    assert z.snapshot(3.0).identity == "alisa"


def test_presence_expires_after_timeout():
    z = _zone()
    z.update(0.0, "alisa", 0.9, True)
    assert z.snapshot(10.0).present is True       # within 30s timeout
    assert z.snapshot(31.0).present is False      # past it
