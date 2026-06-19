"""Auto-refill anti-spam guarantees (FeedController).

Criterion: with N "empty" readings in a row and the min-interval elapsed, exactly
ONE feed is dispatched; no repeat until the bowl returns to "has_food" (or the
confirm timeout elapses); when disabled, nothing feeds; the door must be stable.
"""
from feed_control import FeedController


def _ctrl(**kw):
    defaults = dict(
        enabled=True, grain_num=1, empty_consecutive=2,
        min_interval_sec=1800.0, confirm_timeout_sec=120.0,
    )
    defaults.update(kw)
    return FeedController(**defaults)


def _drive(ctrl, readings, *, door_stable=True, t0=0.0, dt=1.0):
    """Feed a list of food_state strings; return the times feed fired.

    A fired feed is acknowledged as success (record_fed), mirroring main.py."""
    fired = []
    t = t0
    for fs in readings:
        if ctrl.observe(fs, door_stable=door_stable, now=t):
            fired.append(t)
            ctrl.record_fed(t)
        t += dt
    return fired


def test_requires_n_consecutive_empty():
    ctrl = _ctrl(empty_consecutive=2)
    # One empty is not enough.
    assert ctrl.observe("empty", door_stable=True, now=0.0) is False
    # Second consecutive empty triggers.
    assert ctrl.observe("empty", door_stable=True, now=1.0) is True


def test_single_empty_then_has_food_does_not_feed():
    ctrl = _ctrl(empty_consecutive=2)
    fired = _drive(ctrl, ["empty", "has_food", "empty", "has_food"])
    assert fired == []


def test_exactly_one_feed_on_sustained_empty():
    ctrl = _ctrl(empty_consecutive=2)
    # Five empties in a row: must feed exactly once (the rest are blocked by the
    # post-feed confirm wait — bowl hasn't returned to has_food yet).
    fired = _drive(ctrl, ["empty"] * 5)
    assert len(fired) == 1
    assert fired[0] == 1.0  # second reading (index 1)


def test_no_repeat_until_has_food_returns():
    ctrl = _ctrl(empty_consecutive=2, min_interval_sec=0.0)
    # Even with a zero min-interval, the confirm-wait blocks a second feed until
    # the bowl reads has_food again.
    fired = _drive(ctrl, ["empty", "empty", "empty", "empty"])
    assert len(fired) == 1
    # Bowl refills, then empties again past the (zero) interval → one more feed.
    assert ctrl.observe("has_food", door_stable=True, now=10.0) is False
    assert ctrl.observe("empty", door_stable=True, now=11.0) is False  # streak=1
    assert ctrl.observe("empty", door_stable=True, now=12.0) is True   # streak=2


def test_min_interval_blocks_second_feed_even_after_has_food():
    ctrl = _ctrl(empty_consecutive=2, min_interval_sec=1800.0)
    assert ctrl.observe("empty", door_stable=True, now=0.0) is False
    assert ctrl.observe("empty", door_stable=True, now=1.0) is True
    ctrl.record_fed(1.0)
    # Bowl refills and empties again well within the 1800s interval.
    ctrl.observe("has_food", door_stable=True, now=2.0)
    assert ctrl.observe("empty", door_stable=True, now=3.0) is False
    assert ctrl.observe("empty", door_stable=True, now=4.0) is False   # interval not elapsed
    # Past the interval, a sustained empty feeds again.
    ctrl.observe("has_food", door_stable=True, now=1801.0)
    assert ctrl.observe("empty", door_stable=True, now=1802.0) is False
    assert ctrl.observe("empty", door_stable=True, now=1803.0) is True


def test_confirm_timeout_allows_next_feed_without_has_food():
    ctrl = _ctrl(empty_consecutive=1, min_interval_sec=0.0, confirm_timeout_sec=120.0)
    assert ctrl.observe("empty", door_stable=True, now=0.0) is True
    ctrl.record_fed(0.0)
    # Still settling: no has_food, within the confirm window → blocked.
    assert ctrl.observe("empty", door_stable=True, now=60.0) is False
    # Past the confirm timeout, the wait expires and a feed is allowed again.
    assert ctrl.observe("empty", door_stable=True, now=121.0) is True


def test_door_in_motion_blocks_feed():
    ctrl = _ctrl(empty_consecutive=2)
    assert ctrl.observe("empty", door_stable=False, now=0.0) is False
    # Streak keeps counting while the door moves, but no feed until it's stable.
    assert ctrl.observe("empty", door_stable=False, now=1.0) is False
    assert ctrl.observe("empty", door_stable=False, now=2.0) is False
    # Door settles → the sustained empty now feeds.
    assert ctrl.observe("empty", door_stable=True, now=3.0) is True


def test_unknown_is_ignored_no_state_change():
    ctrl = _ctrl(empty_consecutive=2)
    # "unknown" must neither advance nor reset the streak.
    assert ctrl.observe("empty", door_stable=True, now=0.0) is False     # streak=1
    assert ctrl.observe("unknown", door_stable=True, now=1.0) is False   # ignored
    assert ctrl.observe(None, door_stable=True, now=2.0) is False        # ignored
    assert ctrl.observe("empty", door_stable=True, now=3.0) is True      # streak=2 → feed


def test_disabled_never_feeds():
    ctrl = _ctrl(enabled=False, empty_consecutive=1, min_interval_sec=0.0)
    fired = _drive(ctrl, ["empty"] * 10)
    assert fired == []


def test_failed_feed_requires_fresh_streak_no_immediate_retry():
    ctrl = _ctrl(empty_consecutive=2, min_interval_sec=1800.0)
    assert ctrl.observe("empty", door_stable=True, now=0.0) is False
    assert ctrl.observe("empty", door_stable=True, now=1.0) is True
    ctrl.record_feed_failed()                       # I/O failed after retry
    # Streak consumed: a single further empty must NOT immediately retry.
    assert ctrl.observe("empty", door_stable=True, now=2.0) is False
    # Two fresh empties re-trigger (no min-interval lock since we never fed).
    assert ctrl.observe("empty", door_stable=True, now=3.0) is True
