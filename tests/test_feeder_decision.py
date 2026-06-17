"""Decision-layer guarantees: the feeder never opens on a single frame, an
unstable identity, or an unknown cat, and DOES open for a stable allowed cat.
Exercises the real pipeline (ZoneState -> decide -> DoorFSM), not one frame."""
from decision import decide
from door_fsm import DoorFSM
from zone_state import ZoneState


class _NoCooldown:
    def is_ready(self, *_a, **_k):
        return True

    def remaining_hours(self, *_a, **_k):
        return 0.0


def _run(events, *, allowed, window_sec=0.4, open_debounce=3.0):
    """events: list of (wall_t, cat, score, in_action). Returns True if the door
    ever opened."""
    zone = ZoneState(window_sec=window_sec, door_close_timeout_sec=30.0,
                     classifier_min_conf=0.5)
    fsm = DoorFSM(open_debounce_sec=open_debounce, multi_debounce_sec=2.0)
    cooldown = _NoCooldown()
    opened = False
    for wall_t, cat, score, in_action in events:
        zone.update(wall_t, cat, score, in_action)
        snap = zone.snapshot(wall_t)
        action, reason = decide(snap, allowed, cooldown, {})
        cmd = fsm.step(wall_t, snap, action, reason)
        if cmd.kind == "open":
            opened = True
            fsm.confirm_open(cmd.cat, wall_t)
    return opened


def test_single_frame_does_not_open():
    assert _run([(0.0, "alisa", 0.95, True)], allowed=["alisa"]) is False


def test_unstable_identity_does_not_open():
    # Identity flips every frame (short window => snapshot sees only the latest),
    # so no single cat holds the open verdict for open_debounce_sec.
    events = []
    t = 0.0
    for i in range(12):
        events.append((t, "alisa" if i % 2 == 0 else "chuzh", 0.95, True))
        t += 0.5
    assert _run(events, allowed=["alisa", "chuzh"]) is False


def test_unknown_does_not_open():
    events = [(i * 0.5, "unknown", 0.99, True) for i in range(12)]
    assert _run(events, allowed=["alisa"]) is False


def test_stable_allowed_cat_opens():
    events = [(i * 0.5, "alisa", 0.95, True) for i in range(12)]  # 0..5.5s
    assert _run(events, allowed=["alisa"]) is True
