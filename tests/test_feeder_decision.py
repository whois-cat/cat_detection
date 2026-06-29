"""Decision-layer guarantees: the feeder never opens on a single frame, an
unstable identity, or an unknown cat, and DOES open for a stable allowed cat.
Exercises the real pipeline (ZoneState -> decide -> DoorFSM), not one frame."""
import pytest

from decision import DangerousConfusion, decide, parse_dangerous_confusions
from door_fsm import DoorFSM
from zone_state import ZoneState, ZoneSummary


def _run(events, *, allowed, window_sec=0.4, open_debounce=3.0):
    """events: list of (wall_t, cat, score, in_action). Returns True if the door
    ever opened."""
    zone = ZoneState(window_sec=window_sec, door_close_timeout_sec=30.0,
                     classifier_min_conf=0.5)
    fsm = DoorFSM(open_debounce_sec=open_debounce, multi_debounce_sec=2.0)
    opened = False
    for wall_t, cat, score, in_action in events:
        zone.update(wall_t, cat, score, in_action)
        snap = zone.snapshot(wall_t)
        action, reason = decide(snap, allowed)
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


# ---- decide() verdicts (no cooldown) ----

def _present(identity="alisa", n_cats=1, present=True, identity_score=None, margin=None):
    return ZoneSummary(n_cats=n_cats, identity=identity, present=present, meal_sec=0.0,
                       identity_score=identity_score, margin=margin)


def test_allowed_present_single_cat_opens():
    # An allowed, present, single, identified cat opens immediately — there is no
    # cooldown gate anymore.
    action, reason = decide(_present("alisa"), ["alisa"])
    assert action == "open"
    assert reason == "alisa"


def test_not_allowed_cat_closes():
    action, reason = decide(_present("chuzh"), ["alisa"])
    assert action == "close"
    assert reason == "not_allowed:chuzh"


# ---- confidence + margin gates (generic fixtures) ----

def test_low_confidence_blocks_open():
    # Allowed cat, but the winning identity's confidence is below the gate.
    action, reason = decide(_present("cat_a", identity_score=0.6), ["cat_a"],
                            min_confidence=0.85)
    assert action == "close" and reason == "low_confidence:cat_a"


def test_high_confidence_opens():
    action, reason = decide(_present("cat_a", identity_score=0.95), ["cat_a"],
                            min_confidence=0.85)
    assert action == "open" and reason == "cat_a"


def test_confidence_gate_off_by_default():
    # Default min_confidence=0 → no decision-level confidence gate.
    action, reason = decide(_present("cat_a", identity_score=0.10), ["cat_a"])
    assert action == "open" and reason == "cat_a"


def test_confidence_gate_skipped_when_score_unavailable():
    # Non-classifier path (no score) → confidence gate can't apply, still opens.
    action, reason = decide(_present("cat_a", identity_score=None), ["cat_a"],
                            min_confidence=0.85)
    assert action == "open" and reason == "cat_a"


def test_margin_gate_inert_until_margin_available():
    # min_margin set but margin is None (detector emits no top-2) → does NOT block.
    action, _ = decide(_present("cat_a", identity_score=0.95, margin=None), ["cat_a"],
                       min_margin=0.2)
    assert action == "open"


def test_low_margin_blocks_when_margin_present():
    action, reason = decide(_present("cat_a", identity_score=0.95, margin=0.05), ["cat_a"],
                            min_margin=0.2)
    assert action == "close" and reason == "low_margin:cat_a"


# ---- config-driven dangerous confusions (generic fixtures) ----

def test_dangerous_confusion_blocks_open():
    # feeder allows cat_a; config says cat_b can be mistaken for cat_a (forbidden)
    # → refuse to open even though the resolved identity is the allowed cat_a.
    dcs = [DangerousConfusion(actual="cat_b", predicted="cat_a", action="block_open")]
    action, reason = decide(_present("cat_a"), ["cat_a"], dcs)
    assert action == "close"
    assert reason == "dangerous_confusion:cat_b~cat_a"


def test_dangerous_confusion_ignored_when_actual_is_allowed():
    # If the confusable "actual" is itself allowed, there's no safety risk → open.
    dcs = [DangerousConfusion(actual="cat_b", predicted="cat_a", action="block_open")]
    action, reason = decide(_present("cat_a"), ["cat_a", "cat_b"], dcs)
    assert action == "open" and reason == "cat_a"


def test_dangerous_confusion_only_for_matching_prediction():
    dcs = [DangerousConfusion(actual="cat_b", predicted="cat_a", action="block_open")]
    # resolved identity is cat_c (allowed), not the dangerous "predicted" → open.
    action, reason = decide(_present("cat_c"), ["cat_a", "cat_c"], dcs)
    assert action == "open" and reason == "cat_c"


def test_no_dangerous_config_is_unchanged():
    action, reason = decide(_present("cat_a"), ["cat_a"])
    assert action == "open" and reason == "cat_a"


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_decision_is_class_count_agnostic(n):
    # Arbitrary number of identity classes with arbitrary (non-local) names.
    classes = [f"id_{i}" for i in range(n)]
    allowed = classes[: max(1, n // 2)]
    # An allowed identity opens; a non-allowed one closes — for ANY class count.
    assert decide(_present(allowed[-1]), allowed) == ("open", allowed[-1])
    if n > len(allowed):
        outsider = classes[-1]
        assert decide(_present(outsider), allowed) == ("close", f"not_allowed:{outsider}")


def test_decision_handles_unknown_and_unresolved_identity():
    # "unknown" never reaches decide() as an identity (zone_state filters it), but
    # an unresolved vote surfaces as identity=None → no_identity.
    assert decide(_present(None), ["greycat", "spot"]) == ("close", "no_identity")
    # Non-local names work the same as any other.
    assert decide(_present("greycat"), ["greycat", "spot"]) == ("open", "greycat")


def test_parse_dangerous_confusions():
    assert parse_dangerous_confusions("") == []
    assert parse_dangerous_confusions("   ") == []
    dcs = parse_dangerous_confusions(
        '[{"actual":"cat_b","predicted":"cat_a","action":"block_open"},'
        ' {"actual":"cat_d","predicted":"cat_c"}]'
    )
    assert dcs[0] == DangerousConfusion("cat_b", "cat_a", "block_open")
    assert dcs[1].action == "block_open"   # default
    with pytest.raises(ValueError):
        parse_dangerous_confusions("{not json")
