"""Door state-machine debounce / hysteresis tests.

These drive DoorFSM.step() with crafted (action, reason) verdicts and
ZoneSummary snapshots — the FSM is isolated from decide() on purpose, so each
test pins exactly one behaviour. In the CLOSED/ARMING states only (action,
reason) matter; in OPEN/CLOSING only the snapshot matters.
"""
from door_fsm import ARMING, CLOSED, CLOSING, OPEN, DoorFSM
from zone_state import ZoneSummary


def _snap(present=True, identity="alisa", n_cats=1, meal_sec=0.0):
    return ZoneSummary(n_cats=n_cats, identity=identity, present=present, meal_sec=meal_sec)


def _open_for(fsm, cat, *, t0=0.0, debounce=3.0, dt=1.0):
    """Drive an FSM from CLOSED to OPEN(cat) and return the wall_t it opened at."""
    t = t0
    while True:
        cmd = fsm.step(t, _snap(identity=cat), "open", cat)
        if cmd.kind == "open":
            fsm.confirm_open(cmd.cat, t)
            return t
        t += dt


# ---- opening ----

def test_opens_once_after_debounce_then_holds():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)

    # Below the debounce: arming, no physical open.
    assert fsm.step(0.0, _snap(), "open", "alisa").kind is None
    assert fsm.state == ARMING
    assert fsm.step(2.9, _snap(), "open", "alisa").kind is None

    # At/after the debounce: open exactly once.
    cmd = fsm.step(3.0, _snap(), "open", "alisa")
    assert cmd.kind == "open" and cmd.cat == "alisa"
    fsm.confirm_open(cmd.cat, 3.0)
    assert fsm.state == OPEN

    # A steady allowed cat keeps it latched — never re-issues open.
    for t in (4.0, 5.0, 6.0):
        assert fsm.step(t, _snap(), "open", "alisa").kind is None
    assert fsm.state == OPEN


def test_arming_resets_on_glitch_then_full_debounce_again():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    assert fsm.step(0.0, _snap(), "open", "alisa").kind is None  # arm at 0
    # Glitch frame at t=1 (e.g. momentary no_identity) disarms.
    assert fsm.step(1.0, _snap(identity=None), "close", "no_identity").kind is None
    assert fsm.state == CLOSED
    # Re-arm at t=2: must wait a FULL debounce from 2, so t=4 (2s) is too early.
    assert fsm.step(2.0, _snap(), "open", "alisa").kind is None
    assert fsm.step(4.0, _snap(), "open", "alisa").kind is None
    assert fsm.step(5.0, _snap(), "open", "alisa").kind == "open"


def test_arming_candidate_switch_restarts_timer_for_new_cat():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    fsm.step(0.0, _snap(), "open", "alisa")           # arm alisa @0
    fsm.step(1.0, _snap(identity="ellie"), "open", "ellie")  # switch to ellie @1
    assert fsm.step(3.5, _snap(identity="ellie"), "open", "ellie").kind is None  # 2.5s < 3
    cmd = fsm.step(4.0, _snap(identity="ellie"), "open", "ellie")               # 3.0s
    assert cmd.kind == "open" and cmd.cat == "ellie"


def test_open_command_retried_until_confirmed():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    fsm.step(0.0, _snap(), "open", "alisa")
    assert fsm.step(3.0, _snap(), "open", "alisa").kind == "open"  # I/O "fails": no confirm
    assert fsm.state == ARMING
    assert fsm.step(4.0, _snap(), "open", "alisa").kind == "open"  # re-issued


# ---- latch / closing hysteresis ----

def test_transient_no_identity_does_not_close():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    # One frame the classifier returns "unknown" → identity None.
    assert fsm.step(4.0, _snap(identity=None)).kind is None
    assert fsm.state == OPEN
    # Recovers — still open, never closed.
    assert fsm.step(5.0, _snap(identity="alisa")).kind is None
    assert fsm.state == OPEN


def test_transient_multi_cat_does_not_close():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    assert fsm.step(4.0, _snap(n_cats=2)).kind is None   # closing armed
    assert fsm.state == CLOSING
    assert fsm.step(5.0, _snap(n_cats=1)).kind is None   # recovered before 2s
    assert fsm.state == OPEN


def test_sustained_multi_cat_closes_after_debounce():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    assert fsm.step(4.0, _snap(n_cats=2)).kind is None   # closing @4
    assert fsm.step(5.0, _snap(n_cats=2)).kind is None   # 1s < 2
    cmd = fsm.step(6.0, _snap(n_cats=2))                  # 2s
    assert cmd.kind == "close" and cmd.reason == "multi_cat" and cmd.cat == "alisa"
    fsm.confirm_close()
    assert fsm.state == CLOSED


def test_absent_cat_closes_immediately():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    cmd = fsm.step(4.0, _snap(present=False))
    assert cmd.kind == "close" and cmd.reason == "no_cat" and cmd.cat == "alisa"


def test_sustained_identity_change_closes_and_attributes_to_open_cat():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    assert fsm.step(4.0, _snap(identity="ellie")).kind is None   # closing @4
    cmd = fsm.step(6.0, _snap(identity="ellie"))                 # 2s
    assert cmd.kind == "close" and cmd.reason == "identity_change"
    assert cmd.cat == "alisa"   # meal attributed to the cat the door opened for


# ---- stream-silence grace (lost stream vs cat-left) ----

def test_stream_blip_under_grace_does_not_close():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    # Detector goes silent (watchdog / WS blip), but for less than the grace:
    # hold the door open for the current cat — no chatter on a brief stream drop.
    cmd = fsm.step(0.0, _snap(), "open", "alisa")  # arming verdict is irrelevant here
    cmd = fsm.note_silence(10.0, grace_sec=25.0)
    assert cmd.kind is None
    assert fsm.state == OPEN
    assert fsm.door_cat == "alisa"
    # A blip during CLOSING is also held (door is still physically open).
    fsm.step(0.0, _snap(n_cats=2))                 # arm a close trigger → CLOSING
    assert fsm.state == CLOSING
    assert fsm.note_silence(24.9, grace_sec=25.0).kind is None
    assert fsm.state == CLOSING


def test_sustained_silence_closes_with_stream_lost():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    # Silence >= grace → close, with a reason that flags a lost stream (not a cat
    # that walked away). The meal stays attributed to the open cat.
    cmd = fsm.note_silence(25.0, grace_sec=25.0)
    assert cmd.kind == "close"
    assert cmd.reason == "stream_lost"
    assert cmd.cat == "alisa"


def test_silence_is_noop_when_door_closed():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    # Never opened: a silent detector has nothing to close.
    assert fsm.note_silence(1000.0, grace_sec=25.0).kind is None
    assert fsm.state == CLOSED


def test_events_resume_within_grace_relatch_no_close():
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    _open_for(fsm, "alisa")
    # Blip under grace holds...
    assert fsm.note_silence(12.0, grace_sec=25.0).kind is None
    # ...then the same cat's events resume → relatch (hold), door never closed.
    assert fsm.step(13.0, _snap(identity="alisa")).kind is None
    assert fsm.state == OPEN and fsm.door_cat == "alisa"


def test_oscillation_sequence_keeps_door_open():
    """Realistic glitchy stream: an allowed cat with periodic 1-frame dropouts
    must open once and stay open (no chatter)."""
    fsm = DoorFSM(open_debounce_sec=3, multi_debounce_sec=2)
    opened_at = _open_for(fsm, "alisa")
    closes = 0
    t = opened_at + 1.0
    # 20s of mostly-good frames with a glitch every 3rd frame, none sustained.
    for i in range(20):
        if i % 3 == 0:
            snap = _snap(identity=None) if i % 2 == 0 else _snap(n_cats=2)
        else:
            snap = _snap(identity="alisa")
        cmd = fsm.step(t, snap)
        if cmd.kind == "close":
            closes += 1
        t += 1.0
    assert closes == 0
    assert fsm.state in (OPEN, CLOSING)
