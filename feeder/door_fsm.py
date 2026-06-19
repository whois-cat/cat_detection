"""Door state machine with debounce / hysteresis.

`decide()` (decision.py) is a *pure, instantaneous* verdict on the current
ZoneSummary. On its own it makes the door chatter: one glitch frame — the
classifier momentarily returning "unknown" (identity → None → "no_identity"),
or YOLO drawing a second box on a single cat (n_cats → 2 → "multi_cat") —
flips the verdict to "close" for a single event, and the next good frame flips
it back to "open". This module wraps that verdict in an explicit automaton that
debounces transitions so the physical door only moves on *sustained* changes.

States (the door's physical position is implied):

    CLOSED   ── door shut, nothing eligible
    ARMING   ── door shut, an allowed+ready cat is present; waiting for it to
                stay the open-verdict continuously for open_debounce_sec
    OPEN     ── door open and latched for one cat (door_cat)
    CLOSING  ── door still open, a close condition is building; waiting for it
                to hold for its debounce before actually shutting

Only two transitions touch the physical door:
    CLOSED/ARMING → OPEN   (emit "open")
    OPEN/CLOSING  → CLOSED (emit "close")

The caller performs the I/O and, on success, calls confirm_open()/confirm_close()
to commit the physical transition. If the REST call fails the caller skips the
confirm; step() keeps re-emitting the same command on subsequent events, so the
door self-heals (idempotent retry, matching the feeder_client contract).

Hysteresis rules:
  - OPEN-debounce: open only after the dominant allowed+ready identity has been
    the open-verdict continuously for >= open_debounce_sec.
  - Latch: while OPEN for cat X, transient "no_identity" (identity None) and a
    single-frame "multi_cat" are ignored — the door stays open.
  - Close only on sustained conditions:
      (a) present == False           — cat really gone (already debounced by
                                        door_close_timeout_sec in ZoneState);
                                        closes immediately, reason "no_cat".
      (b) multi_cat held             >= multi_debounce_sec, reason "multi_cat".
      (c) a DIFFERENT identity held  >= multi_debounce_sec, reason
                                        "identity_change" (meal attributed to X).
"""
from __future__ import annotations

from dataclasses import dataclass

from zone_state import ZoneSummary

CLOSED = "closed"
ARMING = "arming"
OPEN = "open"
CLOSING = "closing"


@dataclass
class DoorCommand:
    """What the caller should do to the physical door after this step.

    kind == "open"  → open the door for `cat`; on success call confirm_open(cat, wall_t).
    kind == "close" → close the door; on success record the meal for `cat` and
                       call confirm_close().
    kind is None    → no physical change; `reason` is a human-readable status
                      (arming:/hold:/closing:/idle) for logging only.
    """

    kind: str | None
    cat: str | None
    reason: str


class DoorFSM:
    def __init__(self, open_debounce_sec: float, multi_debounce_sec: float) -> None:
        self.open_debounce_sec = open_debounce_sec
        self.multi_debounce_sec = multi_debounce_sec

        self.state: str = CLOSED
        self.door_cat: str | None = None       # identity the door is OPEN/CLOSING for
        self.opened_at: float | None = None     # wall_t the door was confirmed open

        self._arming_cat: str | None = None
        self._arming_since: float | None = None
        self._closing_reason: str | None = None
        self._closing_since: float | None = None

    # ---- driver ----

    def step(
        self,
        wall_t: float,
        snap: ZoneSummary,
        action: str | None = None,
        reason: str = "",
    ) -> DoorCommand:
        """Advance the automaton with one (already-smoothed) ZoneSummary and the
        pure decide() verdict for it. Returns the door command to execute.

        action/reason are the decide() verdict; they steer the CLOSED/ARMING
        states only (OPEN/CLOSING derive everything from the snapshot), so they
        default to a no-open verdict for snapshot-only callers."""
        if self.state in (CLOSED, ARMING):
            return self._step_closed(wall_t, action, reason)
        return self._step_open(wall_t, snap)

    def _step_closed(self, wall_t: float, action: str | None, reason: str) -> DoorCommand:
        # decide() returns ("open", <cat>) only for an allowed, ready, single,
        # present cat — exactly the open candidate. Anything else (cooldown hold,
        # no_cat, multi_cat, no_identity, not_allowed) disarms.
        open_cat = reason if action == "open" else None

        if open_cat is None:
            self.state = CLOSED
            self._arming_cat = None
            self._arming_since = None
            return DoorCommand(None, None, "idle")

        # (Re)start the arming timer when a new candidate becomes eligible.
        if self.state == CLOSED or self._arming_cat != open_cat:
            self.state = ARMING
            self._arming_cat = open_cat
            self._arming_since = wall_t

        assert self._arming_since is not None
        if wall_t - self._arming_since >= self.open_debounce_sec:
            return DoorCommand("open", open_cat, open_cat)
        return DoorCommand(None, None, f"arming:{open_cat}")

    def _step_open(self, wall_t: float, snap: ZoneSummary) -> DoorCommand:
        x = self.door_cat

        # (a) cat really gone — present is already door_close_timeout-debounced.
        if not snap.present:
            return DoorCommand("close", x, "no_cat")

        # Determine a sustained-close trigger, if any. A transient "no_identity"
        # (identity is None) is NOT a trigger — the latch holds the door open.
        trigger: str | None = None
        if snap.n_cats >= 2:
            trigger = "multi_cat"
        elif snap.identity is not None and snap.identity != x:
            trigger = "identity_change"

        if trigger is None:
            # Good frame (or transient glitch) — relatch, cancel any pending close.
            self.state = OPEN
            self._closing_reason = None
            self._closing_since = None
            return DoorCommand(None, None, f"hold:{x}")

        # (Re)start the closing timer when a new trigger appears.
        if self.state == OPEN or self._closing_reason != trigger:
            self.state = CLOSING
            self._closing_reason = trigger
            self._closing_since = wall_t

        assert self._closing_since is not None
        if wall_t - self._closing_since >= self.multi_debounce_sec:
            return DoorCommand("close", x, trigger)
        return DoorCommand(None, None, f"closing:{trigger}")

    def note_silence(self, silence_sec: float, grace_sec: float) -> DoorCommand:
        """Decide what an OPEN/CLOSING door should do during detector SILENCE.

        "Silence" means *no events at all* — a watchdog tick or a WS disconnect —
        as opposed to live events that report ``present == False`` (a cat that
        genuinely left, handled by ``_step_open`` as "no_cat"). A momentary stream
        blip must not chatter the door, so we HOLD it open for the current
        ``door_cat`` until the silence has lasted ``>= grace_sec``; only then do we
        close, with reason "stream_lost" so logs distinguish a dropped stream from
        a departed cat. If events resume within the grace window, the next
        ``step()`` relatches and this never fires.

        No-op unless the door is physically open (OPEN/CLOSING); does not mutate
        state (the caller closes via the normal confirm_close path on success).

        Safety tradeoff: during the blind grace window the door stays open, so in
        theory another cat could sneak a bite; ``grace_sec`` bounds that exposure,
        and a sustained loss still closes."""
        if self.state not in (OPEN, CLOSING):
            return DoorCommand(None, None, "idle")
        if silence_sec < grace_sec:
            return DoorCommand(None, self.door_cat, f"stream_blip:{self.door_cat}")
        return DoorCommand("close", self.door_cat, "stream_lost")

    # ---- commits (called by the caller after a successful door I/O) ----

    def confirm_open(self, cat: str, wall_t: float) -> None:
        self.state = OPEN
        self.door_cat = cat
        self.opened_at = wall_t
        self._arming_cat = None
        self._arming_since = None
        self._closing_reason = None
        self._closing_since = None

    def confirm_close(self) -> None:
        self.state = CLOSED
        self.door_cat = None
        self.opened_at = None
        self._arming_cat = None
        self._arming_since = None
        self._closing_reason = None
        self._closing_since = None
