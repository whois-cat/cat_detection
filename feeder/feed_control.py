"""Auto-refill decision logic: dispense food when the bowl reads empty.

Pure, I/O-free state machine — the same shape as cooldown.py / door_fsm.py, kept
out of main.py so it is unit-testable without env vars or a live WS. main.py owns
the env config and the actual ``client.feed()`` REST call; this class only decides
*whether* to feed and tracks the anti-spam state locally.

The detector publishes ``food_state`` ("empty" | "has_food" | "unknown") only when
a FOOD_REGION is set AND the bowl monitor is calibrated. We act on a SUSTAINED
"empty", never on a single frame, and guard against refilling in a loop:

  - require ``empty_consecutive`` "empty" readings in a row;
  - at most one feed per ``min_interval_sec``;
  - after a successful feed, wait for the bowl to return to "has_food" (or for
    ``confirm_timeout_sec`` to elapse) before another feed is allowed, so we don't
    keep pouring while the grain is still settling;
  - never feed while the door is in motion (caller passes ``door_stable``);
  - "unknown" (uncalibrated / occluded / busy bowl) is ignored — no state change.

Time is injected (``now`` = a monotonic clock) so tests stay deterministic.
"""
from __future__ import annotations


class FeedController:
    def __init__(self, *, enabled: bool, grain_num: int,
                 empty_consecutive: int, min_interval_sec: float,
                 confirm_timeout_sec: float) -> None:
        self.enabled = enabled
        self.grain_num = grain_num
        self.empty_consecutive = max(1, int(empty_consecutive))
        self.min_interval_sec = float(min_interval_sec)
        self.confirm_timeout_sec = float(confirm_timeout_sec)

        self._consecutive_empty = 0
        self._last_feed_monotonic: float | None = None
        # After a successful feed we wait for has_food (or a timeout) before the
        # next feed is allowed, so grain settling can't trigger a refill loop.
        self._awaiting_confirm = False
        self._awaiting_since: float | None = None

    def observe(self, food_state: str | None, *, door_stable: bool, now: float) -> bool:
        """Fold one ``food_state`` reading into the anti-spam state and return
        True iff a feed should be dispatched right now. The caller performs the
        I/O and then reports the outcome via ``record_fed`` / ``record_feed_failed``."""
        if not self.enabled:
            return False

        # Expire a stale confirm-wait by elapsed time, regardless of the current
        # reading — a feed that never registered as has_food must not latch us
        # waiting forever.
        if (
            self._awaiting_confirm
            and self._awaiting_since is not None
            and now - self._awaiting_since >= self.confirm_timeout_sec
        ):
            self._awaiting_confirm = False
            self._awaiting_since = None

        if food_state == "has_food":
            # Bowl refilled (by us or a human): reset the streak and clear the
            # post-feed wait — the next empty run starts fresh.
            self._consecutive_empty = 0
            self._awaiting_confirm = False
            self._awaiting_since = None
            return False

        if food_state != "empty":
            # "unknown" / None / anything unrecognised: no information, ignore.
            return False

        # food_state == "empty"
        self._consecutive_empty += 1

        if self._awaiting_confirm:
            return False                      # still settling from the last feed
        if self._consecutive_empty < self.empty_consecutive:
            return False                      # not sustained yet
        if not door_stable:
            return False                      # never feed while the door is moving
        if (
            self._last_feed_monotonic is not None
            and now - self._last_feed_monotonic < self.min_interval_sec
        ):
            return False                      # min interval not elapsed
        return True

    def record_fed(self, now: float) -> None:
        """Feed succeeded: start the min-interval clock and the confirm-wait, and
        reset the streak so empties must re-accumulate for the next refill."""
        self._last_feed_monotonic = now
        self._awaiting_confirm = True
        self._awaiting_since = now
        self._consecutive_empty = 0

    def record_feed_failed(self) -> None:
        """Feed failed (after feeder_client's own retry): consume the streak so a
        persistent failure can't retry every event — it must re-accumulate
        ``empty_consecutive`` fresh readings. No min-interval lock is set (we
        never actually dispensed), so a transient miss can still recover."""
        self._consecutive_empty = 0
