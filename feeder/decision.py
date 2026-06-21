"""Feeder door decision logic.

Pure function: maps (ZoneSummary, allowed_cats, CooldownState, cooldown_hours)
→ (action, reason).

Decision rules (evaluated in order):
  not present                          → "close",  "no_cat"
  n_cats >= 2 (sustained simultaneous) → "close",  "multi_cat"
  identity is None                     → "close",  "no_identity"
  identity not in allowed_cats         → "close",  "not_allowed:<cat>"
  identity in allowed_cats, cooldown   → None,     "cooldown:<cat> remaining=Xh (Ys)"
  identity in allowed_cats, ready      → "open",   <cat>

Cooldown blocks ONLY while remaining > 0 (strict): is_ready() returns True at
remaining == 0, so an expired cooldown opens. The reason prints both hours and
whole seconds so a near-zero remaining (< ~3 min) is still visible instead of
rounding to 0.0h.
"""
from __future__ import annotations

from cooldown import CooldownState
from zone_state import ZoneSummary


def decide(
    snap: ZoneSummary,
    allowed_cats: list[str],
    cooldown: CooldownState,
    cooldown_hours: dict[str, float],
) -> tuple[str | None, str]:
    if not snap.present:
        return "close", "no_cat"

    if snap.n_cats >= 2:
        return "close", "multi_cat"

    if snap.identity is None:
        return "close", "no_identity"

    cat = snap.identity

    if cat not in allowed_cats:
        return "close", f"not_allowed:{cat}"

    hours = cooldown_hours.get(cat, 0.0)
    if hours > 0 and not cooldown.is_ready(cat, hours):
        remaining = cooldown.remaining_hours(cat, hours)
        # Print hours AND whole seconds: at the boundary the remainder is < 0.05h
        # and would round to "0.0h", hiding that the cooldown is still active.
        return None, f"cooldown:{cat} remaining={remaining:.2f}h ({int(round(remaining * 3600))}s)"

    return "open", cat
