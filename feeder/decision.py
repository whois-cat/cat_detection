"""Feeder door decision logic.

All lists come from config (env vars), never hardcoded.

Rules:
  - multiple cats in action zone (any combination)  → "close"
  - single "unknown" cat                             → "close"
  - single blocked (not in allowed_cats) cat         → "close"
  - single allowed cat + cooldown not ready          → None  (stay closed)
  - single allowed cat + cooldown ready              → "open"
  - empty action zone                                → "close"

Returns (decision, reason):
  decision = "open" | "close" | None
  reason   = human-readable string for logging
"""
from __future__ import annotations

from cooldown import CooldownState


def decide(
    cats_in_zone: set[str],
    allowed_cats: list[str],
    cooldown: CooldownState,
    cooldown_hours: dict[str, float],
) -> tuple[str | None, str]:
    n = len(cats_in_zone)

    if n == 0:
        return "close", "no_cat"

    if n > 1:
        return "close", f"multi_cat:{','.join(sorted(cats_in_zone))}"

    cat = next(iter(cats_in_zone))

    if cat == "unknown":
        return "close", "unknown_cat"

    if cat not in allowed_cats:
        return "close", f"blocked:{cat}"

    hours = cooldown_hours.get(cat, 0.0)
    if hours > 0 and not cooldown.is_ready(cat, hours):
        remaining = cooldown.remaining_hours(cat, hours)
        return None, f"cooldown:{cat} remaining={remaining:.1f}h"

    return "open", cat
