"""Feeder door decision logic.

Pure function: maps (ZoneSummary, allowed_cats) → (action, reason).

Decision rules (evaluated in order):
  not present                          → "close",  "no_cat"
  n_cats >= 2 (sustained simultaneous) → "close",  "multi_cat"
  identity is None                     → "close",  "no_identity"
  identity not in allowed_cats         → "close",  "not_allowed:<cat>"
  identity in allowed_cats             → "open",   <cat>
"""
from __future__ import annotations

from zone_state import ZoneSummary


def decide(
    snap: ZoneSummary,
    allowed_cats: list[str],
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

    return "open", cat
