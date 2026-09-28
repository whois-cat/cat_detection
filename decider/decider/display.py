"""Short live status for the feeder's small display (opt-in: status_display).

Format, at most 4 characters: <state><cat><confidence><block reason>
  state:       C closed · A arming · O open · X closing
  cat:         first letter of the identity (upper case); ? unknown; 2 several
               cats; - nobody
  confidence:  first digit of the identity confidence (0.97 → 9), if known
  block reason (only while a cat is present and the door stays shut):
               N not allowed · L low confidence · D dangerous confusion ·
               M several cats · I no identity
e.g. "OC9" open for chuzh at 0.9x; "CA8N" alisa present, not allowed; "C-" idle.
"""
from __future__ import annotations

from .zone_state import UNKNOWN, ZoneSummary

_STATE = {"closed": "C", "arming": "A", "open": "O", "closing": "X"}
_REASON = {
    "not_allowed": "N", "low_confidence": "L", "low_margin": "L",
    "dangerous_confusion": "D", "multi_cat": "M", "no_identity": "I",
}


def status_text(state: str, snap: ZoneSummary, action: str | None, reason: str) -> str:
    if snap.n_cats >= 2:
        cat = "2"
    elif not snap.present or snap.identity is None:
        cat = "-"
    elif snap.identity == UNKNOWN:
        cat = "?"
    else:
        cat = snap.identity[0].upper()
    conf = ""
    if snap.present and snap.identity_score is not None:
        conf = str(min(9, max(0, int(snap.identity_score * 10))))
    block = ""
    if snap.present and action != "open" and state in ("closed", "arming"):
        block = _REASON.get(reason.split(":", 1)[0], "")
    return f"{_STATE.get(state, '?')}{cat}{conf}{block}"


class DisplayThrottle:
    """Sends status text only when it changes, at most once per min_gap_sec,
    and re-sends the current text every refresh_sec so it doesn't fade."""

    def __init__(self, send, min_gap_sec: float = 1.0, refresh_sec: float = 25.0) -> None:
        self._send = send  # (text) -> bool
        self._min_gap = min_gap_sec
        self._refresh = refresh_sec
        self._want: str | None = None
        self._shown: str | None = None
        self._sent_at = float("-inf")

    def set(self, text: str) -> None:
        self._want = text

    def flush(self, now: float) -> None:
        if self._want is None or now - self._sent_at < self._min_gap:
            return
        if self._want == self._shown and now - self._sent_at < self._refresh:
            return
        if self._send(self._want):
            self._shown, self._sent_at = self._want, now
