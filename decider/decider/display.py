"""Short live status for the feeder's small display (display: status).

Format: "<cat> <door>"
  cat:   first letter of the cat the door is open for, else of the currently
         recognised cat (upper case); ? unknown; 2 several cats; - nobody
  door:  O open (incl. closing), C closed (incl. arming)
e.g. "C O" door open for chuzh; "A C" alisa present, door shut; "- C" idle.
"""
from __future__ import annotations

from .zone_state import UNKNOWN, ZoneSummary

_OPEN_STATES = ("open", "closing")


def _letter(name: str | None) -> str:
    if not name:
        return "-"
    return "?" if name == UNKNOWN else name[0].upper()


def status_text(state: str, door_cat: str | None, snap: ZoneSummary) -> str:
    if state in _OPEN_STATES:
        cat = _letter(door_cat)
    elif snap.present and snap.n_cats >= 2:
        cat = "2"
    else:
        cat = _letter(snap.identity if snap.present else None)
    return f"{cat} {'O' if state in _OPEN_STATES else 'C'}"


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
