"""Action-zone sliding-window aggregator.

The detector sends ONE WebSocket event per detected box; boxes from the same
camera frame share the same wall_ms timestamp. This module groups events by
wall_t to recover the true simultaneous box count, and weights identity votes
by classifier confidence so low-confidence misclassifications cannot dominate.

Usage:
    zone = ZoneState(window_sec=5, door_close_timeout_sec=30, classifier_min_conf=0.5)

    # On every WS event (including stats ticks and clear events):
    zone.update(wall_t, cat, cat_score, in_action)
    snap = zone.snapshot(wall_t)
    # snap.n_cats, snap.identity, snap.present, snap.meal_sec
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoneSummary:
    """Snapshot of the action zone at a given wall_t."""

    n_cats: int           # sustained simultaneous in_action box count
    identity: str | None  # dominant weighted-vote cat identity (None = unresolved)
    present: bool         # any in-zone detection within door_close_timeout_sec
    meal_sec: float       # last_seen − first_seen for the current presence episode


class ZoneState:
    """Sliding-window aggregator for action-zone events.

    Keeps a frame buffer (wall_t → list of (cat, cat_score)) for the last
    max(window_sec, door_close_timeout_sec) seconds.  snapshot() derives
    n_cats and identity from the short window; presence from the long timeout.
    """

    def __init__(
        self,
        window_sec: float,
        door_close_timeout_sec: float,
        classifier_min_conf: float,
    ) -> None:
        self._win = window_sec
        self._timeout = door_close_timeout_sec
        self._min_conf = classifier_min_conf
        # wall_t → list of (cat, cat_score) for in_action detections at that frame
        self._frames: dict[float, list[tuple[str | None, float | None]]] = {}
        self._first_seen: float | None = None  # start of current presence episode
        self._last_seen: float | None = None   # most recent in-zone detection

    def update(
        self,
        wall_t: float,
        cat: str | None,
        cat_score: float | None,
        in_action: bool,
    ) -> None:
        """Record one detection event or a no-detection clock tick.

        Call for every WS event — including stats and clear events — so the
        presence TTL keeps ticking even when no cats are in frame.
        """
        if in_action:
            # Gap longer than timeout means a new presence episode.
            if self._last_seen is not None and wall_t - self._last_seen > self._timeout:
                self._first_seen = wall_t
            if self._first_seen is None:
                self._first_seen = wall_t
            if wall_t not in self._frames:
                self._frames[wall_t] = []
            self._frames[wall_t].append((cat, cat_score))
            self._last_seen = wall_t

        # Prune: retain frames needed by both the short window and the long timeout.
        cutoff = wall_t - max(self._win, self._timeout)
        stale = [t for t in self._frames if t < cutoff]
        for t in stale:
            del self._frames[t]

    def snapshot(self, wall_t: float) -> ZoneSummary:
        """Compute the zone summary at wall_t."""
        win_cutoff = wall_t - self._win
        recent = {t: cats for t, cats in self._frames.items() if t >= win_cutoff}

        # --- n_cats: sustained concurrent in_action box count ---
        # Multi (>=2) requires at least 2 distinct frames — a single-frame
        # double-detection glitch never triggers multi_cat.
        if recent:
            frames_with_multi = sum(1 for cats in recent.values() if len(cats) >= 2)
            any_boxes = any(cats for cats in recent.values())
            n_cats = 2 if frames_with_multi >= 2 else (1 if any_boxes else 0)
        else:
            n_cats = 0

        # --- identity: weighted vote of high-confidence in_action labels ---
        # "unknown" is excluded. Non-classifier paths (cat_score=None) vote
        # with weight 1.0 (blob detector, YOLO without classifier).
        votes: dict[str, float] = {}
        for cats in recent.values():
            for cat, score in cats:
                if not cat or cat == "unknown":
                    continue
                if score is not None and score < self._min_conf:
                    continue
                votes[cat] = votes.get(cat, 0.0) + (score if score is not None else 1.0)
        identity = max(votes, key=votes.__getitem__) if votes else None

        # --- present: any detection within door_close_timeout_sec ---
        present = (
            self._last_seen is not None
            and wall_t - self._last_seen <= self._timeout
        )

        # --- meal_sec: span of current or most recent presence episode ---
        # _first_seen and _last_seen are preserved across the timeout so that
        # the close event at t = last_seen + timeout still has the full meal_sec.
        if self._first_seen is not None and self._last_seen is not None:
            meal_sec = self._last_seen - self._first_seen
        else:
            meal_sec = 0.0

        return ZoneSummary(
            n_cats=n_cats,
            identity=identity,
            present=present,
            meal_sec=meal_sec,
        )
