"""One feeder's decision loop: a port of the previous feeder service (one
process per feeder, module globals) to one object per feeder, owned by one
thread (see run()), so feeders never block each other.

Input changed from the detector's per-box WebSocket events to streamhub CV
results: one result per processed frame of the feeder's camera, with all
boxes of that frame. A result without boxes advances the clock like the old
"clear"/stats events did. Everything after that — ZoneState, decide(), DoorFSM,
the journal, scheduled feeding, the silence watchdog — is unchanged.

Decisions are reported to streamhub (stored next to the video, shown in the
web UI) whenever the decision state changes or the door is commanded.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from .config import FeederConfig
from .decision import decide, parse_dangerous_confusions
from .display import DisplayThrottle, status_text
from .door_fsm import CLOSING, OPEN, DoorFSM
from .journal import FeedJournal
from .schedule_feed import ScheduleFeeder
from .zone_state import UNKNOWN, ZoneState, ZoneSummary

CLOCK_RATE = 90000
SCHEDULE_TICK_SEC = 30.0
TICK_SEC = 1.0
OPEN_RETRY_SEC = 30.0
ALWAYS_OPEN = "always_open"

# FSM verdicts are terse and "no_cat" reads like "no cat was here" when it
# actually means the cat LEFT the zone (present==False) — the normal end of a
# meal. Translate to a clearer journal label at the write boundary.
_CLOSE_REASON_JOURNAL = {"no_cat": "cat_left"}

Observation = tuple[str | None, float | None, bool]  # cat, cat_score, in_action


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray casting; poly in the same (camera-fraction) coordinates."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def prefer_allowed(snap: ZoneSummary, allowed: list[str]) -> ZoneSummary:
    """If any allowed identity got votes, report the allowed group as the
    identity (see FeederConfig.open_if_any_allowed); else leave snap as is."""
    voted = [c for c in allowed if snap.votes.get(c)]
    if not voted:
        return snap
    scores = [s for c in voted if (s := snap.vote_scores.get(c)) is not None]
    return dataclasses.replace(snap, identity=allowed_group(allowed),
                               identity_score=max(scores) if scores else None)


def allowed_group(allowed: list[str]) -> str:
    return "|".join(allowed)


def observations(result: dict[str, Any], unknown_conf: float, polygon: list[list[float]]) -> list[Observation]:
    """Per box: identity as the old detector reported it (top class, or
    "unknown" below unknown_conf; "cat" without a classifier), its probability,
    and whether the box centre is inside the action polygon."""
    out = []
    for d in result.get("dets") or []:
        cats = d.get("cats") or {}
        if cats:
            cat, score = max(cats.items(), key=lambda kv: (kv[1], kv[0]))
            if score < unknown_conf:
                cat = UNKNOWN
        else:
            cat, score = "cat", None
        x, y, w, h = d["box"]
        out.append((cat, score, point_in_polygon(x + w / 2, y + h / 2, polygon)))
    return out


class Feeder:
    def __init__(
        self,
        cfg: FeederConfig,
        client: Any,  # FeederClient
        journal: FeedJournal,
        send_decision: Callable[[dict[str, Any]], None],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.id = cfg.id
        self.camera = cfg.camera
        self.client = client
        self.journal = journal
        self._send_decision = send_decision
        self._monotonic = monotonic
        self._wall = wall
        self.log = logging.getLogger(f"decider.{cfg.id}")
        self.inbox: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)

        try:
            self.dangerous = parse_dangerous_confusions(json.dumps(cfg.dangerous_confusions))
        except Exception as e:  # never let bad config kill the feeder
            self.log.warning("invalid dangerous_confusions (%r); ignoring", e)
            self.dangerous = []
        self.zone = ZoneState(
            window_sec=cfg.presence_window_sec,
            door_close_timeout_sec=cfg.door_close_timeout_sec,
            classifier_min_conf=cfg.classifier_min_conf,
            unknown_votes=UNKNOWN in cfg.allowed_cats,
            allowed=cfg.allowed_cats,
        )
        # With open_if_any_allowed the allowed identities act as one group.
        self.allowed = cfg.allowed_cats + ([allowed_group(cfg.allowed_cats)] if cfg.open_if_any_allowed else [])
        self.fsm = DoorFSM(open_debounce_sec=cfg.open_debounce_sec, multi_debounce_sec=cfg.multi_debounce_sec)
        # Show the open-cat name ONCE on open with a long interval covering the
        # meal (per-event re-pushing flooded the display bridge).
        self.display_open_interval = max(cfg.display_text_interval, int(cfg.door_close_timeout_sec))
        # The status stays on the display until the next (refresh) update, so
        # hold each text longer than the refresh period.
        status_hold = max(self.display_open_interval, int(cfg.display_refresh_min_sec) * 2)
        self.status = DisplayThrottle(
            lambda text: client.set_display_text(text, status_hold),
            refresh_sec=cfg.display_refresh_min_sec,
        ) if cfg.display == "status" else None
        # "name": the cat's name on open (and slow refresh while open).
        self.show_name = cfg.display == "name"

        self.schedule: ScheduleFeeder | None = None
        if cfg.feed.mode == "scheduled":
            tz = _resolve_tz(cfg.feed.tz)
            self.schedule = ScheduleFeeder(
                times=cfg.feed.times, grain_num=cfg.feed.grain_num, tz=tz,
                catchup_max=cfg.feed.catchup_max, journal=journal, feeder_id=cfg.id,
            )
            if not cfg.feed.times:
                self.log.warning("feed.mode=scheduled but feed.times is empty — no feeds will be scheduled")
        self._next_schedule = 0.0

        self.open_session_id: int | None = None
        self.last_event_monotonic: float | None = None
        self.last_event_wall_t: float | None = None
        self._last_display_monotonic: float | None = None
        self._last_display_cat: str | None = None
        self._close_fail_count = 0
        self._close_fail_since: float | None = None
        self._last_not_opening_key: tuple | None = None
        self._last_decision_key: tuple | None = None
        # door: open — the door is held open and the FSM is bypassed.
        self.always_open = cfg.door == "open"
        self._next_open_retry = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        recovered = self.journal.recover_interrupted(self.id)
        if recovered:
            self.log.info("recovered %d interrupted door session(s)", recovered)
        if self.always_open:
            self._hold_open()
        else:
            self.client.force_closed()

    def run(self, stop: threading.Event) -> None:
        """Own this feeder: handle results, and tick the watchdog/schedule."""
        next_tick = self._monotonic()
        while not stop.is_set():
            try:
                self.handle_result(self.inbox.get(timeout=max(0.0, next_tick - self._monotonic())))
            except queue.Empty:
                pass
            except Exception:
                self.log.exception("handling a result failed")
            if self._monotonic() >= next_tick:
                try:
                    self.tick()
                except Exception:
                    self.log.exception("tick failed")
                next_tick = self._monotonic() + TICK_SEC

    # ---- results ----

    def handle_result(self, result: dict[str, Any]) -> None:
        wall_t = result["pts"] / CLOCK_RATE
        self.last_event_monotonic = self._monotonic()
        self.last_event_wall_t = wall_t
        obs = observations(result, self.cfg.unknown_conf, self.cfg.action_polygon)
        if not obs:
            self.zone.update(wall_t, None, None, False)  # no boxes: advance the clock
        for cat, score, in_action in obs:
            self.zone.update(wall_t, cat, score, in_action)
        self._step(wall_t, result["pts"])

    def _step(self, wall_t: float, pts: int) -> None:
        snap = self.zone.snapshot(wall_t)
        if self.cfg.open_if_any_allowed:
            snap = prefer_allowed(snap, self.cfg.allowed_cats)
        action, reason = decide(
            snap, self.allowed, self.dangerous,
            min_confidence=self.cfg.open_min_confidence, min_margin=self.cfg.open_min_margin,
        )
        ctx = self._context(snap, action, reason)

        # A cat is present but the door isn't being opened: log once per change.
        if snap.present and action != "open":
            key = (action, snap.identity, reason)
            if key != self._last_not_opening_key:
                self._last_not_opening_key = key
                self.log.info("decision: %s", ctx)
        else:
            self._last_not_opening_key = None

        if self.always_open:
            text = None
            if self.status is not None:
                text = status_text(self.client.state, None, snap)
                self.status.set(text)
                self.status.flush(self._monotonic())
            self._report(pts, snap, action, reason, ALWAYS_OPEN, None, text)
            return

        cmd = self.fsm.step(wall_t, snap, action, reason)
        event = None
        if cmd.kind == "open":
            if self.client.set_door("open", cmd.reason):
                self.fsm.confirm_open(cmd.cat, wall_t)
                self.open_session_id = self.journal.open_session(self.id, cmd.cat, wall_t)
                self._reset_close_backstop()
                if self.show_name:
                    self._set_display_for_open(cmd.cat)
                self.log.info("door opened: cat=%s", cmd.cat)
                self.log.info("decision: %s", ctx)
                event = "opened"
            else:
                event = "open_failed"
        elif cmd.kind == "close":
            if self.client.set_door("close", cmd.reason):
                door_sec = (wall_t - self.fsm.opened_at) if self.fsm.opened_at is not None else 0.0
                self.log.info("door closed: cat=%s open=%.0fs meal≈%.0fs reason=%s",
                              cmd.cat, door_sec, snap.meal_sec, cmd.reason)
                self._finalize_open_session(wall_t, cmd.reason, meal_sec=snap.meal_sec)
                self.fsm.confirm_close()
                self._reset_close_backstop()
                event = "closed"
            else:
                event = "close_failed"
                self._close_backstop_after_failure(cmd.cat)
        elif self.fsm.state in (OPEN, CLOSING) and self.show_name:
            # No door command this step: a slow display refresh may go out
            # (the door always has priority over the display bridge).
            self._maybe_slow_refresh_display(self.fsm.door_cat)

        text = None
        if self.status is not None:
            text = status_text(self.fsm.state, self.fsm.door_cat, snap)
            self.status.set(text)
            if cmd.kind is None:
                self.status.flush(self._monotonic())
        self._report(pts, snap, action, reason, cmd.reason, event, text)

    # ---- periodic ----

    def tick(self) -> None:
        if self.always_open:
            self._hold_open()
        self._watchdog()
        if self.schedule is not None and self._monotonic() >= self._next_schedule:
            self._next_schedule = self._monotonic() + SCHEDULE_TICK_SEC
            self._schedule_step()
        if self.status is not None:
            self.status.flush(self._monotonic())

    def _watchdog(self) -> None:
        """Close an OPEN door when CV results for the camera stop (hub, camera or
        CV worker gone) for longer than the stream-blip grace. A departing cat
        keeps results coming, so it closes via the presence timeout instead."""
        if self.last_event_monotonic is None or self.fsm.state not in (OPEN, CLOSING):
            return
        silence_sec = self._monotonic() - self.last_event_monotonic
        cmd = self.fsm.note_silence(silence_sec, self.cfg.stream_blip_grace_sec)
        if cmd.kind == "close":
            self.log.info("no CV results for %.0fs >= grace %.0fs while door open for %s — closing (stream_lost)",
                          silence_sec, self.cfg.stream_blip_grace_sec, cmd.cat)
            self._fail_safe_close("stream_lost")

    def _schedule_step(self) -> None:
        """Scheduled feeding (independent of CV). A fed slot is journaled right
        after the dispense so the next plan() excludes it; a failed feed is left
        pending and retried next tick; maintenance slots are always marked."""
        assert self.schedule is not None
        for slot, action in self.schedule.plan(dt.datetime.now(self.schedule.tz)):
            if action == "fed":
                if not self.client.feed(self.schedule.grain_num):
                    self.log.info("scheduled:feed_failed slot=%s — leaving slot pending for retry", slot.time)
                    continue
                self.journal.record_scheduled(self.id, slot.date, slot.time, status="fed",
                                              grain_num=self.schedule.grain_num, acked=True)
                self.log.info("scheduled:fed slot=%s grain=%s", slot.time, self.schedule.grain_num)
            else:
                self.journal.record_scheduled(self.id, slot.date, slot.time, status="maintenance",
                                              grain_num=0, acked=False)
                self.log.info("scheduled:maintenance slot=%s (beyond catch-up cap)", slot.time)

    # ---- door helpers (as in the previous feeder service) ----

    def _hold_open(self) -> None:
        """door: open — open the door, retrying until it worked."""
        now = self._monotonic()
        if self.client.state == "open" or now < self._next_open_retry:
            return
        self._next_open_retry = now + OPEN_RETRY_SEC
        if self.client.set_door("open", ALWAYS_OPEN):
            self.log.info("door held open (door: open)")

    def _fail_safe_close(self, reason: str) -> None:
        if self.fsm.state not in (OPEN, CLOSING) and self.client.state != "open":
            return
        if self.client.set_door("close", reason):
            self.log.info("fail-safe door close: %s", reason)
            self._finalize_open_session(self.last_event_wall_t or self._wall(), reason)
            self.fsm.confirm_close()
            self._report(int(self._wall() * CLOCK_RATE), None, "close", reason, reason, "closed", None)

    def _finalize_open_session(self, closed_at_wall: float, reason: str, *, meal_sec: float | None = None) -> None:
        if self.open_session_id is None:
            return
        self.journal.finalize_session(self.open_session_id, closed_at_wall, meal_sec,
                                      _CLOSE_REASON_JOURNAL.get(reason, reason), self.cfg.min_meal_sec)
        self.open_session_id = None

    def _reset_close_backstop(self) -> None:
        self._close_fail_count = 0
        self._close_fail_since = None

    def _close_backstop_after_failure(self, cat: str | None) -> None:
        """door/close keeps failing for this episode: once over the attempt/time
        budget, assume it physically closed and disarm the FSM, so one cat can't
        latch the door open forever."""
        now = self._monotonic()
        if self._close_fail_since is None:
            self._close_fail_since = now
        self._close_fail_count += 1
        stuck_sec = now - self._close_fail_since
        if self._close_fail_count < self.cfg.close_backstop_max_attempts and stuck_sec < self.cfg.close_backstop_max_sec:
            return
        self.log.warning("door close failed %dx over %.0fs for cat=%s; assuming physically closed and "
                         "disarming FSM (backstop) so the next cat can be served",
                         self._close_fail_count, stuck_sec, cat)
        self._finalize_open_session(self.last_event_wall_t or self._wall(), "backstop")
        self.fsm.confirm_close()
        self._reset_close_backstop()

    def _set_display_for_open(self, cat: str | None) -> None:
        if cat and self.client.set_display_text(cat, self.display_open_interval):
            self._last_display_cat = cat
            self._last_display_monotonic = self._monotonic()

    def _maybe_slow_refresh_display(self, cat: str | None) -> None:
        """For hardware that caps the display interval: re-push the name at most
        once per display_refresh_min_sec."""
        if not cat:
            return
        now = self._monotonic()
        if (self._last_display_cat == cat and self._last_display_monotonic is not None
                and now - self._last_display_monotonic < self.cfg.display_refresh_min_sec):
            return
        if self.client.set_display_text(cat, self.display_open_interval):
            self._last_display_cat = cat
            self._last_display_monotonic = now

    # ---- reporting ----

    def request_report(self) -> None:
        """Report the current state with the next result even if unchanged
        (e.g. after reconnecting: streamhub may have restarted and forgotten it)."""
        self._last_decision_key = None

    def _context(self, snap: ZoneSummary, action: str | None, reason: str) -> str:
        conf = "n/a" if snap.identity_score is None else f"{snap.identity_score:.3f}"
        margin = "n/a" if snap.margin is None else f"{snap.margin:.3f}"
        return (f"feeder={self.id} camera={self.camera} identity={snap.identity} conf={conf} margin={margin} "
                f"n_cats={snap.n_cats} allowed={self.cfg.allowed_cats} action={action} reason={reason}")

    def _report(self, pts: int, snap: ZoneSummary | None, action: str | None, reason: str,
                step: str, event: str | None, display: str | None) -> None:
        """Send a decision record when the decision state changed or the door
        was commanded."""
        d: dict[str, Any] = {
            "type": "decision", "camera": self.camera, "pts": pts, "feeder": self.id,
            # FeederClient.state is "closed" initially but "close" after a close command.
            "state": ALWAYS_OPEN if self.always_open else self.fsm.state, "door": "open" if self.client.state == "open" else "closed",
            "action": action, "reason": reason,
            "step": step,
        }
        if snap is not None:
            d.update(identity=snap.identity, n_cats=snap.n_cats, present=snap.present,
                     conf=None if snap.identity_score is None else round(snap.identity_score, 4))
        if display is not None:
            d["display"] = display
        key = (d["state"], d["door"], action, reason, d.get("identity"), d.get("n_cats"), display)
        if event is None and key == self._last_decision_key:
            return
        self._last_decision_key = key
        if event:
            d["event"] = event
        try:
            self._send_decision(d)
        except Exception as e:
            self.log.debug("decision not sent: %s", e)


def _resolve_tz(name: str) -> dt.tzinfo:
    if not name.strip():
        local = dt.datetime.now().astimezone().tzinfo
        assert local is not None
        return local
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)
