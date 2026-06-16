"""Feeder service: one process per feeder.

Connects to ws://detector-<CAMERA_ID>:8091/ws, processes detection events,
and opens/closes the physical feeder via the REST API.

EVENT FORMAT (one WS message per detected box):
  Regular : {wall_ms, cat, cat_score, boxes:[{…,in_action}], …}
  Clear   : {wall_ms, boxes:[], cat:None}        ← detections stopped
  Stats   : {kind:"stats", wall_ms, …}           ← periodic heartbeat

ZoneState groups same-wall_ms events into frames to count simultaneous
in_action boxes correctly, and weights identity votes by cat_score to
suppress low-confidence misclassifications.

Stats ticks are forwarded to ZoneState so the presence TTL keeps advancing
even when no cats are detected — this guarantees the door closes after
DOOR_CLOSE_TIMEOUT_SEC even if the camera produces no new detection events.
An independent watchdog also advances the clock if the detector stops sending
events entirely, and WS disconnects force the physical door closed before
reconnect.

The physical door is driven by an explicit state machine (door_fsm.DoorFSM,
CLOSED→ARMING→OPEN→CLOSING) that debounces the pure decide() verdict so single
glitch frames never chatter the door. decide() stays a pure function.

Cooldown is recorded at the END of a meal (door close), and only when the
cat was present for at least MIN_MEAL_SEC.

Env vars (all from configure.py — nothing hardcoded):
  CAMERA_ID               detector service to connect to (detector-<id>)
  FEEDER_ID               this feeder's identifier (for logs / cooldown DB name)
  FEEDER_API_BASE_URL     base URL of the feeder REST API
  FEEDER_SERIAL_NUMBER    feeder hardware serial
  ALLOWED_CATS            comma-separated list of allowed cat names
  COOLDOWN                JSON {"cat": cooldown_hours, ...}  (default: {})
  COOLDOWN_DB             SQLite path (default /data/cooldowns/feeder-<FEEDER_ID>.db)
  DOOR_CLOSE_TIMEOUT_SEC  seconds without detection before door closes (default 30)
  MIN_MEAL_SEC            min meal duration (sec) to record a cooldown (default 10)
  PRESENCE_WINDOW_SEC     sliding window (sec) for n_cats / identity smoothing (default 5)
  CLASSIFIER_MIN_CONF     min cat_score to participate in identity vote (default 0.9)
  OPEN_DEBOUNCE_SEC       allowed cat must hold the open-verdict this long before
                          the door opens (default 3)
  MULTI_DEBOUNCE_SEC      a multi_cat / identity-change condition must hold this
                          long before an open door closes (default 2)
  DISPLAY_TEXT_INTERVAL   seconds the feeder display should keep each text update
                          visible; refreshed while the door is open (default 2)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import websockets

from cooldown import CooldownState
from decision import decide
from door_fsm import DoorFSM
from feeder_client import FeederClient
from zone_state import ZoneState

# ---- config (all from env) ----

CAMERA_ID = os.environ["CAMERA_ID"]
FEEDER_ID = os.environ["FEEDER_ID"]
FEEDER_API_BASE_URL = os.environ["FEEDER_API_BASE_URL"]
FEEDER_SERIAL_NUMBER = os.environ["FEEDER_SERIAL_NUMBER"]
ALLOWED_CATS: list[str] = [
    c.strip() for c in os.environ.get("ALLOWED_CATS", "").split(",") if c.strip()
]
COOLDOWN_HOURS: dict[str, float] = json.loads(os.environ.get("COOLDOWN", "{}"))
COOLDOWN_DB = Path(
    os.environ.get("COOLDOWN_DB", f"/data/cooldowns/feeder-{FEEDER_ID}.db")
)
DOOR_CLOSE_TIMEOUT_SEC = float(os.environ.get("DOOR_CLOSE_TIMEOUT_SEC", "30"))
MIN_MEAL_SEC           = float(os.environ.get("MIN_MEAL_SEC", "10"))
PRESENCE_WINDOW_SEC    = float(os.environ.get("PRESENCE_WINDOW_SEC", "5"))
CLASSIFIER_MIN_CONF    = float(os.environ.get("CLASSIFIER_MIN_CONF", "0.9"))
OPEN_DEBOUNCE_SEC      = float(os.environ.get("OPEN_DEBOUNCE_SEC", "3"))
MULTI_DEBOUNCE_SEC     = float(os.environ.get("MULTI_DEBOUNCE_SEC", "2"))
DISPLAY_TEXT_INTERVAL  = max(1, int(os.environ.get("DISPLAY_TEXT_INTERVAL", "2")))
# Show the open-cat name ONCE on open with a long interval that covers the whole
# expected meal, instead of re-pushing it every ~1.5s (that display spam flooded
# the bridge and starved door commands). If the hardware caps the interval and
# lets the name fade, fall back to a slow refresh no more than once per
# DISPLAY_REFRESH_MIN_SEC — and never on an event that also issues a door command
# (the door always has priority over the display bridge).
DISPLAY_OPEN_INTERVAL   = max(DISPLAY_TEXT_INTERVAL, int(DOOR_CLOSE_TIMEOUT_SEC))
DISPLAY_REFRESH_MIN_SEC = float(os.environ.get("DISPLAY_REFRESH_MIN_SEC", "25"))
# Backstop against a permanent clamp: if door/close keeps failing for one open
# episode (this many attempts in a row OR this many seconds), assume the actuator
# has physically closed and disarm the FSM so the next cat can be served.
CLOSE_BACKSTOP_MAX_ATTEMPTS = max(1, int(os.environ.get("CLOSE_BACKSTOP_MAX_ATTEMPTS", "3")))
CLOSE_BACKSTOP_MAX_SEC      = float(os.environ.get("CLOSE_BACKSTOP_MAX_SEC", "30"))

WS_URL = f"ws://detector-{CAMERA_ID}:8091/ws"

# ---- module state (initialised in main()) ----

_zone: ZoneState
_fsm: DoorFSM
_last_event_monotonic: float | None = None
_last_event_wall_t: float | None = None
_last_display_monotonic: float | None = None
_last_display_cat: str | None = None
# Backstop bookkeeping for the currently-open episode.
_close_fail_count: int = 0
_close_fail_since: float | None = None


def _set_display_for_open(client: FeederClient, cat: str | None) -> None:
    """Set the open-cat name ONCE on open, with a long interval covering the
    meal — so it does not need per-event refreshing (which flooded the bridge)."""
    global _last_display_monotonic, _last_display_cat
    if not cat:
        return
    if client.set_display_text(cat, DISPLAY_OPEN_INTERVAL):
        _last_display_cat = cat
        _last_display_monotonic = time.monotonic()


def _maybe_slow_refresh_display(client: FeederClient, cat: str | None) -> None:
    """Fallback for hardware that caps the display interval and lets the name
    fade: re-push it at most once per DISPLAY_REFRESH_MIN_SEC. Only called on
    events that issue no door command, so the door always wins the bridge."""
    global _last_display_monotonic, _last_display_cat
    if not cat:
        return
    now = time.monotonic()
    if (
        _last_display_cat == cat
        and _last_display_monotonic is not None
        and now - _last_display_monotonic < DISPLAY_REFRESH_MIN_SEC
    ):
        return
    if client.set_display_text(cat, DISPLAY_OPEN_INTERVAL):
        _last_display_cat = cat
        _last_display_monotonic = now


def _reset_close_backstop() -> None:
    global _close_fail_count, _close_fail_since
    _close_fail_count = 0
    _close_fail_since = None


def _close_backstop_after_failure(
    cooldown: CooldownState, cat: str | None, meal_sec: float
) -> None:
    """Called when a door/close command failed. Counts consecutive failures for
    this open episode; once they exceed the attempt/time budget, log loudly and
    force the FSM closed (confirm_close) so it can disarm and serve the next cat.
    Better to risk a re-open than to stay latched open on one cat forever."""
    global _close_fail_count, _close_fail_since
    now = time.monotonic()
    if _close_fail_since is None:
        _close_fail_since = now
    _close_fail_count += 1
    stuck_sec = now - _close_fail_since
    if _close_fail_count < CLOSE_BACKSTOP_MAX_ATTEMPTS and stuck_sec < CLOSE_BACKSTOP_MAX_SEC:
        return
    print(
        f"[feeder={FEEDER_ID}] WARNING: door close failed {_close_fail_count}x"
        f" over {stuck_sec:.0f}s for cat={cat}; assuming physically closed and"
        f" disarming FSM (backstop) so the next cat can be served",
        flush=True,
    )
    if cat and meal_sec >= MIN_MEAL_SEC:
        cooldown.record_meal_end(cat)
    _fsm.confirm_close()
    _reset_close_backstop()


def _handle_event(
    ev: dict,
    client: FeederClient,
    cooldown: CooldownState,
    *,
    mark_event: bool = True,
) -> None:
    global _last_event_monotonic, _last_event_wall_t
    wall_t: float = (ev.get("wall_ms") or time.time() * 1000) / 1000.0
    if mark_event:
        _last_event_monotonic = time.monotonic()
        _last_event_wall_t = wall_t

    if ev.get("kind") in {"stats", "watchdog"}:
        # Advance ZoneState clock so the presence TTL keeps ticking between
        # detection bursts — without this, the door would never close if no
        # detection events arrive for DOOR_CLOSE_TIMEOUT_SEC.
        _zone.update(wall_t, None, None, False)
    elif not ev.get("boxes"):
        # Clear event: detector has no detections; advance clock.
        _zone.update(wall_t, None, None, False)
    else:
        # Regular event: exactly one box per message (detector protocol).
        box = ev["boxes"][0]
        _zone.update(wall_t, ev.get("cat"), ev.get("cat_score"), bool(box.get("in_action")))

    snap = _zone.snapshot(wall_t)
    action, reason = decide(snap, ALLOWED_CATS, cooldown, COOLDOWN_HOURS)
    cmd = _fsm.step(wall_t, snap, action, reason)

    if cmd.kind == "open":
        if client.set_door("open", cmd.reason):
            _fsm.confirm_open(cmd.cat, wall_t)
            # Fresh episode: clear any stale close-failure bookkeeping and set
            # the display name once (long interval) — never refreshed per-event.
            _reset_close_backstop()
            _set_display_for_open(client, cmd.cat)
            print(f"[feeder={FEEDER_ID}] door opened: cat={cmd.cat}", flush=True)

    elif cmd.kind == "close":
        if client.set_door("close", cmd.reason):
            door_sec = (wall_t - _fsm.opened_at) if _fsm.opened_at is not None else 0.0
            print(
                f"[feeder={FEEDER_ID}] door closed:"
                f" cat={cmd.cat} open={door_sec:.0f}s meal≈{snap.meal_sec:.0f}s"
                f" reason={cmd.reason}",
                flush=True,
            )
            if cmd.cat and snap.meal_sec >= MIN_MEAL_SEC:
                cooldown.record_meal_end(cmd.cat)
            _fsm.confirm_close()
            _reset_close_backstop()
        else:
            # Close failed (commonly a ReadTimeout to the bridge). Count it and,
            # if it keeps failing for this episode, force the FSM closed so it
            # can't latch open forever on one cat.
            _close_backstop_after_failure(cooldown, cmd.cat, snap.meal_sec)

    # cmd.kind is None: arming / latch hold / closing debounce — no door change.
    # Only on these no-door-command events do we consider a slow display refresh,
    # so a door command always takes priority over the display bridge.
    elif _fsm.state in {"open", "closing"}:
        _maybe_slow_refresh_display(client, _fsm.door_cat)


def _fail_safe_close(client: FeederClient, cooldown: CooldownState, reason: str) -> None:
    """Close the physical door immediately when detector liveness is unknown."""
    if _fsm.state not in {"open", "closing"} and client.state != "open":
        return
    cat = _fsm.door_cat
    meal_sec = 0.0
    if _fsm.opened_at is not None:
        meal_sec = ((_last_event_wall_t or time.time()) - _fsm.opened_at)
    if client.set_door("close", reason):
        print(f"[feeder={FEEDER_ID}] fail-safe door close: {reason}", flush=True)
        if cat and meal_sec >= MIN_MEAL_SEC:
            cooldown.record_meal_end(cat)
        _fsm.confirm_close()


async def _watchdog(client: FeederClient, cooldown: CooldownState) -> None:
    """Close an open door if detector events stop but the WS connection hangs."""
    interval = max(1.0, min(5.0, DOOR_CLOSE_TIMEOUT_SEC / 3.0))
    while True:
        await asyncio.sleep(interval)
        if _last_event_monotonic is None or _last_event_wall_t is None:
            continue
        silence_sec = time.monotonic() - _last_event_monotonic
        if silence_sec < DOOR_CLOSE_TIMEOUT_SEC:
            continue
        wall_t = _last_event_wall_t + silence_sec
        _handle_event(
            {"kind": "watchdog", "wall_ms": int(wall_t * 1000), "boxes": []},
            client,
            cooldown,
            mark_event=False,
        )


async def _run(client: FeederClient, cooldown: CooldownState) -> None:
    watchdog = asyncio.create_task(_watchdog(client, cooldown))
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                print(
                    f"[feeder={FEEDER_ID}] connected to {WS_URL} "
                    f"allowed={ALLOWED_CATS} cooldown={COOLDOWN_HOURS} "
                    f"close_timeout={DOOR_CLOSE_TIMEOUT_SEC}s min_meal={MIN_MEAL_SEC}s "
                    f"presence_win={PRESENCE_WINDOW_SEC}s min_conf={CLASSIFIER_MIN_CONF} "
                    f"open_debounce={OPEN_DEBOUNCE_SEC}s multi_debounce={MULTI_DEBOUNCE_SEC}s",
                    flush=True,
                )
                async for raw in ws:
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    _handle_event(ev, client, cooldown)
                print(
                    f"[feeder={FEEDER_ID}] WS closed; reconnecting in 5s",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[feeder={FEEDER_ID}] WS error: {exc!r}; reconnecting in 5s",
                flush=True,
            )
        _fail_safe_close(client, cooldown, "detector_ws_lost")
        if watchdog.done():
            watchdog.result()
        await asyncio.sleep(5)


def main() -> None:
    global _zone, _fsm
    print(
        f"[feeder={FEEDER_ID}] starting — camera={CAMERA_ID} "
        f"api={FEEDER_API_BASE_URL} serial={FEEDER_SERIAL_NUMBER}",
        flush=True,
    )
    _zone = ZoneState(
        window_sec=PRESENCE_WINDOW_SEC,
        door_close_timeout_sec=DOOR_CLOSE_TIMEOUT_SEC,
        classifier_min_conf=CLASSIFIER_MIN_CONF,
    )
    _fsm = DoorFSM(
        open_debounce_sec=OPEN_DEBOUNCE_SEC,
        multi_debounce_sec=MULTI_DEBOUNCE_SEC,
    )
    cooldown = CooldownState(COOLDOWN_DB)
    client = FeederClient(
        api_base_url=FEEDER_API_BASE_URL,
        serial_number=FEEDER_SERIAL_NUMBER,
        feeder_id=FEEDER_ID,
    )
    client.force_closed()
    asyncio.run(_run(client, cooldown))


if __name__ == "__main__":
    main()
