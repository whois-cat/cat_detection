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
  CLASSIFIER_MIN_CONF     min cat_score to participate in identity vote (default 0.5)
  OPEN_DEBOUNCE_SEC       allowed cat must hold the open-verdict this long before
                          the door opens (default 3)
  MULTI_DEBOUNCE_SEC      a multi_cat / identity-change condition must hold this
                          long before an open door closes (default 2)
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
CLASSIFIER_MIN_CONF    = float(os.environ.get("CLASSIFIER_MIN_CONF", "0.5"))
OPEN_DEBOUNCE_SEC      = float(os.environ.get("OPEN_DEBOUNCE_SEC", "3"))
MULTI_DEBOUNCE_SEC     = float(os.environ.get("MULTI_DEBOUNCE_SEC", "2"))

WS_URL = f"ws://detector-{CAMERA_ID}:8091/ws"

# ---- module state (initialised in main()) ----

_zone: ZoneState
_fsm: DoorFSM


def _handle_event(
    ev: dict,
    client: FeederClient,
    cooldown: CooldownState,
) -> None:
    wall_t: float = (ev.get("wall_ms") or time.time() * 1000) / 1000.0

    if ev.get("kind") == "stats":
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

    # cmd.kind is None: arming / latch hold / closing debounce — no door change.


async def _run(client: FeederClient, cooldown: CooldownState) -> None:
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
        except Exception as exc:
            print(
                f"[feeder={FEEDER_ID}] WS error: {exc!r}; reconnecting in 5s",
                flush=True,
            )
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
