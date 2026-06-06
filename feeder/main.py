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

WS_URL = f"ws://detector-{CAMERA_ID}:8091/ws"

# ---- module state (initialised in main()) ----

_zone: ZoneState
_door_state = "closed"
_door_cat: str | None = None          # identity for whom the door was opened
_door_opened_at: float | None = None  # wall_t when door last opened
_cat_last_seen: float | None = None   # wall_t of most recent _door_cat in-zone detection


def _close_door(
    reason: str,
    wall_t: float,
    client: FeederClient,
    cooldown: CooldownState,
) -> None:
    """Close the door, emit a human-readable log line, record cooldown if warranted."""
    global _door_state, _door_cat, _door_opened_at, _cat_last_seen

    ok = client.set_door("close", reason)
    if not ok:
        return

    _door_state = "closed"
    if _door_opened_at is not None:
        door_sec = wall_t - _door_opened_at
        meal_sec = (_cat_last_seen - _door_opened_at) if _cat_last_seen is not None else 0.0
        print(
            f"[feeder={FEEDER_ID}] door closed:"
            f" cat={_door_cat} open={door_sec:.0f}s meal≈{meal_sec:.0f}s reason={reason}",
            flush=True,
        )
        if _door_cat and meal_sec >= MIN_MEAL_SEC:
            cooldown.record_meal_end(_door_cat)

    _door_cat = None
    _door_opened_at = None
    _cat_last_seen = None


def _handle_event(
    ev: dict,
    client: FeederClient,
    cooldown: CooldownState,
) -> None:
    global _door_state, _door_cat, _door_opened_at, _cat_last_seen

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
        this_cat = ev.get("cat")
        in_action = bool(box.get("in_action"))
        # Track last time the currently-open cat was seen (for meal_sec log).
        if _door_cat and this_cat == _door_cat and in_action:
            _cat_last_seen = wall_t
        _zone.update(wall_t, this_cat, ev.get("cat_score"), in_action)

    snap = _zone.snapshot(wall_t)

    # Identity change while door is open: the dominant cat has switched.
    # Close and attribute the meal to the cat the door was opened for, then
    # fall through to re-evaluate whether to open for the new identity.
    if (
        _door_state == "open"
        and _door_cat is not None
        and snap.identity is not None
        and snap.identity != _door_cat
        and snap.present
    ):
        _close_door("identity_change", wall_t, client, cooldown)

    action, reason = decide(snap, ALLOWED_CATS, cooldown, COOLDOWN_HOURS)

    if action == "open" and _door_state != "open":
        ok = client.set_door("open", reason)
        if ok:
            _door_state = "open"
            _door_cat = reason     # reason IS the identity when action == "open"
            _door_opened_at = wall_t
            _cat_last_seen = wall_t

    elif action == "close" and _door_state != "closed":
        _close_door(reason, wall_t, client, cooldown)

    # action == None: cooldown hold — keep current state unchanged


async def _run(client: FeederClient, cooldown: CooldownState) -> None:
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                print(
                    f"[feeder={FEEDER_ID}] connected to {WS_URL} "
                    f"allowed={ALLOWED_CATS} cooldown={COOLDOWN_HOURS} "
                    f"close_timeout={DOOR_CLOSE_TIMEOUT_SEC}s min_meal={MIN_MEAL_SEC}s "
                    f"presence_win={PRESENCE_WINDOW_SEC}s min_conf={CLASSIFIER_MIN_CONF}",
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
    global _zone
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
