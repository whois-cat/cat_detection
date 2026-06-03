"""Feeder service: one process per feeder.

Connects to ws://detector-<CAMERA_ID>:8091/ws, tracks which cats are
currently in the action zone, and opens/closes the physical feeder via
the REST API.

Active-zone state uses a TTL: a cat is considered "present" for up to
ACTIVE_TTL_MS after its last in_action=true event. A clear event (boxes=[])
resets the state immediately.

Env vars (all from configure.py — nothing hardcoded):
  CAMERA_ID            detector service to connect to (detector-<id>)
  FEEDER_ID            this feeder's identifier (for logs / cooldown DB name)
  FEEDER_API_BASE_URL  base URL of the feeder REST API
  FEEDER_SERIAL_NUMBER feeder hardware serial
  ALLOWED_CATS         comma-separated list of allowed cat names
  COOLDOWN             JSON {"cat": cooldown_hours, ...}  (default: {})
  COOLDOWN_DB          SQLite path (default /data/cooldowns/feeder-<FEEDER_ID>.db)
  ACTIVE_TTL_MS        how long a cat stays "active" after last seen (default 3000)
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
ACTIVE_TTL_MS = int(os.environ.get("ACTIVE_TTL_MS", "3000"))

WS_URL = f"ws://detector-{CAMERA_ID}:8091/ws"

# ---- event loop state ----

# cat_name → last wall_ms when seen in action zone
_active: dict[str, int] = {}
_door_state = "closed"


def _handle_event(
    ev: dict,
    client: FeederClient,
    cooldown: CooldownState,
) -> None:
    global _active, _door_state

    # Ignore stats-only frames.
    if ev.get("kind") == "stats":
        return

    wall_ms: int = ev.get("wall_ms") or int(time.time() * 1000)
    boxes: list[dict] = ev.get("boxes") or []

    if not boxes:
        # Clear event: no detections in this frame → action zone is empty.
        _active.clear()
    else:
        for box in boxes:
            if box.get("in_action"):
                cat = ev.get("cat")
                if cat:  # skip None (blob detector without identity)
                    _active[cat] = wall_ms

    # Expire entries older than ACTIVE_TTL_MS.
    cutoff = wall_ms - ACTIVE_TTL_MS
    _active = {c: t for c, t in _active.items() if t >= cutoff}

    cats_in_zone = set(_active.keys())
    action, reason = decide(cats_in_zone, ALLOWED_CATS, cooldown, COOLDOWN_HOURS)

    if action == "open" and _door_state != "open":
        ok = client.set_door("open", reason)
        if ok:
            _door_state = "open"
            # reason is the cat name when action == "open" (see decision.py)
            cooldown.record_open(reason)
    elif action == "close" and _door_state != "closed":
        ok = client.set_door("close", reason)
        if ok:
            _door_state = "closed"
    # action == None: cooldown in effect, keep current state


async def _run(client: FeederClient, cooldown: CooldownState) -> None:
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                print(
                    f"[feeder={FEEDER_ID}] connected to {WS_URL} "
                    f"allowed={ALLOWED_CATS} cooldown={COOLDOWN_HOURS}",
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
    print(
        f"[feeder={FEEDER_ID}] starting — camera={CAMERA_ID} "
        f"api={FEEDER_API_BASE_URL} serial={FEEDER_SERIAL_NUMBER}",
        flush=True,
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
