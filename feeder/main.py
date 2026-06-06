"""Feeder service: one process per feeder.

Connects to ws://detector-<CAMERA_ID>:8091/ws, tracks which cats are
currently in the action zone, and opens/closes the physical feeder via
the REST API.

Presence is stable: a cat is considered "present" for DOOR_CLOSE_TIMEOUT_SEC
after its last in-zone detection. Missed frames and brief pose-change gaps do
not trigger a close.

Identity is smoothed: a cat name is only counted toward the zone if it has
been observed for at least PRESENCE_CONFIRM_SEC seconds of span within a
recent PRESENCE_WINDOW_SEC window. A single misclassified frame ("ghost cat")
never reaches confirmation. "unknown" detections are excluded from multi_cat
counting regardless.

Cooldown is recorded at the END of a meal (door close), and only when the
cat was present for at least MIN_MEAL_SEC — a short blip does not lock the
cat out for hours.

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
  PRESENCE_WINDOW_SEC     sliding window for identity evidence collection (default 10)
  PRESENCE_CONFIRM_SEC    min span of evidence (sec) to confirm a cat's identity (default 3)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
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
DOOR_CLOSE_TIMEOUT_SEC = float(os.environ.get("DOOR_CLOSE_TIMEOUT_SEC", "30"))
MIN_MEAL_SEC           = float(os.environ.get("MIN_MEAL_SEC", "10"))
PRESENCE_WINDOW_SEC    = float(os.environ.get("PRESENCE_WINDOW_SEC", "10"))
PRESENCE_CONFIRM_SEC   = float(os.environ.get("PRESENCE_CONFIRM_SEC", "3"))

WS_URL = f"ws://detector-{CAMERA_ID}:8091/ws"

# ---- event loop state ----

# Presence: cat → wall_t of last in-zone detection. Expires after DOOR_CLOSE_TIMEOUT_SEC.
_active: dict[str, float] = {}

# Identity evidence: cat → deque of wall_t detections within PRESENCE_WINDOW_SEC.
_cat_history: dict[str, deque[float]] = {}

# Confirmed identities: cats whose evidence span reached PRESENCE_CONFIRM_SEC.
# Sticky — cleared only when the cat expires from _active, not when evidence window shrinks.
_confirmed_cats: set[str] = set()

_door_state = "closed"

# Meal / session tracking (reset on each open→close cycle).
_door_cat: str | None = None          # cat that triggered the open
_door_opened_at: float | None = None  # wall_t when door opened
_cat_first_seen: float | None = None  # wall_t of first detection after open
_cat_last_seen: float | None = None   # wall_t of most recent detection of _door_cat


def _handle_event(
    ev: dict,
    client: FeederClient,
    cooldown: CooldownState,
) -> None:
    global _active, _cat_history, _confirmed_cats, _door_state
    global _door_cat, _door_opened_at, _cat_first_seen, _cat_last_seen

    if ev.get("kind") == "stats":
        return

    wall_t: float = (ev.get("wall_ms") or time.time() * 1000) / 1000.0

    # Record in-zone detections.
    # Empty frames are ignored — one missed frame never resets presence.
    for box in ev.get("boxes") or []:
        if box.get("in_action"):
            cat = ev.get("cat")
            if cat:
                _active[cat] = wall_t
                if cat not in _cat_history:
                    _cat_history[cat] = deque()
                _cat_history[cat].append(wall_t)
                if cat == _door_cat:
                    _cat_last_seen = wall_t

    # Expire presence (DOOR_CLOSE_TIMEOUT_SEC).
    pres_cutoff = wall_t - DOOR_CLOSE_TIMEOUT_SEC
    expired = [c for c, t in _active.items() if t <= pres_cutoff]
    for c in expired:
        del _active[c]
        _confirmed_cats.discard(c)  # confirmation is tied to presence

    # Prune identity history (PRESENCE_WINDOW_SEC).
    conf_cutoff = wall_t - PRESENCE_WINDOW_SEC
    for cat in list(_cat_history):
        h = _cat_history[cat]
        while h and h[0] < conf_cutoff:
            h.popleft()
        if not h:
            del _cat_history[cat]

    # Promote to confirmed: sufficient evidence span accumulated.
    for cat in _active:
        if cat in _confirmed_cats:
            continue
        h = _cat_history.get(cat)
        if h and len(h) >= 2:
            span = h[-1] - h[0]
            if span >= PRESENCE_CONFIRM_SEC:
                _confirmed_cats.add(cat)
                print(f"[feeder={FEEDER_ID}] identity confirmed: {cat}", flush=True)

    # cats_in_zone: present + confirmed; "unknown" excluded (never triggers multi_cat).
    cats_in_zone = {c for c in _active if c in _confirmed_cats and c != "unknown"}

    action, reason = decide(cats_in_zone, ALLOWED_CATS, cooldown, COOLDOWN_HOURS)

    if action == "open" and _door_state != "open":
        ok = client.set_door("open", reason)
        if ok:
            _door_state = "open"
            # reason IS the cat name when action == "open" (see decision.py).
            _door_cat = reason
            _door_opened_at = wall_t
            _cat_first_seen = wall_t
            _cat_last_seen = wall_t

    elif action == "close" and _door_state != "closed":
        ok = client.set_door("close", reason)
        if ok:
            _door_state = "closed"
            if _door_opened_at is not None:
                door_sec = wall_t - _door_opened_at
                meal_sec = (
                    (_cat_last_seen - _cat_first_seen)
                    if (_cat_last_seen is not None and _cat_first_seen is not None)
                    else 0.0
                )
                print(
                    f"[feeder={FEEDER_ID}] door closed:"
                    f" cat={_door_cat} open={door_sec:.0f}s meal≈{meal_sec:.0f}s"
                    f" reason={reason}",
                    flush=True,
                )
                if _door_cat and meal_sec >= MIN_MEAL_SEC:
                    cooldown.record_meal_end(_door_cat)
            _door_cat = None
            _door_opened_at = None
            _cat_first_seen = None
            _cat_last_seen = None

    # action == None: cooldown in effect, keep current state


async def _run(client: FeederClient, cooldown: CooldownState) -> None:
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                print(
                    f"[feeder={FEEDER_ID}] connected to {WS_URL} "
                    f"allowed={ALLOWED_CATS} cooldown={COOLDOWN_HOURS} "
                    f"close_timeout={DOOR_CLOSE_TIMEOUT_SEC}s min_meal={MIN_MEAL_SEC}s "
                    f"confirm={PRESENCE_CONFIRM_SEC}s/{PRESENCE_WINDOW_SEC}s",
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
