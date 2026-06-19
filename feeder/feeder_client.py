"""Feeder REST API client with retry logic.

Port of donor live/feeder_client.py — metrics push removed (no VictoriaMetrics
in live2). All state management is local; idempotent: a door already in the
desired state still gets the API call (server is authoritative).
"""
from __future__ import annotations

import time

_RETRY_BACKOFF_SEC = 0.25
# Door commands drive the physical actuator and can be slow to ack — give them a
# generous read timeout so a single slow response is retried, not failed. The
# display endpoint is best-effort cosmetic text that must never hold up the
# bridge to the hardware, so it gets a short timeout. Per-request `timeout=`
# overrides keep these independent; the base client never caps the door at the
# display's short timeout.
_DOOR_TIMEOUT_SEC = 15.0
_DISPLAY_TIMEOUT_SEC = 3.0


class FeederClient:
    def __init__(self, api_base_url: str, serial_number: str, feeder_id: str) -> None:
        import httpx

        self._base = api_base_url.rstrip("/")
        self._serial = serial_number
        self._fid = feeder_id
        self._state = "closed"
        # Base default is the long (door) timeout; display calls pass the short
        # one explicitly. This way an unspecified call never starves the door.
        self._http = httpx.Client(timeout=_DOOR_TIMEOUT_SEC)

    # ---- internal ----

    def _api_path(self, suffix: str) -> str:
        return f"/api/{self._serial}{suffix}"

    def _call(
        self, path: str, *, json_body: dict | None = None, timeout: float = _DOOR_TIMEOUT_SEC
    ) -> bool:
        """POST with one retry on transient errors; immediate fail on 4xx.

        `timeout` is per-request so door and display calls keep independent
        budgets regardless of the base client default."""
        import httpx

        url = f"{self._base}{path}"
        def body_preview(resp) -> str:
            text = (resp.text or "").strip()
            return f" body={text[:200]!r}" if text else ""

        for attempt in range(2):
            try:
                if json_body is None:
                    resp = self._http.post(url, timeout=timeout)
                else:
                    resp = self._http.post(url, json=json_body, timeout=timeout)
                if resp.is_success:
                    return True
                if 400 <= resp.status_code < 500:
                    print(
                        f"[feeder={self._fid}] POST {path} → {resp.status_code}"
                        f" (4xx, no retry){body_preview(resp)}",
                        flush=True,
                    )
                    return False
                if attempt == 0:
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                print(
                    f"[feeder={self._fid}] POST {path} → {resp.status_code}"
                    f"{body_preview(resp)}",
                    flush=True,
                )
                return False
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                # httpx.TimeoutException covers Connect/Read/Write/Pool timeouts.
                # A single slow ack must be retried inside the call, not failed
                # immediately — that's what was leaving the door stuck.
                if attempt == 0:
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                print(
                    f"[feeder={self._fid}] POST {path} timeout/conn error: {exc!r}",
                    flush=True,
                )
                return False
            except Exception as exc:
                print(f"[feeder={self._fid}] POST {path} error: {exc!r}", flush=True)
                return False
        return False

    # ---- public ----

    def force_closed(self) -> None:
        print(f"[feeder={self._fid}] startup: forcing door closed", flush=True)
        ok = self._call(self._api_path("/door/close"), timeout=_DOOR_TIMEOUT_SEC)
        if not ok:
            print(f"[feeder={self._fid}] force-close failed — assuming closed", flush=True)

    def set_door(self, desired: str, reason: str) -> bool:
        path = self._api_path("/door/open" if desired == "open" else "/door/close")
        ok = self._call(path, timeout=_DOOR_TIMEOUT_SEC)
        if ok:
            prev, self._state = self._state, desired
            print(f"[feeder={self._fid}] door {prev} → {desired} ({reason})", flush=True)
        else:
            print(
                f"[feeder={self._fid}] door {desired} failed — will retry next event",
                flush=True,
            )
        return ok

    def feed(self, grain_num: int = 1) -> bool:
        """Dispense `grain_num` portions of food. Transient errors get one retry
        inside `_call`; a 4xx fails immediately without retry (server rejected the
        request, e.g. bad grain_num)."""
        ok = self._call(
            self._api_path("/feed"),
            json_body={"grain_num": grain_num},
            timeout=_DOOR_TIMEOUT_SEC,
        )
        print(
            f"[feeder={self._fid}] {'fed' if ok else 'feed FAILED'} "
            f"grain_num={grain_num}",
            flush=True,
        )
        return ok

    def set_display_text(self, text: str, interval: int = 1) -> bool:
        ok = self._call(
            self._api_path("/display/text"),
            json_body={"text": text, "interval": interval},
            timeout=_DISPLAY_TIMEOUT_SEC,
        )
        if ok:
            print(f"[feeder={self._fid}] display text: {text!r}", flush=True)
        else:
            print(f"[feeder={self._fid}] display text failed: {text!r}", flush=True)
        return ok

    @property
    def state(self) -> str:
        return self._state

    def close(self) -> None:
        self._http.close()
