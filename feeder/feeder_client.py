"""Feeder REST API client with retry logic.

Port of donor live/feeder_client.py — metrics push removed (no VictoriaMetrics
in live2). All state management is local; idempotent: a door already in the
desired state still gets the API call (server is authoritative).
"""
from __future__ import annotations

import time

_RETRY_BACKOFF_SEC = 0.25


class FeederClient:
    def __init__(self, api_base_url: str, serial_number: str, feeder_id: str) -> None:
        import httpx

        self._base = api_base_url.rstrip("/")
        self._serial = serial_number
        self._fid = feeder_id
        self._state = "closed"
        self._http = httpx.Client(timeout=3.0)

    # ---- internal ----

    def _api_path(self, suffix: str) -> str:
        return f"/api/{self._serial}{suffix}"

    def _call(self, path: str) -> bool:
        """POST with one retry on transient errors; immediate fail on 4xx."""
        import httpx

        url = f"{self._base}{path}"
        for attempt in range(2):
            try:
                resp = self._http.post(url)
                if resp.is_success:
                    return True
                if 400 <= resp.status_code < 500:
                    print(
                        f"[feeder={self._fid}] POST {path} → {resp.status_code} (4xx, no retry)",
                        flush=True,
                    )
                    return False
                if attempt == 0:
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                print(f"[feeder={self._fid}] POST {path} → {resp.status_code}", flush=True)
                return False
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt == 0:
                    time.sleep(_RETRY_BACKOFF_SEC)
                    continue
                print(f"[feeder={self._fid}] POST {path} conn error: {exc!r}", flush=True)
                return False
            except Exception as exc:
                print(f"[feeder={self._fid}] POST {path} error: {exc!r}", flush=True)
                return False
        return False

    # ---- public ----

    def force_closed(self) -> None:
        print(f"[feeder={self._fid}] startup: forcing door closed", flush=True)
        ok = self._call(self._api_path("/door/close"))
        if not ok:
            print(f"[feeder={self._fid}] force-close failed — assuming closed", flush=True)

    def set_door(self, desired: str, reason: str) -> bool:
        path = self._api_path("/door/open" if desired == "open" else "/door/close")
        ok = self._call(path)
        if ok:
            prev, self._state = self._state, desired
            print(f"[feeder={self._fid}] door {prev} → {desired} ({reason})", flush=True)
        else:
            print(
                f"[feeder={self._fid}] door {desired} failed — will retry next event",
                flush=True,
            )
        return ok

    def set_display_text(self, text: str, interval: int = 1) -> bool:
        import httpx

        url = f"{self._base}{self._api_path('/display/text')}"
        try:
            resp = self._http.post(url, json={"text": text, "interval": interval})
            return resp.is_success
        except Exception as exc:
            print(f"[feeder={self._fid}] display error: {exc!r}", flush=True)
            return False

    @property
    def state(self) -> str:
        return self._state

    def close(self) -> None:
        self._http.close()
