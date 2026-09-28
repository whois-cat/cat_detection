"""Client for streamhub's hub protocol.

Transport: TCP; each message is a 4-byte big-endian length followed by a
msgpack map with a "type" key. The client sends "hello" first; both sides send
"ping" periodically. See streamhub/internal/hub for the message catalogue.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

import msgpack

__all__ = ["HubConnection", "run_forever"]

PING_EVERY = 10.0
# The hub pings every PING_EVERY; silence this long means it's gone.
READ_TIMEOUT = 3 * PING_EVERY
MAX_MESSAGE = 16 << 20

log = logging.getLogger("hubclient")


class HubConnection:
    """One connection to the hub. send() is thread-safe; recv() is not."""

    def __init__(self, addr: str, hello: dict[str, Any], connect_timeout: float = 10.0) -> None:
        host, _, port = addr.rpartition(":")
        self._sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout=connect_timeout)
        self._sock.settimeout(READ_TIMEOUT)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._wlock = threading.Lock()
        self._closed = threading.Event()
        self.send({**hello, "type": "hello"})
        threading.Thread(target=self._ping_loop, name="hub-ping", daemon=True).start()

    def send(self, msg: dict[str, Any]) -> None:
        body = msgpack.packb(msg, use_bin_type=True)
        with self._wlock:
            self._sock.sendall(struct.pack(">I", len(body)) + body)

    def recv(self) -> dict[str, Any]:
        (n,) = struct.unpack(">I", self._read_exactly(4))
        if n > MAX_MESSAGE:
            raise ConnectionError(f"message of {n} bytes exceeds limit")
        return msgpack.unpackb(self._read_exactly(n), raw=False)

    def close(self) -> None:
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def __enter__(self) -> HubConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _read_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("hub closed the connection")
            buf += chunk
        return bytes(buf)

    def _ping_loop(self) -> None:
        while not self._closed.wait(PING_EVERY):
            try:
                self.send({"type": "ping"})
            except OSError:
                return


def run_forever(
    addr: str,
    hello: dict[str, Any],
    session: Callable[[HubConnection], None],
    max_backoff: float = 30.0,
) -> None:
    """Connect, run session(conn) until the connection fails, reconnect with
    backoff. Never returns (except via exceptions other than connection errors)."""
    backoff = 1.0
    while True:
        started = time.monotonic()
        try:
            with HubConnection(addr, hello) as conn:
                log.info("connected to hub %s", addr)
                session(conn)
        except (OSError, ConnectionError) as e:
            log.warning("hub connection to %s lost: %s", addr, e)
        if time.monotonic() - started > max_backoff:
            backoff = 1.0
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)
