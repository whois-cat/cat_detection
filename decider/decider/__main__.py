"""decider: consume CV results from streamhub, drive the feeders, report
decisions back to streamhub."""
from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from typing import Any

import hubclient

from .config import load
from .feeder import Feeder
from .feeder_client import DryRunClient, FeederClient
from .journal import FeedJournal

log = logging.getLogger("decider")


class _Sender:
    """Sends decisions over the current hub connection, if any."""

    def __init__(self) -> None:
        self.conn: hubclient.HubConnection | None = None

    def __call__(self, msg: dict[str, Any]) -> None:
        conn = self.conn
        if conn is not None:
            conn.send(msg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.environ.get("CONFIG", "config.yaml"))
    p.add_argument("--hub", default=os.environ.get("STREAMHUB_HUB"), help="default: from config")
    p.add_argument("--id", default=os.environ.get("DECIDER_ID", socket.gethostname()))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load(args.config)
    if not cfg.feeders:
        log.warning("no feeders configured; nothing to do")
    if cfg.dry_run:
        log.warning("DRY RUN: deciding and journaling, but not calling any feeder API")
    send = _Sender()
    stop = threading.Event()
    by_camera: dict[str, list[Feeder]] = {}
    for fc in cfg.feeders:
        client = (DryRunClient(fc.id) if cfg.dry_run else
                  FeederClient(api_base_url=fc.api_base_url, serial_number=fc.serial_number, feeder_id=fc.id))
        # One journal connection per feeder: each is used only by its feeder's thread.
        f = Feeder(fc, client, FeedJournal(cfg.journal_db), send)
        log.info("feeder %s: camera=%s allowed=%s feed=%s display=%s door=%s",
                 fc.id, fc.camera, fc.allowed_cats, fc.feed.mode, fc.display or "off", fc.door)
        f.start()
        threading.Thread(target=f.run, args=(stop,), name=f"feeder-{fc.id}", daemon=True).start()
        by_camera.setdefault(fc.camera, []).append(f)

    def session(conn: hubclient.HubConnection) -> None:
        send.conn = conn
        for fs in by_camera.values():
            for f in fs:
                f.request_report()
        try:
            while True:
                msg = conn.recv()
                if msg.get("type") != "result":
                    continue
                for f in by_camera.get(msg.get("camera"), ()):
                    try:
                        f.inbox.put_nowait(msg)
                    except Exception:
                        log.warning("feeder %s is falling behind; dropping a result", f.id)
        finally:
            send.conn = None

    hello = {"role": "decider", "id": args.id, "cameras": sorted(by_camera)}
    hubclient.run_forever(args.hub or cfg.hub, hello, session)


if __name__ == "__main__":
    main()
