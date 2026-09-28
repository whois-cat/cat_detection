"""cv-worker: connect to streamhub's hub port and serve CV for its cameras."""
from __future__ import annotations

import argparse
import logging
import os
import socket

import hubclient

from .harness import Harness
from .models import build


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub", default=os.environ.get("STREAMHUB_HUB", "127.0.0.1:9000"), help="streamhub hub address")
    p.add_argument("--model", default=os.environ.get("CV_MODEL", "yolo_cat"), help="blob, yolo or yolo_cat")
    p.add_argument("--cameras", default=os.environ.get("CV_CAMERAS", "*"), help="comma-separated ids, or *")
    p.add_argument("--id", default=os.environ.get("CV_WORKER_ID", socket.gethostname()), help="worker id")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    model = build(args.model)
    logging.getLogger("cv_worker").info("model %s@%s", model.name, model.version)
    hello = {"role": "cv", "id": args.id, "cameras": [c for c in args.cameras.split(",") if c],
             "model": {"name": model.name, "version": model.version}}
    hubclient.run_forever(args.hub, hello, Harness(model).session)


if __name__ == "__main__":
    main()
