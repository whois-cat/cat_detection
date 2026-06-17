"""Tiny RSS probe for compact memory logging during training.

No hard dependency on psutil: falls back to /proc/self/status, then to
resource.getrusage. Returns 0.0 if none are available.
"""
from __future__ import annotations

import logging


def rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes.
        return kb / 1024.0 if kb > 10_000_000 else kb / 1024.0
    except Exception:
        return 0.0


def log_rss(log: logging.Logger, stage: str, **extra) -> None:
    suffix = "".join(f" {k}={v}" for k, v in extra.items())
    log.info("rss %.0f MB @ %s%s", rss_mb(), stage, suffix)
