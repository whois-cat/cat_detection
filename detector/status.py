"""Runtime status assembly for the detector `/status` endpoint.

Kept dependency-free (stdlib only) so it is unit-testable without aiohttp / av /
OpenVINO. `build_status` assembles a small, inspectable snapshot of runtime
state for MVP/demo debugging — no frames, no labels, no secrets, no heavy data.
"""
from __future__ import annotations

SERVICE_NAME = "detector"

# Status fragment used when the detector hasn't been built yet (or failed to).
_NO_DETECTOR = {
    "detector": {"enabled": False, "backend": "unknown", "model": None, "min_score": None},
    "classifier": {"enabled": False},
}


def _age(now: float, t: float | None) -> float | None:
    """Seconds since monotonic timestamp `t`, or None if never set."""
    return round(now - t, 3) if t is not None else None


def build_status(
    detector,
    *,
    camera_id: str,
    now: float,
    start_monotonic: float,
    last_frame_monotonic: float | None = None,
    last_detection_monotonic: float | None = None,
    detections_total: int = 0,
    last_error: str | None = None,
) -> dict:
    """Assemble the runtime status dict.

    `detector` is any object exposing `.status()` returning {"detector": {...},
    "classifier": {...}} (see detectors.Detector.status), or None before it is
    built. All time inputs are monotonic clock values; ages are derived here so
    the snapshot is wall-clock-independent.
    """
    frag = detector.status() if detector is not None else _NO_DETECTOR
    status = {
        "service": SERVICE_NAME,
        "camera_id": camera_id,
        "uptime_sec": round(now - start_monotonic, 3),
        "classifier": frag["classifier"],
        "detector": frag["detector"],
        "runtime": {
            "last_frame_age_sec": _age(now, last_frame_monotonic),
            "last_detection_age_sec": _age(now, last_detection_monotonic),
            "detections_total": detections_total,
        },
    }
    if last_error:
        status["last_error"] = last_error
    return status
