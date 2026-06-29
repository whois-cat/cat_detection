"""Detector runtime /status assembly.

Exercises the dependency-free status builder (detector/status.py), the generic
Detector.status() fragment, and classifier version resolution — without aiohttp,
av, OpenVINO, or any real model. No hardcoded cat/camera/class/feeder names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DETECTOR = Path(__file__).resolve().parents[1] / "detector"
if str(DETECTOR) not in sys.path:
    sys.path.insert(0, str(DETECTOR))

from classifier import resolve_model_version  # noqa: E402
from detectors import Detector  # noqa: E402
from status import build_status  # noqa: E402


class _ClfStub:
    """Stands in for CatClassifier.status() — no OpenVINO needed."""

    def __init__(self, n=3, version="20260629-001122",
                 model_dir="/opt/models/classifier/current"):
        self._n, self._v, self._d = n, version, model_dir

    def status(self):
        return {
            "enabled": True,
            "model_dir": self._d,
            "resolved_version": self._v,
            "classes_count": self._n,
            "format": "openvino",
        }


class _EnabledDet(Detector):
    model_name = "yolov8n_int8_openvino_model+cat"
    backend = "openvino"
    conf = 0.25

    def __init__(self, clf):
        self._classifier = clf

    def detect(self, img):  # pragma: no cover - not exercised
        return []


class _DisabledDet(Detector):
    """No classifier and no conf — e.g. the blob/plain-yolo path."""

    model_name = "blob-dummy"
    backend = "opencv"

    def detect(self, img):  # pragma: no cover - not exercised
        return []


def _status(detector, **kw):
    base = dict(camera_id="camera_1", now=100.0, start_monotonic=0.0)
    base.update(kw)
    return build_status(detector, **base)


def test_status_with_classifier_enabled():
    s = _status(_EnabledDet(_ClfStub(n=3)))
    assert s["service"] == "detector"
    assert s["camera_id"] == "camera_1"
    assert s["uptime_sec"] == 100.0
    assert s["classifier"] == {
        "enabled": True,
        "model_dir": "/opt/models/classifier/current",
        "resolved_version": "20260629-001122",
        "classes_count": 3,
        "format": "openvino",
    }
    assert s["detector"] == {
        "enabled": True, "backend": "openvino",
        "model": "yolov8n_int8_openvino_model+cat", "min_score": 0.25,
    }


def test_status_with_classifier_disabled():
    s = _status(_DisabledDet())
    assert s["classifier"] == {"enabled": False}
    # min_score is None when the detector has no confidence threshold.
    assert s["detector"] == {
        "enabled": True, "backend": "opencv", "model": "blob-dummy", "min_score": None,
    }


def test_status_missing_detector_is_graceful():
    s = _status(None)
    assert s["detector"]["enabled"] is False
    assert s["classifier"] == {"enabled": False}
    assert s["camera_id"] == "camera_1"


def test_status_exposes_count_not_labels():
    # Class COUNT is visible; no label strings are leaked into the snapshot.
    s = _status(_EnabledDet(_ClfStub(n=5)))
    assert s["classifier"]["classes_count"] == 5
    assert "classes" not in s["classifier"]
    assert "class_names" not in s["classifier"]


def test_resolve_version_follows_current_symlink(tmp_path):
    versions = tmp_path / "versions" / "20260629-001122"
    versions.mkdir(parents=True)
    current = tmp_path / "current"
    current.symlink_to("versions/20260629-001122")
    assert resolve_model_version(current) == "20260629-001122"


def test_resolve_version_unresolvable_is_safe(tmp_path):
    # A dangling symlink resolves to its own basename rather than crashing.
    assert isinstance(resolve_model_version(tmp_path / "nope"), str)


def test_runtime_ages_and_totals():
    s = _status(
        _EnabledDet(_ClfStub()),
        now=100.0,
        last_frame_monotonic=98.8,
        last_detection_monotonic=95.2,
        detections_total=382,
    )
    assert s["runtime"] == {
        "last_frame_age_sec": 1.2,
        "last_detection_age_sec": 4.8,
        "detections_total": 382,
    }


def test_runtime_ages_none_before_first_frame():
    s = _status(_EnabledDet(_ClfStub()))
    assert s["runtime"]["last_frame_age_sec"] is None
    assert s["runtime"]["last_detection_age_sec"] is None
    assert s["runtime"]["detections_total"] == 0


def test_last_error_only_present_when_set():
    assert "last_error" not in _status(_EnabledDet(_ClfStub()))
    s = _status(_EnabledDet(_ClfStub()), last_error="frame decode failed: ValueError()")
    assert s["last_error"] == "frame decode failed: ValueError()"


def test_no_secrets_exposed():
    # Top-level keys are a fixed, small allow-list; nothing in the serialized
    # snapshot looks like an RTSP URL or credential.
    s = _status(_EnabledDet(_ClfStub()), last_error="boom")
    assert set(s) <= {
        "service", "camera_id", "uptime_sec",
        "classifier", "detector", "runtime", "last_error",
    }
    blob = json.dumps(s).lower()
    for needle in ("rtsp", "://", "password", "secret", "@", "passwd", "token"):
        assert needle not in blob
