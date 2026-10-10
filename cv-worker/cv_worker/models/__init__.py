"""Models: turn an image into detections. Everything else (streams, timing,
geometry) is the harness's job, so models stay small and swappable."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Det:
    box: tuple[float, float, float, float]  # x, y, w, h in model-input pixels
    score: float  # detector confidence that this is a cat
    cats: dict[str, float] = field(default_factory=dict)  # identity probabilities


class Model(Protocol):
    name: str
    version: str

    # config: the camera's cv settings from config.yaml (model-specific keys,
    # e.g. yolo_conf; geometry keys are already applied by the harness).
    def infer(self, img_bgr: np.ndarray, config: dict) -> list[Det]: ...


# Deployed with `just deploy detector` (tools/models.py); until then, the COCO
# yolov8n exported into the image at build time.
DEPLOYED_DETECTOR = "/opt/models/detector/current"
BUILTIN_DETECTOR = "/opt/models/yolov8n_int8_openvino_model/"


def detector_weights(env=os.environ.get) -> tuple[str, str | None]:
    """(weights, label): a YOLO_WEIGHTS override, else the deployed version's
    OpenVINO export labelled with its run name, else the built-in model."""
    if env("YOLO_WEIGHTS"):
        return env("YOLO_WEIGHTS"), None
    if os.path.exists(DEPLOYED_DETECTOR):
        exports = glob.glob(os.path.join(DEPLOYED_DETECTOR, "*_openvino_model"))
        if len(exports) != 1:
            raise RuntimeError(
                f"{DEPLOYED_DETECTOR} must hold exactly one *_openvino_model export, got {exports}")
        return exports[0], os.path.basename(os.path.realpath(DEPLOYED_DETECTOR))
    return BUILTIN_DETECTOR, None


def build(kind: str) -> Model:
    """Build a model by name; model-level settings come from the environment."""
    env = os.environ.get
    if kind == "blob":
        from .blob import BlobModel
        return BlobModel(threshold=int(env("BLOB_THRESHOLD", "240")), min_area=int(env("BLOB_MIN_AREA", "500")))
    if kind in ("yolo", "yolo_cat"):
        from .yolo import YoloModel
        weights, label = detector_weights(env)
        return YoloModel(
            weights=weights,
            label=label,
            conf=float(env("YOLO_CONF", "0.25")),
            classifier_dir=env("CLASSIFIER_DIR", "/opt/models/classifier/current") if kind == "yolo_cat" else None,
            # Must match the padding used when training the classifier.
            pad_frac=float(env("CLASSIFIER_PAD_FRAC", "0.05")),
        )
    raise ValueError(f"unknown model {kind!r} (expected blob, yolo, yolo_cat)")
