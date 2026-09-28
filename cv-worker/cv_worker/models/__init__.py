"""Models: turn an image into detections. Everything else (streams, timing,
geometry) is the harness's job, so models stay small and swappable."""
from __future__ import annotations

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

    def infer(self, img_bgr: np.ndarray) -> list[Det]: ...


def build(kind: str) -> Model:
    """Build a model by name; model-level settings come from the environment."""
    env = os.environ.get
    if kind == "blob":
        from .blob import BlobModel
        return BlobModel(threshold=int(env("BLOB_THRESHOLD", "240")), min_area=int(env("BLOB_MIN_AREA", "500")))
    if kind in ("yolo", "yolo_cat"):
        from .yolo import YoloModel
        return YoloModel(
            weights=env("YOLO_WEIGHTS", "/opt/models/yolov8n_int8_openvino_model/"),
            conf=float(env("YOLO_CONF", "0.25")),
            classifier_dir=env("CLASSIFIER_DIR", "/opt/models/classifier/current") if kind == "yolo_cat" else None,
            # Must match the padding used when training the classifier.
            pad_frac=float(env("CLASSIFIER_PAD_FRAC", "0.05")),
        )
    raise ValueError(f"unknown model {kind!r} (expected blob, yolo, yolo_cat)")
