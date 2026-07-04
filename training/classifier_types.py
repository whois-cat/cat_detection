"""Shared training data types for the identity-classifier pipeline."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Meta:
    label: str
    camera: str
    wall_ms: int
    rowid: int | None = None
    duplicate_group_id: str | None = None
    suspicious_score: float = 0.0
    sampling_reason: str | None = None


CropRefLite = namedtuple("CropRefLite", ["camera_id", "wall_ms", "box", "rotate_deg"])


TrainItem = namedtuple("TrainItem", ["meta", "ref", "image", "replay"])
