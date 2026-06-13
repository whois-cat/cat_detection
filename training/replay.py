"""Compact replay-memory helpers for classifier retraining.

Replay crops are stored as compressed numpy arrays, not JPEGs. They are a small,
balanced memory of human-reviewed examples that survives video pruning and is
used as train-only data during weekly fine-tuning.
"""
from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import numpy as np


DROP_LABELS = {"discard", "unknown"}
ReplayMeta = namedtuple("ReplayMeta", ["label", "camera", "wall_ms"])


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_replay_set(path: Path) -> list[tuple[np.ndarray, ReplayMeta]]:
    """Return replay crops as (BGR image, Meta) pairs."""
    path = Path(path)
    manifest = path / "manifest.jsonl" if path.is_dir() else path
    root = manifest.parent
    out: list[tuple[np.ndarray, Meta]] = []
    if not manifest.exists():
        return out
    for row in iter_manifest(manifest):
        label = row["label"]
        if label in DROP_LABELS:
            continue
        crop_path = root / row["path"]
        with np.load(crop_path) as data:
            image = data["image"].astype(np.uint8, copy=False)
        out.append((
            image,
            ReplayMeta(
                label=label,
                camera=f"replay:{row.get('camera', 'unknown')}",
                wall_ms=int(row.get("wall_ms", 0)),
            ),
        ))
    return out
