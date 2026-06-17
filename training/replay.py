"""Compact replay-memory helpers for classifier retraining.

Replay crops are stored as compressed numpy arrays, not JPEGs. They are a small,
balanced memory of human-reviewed examples that survives video pruning and is
used as train-only data during weekly fine-tuning.

Loading is LAZY: the manifest is parsed into lightweight ``ReplayItem`` records
(label + camera + wall_ms + npz path), and pixels are decoded only on demand,
one crop at a time. This keeps RAM flat regardless of how big the replay set is;
the old eager ``load_replay_set`` held every decoded crop at once.
"""
from __future__ import annotations

import json
import logging
import random
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


log = logging.getLogger(__name__)

DROP_LABELS = {"discard", "unknown"}
ReplayMeta = namedtuple("ReplayMeta", ["label", "camera", "wall_ms"])


class ReplayImageError(RuntimeError):
    """A replay crop's .npz is missing, unreadable, or malformed."""


@dataclass(slots=True)
class ReplayItem:
    """One replay sample's metadata — NO pixels. Decode via
    :func:`decode_replay_image` when the crop is actually needed for a batch."""
    label: str
    camera: str                       # RAW camera id (used for leakage checks)
    wall_ms: int
    npz_path: Path
    src_event_key: int | None = None
    embedding: tuple[float, ...] | None = None

    @property
    def meta(self) -> ReplayMeta:
        """Training metadata. The camera is namespaced ``replay:<cam>`` so replay
        crops form their own episodes and never merge with fresh ones."""
        return ReplayMeta(label=self.label, camera=f"replay:{self.camera}",
                          wall_ms=self.wall_ms)


def iter_manifest(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _manifest_and_root(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    manifest = path / "manifest.jsonl" if path.is_dir() else path
    return manifest, manifest.parent


def iter_replay_items(path: Path) -> Iterator[ReplayItem]:
    """Yield :class:`ReplayItem` metadata from a replay set, decoding nothing."""
    manifest, root = _manifest_and_root(path)
    if not manifest.exists():
        return
    for row in iter_manifest(manifest):
        label = row["label"]
        if label in DROP_LABELS:
            continue
        emb = row.get("embedding")
        yield ReplayItem(
            label=label,
            camera=str(row.get("camera", "unknown")),
            wall_ms=int(row.get("wall_ms", 0)),
            npz_path=root / row["path"],
            src_event_key=(int(row["src_event_key"])
                           if row.get("src_event_key") is not None else None),
            embedding=tuple(float(v) for v in emb) if emb is not None else None,
        )


def limit_replay_items(items: list[ReplayItem], max_items: int | None,
                       seed: int | None) -> list[ReplayItem]:
    """Cap the replay set to ``max_items``, sampling deterministically when a
    ``seed`` is given. Order of the kept items is stable (manifest order), so
    downstream indices/episodes are reproducible.

    A balanced cap: we round-robin across labels so a big class can't crowd the
    others out of the budget."""
    if max_items is None or max_items < 0 or len(items) <= max_items:
        return items
    rng = random.Random(seed) if seed is not None else random.Random()
    by_label: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_label.setdefault(it.label, []).append(i)
    if seed is not None:
        for idxs in by_label.values():
            rng.shuffle(idxs)
    # Round-robin pick across labels until the budget is hit.
    chosen: list[int] = []
    labels = sorted(by_label)
    cursors = {label: 0 for label in labels}
    while len(chosen) < max_items:
        progressed = False
        for label in labels:
            if len(chosen) >= max_items:
                break
            cur = cursors[label]
            if cur < len(by_label[label]):
                chosen.append(by_label[label][cur])
                cursors[label] = cur + 1
                progressed = True
        if not progressed:
            break
    chosen.sort()                                    # restore stable order
    return [items[i] for i in chosen]


def load_replay_items(path: Path, *, max_items: int | None = None,
                      seed: int | None = None) -> list[ReplayItem]:
    """Lazy load: parse the manifest into metadata records (no pixels) and
    optionally cap to ``max_items`` (deterministic with ``seed``)."""
    items = list(iter_replay_items(path))
    return limit_replay_items(items, max_items, seed)


def decode_replay_image(item: ReplayItem, *, missing_ok: bool = False) -> np.ndarray | None:
    """Decode ONE replay crop's pixels from its .npz, on demand.

    Returns a BGR uint8 ndarray. Raises :class:`ReplayImageError` on a missing or
    corrupted file unless ``missing_ok`` (then returns None so a batch can skip
    it)."""
    try:
        with np.load(item.npz_path) as data:
            image = data["image"]
            return np.ascontiguousarray(image.astype(np.uint8, copy=False))
    except (FileNotFoundError, OSError, ValueError, KeyError, EOFError) as e:
        if missing_ok:
            log.warning("skipping unreadable replay crop %s: %r", item.npz_path, e)
            return None
        raise ReplayImageError(f"cannot read replay crop {item.npz_path}: {e!r}") from e


def load_replay_set(path: Path) -> list[tuple[np.ndarray, ReplayMeta]]:
    """Eager loader kept for compatibility: decode every replay crop into RAM.

    Prefer :func:`load_replay_items` + :func:`decode_replay_image` for training —
    this materialises the whole set at once."""
    out: list[tuple[np.ndarray, ReplayMeta]] = []
    for item in iter_replay_items(path):
        image = decode_replay_image(item, missing_ok=True)
        if image is None:
            continue
        out.append((image, item.meta))
    return out
