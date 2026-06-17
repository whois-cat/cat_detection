"""Byte-bounded LRU cache for decoded training crops.

Multi-epoch classifier training re-reads the same crops every epoch. Decoding
from the recordings each time is slow; materialising *all* crops in RAM (the old
TorchCachedDataset) blows up memory on CPU boxes. This is the middle ground: a
lazy cache that holds only resized, contiguous ``uint8`` crops up to a hard byte
budget, evicting least-recently-used entries past it. Full frames are never
stored (the producer must already have copied the crop out of its frame).

Nothing is written to disk. ``max_mb <= 0`` disables caching entirely (every
access is a miss and the value is recomputed by the caller's loader).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CacheStats:
    current_bytes: int = 0
    peak_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def as_dict(self) -> dict:
        return {
            "current_bytes": self.current_bytes,
            "peak_bytes": self.peak_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class BoundedCropCache:
    """LRU cache keyed by sample index, bounded by total stored bytes.

    Values are ``uint8`` ndarrays (resized crops). The cache stores them as-is;
    callers must hand in contiguous arrays that don't alias a larger buffer
    (otherwise ``nbytes`` under-counts the real footprint).
    """

    def __init__(self, max_mb: float) -> None:
        self.max_bytes = int(max(0.0, max_mb) * 1024 * 1024)
        self._store: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: int, loader: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the cached crop for ``key``, computing it via ``loader`` on a
        miss and inserting it (subject to the byte budget)."""
        if not self.enabled:
            self.stats.misses += 1
            return loader()
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return hit
        self.stats.misses += 1
        value = loader()
        self._insert(key, value)
        return value

    def _insert(self, key: int, value: np.ndarray) -> None:
        nbytes = int(value.nbytes)
        # A single item larger than the whole budget is served but not retained.
        if nbytes > self.max_bytes:
            return
        if key in self._store:
            self.stats.current_bytes -= int(self._store[key].nbytes)
            del self._store[key]
        self._store[key] = value
        self.stats.current_bytes += nbytes
        while self.stats.current_bytes > self.max_bytes and self._store:
            _ek, ev = self._store.popitem(last=False)  # LRU end
            self.stats.current_bytes -= int(ev.nbytes)
            self.stats.evictions += 1
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.current_bytes)

    def clear(self) -> None:
        self._store.clear()
        self.stats.current_bytes = 0
