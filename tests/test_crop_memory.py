"""P0: crops must not share memory with (or keep alive) the decoded frame, and
the bounded LRU crop cache must respect its byte budget."""
import numpy as np

from training.crop_cache import BoundedCropCache
from training.sources import Box, _pad_crop


def test_crop_is_independent_copy():
    frame = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    box = Box(x=50, y=40, w=30, h=30, cat="alisa", score=0.9, track_id=None)
    crop, _local = _pad_crop(frame, box, pad_frac=0.0)
    assert crop is not None
    assert crop.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(crop, frame)
    assert crop.base is None


def test_cache_respects_byte_limit_and_evicts_lru():
    # ~1 MB budget; each crop is 256*256*3 ≈ 192 KB → ~5 fit.
    cache = BoundedCropCache(max_mb=1.0)
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(20):
        cache.get(i, lambda c=crop.copy(): c)
    assert cache.stats.current_bytes <= cache.max_bytes
    assert cache.stats.evictions > 0
    assert cache.stats.peak_bytes <= cache.max_bytes
    assert len(cache) <= cache.max_bytes // crop.nbytes + 1


def test_cache_hit_then_evict_drops_reference():
    cache = BoundedCropCache(max_mb=0.4)  # ~2 crops of 192 KB
    crop = np.zeros((256, 256, 3), dtype=np.uint8)
    cache.get(0, lambda: crop.copy())
    assert cache.get(0, lambda: (_ for _ in ()).throw(AssertionError("hit")) ) is not None
    assert cache.stats.hits == 1
    # Push enough to evict key 0; its slot must be gone.
    for i in range(1, 10):
        cache.get(i, lambda: crop.copy())
    assert cache.stats.evictions > 0
    assert 0 not in cache._store


def test_cache_disabled_passthrough():
    cache = BoundedCropCache(max_mb=0)
    assert not cache.enabled
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    cache.get(0, loader)
    cache.get(0, loader)  # no retention → recomputed
    assert calls["n"] == 2
    assert len(cache) == 0
    assert cache.stats.misses == 2
