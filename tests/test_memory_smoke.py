"""Memory smoke test: iterating far more samples than the cache holds must keep
RSS flat (bounded by the cache), not grow with the whole dataset.

This is the regression guard for the original leak, where every decoded crop was
pinned in the dataset's ``_items`` and the "bounded" cache did nothing.
"""
import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.ram import rss_mb  # noqa: E402
from training.sources import Box, CropRef, _pad_crop  # noqa: E402
from training.torch_dataset import TorchLazyCachedDataset  # noqa: E402

# One big frame all crops are cut+copied from (mirrors decode_one_crop).
FRAME = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)


def _ref(i):
    x = (i * 53) % 1600
    y = (i * 31) % 900
    box = Box(x=x, y=y, w=224, h=224, cat="alisa", score=0.9, track_id=None, rowid=i)
    return CropRef("grey", 1000 + i, box, 0)


def _decode(ref):
    crop, _ = _pad_crop(FRAME, ref.box, 0.0)
    return crop


def test_rss_plateaus_below_full_dataset():
    n = 2000
    cache_mb = 16
    ds = TorchLazyCachedDataset.from_refs(
        [f"c{i % 4}" for i in range(n)], [_ref(i) for i in range(n)], _decode,
        transform=lambda b: torch.from_numpy(np.ascontiguousarray(b)),
        cache_max_mb=cache_mb, resize_max_side=224,
    )

    gc.collect()
    rss_before = rss_mb()
    for i in range(n):
        x, _y = ds[i]
        del x
    gc.collect()
    rss_after = rss_mb()

    retained = sum(c.nbytes for c in ds.cache._store.values())
    full_if_held_mb = n * 224 * 224 * 3 / 1e6   # what the old code would pin (MB)

    # Cache stayed within budget and evicted the rest.
    assert retained <= ds.cache.max_bytes
    assert ds.cache.stats.evictions > 0
    # RSS growth is bounded by the cache, nowhere near the full dataset. Lenient
    # bound (full/4) so the test isn't flaky on shared CI; the real delta is ~1x
    # the cache budget.
    if rss_before > 0 and rss_after > 0:        # rss probe available
        assert (rss_after - rss_before) < full_if_held_mb / 4
