"""Lazy, byte-bounded classifier dataset: prove the dataset owns only
lightweight refs, decodes one crop per access, and never grows with the whole
dataset.

These were the actual leak in the old TorchLazyCachedDataset, which stored
``lambda c=crop: c`` closures holding every decoded crop in ``_items`` — the
"bounded" cache was a no-op because the dataset itself pinned the data.
"""
import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.crop_cache import BoundedCropCache  # noqa: E402
from training.sources import Box, CropRef, _pad_crop  # noqa: E402
from training.torch_dataset import TorchLazyCachedDataset  # noqa: E402


# A single big synthetic "video frame" all crops are cut from. The decode_fn
# below mimics decode_one_crop: it cuts ONE crop out of this frame and copies it
# (contiguous), so a crop must never share memory with FRAME.
FRAME = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3) % 251


def _ref(i: int) -> CropRef:
    # Vary box position so each crop is distinct content.
    x = (i * 7) % 400
    y = (i * 11) % 300
    box = Box(x=x, y=y, w=80, h=80, cat="alisa", score=0.9, track_id=None, rowid=i)
    return CropRef(camera_id="grey", wall_ms=1000 + i, box=box, rotate_deg=0)


def _decode_from_frame(ref: CropRef) -> np.ndarray:
    crop, _local = _pad_crop(FRAME, ref.box, pad_frac=0.0)
    assert crop is not None
    return crop


def _make_ds(n: int, *, cache_max_mb: float, resize_max_side: int = 64):
    labels = [f"cat{i % 3}" for i in range(n)]
    refs = [_ref(i) for i in range(n)]
    return TorchLazyCachedDataset.from_refs(
        labels, refs, _decode_from_frame,
        transform=lambda bgr: torch.from_numpy(np.ascontiguousarray(bgr)),
        cache_max_mb=cache_max_mb, resize_max_side=resize_max_side,
    )


def test_refs_hold_no_decoded_images_or_closures():
    ds = _make_ds(50, cache_max_mb=1.0)
    len(ds)  # force indexing
    assert ds._refs is not None
    for r in ds._refs:
        assert isinstance(r, CropRef)
        assert not isinstance(r, np.ndarray)
        assert not callable(r)
    # Labels are plain strings, never arrays.
    assert all(isinstance(label, str) for label in ds._labels)
    # No attribute on the dataset holds an ndarray dataset-wide.
    for name, val in vars(ds).items():
        if isinstance(val, list):
            assert not any(isinstance(v, np.ndarray) for v in val), name


def test_deterministic_order_and_labels():
    ds = _make_ds(10, cache_max_mb=1.0)
    labels_a = [ds[i][1] for i in range(len(ds))]
    labels_b = [ds[i][1] for i in range(len(ds))]
    assert labels_a == labels_b == [f"cat{i % 3}" for i in range(10)]


def test_crops_do_not_share_memory_with_frame():
    ds = _make_ds(5, cache_max_mb=1.0)
    for i in range(len(ds)):
        ds[i]
    # Every cached crop must be an independent contiguous copy.
    for crop in ds.cache._store.values():
        assert crop.flags["C_CONTIGUOUS"]
        assert not np.shares_memory(crop, FRAME)


def test_cache_never_exceeds_capacity_and_evicts():
    # ~0.2 MB budget; each 64x64x3 crop ≈ 12 KB → ~16 fit. Iterate 200.
    ds = _make_ds(200, cache_max_mb=0.2, resize_max_side=64)
    for i in range(len(ds)):
        ds[i]
        assert ds.cache.stats.current_bytes <= ds.cache.max_bytes
    assert ds.cache.stats.evictions > 0
    assert ds.cache.stats.peak_bytes <= ds.cache.max_bytes
    # The retained set is far smaller than the dataset.
    assert len(ds.cache) < len(ds)


def test_cache_size_zero_disables_retention():
    ds = _make_ds(20, cache_max_mb=0.0)
    assert not ds.cache.enabled
    for i in range(len(ds)):
        ds[i]
    assert len(ds.cache) == 0
    assert ds.cache.stats.current_bytes == 0
    # Every access was a miss (no retention).
    assert ds.cache.stats.misses == len(ds)


def test_memory_plateaus_not_linear_in_dataset():
    # Iterating many more samples than fit in the cache must not retain them.
    ds = _make_ds(500, cache_max_mb=0.2, resize_max_side=64)
    for i in range(len(ds)):
        ds[i]
    gc.collect()
    retained = sum(crop.nbytes for crop in ds.cache._store.values())
    assert retained <= ds.cache.max_bytes
    # Whole dataset decoded once would be ~500 * 12 KB ≈ 6 MB; retained is < 1 MB.
    full_estimate = 500 * 64 * 64 * 3
    assert retained < full_estimate // 4


def test_bounded_cache_single_oversized_item_not_retained():
    cache = BoundedCropCache(max_mb=0.001)  # 1 KB
    big = np.zeros((100, 100, 3), dtype=np.uint8)  # 30 KB > budget
    out = cache.get(0, lambda: big)
    assert out is big                      # served
    assert len(cache) == 0                  # but not retained
    assert cache.stats.current_bytes == 0


def test_iter_crop_refs_enumerates_without_decoding(tmp_path):
    """CropSource.iter_crop_refs yields lightweight (stub, CropRef) pairs from
    the DB alone — no recordings touched, no pixels decoded."""
    from storage import init_db, insert_event  # detector storage (on sys.path)

    from training.sources import CropSource

    db = tmp_path / "events.db"
    conn = init_db(db)
    for i in range(6):
        insert_event(
            conn, camera_id="grey", model="yolov8n", wall_ms=1000 + i,
            pts=None, tb_num=None, tb_den=None, media_t=None,
            frame_w=320, frame_h=240, rotate_deg=0,
            cat="alisa" if i % 2 == 0 else "chuzh", cat_score=0.9,
            box_x=10 + i, box_y=20, box_w=40, box_h=40, score=0.8,
        )
    conn.close()

    # recordings_root is never read by iter_crop_refs; a bogus path is fine.
    src = CropSource(db_path=db, recordings_root=tmp_path / "nope",
                     min_score=0.5, pad_frac=0.15)
    pairs = list(src.iter_crop_refs())
    assert len(pairs) == 6
    for stub, ref in pairs:
        assert stub.image is None                    # nothing decoded
        assert isinstance(ref, CropRef)
        assert ref.camera_id == "grey"
        assert ref.box.cat in {"alisa", "chuzh"}
        # default target_fn reads stub.boxes[0].cat
        assert stub.boxes[0].cat == ref.box.cat
