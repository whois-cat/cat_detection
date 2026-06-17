"""Replay loading must be lazy: metadata in RAM, pixels decoded on demand."""
import dataclasses
import json

import numpy as np
import pytest

from training.replay import (
    ReplayImageError,
    ReplayItem,
    decode_replay_image,
    iter_replay_items,
    limit_replay_items,
    load_replay_items,
)


def _write_replay(tmp_path, rows, *, write_npz=True):
    for row in rows:
        if not write_npz:
            continue
        crop_path = tmp_path / row["path"]
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((6, 7, 3), row.get("fill", 1), dtype=np.uint8)
        np.savez_compressed(crop_path, image=img)
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _rows(n, label="alisa"):
    return [
        {"src_event_key": i, "label": label, "camera": "grey",
         "wall_ms": 1000 + i, "path": f"crops/{label}/{i}.npz", "fill": i % 7}
        for i in range(n)
    ]


def test_iter_replay_items_holds_no_pixels(tmp_path):
    _write_replay(tmp_path, _rows(5))
    items = list(iter_replay_items(tmp_path))
    assert len(items) == 5
    for it in items:
        assert isinstance(it, ReplayItem)
        # No decoded image anywhere on the record (slots dataclass → use fields).
        for f in dataclasses.fields(it):
            assert not isinstance(getattr(it, f.name), np.ndarray)
        assert it.meta.camera == "replay:grey"   # namespaced for episode split
        assert it.camera == "grey"               # raw kept for leakage checks


def test_decode_is_on_demand_and_contiguous(tmp_path):
    _write_replay(tmp_path, _rows(3))
    items = load_replay_items(tmp_path)
    img = decode_replay_image(items[0])
    assert img.shape == (6, 7, 3)
    assert img.dtype == np.uint8
    assert img.flags["C_CONTIGUOUS"]


def test_replay_max_items_limits(tmp_path):
    _write_replay(tmp_path, _rows(20))
    items = load_replay_items(tmp_path, max_items=5, seed=1)
    assert len(items) == 5


def test_replay_sampling_is_deterministic_with_seed(tmp_path):
    _write_replay(tmp_path, _rows(30))
    a = [it.src_event_key for it in load_replay_items(tmp_path, max_items=7, seed=42)]
    b = [it.src_event_key for it in load_replay_items(tmp_path, max_items=7, seed=42)]
    c = [it.src_event_key for it in load_replay_items(tmp_path, max_items=7, seed=99)]
    assert a == b
    assert a != c  # different seed → different sample (overwhelmingly likely)
    # Kept order is stable (manifest order), not shuffled.
    assert a == sorted(a)


def test_limit_balances_across_classes(tmp_path):
    items = [
        ReplayItem(label="alisa", camera="g", wall_ms=i, npz_path=tmp_path / f"{i}")
        for i in range(100)
    ] + [
        ReplayItem(label="chuzh", camera="g", wall_ms=1000 + i, npz_path=tmp_path / f"c{i}")
        for i in range(3)
    ]
    kept = limit_replay_items(items, max_items=6, seed=0)
    labels = [it.label for it in kept]
    # Round-robin keeps the rare class from being crowded out.
    assert labels.count("chuzh") == 3
    assert labels.count("alisa") == 3


def test_missing_replay_image_raises_or_skips(tmp_path):
    rows = _rows(1)
    _write_replay(tmp_path, rows, write_npz=False)  # manifest only, no .npz
    item = load_replay_items(tmp_path)[0]
    with pytest.raises(ReplayImageError):
        decode_replay_image(item)
    assert decode_replay_image(item, missing_ok=True) is None


def test_corrupt_replay_image_handled(tmp_path):
    rows = _rows(1)
    _write_replay(tmp_path, rows, write_npz=False)
    crop_path = tmp_path / rows[0]["path"]
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"not a real npz file")
    item = load_replay_items(tmp_path)[0]
    with pytest.raises(ReplayImageError):
        decode_replay_image(item)
    assert decode_replay_image(item, missing_ok=True) is None
