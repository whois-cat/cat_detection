from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from training.streamhub_dataset import open_catalog
from training.yolo_build_version import build_version
from training.yolo_common import load_dataset


def _write_jpeg(path: Path, value: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 24, 3), value % 256, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return encoded.tobytes()


def _insert_sample(conn, root: Path, *, sample_id: str, camera: str, wall_ms: int,
                   sha256: str, dhash: int, boxes: list[tuple], width: int = 24,
                   height: int = 32) -> None:
    rel = f"images/{camera}/{sample_id}.jpg"
    _write_jpeg(root / rel, hash(sha256) & 0xFF)
    conn.execute(
        """INSERT INTO samples(
             sample_id,camera,segment_relpath,pts,wall_ms,image_relpath,width,height,
             jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,status,protected,created_at_ms
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sample_id, camera, f"{camera}/seg.mp4", wall_ms * 90, wall_ms, rel, width, height,
         100, sha256, f"{dhash:016x}", "[]", 90, "verified", 1, int(time.time() * 1000)),
    )
    for index, (x, y, w, h) in enumerate(boxes, 1):
        conn.execute(
            "INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?)",
            (sample_id, index, "cat", x, y, w, h, "human_cvat", 0),
        )


def _build(tmp_path: Path, groups: int = 40):
    root = tmp_path / "yolo_dataset"
    root.mkdir()
    conn = open_catalog(root / "catalog.sqlite3")
    try:
        for g in range(groups):
            wall = 1_000_000 + g * 10_000_000  # each visit far beyond the 120s gap
            # Two cameras in the SAME visit must stay in one split.
            _insert_sample(conn, root, sample_id=f"s{g}a", camera="black", wall_ms=wall,
                           sha256=f"hash-{g}-a", dhash=g * 3, boxes=[(4, 4, 10, 12)])
            _insert_sample(conn, root, sample_id=f"s{g}b", camera="grey", wall_ms=wall + 500,
                           sha256=f"hash-{g}-b", dhash=g * 3 + 1,
                           boxes=[] if g % 7 == 0 else [(2, 2, 8, 8)])
        conn.commit()
    finally:
        conn.close()
    return build_version(
        root / "catalog.sqlite3", root, root / "versions",
        val_frac=0.15, test_frac=0.15, group_gap_sec=120.0, dup_threshold=3,
        min_groups=6, seed=1,
    )


def test_build_version_is_loadable_and_group_split(tmp_path: Path):
    result = _build(tmp_path)
    version_dir = Path(result["path"])

    dataset = load_dataset(version_dir, require_splits=("train", "val", "test"))
    assert dataset.cat_id == 0
    assert dataset.summary["images"] == result["summary"]["images"]

    # No visit group may appear in more than one split (cross-camera included).
    group_to_split: dict[str, str] = {}
    for sample in dataset.samples:
        prior = group_to_split.setdefault(sample["group_id"], sample["split"])
        assert prior == sample["split"], sample["group_id"]

    # Both cameras of a visit share the visit's group and therefore its split.
    by_id = {s["sample_id"]: s for s in dataset.samples}
    assert by_id["s0a"]["group_id"] == by_id["s0b"]["group_id"]

    # Confirmed-empty frames are negatives (empty label file), not dropped.
    assert result["summary"]["negative"] > 0


def test_build_version_dedupes_exact_duplicate_images(tmp_path: Path):
    root = tmp_path / "yolo_dataset"
    root.mkdir()
    conn = open_catalog(root / "catalog.sqlite3")
    try:
        for g in range(8):
            wall = 1_000_000 + g * 10_000_000
            _insert_sample(conn, root, sample_id=f"s{g}", camera="black", wall_ms=wall,
                           sha256="same-pixels", dhash=0, boxes=[(4, 4, 10, 12)])
        conn.commit()
    finally:
        conn.close()
    result = build_version(
        root / "catalog.sqlite3", root, root / "versions",
        val_frac=0.15, test_frac=0.15, group_gap_sec=120.0, dup_threshold=3,
        min_groups=1, seed=1,
    )
    # Eight byte-identical images collapse to one.
    assert result["summary"]["images"] == 1
    assert result["summary"]["exact_duplicates_removed"] == 7


def test_build_version_refuses_without_reviewed_samples(tmp_path: Path):
    root = tmp_path / "yolo_dataset"
    root.mkdir()
    open_catalog(root / "catalog.sqlite3").close()
    import pytest

    from training.yolo_build_version import BuildError

    with pytest.raises(BuildError):
        build_version(root / "catalog.sqlite3", root, root / "versions",
                      val_frac=0.15, test_frac=0.15, group_gap_sec=120.0,
                      dup_threshold=3, min_groups=1, seed=1)
