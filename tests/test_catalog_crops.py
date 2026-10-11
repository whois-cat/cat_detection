"""Classifier crops from the YOLO catalog's human-verified boxes (no events.db)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.catalog_crops import (
    crop_key,
    cut_crop,
    load_catalog_crops,
    read_catalog_crop,
)
from training.sources import CropUnavailable
from training.streamhub_dataset import open_catalog


def _sample(conn, root: Path, sid: str, status: str, wall_ms: int, camera: str = "black",
            boxes=((10, 20, 30, 40),)) -> None:
    rel = f"images/{camera}/{sid}.jpg"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), (200, 50, 50)).save(root / rel)
    conn.execute(
        """INSERT INTO samples(
             sample_id,camera,segment_relpath,pts,wall_ms,visit_group,image_relpath,width,height,
             jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,status,protected,created_at_ms
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, camera, f"{camera}/seg.mp4", wall_ms * 90, wall_ms, f"{camera}:v1", rel,
         100, 80, 100, f"sha-{sid}", "0" * 16, "[]", 90, status, 1, int(time.time() * 1000)),
    )
    for index, (x, y, w, h) in enumerate(boxes, 1):
        conn.execute("INSERT INTO annotations(sample_id,annotation_id,class_name,x,y,w,h,source,"
            "updated_at_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                     (sid, index, "cat", x, y, w, h, "label_studio", 0))


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.sqlite3"
    conn = open_catalog(path)
    _sample(conn, tmp_path, "two_cats", "verified", 2_000, boxes=((10, 20, 30, 40), (50, 5, 20, 20)))
    _sample(conn, tmp_path, "empty", "verified", 3_000, boxes=())
    _sample(conn, tmp_path, "unreviewed", "unreviewed", 1_000)
    _sample(conn, tmp_path, "grey_cat", "verified", 500, camera="grey")
    conn.commit()
    conn.close()
    return path


def test_crop_key_is_stable_and_distinct():
    assert crop_key("s1", 1) == crop_key("s1", 1)
    assert crop_key("s1", 1) != crop_key("s1", 2)
    assert 0 < crop_key("s1", 1) < 2**63


def test_only_verified_boxes_become_crops(catalog: Path):
    crops = load_catalog_crops(catalog)
    assert [c.crop_id for c in crops] == ["two_cats:1", "two_cats:2", "grey_cat:1"]
    assert [c.crop_id for c in load_catalog_crops(catalog, camera="grey")] == ["grey_cat:1"]


def test_crop_uses_runtime_padding_and_clamps_to_frame():
    frame = np.zeros((80, 100, 3), np.uint8)
    # pad = int(0.1 * max(30, 40)) = 4 on every side
    assert cut_crop(frame, (10, 20, 30, 40), 0.1).shape == (48, 38, 3)
    # clamped at the left/top edges
    assert cut_crop(frame, (0, 0, 30, 40), 0.1).shape == (44, 34, 3)


def test_read_crop_from_frame_and_missing_frame_raises(catalog: Path):
    crop = load_catalog_crops(catalog)[0]
    img = read_catalog_crop(catalog.parent, crop.image_relpath, crop.box, 0.0)
    assert img.shape == (40, 30, 3)
    assert tuple(img[0, 0]) == pytest.approx((50, 50, 200), abs=3)   # BGR of (200,50,50)
    with pytest.raises(CropUnavailable):
        read_catalog_crop(catalog.parent, "images/black/gone.jpg", crop.box, 0.0)


def test_crops_carry_the_cat_name_picked_in_label_studio(catalog: Path):
    import sqlite3
    from training.catalog_crops import name_counts
    conn = sqlite3.connect(catalog)
    conn.execute("UPDATE annotations SET cat_name='ellie' WHERE sample_id='two_cats' AND annotation_id=1")
    conn.execute("UPDATE annotations SET cat_name='unknown' WHERE sample_id='grey_cat'")
    conn.commit()
    conn.close()
    names = {c.crop_id: c.name for c in load_catalog_crops(catalog)}
    assert names == {"two_cats:1": "ellie", "two_cats:2": None, "grey_cat:1": "unknown"}
    table = name_counts(load_catalog_crops(catalog))
    assert table.splitlines()[1].split() == ["ellie", "1", "1", "0"]   # black, grey
