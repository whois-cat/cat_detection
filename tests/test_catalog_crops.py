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
    manifest_item,
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
        conn.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?)",
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


def test_manifest_item_points_review_app_at_catalog(catalog: Path):
    crop = load_catalog_crops(catalog)[0]
    item = manifest_item(crop, 0.05)
    assert item["src_event_key"] == crop_key("two_cats", 1)
    assert item["catalog_image"] == "images/black/two_cats.jpg"
    assert item["rotate_deg"] == 0      # catalog frames are already in model orientation


def test_time_manifest_from_catalog(catalog: Path, tmp_path: Path, monkeypatch):
    from training import build_cluster_manifest

    out = tmp_path / "clusters.json"
    monkeypatch.setattr(sys, "argv", [
        "build_cluster_manifest", "--catalog", str(catalog), "--out", str(out),
        "--mode", "time", "--pad-frac", "0.05", "--dedupe-window-sec", "0",
    ])
    build_cluster_manifest.main()
    manifest = json.loads(out.read_text())
    assert manifest["params"]["source"] == "catalog"
    assert manifest["params"]["catalog"] == str(catalog)
    assert sorted(i["crop_id"] for i in manifest["items"]) == [
        "grey_cat:1", "two_cats:1", "two_cats:2"]
    # one visit per camera -> one cluster per camera
    assert len(manifest["clusters"]) == 2


def test_review_app_serves_catalog_crops(catalog: Path, tmp_path: Path, monkeypatch):
    import importlib.util
    import io

    from starlette.testclient import TestClient

    from training import build_cluster_manifest

    out = tmp_path / "clusters.json"
    monkeypatch.setattr(sys, "argv", [
        "build_cluster_manifest", "--catalog", str(catalog), "--out", str(out),
        "--mode", "time", "--pad-frac", "0.0",
    ])
    build_cluster_manifest.main()
    monkeypatch.setenv("CLUSTER_MANIFEST", str(out))
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("RECORDINGS_ROOT", str(tmp_path / "no-recordings"))
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cluster_app_catalog",
                                                  repo / "review" / "cluster_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with TestClient(mod.app) as client:
        resp = client.get("/api/crop/two_cats:1", params={"thumb": 64})
    assert resp.status_code == 200, resp.text
    thumb = Image.open(io.BytesIO(resp.content)).convert("RGB")
    # centre pixel is the frame's red, not the "missing recording" placeholder
    assert thumb.getpixel((32, 32))[0] > 150
