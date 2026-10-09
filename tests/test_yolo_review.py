from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from training.streamhub_dataset import open_catalog
from training.yolo_review import export_batch, import_batch, queue_status, requeue_batch


def add_sample(conn, root: Path, sid: str, prediction: list[dict] | None = None):
    rel = Path("images/black") / f"{sid}.jpg"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"jpeg-placeholder"
    path.write_bytes(payload)
    offset = sum(sid.encode())
    conn.execute(
        """INSERT INTO samples(
          sample_id,camera,segment_relpath,pts,wall_ms,image_relpath,width,height,jpeg_bytes,
          sha256,dhash,reasons_json,rotate_deg,created_at_ms
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, "black", "seg.mp4", 90_000 + offset, 1_000 + offset, rel.as_posix(), 100, 80, len(payload),
         hashlib.sha256(payload).hexdigest(), "0", '["regular"]', 90, 1_000),
    )
    if prediction is not None:
        conn.execute(
            "INSERT INTO predictions(sample_id,detections_json) VALUES(?,?)",
            (sid, json.dumps(prediction)),
        )
    conn.commit()


def rewrite_coco(package: Path, mutate):
    with zipfile.ZipFile(package) as source:
        files = {name: source.read(name) for name in source.namelist()}
    coco = json.loads(files["annotations/instances_default.json"])
    mutate(coco)
    files["annotations/instances_default.json"] = json.dumps(coco).encode()
    with zipfile.ZipFile(package, "w") as target:
        for name, payload in files.items():
            target.writestr(name, payload)


def rewrite_manifest(package: Path, mutate):
    with zipfile.ZipFile(package) as source:
        files = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(files["manifest.json"])
    mutate(manifest)
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(package, "w") as target:
        for name, payload in files.items():
            target.writestr(name, payload)


def test_cvat_roundtrip_keeps_suggestions_separate_until_import(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample-a", [{"box": [0.1, 0.2, 0.3, 0.4], "score": 0.2}])
    conn.close()
    package = tmp_path / "batch.zip"
    result = export_batch(catalog, tmp_path, package, 10)
    assert result["suggestions"] == 1
    conn = open_catalog(catalog)
    assert conn.execute("SELECT count(*) FROM annotations").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM samples").fetchone()[0] == "exported"
    conn.close()
    imported = import_batch(catalog, package, confirm_empty=False)
    assert imported["objects"] == 1
    conn = open_catalog(catalog)
    row = conn.execute("SELECT class_name,x,y,w,h,source FROM annotations").fetchone()
    assert tuple(row) == ("cat", 10.0, 16.0, 30.0, 32.0, "human_cvat")
    assert conn.execute("SELECT status FROM samples").fetchone()[0] == "verified"
    conn.close()


def test_export_auto_names_under_exports_dir(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample-a", [{"box": [0.1, 0.2, 0.3, 0.4], "score": 0.2}])
    conn.close()
    result = export_batch(catalog, tmp_path, None, 10)  # no filename given
    written = Path(result["path"])
    assert written.parent == tmp_path / "exports"
    assert written.name == f"{result['batch_id']}.zip"
    assert written.is_file()
    # A second auto-named export of fresh samples is a different, non-colliding file.
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample-b", [{"box": [0.1, 0.2, 0.3, 0.4], "score": 0.3}])
    conn.close()
    second = Path(export_batch(catalog, tmp_path, None, 10)["path"])
    assert second != written and second.is_file()


def test_empty_frame_requires_explicit_human_confirmation(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "empty")
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 10, include_suggestions=False)
    with pytest.raises(ValueError, match="confirm-empty"):
        import_batch(catalog, package, confirm_empty=False)
    result = import_batch(catalog, package, confirm_empty=True)
    assert result["confirmed_empty"] == 1


def test_import_rejects_out_of_bounds_box(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "bad")
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 10, include_suggestions=False)

    def add_bad(coco):
        coco["annotations"] = [{"id": 1, "image_id": coco["images"][0]["id"],
                                "category_id": 1, "bbox": [90, 0, 20, 20]}]

    rewrite_coco(package, add_bad)
    with pytest.raises(ValueError, match="invalid bbox"):
        import_batch(catalog, package, confirm_empty=False)


def test_import_matches_cvat_export_without_custom_manifest(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "from-cvat", [{"box": [0.1, 0.1, 0.2, 0.2]}])
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 10)
    with zipfile.ZipFile(package) as source:
        files = {name: source.read(name) for name in source.namelist() if name != "manifest.json"}
    with zipfile.ZipFile(package, "w") as target:
        for name, payload in files.items():
            target.writestr(name, payload)
    assert import_batch(catalog, package, confirm_empty=False)["objects"] == 1


def test_export_requires_positive_limit_and_never_overwrites(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample")
    conn.close()
    package = tmp_path / "batch.zip"
    package.write_bytes(b"keep-me")
    with pytest.raises(ValueError, match="greater than zero"):
        export_batch(catalog, tmp_path, tmp_path / "other.zip", 0)
    with pytest.raises(FileExistsError, match="overwrite"):
        export_batch(catalog, tmp_path, package, 1)
    assert package.read_bytes() == b"keep-me"
    conn = open_catalog(catalog)
    assert conn.execute("SELECT status FROM samples").fetchone()[0] == "unreviewed"
    conn.close()


def test_failed_export_leaves_no_archive_or_batch(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "changed")
    conn.close()
    (tmp_path / "images/black/changed.jpg").write_bytes(b"changed-after-catalog")
    package = tmp_path / "batch.zip"
    with pytest.raises(RuntimeError, match="changed source image"):
        export_batch(catalog, tmp_path, package, 1)
    assert not package.exists()
    assert list(tmp_path.glob(".batch.zip.*.tmp")) == []
    conn = open_catalog(catalog)
    assert conn.execute("SELECT count(*) FROM review_batches").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM samples").fetchone()[0] == "unreviewed"
    conn.close()


def test_import_rejects_tampered_manifest_hash(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample", [{"box": [0.1, 0.1, 0.2, 0.2]}])
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 1)
    rewrite_manifest(package, lambda manifest: manifest["samples"][0].update({"camera": "other"}))
    with pytest.raises(ValueError, match="manifest hash"):
        import_batch(catalog, package, confirm_empty=False)


def test_import_rejects_duplicate_image_ids(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "one")
    add_sample(conn, tmp_path, "two")
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 2, include_suggestions=False)

    def duplicate(coco):
        coco["images"][1]["id"] = coco["images"][0]["id"]

    rewrite_coco(package, duplicate)
    with pytest.raises(ValueError, match="duplicate COCO image id"):
        import_batch(catalog, package, confirm_empty=True)


def test_import_rejects_annotation_for_unknown_image(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample")
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 1, include_suggestions=False)

    def unknown(coco):
        coco["annotations"] = [
            {"id": 1, "image_id": -1, "category_id": 1, "bbox": [1, 1, 2, 2]}
        ]

    rewrite_coco(package, unknown)
    with pytest.raises(ValueError, match="unknown image id"):
        import_batch(catalog, package, confirm_empty=False)


def test_import_rejects_non_finite_bbox(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample")
    conn.close()
    package = tmp_path / "batch.zip"
    export_batch(catalog, tmp_path, package, 1, include_suggestions=False)

    def non_finite(coco):
        coco["annotations"] = [{
            "id": 1, "image_id": coco["images"][0]["id"],
            "category_id": 1, "bbox": [0, 0, float("nan"), 10],
        }]

    rewrite_coco(package, non_finite)
    with pytest.raises(ValueError, match="invalid bbox"):
        import_batch(catalog, package, confirm_empty=False)


def test_requeue_recovers_only_unimported_export_and_restores_protection(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample")
    conn.execute("UPDATE samples SET protected=1 WHERE sample_id='sample'")
    conn.commit()
    conn.close()
    package = tmp_path / "batch.zip"
    batch = export_batch(catalog, tmp_path, package, 1, include_suggestions=False)
    result = requeue_batch(catalog, batch["batch_id"])
    assert result["status"] == "requeued"
    conn = open_catalog(catalog)
    row = conn.execute("SELECT status,protected FROM samples").fetchone()
    assert tuple(row) == ("unreviewed", 1)
    conn.close()
    with pytest.raises(ValueError, match="not exported"):
        requeue_batch(catalog, batch["batch_id"])


def test_status_reports_queue_and_total_protected_storage(tmp_path: Path):
    catalog = tmp_path / "catalog.sqlite3"
    conn = open_catalog(catalog)
    add_sample(conn, tmp_path, "sample")
    conn.execute("UPDATE samples SET protected=1 WHERE sample_id='sample'")
    conn.commit()
    conn.close()
    cache = tmp_path / "model_images/key/sample.jpg"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    result = queue_status(catalog, tmp_path, budget_gb=1.0)
    assert result["by_status"] == {"unreviewed": 1}
    assert result["by_camera"] == {"black": 1}
    assert result["by_reason"] == {"regular": 1}
    assert result["cache_version_bytes"] == 5
    assert result["total_bytes"] == len(b"jpeg-placeholder") + 5
    assert result["protected_bytes"] == result["total_bytes"]
    assert result["remaining_bytes"] == 1_000_000_000 - result["total_bytes"]
