from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import training.streamhub_dataset as dataset_module
from training.streamhub_dataset import (
    CameraState,
    BudgetExceeded,
    Segment,
    collect_segment,
    derived_storage_bytes,
    difference_hash,
    frame_reasons,
    enforce_budget,
    main,
    nearest_prediction,
    open_catalog,
    load_state,
    parse_segment,
    scan_is_complete,
    segment_source_fingerprint,
    selection_fingerprint,
    stable_id,
    validate_collection_args,
    write_sample,
)


def test_parse_streamhub_segment(tmp_path: Path):
    root = tmp_path / "recordings"
    path = root / "black/2026-10-08/16/2026-10-08T16-46-27.232Z_11999ms.mp4"
    path.parent.mkdir(parents=True)
    path.touch()
    seg = parse_segment(root, path)
    assert seg is not None
    assert seg.camera == "black"
    assert seg.duration_ms == 11999
    assert seg.sidecar.name == "2026-10-08T16-46-27.232Z.labels.jsonl"


def test_nearest_prediction_respects_tolerance():
    rows = [{"pts": 90_000}, {"pts": 180_000}]
    assert nearest_prediction(rows, 170_000, 20_000) == rows[1]
    assert nearest_prediction(rows, 140_000, 10_000) is None


def test_project_prediction_rotates_boxes_and_keeps_scores():
    pred = {"model": "yolo", "dets": [{"box": [0.2, 0.2, 0.3, 0.4], "score": 0.8,
                                        "cats": {"alisa": 0.9}}]}
    out = dataset_module.project_prediction(pred, 60, 40, 90, None)
    assert out["model"] == "yolo"
    [det] = out["dets"]
    assert det["score"] == 0.8 and det["cats"] == {"alisa": 0.9}
    # 90° CW swaps axes; the box must differ from the camera-frame box.
    assert det["box"] != [0.2, 0.2, 0.3, 0.4]
    assert all(0.0 <= value <= 1.0 for value in det["box"])


def test_project_prediction_drops_boxes_outside_roi():
    pred = {"dets": [{"box": [0.0, 0.0, 0.3, 1.0], "score": 0.5}]}
    out = dataset_module.project_prediction(pred, 60, 40, 0, [0.5, 0.0, 1.0, 1.0])
    assert out["dets"] == []
    assert dataset_module.project_prediction(None, 60, 40, 90, None) is None


def test_stable_id_is_path_independent():
    assert stable_id("black", 123) == stable_id("black", 123)
    assert stable_id("black", 123) != stable_id("grey", 123)


def test_regular_sampling_does_not_depend_on_detection():
    state = CameraState(last_hash=difference_hash(np.zeros((16, 16, 3), dtype=np.uint8)))
    reasons, group = frame_reasons(
        wall_ms=1_000, prediction={"dets": []}, image_hash=state.last_hash,
        state=state, regular_interval_ms=300_000, motion_threshold=10,
        low_conf=0.35, visit_gap_ms=90_000, visit_sample_ms=20_000,
        max_visit_samples=6,
    )
    assert reasons == ["regular"]
    assert group is None


def test_motion_starts_visit_without_yolo_detection():
    state = CameraState(last_hash=0)
    reasons, group = frame_reasons(
        wall_ms=5_000, prediction=None, image_hash=(1 << 16) - 1,
        state=state, regular_interval_ms=300_000, motion_threshold=10,
        low_conf=0.35, visit_gap_ms=90_000, visit_sample_ms=20_000,
        max_visit_samples=6,
    )
    assert "visual_change" in reasons
    assert "visit_sample" in reasons
    assert group is not None


def test_catalog_keeps_predictions_separate_from_samples(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"samples", "predictions", "segments", "segment_scans", "collector_state"} <= tables
    columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    assert "status" in columns
    assert "tags_json" in columns
    assert "detections_json" not in columns
    conn.close()


def _collector_args(**overrides):
    values = {
        "candidate_interval_sec": 5.0,
        "regular_interval_sec": 300.0,
        "visit_gap_sec": 90.0,
        "visit_sample_interval_sec": 20.0,
        "max_visit_samples": 6,
        "motion_threshold": 10,
        "dedupe_threshold": 3,
        "low_confidence": 0.35,
        "prediction_tolerance_sec": 1.0,
        "jpeg_quality": 90,
        "budget_gb": 5.0,
        "min_per_camera_day": 2,
        "reviewed_budget_gb": 4.0,
        "keep_min_reviewed": 20,
        "cameras": ["black"],
        "tags": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_source_fingerprint_includes_late_and_changed_sidecar(tmp_path: Path):
    root = tmp_path / "recordings"
    video = root / "black/2026-10-08/16/2026-10-08T16-46-27.232Z_11999ms.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    seg = parse_segment(root, video)
    assert seg is not None
    without_labels = segment_source_fingerprint(seg)
    seg.sidecar.write_text('{"t":"cv","pts":1}\n')
    with_labels = segment_source_fingerprint(seg)
    seg.sidecar.write_text('{"t":"cv","pts":2}\n')
    changed_labels = segment_source_fingerprint(seg)
    assert len({without_labels, with_labels, changed_labels}) == 3


def test_scan_completion_requires_covering_range_source_and_config(tmp_path: Path):
    root = tmp_path / "recordings"
    video = root / "black/2026-10-08/16/2026-10-08T16-46-27.232Z_11999ms.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    seg = parse_segment(root, video)
    assert seg is not None
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    conn.execute(
        "INSERT INTO segment_scans VALUES(?,?,?,?,?,?)",
        (seg.relpath, seg.start_ms + 1_000, seg.start_ms + 5_000, "source-a", "config-a", 1),
    )
    conn.commit()
    assert scan_is_complete(
        conn, seg, seg.start_ms + 2_000, seg.start_ms + 4_000, "source-a", "config-a"
    )
    assert not scan_is_complete(
        conn, seg, seg.start_ms, seg.start_ms + 4_000, "source-a", "config-a"
    )
    assert not scan_is_complete(
        conn, seg, seg.start_ms + 2_000, seg.start_ms + 4_000, "source-b", "config-a"
    )
    assert not scan_is_complete(
        conn, seg, seg.start_ms + 2_000, seg.start_ms + 4_000, "source-a", "config-b"
    )
    conn.close()


def test_selection_fingerprint_changes_with_sampling_and_tags():
    base = selection_fingerprint(_collector_args(), {"rotate_deg": 90})
    assert base != selection_fingerprint(
        _collector_args(candidate_interval_sec=6.0), {"rotate_deg": 90}
    )
    assert base != selection_fingerprint(_collector_args(tags=["shaved"]), {"rotate_deg": 90})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_interval_sec", 0),
        ("regular_interval_sec", -1),
        ("prediction_tolerance_sec", float("nan")),
        ("motion_threshold", 65),
        ("dedupe_threshold", -1),
        ("low_confidence", 1.1),
        ("jpeg_quality", 0),
        ("budget_gb", -1),
        ("min_per_camera_day", -1),
        ("max_visit_samples", 0),
    ],
)
def test_collector_rejects_invalid_numeric_options(field: str, value):
    with pytest.raises(ValueError):
        validate_collection_args(_collector_args(**{field: value}))


def test_collector_rejects_invalid_tag():
    with pytest.raises(ValueError, match="--tag"):
        validate_collection_args(_collector_args(tags=["has spaces"]))


def test_collector_state_resets_when_collecting_backwards(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    conn.execute(
        "INSERT INTO collector_state(camera,state_json,updated_at_ms) VALUES(?,?,?)",
        ("black", json.dumps({"last_seen_ms": 20_000, "last_regular_ms": 20_000}), 20_000),
    )
    conn.commit()
    assert load_state(conn, "black", 10_000) == CameraState()
    assert load_state(conn, "black", 20_000).last_regular_ms == 20_000
    conn.close()


def _insert_sample(conn, root: Path, sid: str, *, wall_ms: int, size: int,
                   reasons: list[str], status: str = "unreviewed", protected: int = 0):
    rel = Path("images/black") / f"{sid}.jpg"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    conn.execute(
        """INSERT INTO samples(
          sample_id,camera,segment_relpath,pts,wall_ms,visit_group,image_relpath,
          width,height,jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,status,protected,created_at_ms
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, "black", "source.mp4", wall_ms * 90, wall_ms, None, rel.as_posix(),
         1, 1, size, sid, "0", json.dumps(reasons), 90, status, protected, wall_ms),
    )
    conn.commit()


def test_budget_evicts_old_low_value_candidates_first(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert_sample(conn, tmp_path, "old-regular", wall_ms=1_000, size=60, reasons=["regular"])
    _insert_sample(conn, tmp_path, "valuable", wall_ms=2_000, size=60, reasons=["low_confidence"])
    deleted, freed = enforce_budget(conn, tmp_path, 60, min_per_camera_day=0)
    assert (deleted, freed) == (1, 60)
    assert conn.execute("SELECT sample_id FROM samples").fetchone()[0] == "valuable"
    assert not (tmp_path / "images/black/old-regular.jpg").exists()
    conn.close()


def test_budget_never_deletes_reviewed_or_protected_images(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert_sample(conn, tmp_path, "reviewed", wall_ms=1_000, size=70,
                   reasons=["regular"], status="verified")
    _insert_sample(conn, tmp_path, "protected", wall_ms=2_000, size=70,
                   reasons=["regular"], protected=1)
    try:
        enforce_budget(conn, tmp_path, 100)
    except BudgetExceeded as exc:
        assert "protected/reviewed" in str(exc)
    else:
        raise AssertionError("protected data above budget must fail")
    assert conn.execute("SELECT count(*) FROM samples").fetchone()[0] == 2
    assert (tmp_path / "images/black/reviewed.jpg").exists()
    assert (tmp_path / "images/black/protected.jpg").exists()
    conn.close()


def test_budget_counts_persistent_version_cache_and_evicts_only_queue(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert_sample(conn, tmp_path, "candidate", wall_ms=1_000, size=60, reasons=["regular"])
    cache = tmp_path / "model_images/key/cached.jpg"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"c" * 70)
    assert derived_storage_bytes(tmp_path) == 70
    deleted, freed = enforce_budget(conn, tmp_path, 100, min_per_camera_day=0)
    assert (deleted, freed) == (1, 60)
    assert cache.exists()
    with pytest.raises(BudgetExceeded, match="protected/reviewed"):
        enforce_budget(conn, tmp_path, 60, min_per_camera_day=0)
    conn.close()


def test_repeated_sample_adds_tags_without_replacing_existing_tags(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    segment_path = tmp_path / "recordings/black/2026-10-08/16/2026-10-08T16-46-27.232Z_11999ms.mp4"
    segment_path.parent.mkdir(parents=True)
    segment_path.touch()
    seg = parse_segment(tmp_path / "recordings", segment_path)
    assert seg is not None
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert write_sample(
        conn, tmp_path, seg, 90_000, image, ["regular"], None, None,
        {}, 90, ["shaved"],
    )
    assert not write_sample(
        conn, tmp_path, seg, 90_000, image, ["regular"], None, None,
        {}, 90, ["winter"],
    )
    tags = json.loads(conn.execute("SELECT tags_json FROM samples").fetchone()[0])
    assert tags == ["shaved", "winter"]
    conn.close()


def test_main_writes_atomic_completed_report_without_video(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("cameras:\n  - id: black\n    cv: {}\n")
    report_path = tmp_path / "reports/collector.json"
    result = main([
        "--recordings", str(tmp_path / "recordings"),
        "--config", str(config),
        "--out", str(tmp_path / "dataset"),
        "--camera", "black",
        "--from", "2026-10-08T00:00:00Z",
        "--to", "2026-10-08T01:00:00Z",
        "--tag", "shaved",
        "--report-json", str(report_path),
    ])
    assert result == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["cameras"] == ["black"]
    assert report["tags"] == ["shaved"]
    assert report["segments"] == report["decoded_frames"] == 0
    assert report["managed_bytes"] == 0
    assert list(report_path.parent.glob("*.tmp")) == []


def test_main_last_days_on_all_configured_cameras(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("cameras:\n  - id: black\n    cv: {}\n  - id: grey\n    cv: {}\n")
    report_path = tmp_path / "collector.json"
    common = ["--recordings", str(tmp_path / "recordings"), "--config", str(config),
              "--out", str(tmp_path / "dataset"), "--report-json", str(report_path)]
    assert main([*common, "--days", "7"]) == 0
    report = json.loads(report_path.read_text())
    assert report["cameras"] == ["black", "grey"]
    span = datetime.fromisoformat(report["to"]) - datetime.fromisoformat(report["from"])
    assert span == timedelta(days=7)
    with pytest.raises(ValueError, match="either --days"):
        main([*common, "--days", "7", "--from", "2026-10-01"])


def test_main_records_interruption(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cameras:\n  - id: black\n    cv: {}\n")
    report_path = tmp_path / "collector.json"

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(dataset_module, "iter_segments", interrupted)
    result = main([
        "--recordings", str(tmp_path / "recordings"),
        "--config", str(config),
        "--out", str(tmp_path / "dataset"),
        "--camera", "black",
        "--from", "2026-10-08T00:00:00Z",
        "--to", "2026-10-08T01:00:00Z",
        "--report-json", str(report_path),
    ])
    assert result == 130
    assert json.loads(report_path.read_text())["status"] == "interrupted"


def test_main_skips_a_segment_the_pruner_deleted_meanwhile(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("cameras:\n  - id: black\n    cv: {}\n")
    report_path = tmp_path / "collector.json"
    gone = Segment(path=tmp_path / "recordings/black/gone.mp4", camera="black",
                   relpath="black/gone.mp4", start_ms=0, duration_ms=12_000)
    monkeypatch.setattr(dataset_module, "iter_segments", lambda *_a, **_k: iter([gone, gone]))
    result = main([
        "--recordings", str(tmp_path / "recordings"),
        "--config", str(config),
        "--out", str(tmp_path / "dataset"),
        "--from", "2026-10-08T00:00:00Z",
        "--to", "2026-10-08T01:00:00Z",
        "--report-json", str(report_path),
    ])
    assert result == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["segments_vanished"] == 2


def _reviewed(conn, root: Path, sid: str, *, wall_ms: int, name: str | None = "ellie",
              dhash: str = "0" * 16, visit: str = "v1", corrected: bool = False,
              camera: str = "black") -> None:
    rel = Path("images") / camera / f"{sid}.jpg"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(b"x" * 100)
    conn.execute(
        """INSERT INTO samples(sample_id,camera,segment_relpath,pts,wall_ms,visit_group,
             image_relpath,width,height,jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,
             status,protected,created_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, camera, "s.mp4", wall_ms * 90, wall_ms, visit, rel.as_posix(), 100, 100, 100,
         "h" + sid, dhash, "[]", 0, "verified", 1, 0),
    )
    if name is not None:
        conn.execute(
            "INSERT INTO annotations(sample_id,annotation_id,class_name,x,y,w,h,source,"
            "updated_at_ms,cat_name) VALUES(?,1,'cat',10,10,40,40,'ls',0,?)", (sid, name))
    suggested = [{"box": [0.1, 0.1, 0.4, 0.4], "cats": {"alisa" if corrected else (name or "x"): 0.9}}]
    conn.execute("INSERT INTO predictions(sample_id,detections_json) VALUES(?,?)",
                 (sid, json.dumps(suggested if name is not None else [])))
    conn.commit()


def test_thinning_retires_duplicates_first_and_keeps_corrections(tmp_path: Path):
    from training.streamhub_dataset import thin_reviewed
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    day = 86_400_000
    _reviewed(conn, tmp_path, "first", wall_ms=1 * day)
    _reviewed(conn, tmp_path, "dup", wall_ms=1 * day + 1000)               # same look, same visit
    _reviewed(conn, tmp_path, "other_visit", wall_ms=2 * day, visit="v2", dhash="f" * 16)
    _reviewed(conn, tmp_path, "fixed", wall_ms=3 * day, visit="v3", dhash="0f" * 8, corrected=True)
    retired, freed = thin_reviewed(conn, tmp_path, budget_bytes=300, keep_min=1)
    assert (retired, freed) == (1, 100)
    status = dict(conn.execute("SELECT sample_id,status FROM samples"))
    assert status["dup"] == "retired" and not (tmp_path / "images/black/dup.jpg").exists()
    # Over budget again: oldest first, the correction never (it also keeps the
    # bucket at keep_min on its own).
    thin_reviewed(conn, tmp_path, budget_bytes=100, keep_min=1)
    status = dict(conn.execute("SELECT sample_id,status FROM samples"))
    assert status["fixed"] == "verified"
    assert status["first"] == status["other_visit"] == "retired"
    conn.close()


def test_thinning_keeps_a_minimum_per_cat_camera_and_month(tmp_path: Path):
    from training.streamhub_dataset import thin_reviewed
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    for i in range(5):
        _reviewed(conn, tmp_path, f"e{i}", wall_ms=i * 3_600_000, visit=f"e{i}", dhash=f"{i:016x}")
    _reviewed(conn, tmp_path, "rare", wall_ms=10, name="alisa", visit="r", dhash="f" * 16)
    _reviewed(conn, tmp_path, "empty", wall_ms=20, name=None, visit="n", dhash="e" * 16)
    thin_reviewed(conn, tmp_path, budget_bytes=0, keep_min=2)
    left = {r[0] for r in conn.execute("SELECT sample_id FROM samples WHERE status='verified'")}
    assert left == {"e3", "e4", "rare", "empty"}     # newest ellie kept, rare cat and empty kept
    conn.close()


def test_retired_frames_free_budget_and_are_never_pushed_again(tmp_path: Path):
    from training.yolo_label_studio import _pending_tasks
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _reviewed(conn, tmp_path, "gone", wall_ms=1)
    conn.execute("UPDATE samples SET status='retired' WHERE sample_id='gone'")
    conn.commit()
    assert _pending_tasks(conn, ["ellie", "unknown"]) == []
    assert enforce_budget(conn, tmp_path, budget_bytes=0) == (0, 0)
    conn.close()


def test_version_hardlinks_to_catalog_images_take_no_extra_space(tmp_path: Path):
    image = tmp_path / "images/black/a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x" * 100)
    linked = tmp_path / "versions/v1/images/train/a.jpg"
    linked.parent.mkdir(parents=True)
    os.link(image, linked)
    assert derived_storage_bytes(tmp_path) == 0
    image.unlink()                                  # only the version holds it now
    assert derived_storage_bytes(tmp_path) == 100
