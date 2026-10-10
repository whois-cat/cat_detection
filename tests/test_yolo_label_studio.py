from __future__ import annotations

import time
from pathlib import Path

from training.streamhub_dataset import open_catalog
from training.yolo_label_studio import (
    LABEL_NAME,
    LOCAL_STORAGE_PATH,
    _ensure_local_storage,
    _pending_tasks,
    annotation_boxes,
    task_priority,
    build_label_config,
    image_url,
    prediction_results,
    sample_task,
)


def _insert(conn, sample_id: str, status: str, pts: int, reasons: str = "[]") -> None:
    conn.execute(
        """INSERT INTO samples(
             sample_id,camera,segment_relpath,pts,wall_ms,image_relpath,width,height,
             jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,status,protected,created_at_ms
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sample_id, "black", "black/seg.mp4", pts, pts // 90, f"images/black/{sample_id}.jpg",
         24, 32, 100, f"sha-{sample_id}", "0000000000000000", reasons, 90, status, 0,
         int(time.time() * 1000)),
    )


def test_label_config_is_single_cat_rectangle():
    config = build_label_config()
    assert "RectangleLabels" in config and f'value="{LABEL_NAME}"' in config


def test_image_url_points_at_local_files():
    assert image_url("images/black/abc.jpg") == "/data/local-files/?d=images/black/abc.jpg"


class _FakeProject:
    id = 7

    def __init__(self, storages):
        self.storages = storages
        self.posted = []

    def make_request(self, method, url, params=None, json=None):
        assert url == "/api/storages/localfiles"
        if method == "POST":
            self.posted.append(json)
        storages = self.storages

        class _Resp:
            def json(self):
                return storages
        return _Resp()


def test_local_storage_created_so_local_files_urls_are_served():
    project = _FakeProject([])
    _ensure_local_storage(project)
    assert project.posted == [{
        "project": 7, "path": LOCAL_STORAGE_PATH, "title": "catalog frames",
        "use_blob_urls": True, "regex_filter": "",
    }]
    # The served path must be the storage root + the relpath used in task URLs.
    assert LOCAL_STORAGE_PATH.endswith("/images")


def test_local_storage_not_duplicated_when_already_covered():
    project = _FakeProject([{"path": "/label-studio/files"}])
    _ensure_local_storage(project)
    assert project.posted == []


def test_box_fraction_round_trips_through_label_studio_percent():
    # fraction box -> LS percent prediction -> back to pixels must match.
    width, height = 100, 50
    [result] = prediction_results([{"box": [0.2, 0.1, 0.3, 0.4], "score": 0.8}])
    value = result["value"]
    assert (value["x"], value["y"], value["width"], value["height"]) == (20.0, 10.0, 30.0, 40.0)
    assert value["rectanglelabels"] == [LABEL_NAME]
    [(x, y, w, h)] = annotation_boxes([result], width, height)
    assert (x, y, w, h) == (0.2 * width, 0.1 * height, 0.3 * width, 0.4 * height)


def test_prediction_skips_degenerate_boxes():
    assert prediction_results([{"box": [0.1, 0.1, 0.0, 0.3]}]) == []
    assert prediction_results([{"box": [0.1, 0.1]}]) == []


def test_sample_task_carries_id_image_and_suggestions():
    sample = {"sample_id": "s1", "camera": "black", "wall_ms": 123,
              "image_relpath": "images/black/s1.jpg",
              "reasons_json": '["regular","low_confidence"]'}
    task = sample_task(sample, [{"box": [0.1, 0.2, 0.3, 0.4]}])
    assert task["data"]["sample_id"] == "s1"
    assert task["data"]["reasons"] == "regular,low_confidence"
    assert task["data"]["priority"] == 0
    assert task["data"]["image"] == "/data/local-files/?d=images/black/s1.jpg"
    assert task["predictions"][0]["result"][0]["value"]["rectanglelabels"] == [LABEL_NAME]


def test_empty_annotation_is_a_confirmed_negative():
    # A submitted annotation with no rectangles yields zero boxes (negative).
    assert annotation_boxes([], 100, 50) == []


def test_priority_puts_likely_model_mistakes_first():
    assert task_priority(["low_confidence"]) < task_priority(["multiple_boxes"])
    assert task_priority(["multiple_boxes"]) < task_priority(["visual_change"])
    assert task_priority(["visit_sample"]) < task_priority(["regular"])
    assert task_priority(["regular", "low_confidence"]) == task_priority(["low_confidence"])
    assert task_priority([]) > task_priority(["regular"])


def test_pending_tasks_ordered_by_priority_then_time(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert(conn, "early_regular", "unreviewed", 90, '["regular"]')
    _insert(conn, "late_unsure", "unreviewed", 900, '["low_confidence"]')
    _insert(conn, "mid_unsure", "unreviewed", 450, '["low_confidence"]')
    conn.commit()
    order = [task["data"]["sample_id"] for task in _pending_tasks(conn)]
    conn.close()
    assert order == ["mid_unsure", "late_unsure", "early_regular"]


def test_pending_tasks_include_exported_but_not_verified(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert(conn, "fresh", "unreviewed", 90)
    _insert(conn, "from_cvat", "exported", 180)   # left over from a CVAT export
    _insert(conn, "done", "verified", 270)
    conn.commit()
    ids = {task["data"]["sample_id"] for task in _pending_tasks(conn)}
    conn.close()
    assert ids == {"fresh", "from_cvat"}
