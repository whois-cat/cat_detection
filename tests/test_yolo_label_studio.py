from __future__ import annotations

import time
from pathlib import Path

import pytest

from training.streamhub_dataset import open_catalog
from training.yolo_label_studio import (
    LABEL_NAME,
    LOCAL_STORAGE_PATH,
    NAME_CONTROL,
    _unnamed_tasks,
    name_choices,
    push_message,
    verified_dets,
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


NAMES = name_choices(["alisa", "ellie"])


def test_label_config_is_single_cat_rectangle_with_a_name_per_box():
    config = build_label_config(["alisa", "ellie"])
    assert "RectangleLabels" in config and f'value="{LABEL_NAME}"' in config
    assert f'name="{NAME_CONTROL}"' in config and 'perRegion="true"' in config
    assert '<Choice value="alisa" hotkey="1"/>' in config
    assert '<Choice value="unknown" hotkey="3"/>' in config


def test_names_need_configuring_and_unknown_comes_last():
    assert name_choices([" ellie", "alisa", "ellie", "unknown", ""]) == ["ellie", "alisa", "unknown"]
    with pytest.raises(SystemExit, match="CAT_NAMES"):
        name_choices([""])


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
    [(x, y, w, h, name)] = annotation_boxes([result], width, height)
    assert (x, y, w, h) == (0.2 * width, 0.1 * height, 0.3 * width, 0.4 * height)
    assert name is None


def test_suggested_name_rides_along_with_its_box_and_comes_back():
    dets = [{"box": [0.1, 0.1, 0.2, 0.2], "cats": {"alisa": 0.1, "ellie": 0.9}},
            {"box": [0.5, 0.5, 0.2, 0.2], "cats": {"felix": 0.99}}]   # not a configured name
    results = prediction_results(dets, NAMES)
    rects = [r for r in results if r["type"] == "rectanglelabels"]
    names = [r for r in results if r["from_name"] == NAME_CONTROL]
    assert len(rects) == 2 and len(names) == 1
    assert names[0]["id"] == rects[0]["id"] and names[0]["value"]["choices"] == ["ellie"]
    boxes = annotation_boxes(results, 100, 100)
    assert [b[4] for b in boxes] == ["ellie", None]


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


def test_unsure_name_puts_the_frame_up_front():
    sample = {"sample_id": "s1", "camera": "black", "wall_ms": 1,
              "image_relpath": "images/black/s1.jpg", "reasons_json": '["regular"]'}
    sure = sample_task(sample, [{"box": [0.1, 0.1, 0.2, 0.2], "cats": {"ellie": 0.95}}], NAMES)
    unsure = sample_task(sample, [{"box": [0.1, 0.1, 0.2, 0.2], "cats": {"ellie": 0.55}}], NAMES)
    assert "name_unsure" in unsure["data"]["reasons"] and "name_unsure" not in sure["data"]["reasons"]
    assert unsure["data"]["priority"] < sure["data"]["priority"]


def test_priority_puts_likely_model_mistakes_first():
    assert task_priority(["low_confidence"]) < task_priority(["name_unsure"])
    assert task_priority(["name_unsure"]) < task_priority(["multiple_boxes"])
    assert task_priority(["regular"]) < task_priority(["names_only"])
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
    order = [task["data"]["sample_id"] for task in _pending_tasks(conn, NAMES)]
    conn.close()
    assert order == ["mid_unsure", "late_unsure", "early_regular"]


def test_pending_tasks_include_exported_but_not_verified(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert(conn, "fresh", "unreviewed", 90)
    _insert(conn, "from_cvat", "exported", 180)   # left over from a CVAT export
    _insert(conn, "done", "verified", 270)
    conn.commit()
    ids = {task["data"]["sample_id"] for task in _pending_tasks(conn, NAMES)}
    conn.close()
    assert ids == {"fresh", "from_cvat"}


def test_verified_boxes_get_the_overlapping_model_name():
    model = [{"box": [0.1, 0.1, 0.2, 0.2], "cats": {"ellie": 0.9}},
             {"box": [0.7, 0.7, 0.1, 0.1], "cats": {"alisa": 0.9}}]
    dets = verified_dets([(10, 10, 20, 21), (40, 40, 10, 10)], 100, 100, model)
    assert dets[0]["cats"] == {"ellie": 0.9}    # overlaps the first model box
    assert dets[1]["cats"] == {}                # overlaps nothing: no suggestion


def test_frames_reviewed_before_names_are_queued_for_names(tmp_path: Path):
    conn = open_catalog(tmp_path / "catalog.sqlite3")
    _insert(conn, "old", "verified", 90)
    _insert(conn, "named", "verified", 180)
    _insert(conn, "empty", "verified", 270)     # confirmed negative: nothing to name
    cols = "sample_id,annotation_id,class_name,x,y,w,h,source,updated_at_ms,cat_name"
    conn.execute(f"INSERT INTO annotations({cols}) VALUES('old',1,'cat',2,3,10,12,'ls',0,NULL)")
    conn.execute(f"INSERT INTO annotations({cols}) VALUES('named',1,'cat',2,3,10,12,'ls',0,'ellie')")
    conn.commit()
    [task] = _unnamed_tasks(conn, NAMES)
    conn.close()
    assert task["data"]["sample_id"] == "old"
    assert "names_only" in task["data"]["reasons"]
    [rect] = task["predictions"][0]["result"]
    assert rect["value"]["x"] == pytest.approx(2 / 24 * 100)


def test_push_message_says_what_waits():
    result = {"waiting": 40, "pushed": 12, "needs_a_look": 5}
    text = push_message(result, "http://ls/projects/1/data")
    assert "40 frames" in text and "12 new" in text and "5 need a look" in text
    assert text.endswith("http://ls/projects/1/data")
    assert "nothing" in push_message({"waiting": 0, "pushed": 0, "needs_a_look": 0}, "x")
