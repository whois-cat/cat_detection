from __future__ import annotations

from training.yolo_label_studio import (
    LABEL_NAME,
    annotation_boxes,
    build_label_config,
    image_url,
    prediction_results,
    sample_task,
)


def test_label_config_is_single_cat_rectangle():
    config = build_label_config()
    assert "RectangleLabels" in config and f'value="{LABEL_NAME}"' in config


def test_image_url_points_at_local_files():
    assert image_url("images/black/abc.jpg") == "/data/local-files/?d=images/black/abc.jpg"


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
              "image_relpath": "images/black/s1.jpg", "reasons": "[]"}
    task = sample_task(sample, [{"box": [0.1, 0.2, 0.3, 0.4]}])
    assert task["data"]["sample_id"] == "s1"
    assert task["data"]["image"] == "/data/local-files/?d=images/black/s1.jpg"
    assert task["predictions"][0]["result"][0]["value"]["rectanglelabels"] == [LABEL_NAME]


def test_empty_annotation_is_a_confirmed_negative():
    # A submitted annotation with no rectangles yields zero boxes (negative).
    assert annotation_boxes([], 100, 50) == []
