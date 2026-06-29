"""configure.py config validation: threshold ranges, allowed_cats element typing,
and that the rendered compose is valid YAML. Generic labels only (no real names).
"""
from __future__ import annotations

import pytest
import yaml

from tools import configure


def _write_cfg(tmp_path, monkeypatch, cfg) -> None:
    p = tmp_path / "cameras.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(configure, "CAMERAS_YAML", p)


def _cam(**extra) -> dict:
    cam = {"id": "grey", "rtsp": "rtsp://host/grey"}
    cam.update(extra)
    return {"cameras": [cam]}


def _feeder(**extra) -> dict:
    f = {"id": "feeder1", "api_base_url": "http://host",
         "serial_number": "S1", "allowed_cats": ["cat_a", "cat_b"]}
    f.update(extra)
    return {"cameras": [{"id": "grey", "rtsp": "rtsp://host/grey", "feeder": f}]}


# ---- _check_ranges (pure helper) -------------------------------------------

def test_check_ranges_accepts_in_range():
    configure._check_ranges("x", {"yolo_conf": 0.4}, configure.CAMERA_RANGES)


def test_check_ranges_rejects_out_of_range():
    with pytest.raises(SystemExit, match=r"yolo_conf=5.*out of range"):
        configure._check_ranges("x", {"yolo_conf": 5}, configure.CAMERA_RANGES)


def test_check_ranges_rejects_non_number():
    with pytest.raises(SystemExit, match="must be a number"):
        configure._check_ranges("x", {"yolo_conf": "high"}, configure.CAMERA_RANGES)


def test_check_ranges_rejects_bool():
    with pytest.raises(SystemExit, match="must be a number"):
        configure._check_ranges("x", {"yolo_conf": True}, configure.CAMERA_RANGES)


def test_check_ranges_rejects_non_integer():
    with pytest.raises(SystemExit, match="whole number"):
        configure._check_ranges("x", {"food_tiles": 8.5}, configure.CAMERA_RANGES)


# ---- load_config integration -----------------------------------------------

def test_valid_config_passes(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _cam(
        yolo_conf=0.3, rotate_deg=90, food_empty_below=0.3, food_full_above=0.6))
    cfg = configure.load_config()
    assert cfg["cameras"][0]["id"] == "grey"


def test_camera_threshold_out_of_range_rejected(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _cam(yolo_conf=1.5))
    with pytest.raises(SystemExit, match="out of range"):
        configure.load_config()


def test_bad_rotate_deg_rejected(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _cam(rotate_deg=45))
    with pytest.raises(SystemExit, match="rotate_deg"):
        configure.load_config()


def test_food_hysteresis_must_be_ordered(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _cam(food_empty_below=0.7, food_full_above=0.5))
    with pytest.raises(SystemExit, match="food_empty_below"):
        configure.load_config()


def test_allowed_cats_non_string_rejected(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _feeder(allowed_cats=[1, 2]))
    with pytest.raises(SystemExit, match="non-empty strings"):
        configure.load_config()


def test_allowed_cats_blank_string_rejected(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _feeder(allowed_cats=["cat_a", "  "]))
    with pytest.raises(SystemExit, match="non-empty strings"):
        configure.load_config()


def test_feeder_threshold_out_of_range_rejected(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _feeder(classifier_min_conf=1.4))
    with pytest.raises(SystemExit, match="out of range"):
        configure.load_config()


def test_valid_feeder_passes(tmp_path, monkeypatch):
    _write_cfg(tmp_path, monkeypatch, _feeder(classifier_min_conf=0.9, feed_grain_num=2))
    cfg = configure.load_config()
    assert cfg["cameras"][0]["feeder"]["id"] == "feeder1"


# ---- generated compose is valid YAML (#6) ----------------------------------

def test_render_compose_is_valid_yaml_with_feeder():
    cfg = {"cameras": [
        {"id": "grey", "rtsp": "rtsp://host/grey", "detector_type": "yolo_cat",
         "feeder": {"id": "feeder1", "api_base_url": "http://host",
                    "serial_number": "S1", "allowed_cats": ["cat_a", "cat_b"]}},
        {"id": "beige", "rtsp": "rtsp://host/beige"},
    ]}
    doc = yaml.safe_load(configure.render_compose(cfg))
    assert set(doc["services"]) == {"detector-grey", "feeder-feeder1", "detector-beige"}
    assert doc["services"]["feeder-feeder1"]["depends_on"] == ["detector-grey"]
    assert doc["services"]["detector-grey"]["depends_on"] == ["mediamtx"]
