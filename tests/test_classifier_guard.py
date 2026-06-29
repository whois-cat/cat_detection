"""Runtime classifier guard: clear, actionable errors for bad/missing artifacts.

These run without OpenVINO — the file/classes checks happen before the OpenVINO
import in CatClassifier.__init__, and the output-dim guard is a pure function.
"""
from __future__ import annotations

import json

import pytest

from classifier import CatClassifier, _check_output_dim, _load_class_names


def _touch_ir(model_dir):
    # Minimal fake IR files so the .xml/.bin existence checks pass and we reach
    # the classes.json validation (no OpenVINO is loaded along this path).
    (model_dir / "cat_classifier.xml").write_text("<net/>", encoding="utf-8")
    (model_dir / "cat_classifier.bin").write_bytes(b"")


def test_missing_xml_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"cat_classifier\.xml"):
        CatClassifier(tmp_path)


def test_missing_classes_json_is_actionable(tmp_path):
    _touch_ir(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        CatClassifier(tmp_path)
    assert "classes.json" in str(e.value)
    assert "just classifier-promote" in str(e.value)   # tells the operator what to run


def test_empty_classes_json_rejected(tmp_path):
    _touch_ir(tmp_path)
    (tmp_path / "classes.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        CatClassifier(tmp_path)


def test_non_string_classes_rejected(tmp_path):
    _touch_ir(tmp_path)
    (tmp_path / "classes.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="label strings"):
        CatClassifier(tmp_path)


def test_load_class_names_accepts_arbitrary_labels(tmp_path):
    p = tmp_path / "classes.json"
    p.write_text(json.dumps(["whiskers", "id_7", "spot"]), encoding="utf-8")
    assert _load_class_names(p, tmp_path) == ["whiskers", "id_7", "spot"]


def test_output_dim_guard_matches_and_mismatches(tmp_path):
    _check_output_dim(4, 4, tmp_path)                  # ok, no raise
    with pytest.raises(ValueError) as e:
        _check_output_dim(3, 4, tmp_path)
    assert "3 classes" in str(e.value) and "lists 4" in str(e.value)
    assert "just classifier-promote" in str(e.value)
