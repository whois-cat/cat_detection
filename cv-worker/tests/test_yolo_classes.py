import pytest

from cv_worker.models.yolo import cat_class_id


def test_coco_base_model_cat_is_15():
    names = {i: f"class{i}" for i in range(80)} | {15: "cat"}
    assert cat_class_id(names) == 15


def test_model_without_cat_fails_loudly():
    with pytest.raises(ValueError, match="no 'cat' class"):
        cat_class_id({0: "dog"})


def test_detector_weights_prefers_deployed_then_builtin(monkeypatch, tmp_path):
    import cv_worker.models as models

    deployed = tmp_path / "detector" / "current"
    monkeypatch.setattr(models, "DEPLOYED_DETECTOR", str(deployed))
    no_env = {}.get
    assert models.detector_weights(no_env) == (models.BUILTIN_DETECTOR, None)

    export = tmp_path / "detector" / "versions" / "yolo-20261010-011816" / "best_int8_openvino_model"
    export.mkdir(parents=True)
    deployed.symlink_to("versions/yolo-20261010-011816")
    assert models.detector_weights(no_env) == (
        str(deployed / "best_int8_openvino_model"), "yolo-20261010-011816")
    assert models.detector_weights({"YOLO_WEIGHTS": "/x"}.get) == ("/x", None)


def test_deployed_detector_without_one_export_fails_loudly(monkeypatch, tmp_path):
    import cv_worker.models as models

    (tmp_path / "current").mkdir()
    monkeypatch.setattr(models, "DEPLOYED_DETECTOR", str(tmp_path / "current"))
    with pytest.raises(RuntimeError, match="exactly one"):
        models.detector_weights({}.get)
