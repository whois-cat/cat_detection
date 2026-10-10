import pytest

from cv_worker.models.yolo import cat_class_id


def test_coco_base_model_cat_is_15():
    names = {i: f"class{i}" for i in range(80)} | {15: "cat"}
    assert cat_class_id(names) == 15


def test_model_without_cat_fails_loudly():
    with pytest.raises(ValueError, match="no 'cat' class"):
        cat_class_id({0: "dog"})


def test_weights_label_names_finetunes_after_their_run():
    from cv_worker.models.yolo import weights_label

    assert weights_label("/opt/models/yolov8n_int8_openvino_model/") == "yolov8n_int8_openvino_model"
    assert weights_label(
        "/opt/models/trained/yolo-20261009-183756/weights/best_int8_openvino_model"
    ) == "yolo-20261009-183756-best"
