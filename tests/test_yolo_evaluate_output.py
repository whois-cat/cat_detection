"""YOLO evaluation plumbing: where Ultralytics writes, which data.yaml it reads,
and that model and dataset agree on the (COCO) cat class id."""
from pathlib import Path

import pytest
import yaml

from training.yolo_common import CAT_CLASS_ID, COCO_NAMES, DatasetInfo, load_dataset, ultralytics_data_yaml
from training.yolo_evaluate import evaluate_artifact, val_output_kwargs


def test_val_output_is_pinned_to_the_given_dir(tmp_path: Path):
    kwargs = val_output_kwargs(tmp_path / "run" / "test_eval")
    assert kwargs == {"project": str((tmp_path / "run").resolve()),
                      "name": "test_eval", "exist_ok": True}


def test_no_dir_keeps_ultralytics_default():
    assert val_output_kwargs(None) == {}


def _dataset(tmp_path: Path, data_yaml: str | None = None) -> DatasetInfo:
    version = tmp_path / "versions" / "v1"
    (version / "images" / "test").mkdir(parents=True)
    (version / "labels" / "test").mkdir(parents=True)
    (version / "images" / "test" / "a.jpg").write_bytes(b"jpeg")
    (version / "labels" / "test" / "a.txt").write_text("15 0.5 0.5 0.2 0.2\n")
    (version / "data.yaml").write_text(data_yaml or yaml.safe_dump(
        {"path": str(version), "test": "images/test", "names": dict(enumerate(COCO_NAMES))}))
    return DatasetInfo(version_dir=version, data_yaml=version / "data.yaml",
                       manifest_path=version / "manifest.json",
                       manifest={"samples": [{"sample_id": "a", "split": "test"}]},
                       dataset_sha256="0" * 64, manifest_file_sha256="", names={},
                       cat_id=CAT_CLASS_ID, summary={})


def test_cat_is_the_coco_id():
    assert CAT_CLASS_ID == 15 and len(COCO_NAMES) == 80


def test_data_yaml_follows_the_version_when_it_moved(tmp_path: Path):
    dataset = _dataset(tmp_path)
    assert ultralytics_data_yaml(dataset) == dataset.data_yaml

    # built on another machine / outside the container
    raw = yaml.safe_load(dataset.data_yaml.read_text()) | {"path": "/home/elsewhere/versions/v1"}
    dataset.data_yaml.write_text(yaml.safe_dump(raw))
    moved = ultralytics_data_yaml(dataset)
    assert moved != dataset.data_yaml
    data = yaml.safe_load(moved.read_text())
    assert data["path"] == str(dataset.version_dir.resolve())
    assert data["names"][15] == "cat" and data["test"] == "images/test"
    assert "elsewhere" in dataset.data_yaml.read_text()      # version left untouched


def test_single_class_version_is_rejected_with_rebuild_hint(tmp_path: Path):
    import hashlib
    import json

    dataset = _dataset(tmp_path, "path: x\nnames:\n  0: cat\n")
    samples = dataset.manifest["samples"]
    (dataset.version_dir / "manifest.json").write_text(json.dumps({
        "samples": samples,
        "manifest_sha256": hashlib.sha256(
            json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }))
    with pytest.raises(ValueError, match="box-build"):
        load_dataset(dataset.version_dir)


class _Model:
    def __init__(self, names):
        self.names = names

    def val(self, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("evaluated a mismatched model")


def test_single_class_model_is_rejected_not_renumbered(tmp_path: Path):
    dataset = _dataset(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    with pytest.raises(ValueError, match="retraining"):
        evaluate_artifact(weights, dataset, model_factory=lambda _p: _Model({0: "cat"}))
