"""Ultralytics val output must land next to our report, not in the global runs_dir."""
from pathlib import Path

from training.yolo_evaluate import val_output_kwargs


def test_val_output_is_pinned_to_the_given_dir(tmp_path: Path):
    kwargs = val_output_kwargs(tmp_path / "run" / "test_eval")
    assert kwargs == {"project": str((tmp_path / "run").resolve()),
                      "name": "test_eval", "exist_ok": True}


def test_no_dir_keeps_ultralytics_default():
    assert val_output_kwargs(None) == {}


def _dataset(tmp_path: Path):
    from training.yolo_common import DatasetInfo

    version = tmp_path / "versions" / "v1"
    for sid, rows in (("a", "0 0.5 0.5 0.2 0.2\n"), ("b", "")):
        (version / "images" / "test").mkdir(parents=True, exist_ok=True)
        (version / "labels" / "test").mkdir(parents=True, exist_ok=True)
        (version / "images" / "test" / f"{sid}.jpg").write_bytes(b"jpeg")
        (version / "labels" / "test" / f"{sid}.txt").write_text(rows)
    (version / "data.yaml").write_text(f"path: {version}\nnames:\n  0: cat\n")
    manifest = {"samples": [{"sample_id": "a", "split": "test"},
                            {"sample_id": "b", "split": "test"},
                            {"sample_id": "c", "split": "train"}]}
    return DatasetInfo(version_dir=version, data_yaml=version / "data.yaml",
                       manifest_path=version / "manifest.json", manifest=manifest,
                       dataset_sha256="", manifest_file_sha256="", names={0: "cat"},
                       cat_id=0, summary={})


def test_same_cat_id_uses_the_version_as_is(tmp_path: Path):
    from training.yolo_evaluate import official_val_data

    dataset = _dataset(tmp_path)
    with official_val_data(dataset, 0) as data_yaml:
        assert data_yaml == dataset.data_yaml


def test_coco_model_gets_labels_renumbered_to_its_cat_id(tmp_path: Path):
    from training.yolo_evaluate import official_val_data

    dataset = _dataset(tmp_path)
    with official_val_data(dataset, 15) as data_yaml:
        root = data_yaml.parent
        assert (root / "labels" / "test" / "a.txt").read_text() == "15 0.5 0.5 0.2 0.2\n"
        assert (root / "labels" / "test" / "b.txt").read_text() == ""   # negative stays empty
        assert sorted(p.name for p in (root / "images" / "test").iterdir()) == ["a.jpg", "b.jpg"]
        text = data_yaml.read_text()
        assert "  15: cat" in text and "test: images/test" in text
    assert not root.exists()                                     # throwaway copy removed
    assert (dataset.version_dir / "labels" / "test" / "a.txt").read_text().startswith("0 ")


def test_data_yaml_follows_the_version_when_it_moved(tmp_path: Path):
    import yaml

    from training.yolo_common import ultralytics_data_yaml

    dataset = _dataset(tmp_path)
    # built on another machine / outside the container
    dataset.data_yaml.write_text("path: /home/elsewhere/versions/v1\ntrain: images/train\n"
                                 "val: images/val\ntest: images/test\nnames:\n  0: cat\n")
    moved = ultralytics_data_yaml(dataset)
    assert moved != dataset.data_yaml
    data = yaml.safe_load(moved.read_text())
    assert data["path"] == str(dataset.version_dir.resolve())
    assert data["names"] == {0: "cat"} and data["test"] == "images/test"
    assert "elsewhere" in dataset.data_yaml.read_text()      # version left untouched

    dataset.data_yaml.write_text(f"path: {dataset.version_dir}\nnames:\n  0: cat\n")
    assert ultralytics_data_yaml(dataset) == dataset.data_yaml
