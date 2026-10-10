from __future__ import annotations

from pathlib import Path

from training.yolo_common import COCO_NAMES, DatasetInfo
from training.yolo_train import merge_training_dataset


def _make_version(root: Path, version_id: str, samples: list[dict]) -> DatasetInfo:
    for sample in samples:
        for kind, ext in (("images", "jpg"), ("labels", "txt")):
            path = root / kind / sample["split"] / f"{sample['sample_id']}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" if ext == "jpg" else b"15 0.5 0.5 0.2 0.2\n")
    return DatasetInfo(
        version_dir=root, data_yaml=root / "data.yaml", manifest_path=root / "manifest.json",
        manifest={"version_id": version_id, "samples": samples},
        dataset_sha256="d", manifest_file_sha256="m", names=dict(enumerate(COCO_NAMES)), cat_id=15, summary={},
    )


def test_merge_mixes_replay_train_and_drops_eval_collisions(tmp_path: Path):
    current = _make_version(tmp_path / "cur", "cur", [
        {"sample_id": "t1", "split": "train", "group_id": "g1", "sha256": "a1"},
        {"sample_id": "t2", "split": "train", "group_id": "g2", "sha256": "a2"},
        {"sample_id": "v1", "split": "val", "group_id": "gV", "sha256": "shaV"},
        {"sample_id": "e1", "split": "test", "group_id": "gE", "sha256": "shaE"},
    ])
    replay = _make_version(tmp_path / "old", "old", [
        {"sample_id": "r1", "split": "train", "group_id": "gR1", "sha256": "b1"},   # added
        {"sample_id": "r5", "split": "train", "group_id": "gR5", "sha256": "b5"},   # added
        {"sample_id": "rG", "split": "train", "group_id": "gV", "sha256": "b2"},    # group collides -> skip
        {"sample_id": "rS", "split": "train", "group_id": "gR3", "sha256": "shaE"}, # sha collides -> skip
        {"sample_id": "t1", "split": "train", "group_id": "gR4", "sha256": "b4"},   # id already present -> skip
        {"sample_id": "rv", "split": "val", "group_id": "gR6", "sha256": "b6"},     # replay val never pulled
    ])

    dest = tmp_path / "merged"
    summary = merge_training_dataset(current, [replay], dest)

    train_ids = {p.stem for p in (dest / "images" / "train").glob("*.jpg")}
    val_ids = {p.stem for p in (dest / "images" / "val").glob("*.jpg")}
    assert train_ids == {"t1", "t2", "r1", "r5"}
    assert val_ids == {"v1"}
    assert summary["current_train_images"] == 2
    assert summary["replay_added"] == 2
    assert summary["replay_skipped_leakage"] == 3
    assert summary["merged_train_images"] == 4
    assert summary["replay_versions"] == ["old"]

    data_yaml = (dest / "data.yaml").read_text()
    assert "15: cat" in data_yaml and "train: images/train" in data_yaml
    # No evaluation sample may leak into the merged training set.
    assert not ({"v1", "e1"} & train_ids)
