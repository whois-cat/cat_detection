"""A classifier run records its test crops; compare can score models on those only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.catalog_crops import TEST_CROPS_FILE
from training.compare_classifiers import held_out_keys as read_held_out
from training.train_classifier import Meta, held_out_keys


def test_run_records_the_crop_keys_of_its_test_split():
    metas = [Meta(label="ellie", camera="black", wall_ms=i, rowid=100 + i) for i in range(5)]
    metas.append(Meta(label="ellie", camera="black", wall_ms=9))   # legacy crop: no key
    assert held_out_keys(metas, [3, 1, 5]) == [101, 103]


def test_compare_reads_them_from_the_run_dir_or_its_checkpoint(tmp_path: Path):
    run = tmp_path / "20261015-120000"
    run.mkdir()
    (run / "cat_classifier.pt").write_bytes(b"")
    (run / TEST_CROPS_FILE).write_text(json.dumps({"catalog_crop_keys": [7, 9]}))
    assert read_held_out(run) == {7, 9}
    assert read_held_out(run / "cat_classifier.pt") == {7, 9}


def test_compare_explains_a_run_without_held_out_crops(tmp_path: Path):
    with pytest.raises(SystemExit, match="--test-of needs a run trained from the catalog"):
        read_held_out(tmp_path)
