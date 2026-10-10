"""Ultralytics val output must land next to our report, not in the global runs_dir."""
from pathlib import Path

from training.yolo_evaluate import val_output_kwargs


def test_val_output_is_pinned_to_the_given_dir(tmp_path: Path):
    kwargs = val_output_kwargs(tmp_path / "run" / "test_eval")
    assert kwargs == {"project": str((tmp_path / "run").resolve()),
                      "name": "test_eval", "exist_ok": True}


def test_no_dir_keeps_ultralytics_default():
    assert val_output_kwargs(None) == {}
