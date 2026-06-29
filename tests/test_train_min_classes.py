"""Single-class training guard: refuse a degenerate identity classifier.

Generic labels only (cat_a/cat_b/...) — no assumptions about real class names.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")  # train_classifier imports torch at module load

from training.train_classifier import require_min_classes  # noqa: E402


def test_two_classes_pass():
    # No raise — the normal case.
    require_min_classes(["cat_a", "cat_b"])


def test_three_classes_pass():
    require_min_classes(["cat_a", "cat_b", "cat_c"])


def test_one_class_refuses_before_model_build():
    with pytest.raises(SystemExit) as e:
        require_min_classes(["cat_a"])
    msg = str(e.value)
    assert "at least 2 labeled classes" in msg
    assert "cat_a" in msg               # reports the found classes dynamically
    assert "label" in msg               # points the operator at the next step


def test_zero_classes_refuses():
    with pytest.raises(SystemExit, match="at least 2 labeled classes"):
        require_min_classes([])
