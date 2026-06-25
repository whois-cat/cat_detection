"""Tests for safe, identity-preserving classifier augmentation.

The pure-Python AugmentSpec tests run anywhere; the builder tests need
torch/torchvision and skip if it isn't installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from training.augment import LEVELS, augment_spec, build_eval_transform, build_train_transform


# --- pure spec invariants (no torch) -----------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_spec_never_enables_identity_destroying_ops(level):
    s = augment_spec(level)
    # These are the augmentations that can erase left/right markings, tail/ear
    # shape, etc. They must never be on for identity classification.
    assert s.horizontal_flip is False
    assert s.vertical_flip is False
    assert s.quarter_turns is False
    assert s.random_resized_crop is False
    assert s.random_erasing is False


def test_off_is_deterministic_and_others_are_random():
    assert augment_spec("off").is_random is False
    assert augment_spec("light").is_random is True
    assert augment_spec("medium").is_random is True


def test_light_rotation_and_scale_are_within_safe_envelope():
    s = augment_spec("light")
    assert 5.0 <= s.rotation_deg <= 7.0          # spec: small rotation only ±5–7°
    assert 0.90 <= s.scale_min and s.scale_max <= 1.05   # spec: ~0.90–1.05, no hard crop
    assert s.translate_frac <= 0.10


def test_levels_are_monotonic_light_to_medium():
    light, medium = augment_spec("light"), augment_spec("medium")
    assert medium.rotation_deg >= light.rotation_deg
    assert medium.brightness >= light.brightness
    # still safe: medium never reaches a quarter turn
    assert medium.rotation_deg < 45.0


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        augment_spec("aggressive")


# --- torchvision builder behaviour -------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402


def _img():
    # Asymmetric pattern so a flip would actually change the pixels.
    a = np.zeros((120, 90, 3), dtype=np.uint8)
    a[:, :45] = 200          # bright left half, dark right half
    return Image.fromarray(a)


def test_eval_transform_is_deterministic():
    ev = build_eval_transform()
    img = _img()
    assert torch.equal(ev(img), ev(img))
    assert tuple(ev(img).shape) == (3, 224, 224)


def test_augment_off_disables_random_augmentation():
    off = build_train_transform("off")
    img = _img()
    assert torch.equal(off(img), off(img))   # identical across calls → no randomness


@pytest.mark.parametrize("level", ["light", "medium"])
def test_augment_levels_are_random(level):
    tf = build_train_transform(level)
    img = _img()
    assert not torch.equal(tf(img), tf(img))
    assert tuple(tf(img).shape) == (3, 224, 224)


@pytest.mark.parametrize("level", ["off", "light", "medium"])
def test_no_flip_or_quarter_turn_or_hard_crop_ops(level):
    tf = build_train_transform(level)
    names = {type(t).__name__ for t in tf.transforms}
    forbidden = {
        "RandomHorizontalFlip", "RandomVerticalFlip",
        "RandomResizedCrop", "RandomErasing", "RandomRotation",
    }
    assert not (names & forbidden), f"{level} pipeline has forbidden op(s): {names & forbidden}"


def test_affine_rotation_stays_small_no_quarter_turns():
    import torchvision.transforms as T
    for level in ("light", "medium"):
        tf = build_train_transform(level)
        affines = [t for t in tf.transforms if isinstance(t, T.RandomAffine)]
        assert affines, f"{level} should use a small RandomAffine"
        lo, hi = affines[0].degrees
        assert abs(lo) <= 10 and abs(hi) <= 10   # never near 90/180
