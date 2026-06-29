"""--image-size training option: accepted, default unchanged, used in transforms."""
from __future__ import annotations

import pytest

from training.augment import INPUT_SIZE, RESIZE_SHORT, resize_short_for


# ---- short-side resize derivation ------------------------------------------

def test_resize_short_for_default_matches_runtime():
    assert resize_short_for(INPUT_SIZE) == RESIZE_SHORT == 256


def test_resize_short_for_384():
    assert resize_short_for(384) == 439  # round(384 * 256 / 224), keeps the ratio


def test_resize_short_for_rejects_nonpositive():
    with pytest.raises(ValueError):
        resize_short_for(0)


# ---- CLI acceptance + default ----------------------------------------------

def test_parser_accepts_image_size_384():
    from training.train_classifier import build_parser
    args = build_parser().parse_args(
        ["--db", "e.db", "--recordings", "rec", "--image-size", "384"])
    assert args.image_size == 384


def test_parser_image_size_defaults_to_current_value():
    from training.train_classifier import build_parser
    args = build_parser().parse_args(["--db", "e.db", "--recordings", "rec"])
    assert args.image_size == INPUT_SIZE == 224


# ---- transforms actually use the size --------------------------------------

def test_transforms_output_chosen_image_size():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import numpy as np
    from PIL import Image

    from training.augment import build_eval_transform, build_train_transform

    img = Image.fromarray(np.zeros((300, 220, 3), dtype=np.uint8))
    size, resize = 384, resize_short_for(384)

    ev = build_eval_transform(size=size, resize=resize)(img)
    assert tuple(ev.shape) == (3, 384, 384)

    tr = build_train_transform("light", size=size, resize=resize)(img)
    assert tuple(tr.shape) == (3, 384, 384)

    # default stays at the current 224×224
    assert tuple(build_eval_transform()(img).shape) == (3, 224, 224)
