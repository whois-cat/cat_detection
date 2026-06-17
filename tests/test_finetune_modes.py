"""configure_finetune must set requires_grad to exactly the intended params.

Uses EfficientNet-B0 with weights=None (random init, no network download) — the
architecture is what matters for which params are (un)frozen.
"""
import pytest

torch = pytest.importorskip("torch")
from torchvision import models  # noqa: E402

from training.train_classifier import configure_finetune  # noqa: E402


def _model(n_classes=4):
    import torch.nn as nn
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


def _head_param_ids(model):
    return {id(p) for p in model.classifier.parameters()}


def test_head_only_trains_only_the_head():
    m = _model()
    params, mode, n_train, n_frozen = configure_finetune(
        m, head_only=True, full_finetune=False)
    assert mode == "head"
    head_ids = _head_param_ids(m)
    trainable = [p for p in m.parameters() if p.requires_grad]
    # Every trainable param is a head param, and nothing else trains.
    assert all(id(p) in head_ids for p in trainable)
    assert len(trainable) == len(list(m.classifier.parameters()))
    # Optimizer would receive exactly these.
    assert {id(p) for p in params} == {id(p) for p in trainable}
    assert n_frozen > 0 and n_train > 0
    # Backbone features must be fully frozen.
    assert all(not p.requires_grad for p in m.features.parameters())


def test_partial_unfreezes_head_plus_last_two_blocks():
    m = _model()
    _params, mode, n_train_partial, _ = configure_finetune(
        m, head_only=False, full_finetune=False)
    assert mode == "partial"
    # Head trains.
    assert all(p.requires_grad for p in m.classifier.parameters())
    # Last two feature blocks train; an earlier block does not.
    assert all(p.requires_grad for p in m.features[-1].parameters())
    assert all(p.requires_grad for p in m.features[-2].parameters())
    assert all(not p.requires_grad for p in m.features[0].parameters())
    # Strictly more trainable than head-only.
    m2 = _model()
    _, _, n_train_head, _ = configure_finetune(m2, head_only=True, full_finetune=False)
    assert n_train_partial > n_train_head


def test_full_finetune_trains_everything():
    m = _model()
    _params, mode, n_train, n_frozen = configure_finetune(
        m, head_only=False, full_finetune=True)
    assert mode == "full"
    assert n_frozen == 0
    assert all(p.requires_grad for p in m.parameters())


def test_head_only_and_full_are_mutually_exclusive():
    m = _model()
    with pytest.raises(ValueError):
        configure_finetune(m, head_only=True, full_finetune=True)


def test_param_counts_are_consistent():
    m = _model()
    _p, _mode, n_train, n_frozen = configure_finetune(
        m, head_only=True, full_finetune=False)
    total = sum(p.numel() for p in m.parameters())
    assert n_train + n_frozen == total
