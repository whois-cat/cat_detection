"""Loss-curve logging: the reported train/val loss uses an UNWEIGHTED mean CE
accumulated as sum(mean_b * size_b) / sum(size_b), which is independent of how
samples are grouped into batches — so train_loss (shuffled batches) and val_loss
(episode-ordered batches) are comparable. The weighted criterion is kept for
gradients only and is NOT grouping-invariant, which is the asymmetry fix-1 avoids.
"""
from __future__ import annotations

import pytest

from training.train_classifier import format_epoch_line, mean_from_batch_means


def test_mean_from_batch_means_basic():
    assert mean_from_batch_means([(1.0, 2), (4.0, 1)]) == pytest.approx((1.0 * 2 + 4.0 * 1) / 3)
    assert mean_from_batch_means([]) == 0.0  # empty set → no crash


def _acc(crit, logits, targets, batches):
    return mean_from_batch_means(
        [(float(crit(logits[idx], targets[idx])), len(idx)) for idx in batches])


def test_epoch_line_omits_dangerous_when_not_configured():
    line = format_epoch_line(3, 0.4127, 0.5012, 0.942, 0, dangerous_configured=False)
    assert "dangerous_errors" not in line
    assert line == (
        "epoch 03  train_loss=0.4127  val_loss=0.5012  val_macro_recall=0.942")


def test_epoch_line_shows_dangerous_when_configured():
    line = format_epoch_line(12, 0.1000, 0.2000, 0.990, 4, dangerous_configured=True)
    assert line == (
        "epoch 12  train_loss=0.1000  val_loss=0.2000  val_macro_recall=0.990  "
        "dangerous_errors=4")


def test_epoch_line_shows_zero_dangerous_when_configured_but_none_hit():
    # Configured pairs with no errors this epoch still report 0 (not omitted).
    line = format_epoch_line(1, 0.5, 0.5, 0.5, 0, dangerous_configured=True)
    assert line.endswith("dangerous_errors=0")


def test_unweighted_loss_is_grouping_invariant_weighted_is_not():
    torch = pytest.importorskip("torch")
    from torch import nn

    # 4 samples, 2 classes, with deliberately different per-sample losses.
    logits = torch.tensor([
        [2.0, 0.0],   # target 0 → easy
        [0.0, 2.0],   # target 0 → hard
        [0.0, 3.0],   # target 1 → easy
        [3.0, 0.0],   # target 1 → hard
    ])
    targets = torch.tensor([0, 0, 1, 1])

    mixed = [[0, 2], [1, 3]]      # each batch mixes both classes (≈ shuffled train)
    grouped = [[0, 1], [2, 3]]    # each batch is one class (≈ episode-ordered val)

    unweighted = nn.CrossEntropyLoss()
    weighted = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))

    # Unweighted: the logged loss is the SAME no matter how batches are grouped.
    assert _acc(unweighted, logits, targets, mixed) == pytest.approx(
        _acc(unweighted, logits, targets, grouped))

    # Weighted: the same accumulation depends on grouping → train/val asymmetry.
    assert _acc(weighted, logits, targets, mixed) != pytest.approx(
        _acc(weighted, logits, targets, grouped))
