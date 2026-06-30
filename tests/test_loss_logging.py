"""Loss-curve logging: the reported train/val loss uses an UNWEIGHTED mean CE
accumulated as sum(mean_b * size_b) / sum(size_b), which is independent of how
samples are grouped into batches — so train_loss (shuffled batches) and val_loss
(episode-ordered batches) are comparable. The weighted criterion is kept for
gradients only and is NOT grouping-invariant, which is the asymmetry fix-1 avoids.
"""
from __future__ import annotations

import pytest

from training.train_classifier import mean_from_batch_means


def test_mean_from_batch_means_basic():
    assert mean_from_batch_means([(1.0, 2), (4.0, 1)]) == pytest.approx((1.0 * 2 + 4.0 * 1) / 3)
    assert mean_from_batch_means([]) == 0.0  # empty set → no crash


def _acc(crit, logits, targets, batches):
    return mean_from_batch_means(
        [(float(crit(logits[idx], targets[idx])), len(idx)) for idx in batches])


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
