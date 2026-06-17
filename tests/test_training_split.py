import numpy as np
import pytest

from training.train_classifier import (
    Meta,
    build_episodes,
    check_split_leakage,
    split_episodes,
    supported_macro_recall,
)


def test_episode_split_keeps_neighbours_together():
    metas = [
        Meta("alisa", "grey", 1_000),
        Meta("alisa", "grey", 2_000),
        Meta("chuzh", "grey", 200_000),
        Meta("chuzh", "grey", 201_000),
        Meta("ellie", "other", 1_000),
        Meta("ellie", "other", 2_000),
    ]

    episodes = build_episodes(metas, gap_ms=60_000)
    assert episodes == [[0, 1], [2, 3], [4, 5]]

    train, val, test = split_episodes(
        episodes,
        metas,
        val_frac=0.2,
        test_frac=0.2,
        required={"alisa", "chuzh", "ellie"},
        seed=7,
    )

    owners = {}
    for split_name, idxs in {"train": train, "val": val, "test": test}.items():
        for idx in idxs:
            owners[idx] = split_name

    for episode in episodes:
        assert len({owners[idx] for idx in episode}) == 1

    # The real split must pass the leakage guard.
    check_split_leakage(episodes, train, val, test)


def test_leakage_check_catches_split_violations():
    episodes = [[0, 1], [2, 3]]
    # crop 1 placed in both train and val → must raise.
    with pytest.raises(AssertionError):
        check_split_leakage(episodes, train_idx=[0, 1], val_idx=[1], test_idx=[2, 3])
    # episode [0,1] split across train/val → must raise.
    with pytest.raises(AssertionError):
        check_split_leakage(episodes, train_idx=[0], val_idx=[1], test_idx=[2, 3])
    # clean split → no raise.
    check_split_leakage(episodes, train_idx=[0, 1], val_idx=[2, 3], test_idx=[])


def test_macro_recall_ignores_classes_absent_from_eval_split():
    # Middle class has no validation samples. This happens in weekly fine-tunes
    # when a cat is present only in replay memory, which is train-only.
    cm = np.array([
        [3, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ])

    assert supported_macro_recall(cm) == 1.0
