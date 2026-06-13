from training.train_classifier import Meta, build_episodes, split_episodes


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
