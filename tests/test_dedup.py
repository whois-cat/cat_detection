from training.dedup import DedupConfig, DedupSample, deduplicate, hamming


def _s(sid, h, *, label="alisa", wall=1000, hard=False, **kw):
    return DedupSample(sample_id=sid, label=label, camera="grey", video="v1",
                       wall_ms=wall, dhash=h, hard=hard, **kw)


def test_hamming():
    assert hamming(0b0, 0b0) == 0
    assert hamming(0b1011, 0b0001) == 2


def test_near_duplicates_collapse():
    # Same group, hashes 1 bit apart → one representative kept.
    s = [_s(1, 0b0, bbox_area=900), _s(2, 0b1, wall=1200, bbox_area=400)]
    res = deduplicate(s, DedupConfig(hash_distance=6, keep_hard_fraction=0.0))
    assert res.kept_ids == [1]               # larger bbox wins as representative
    assert res.report["near_duplicates"] == 1
    assert res.report["samples_after"] == 1


def test_distinct_poses_are_kept():
    # Same group, hashes far apart (64-bit max) → both survive.
    s = [_s(1, 0), _s(2, (1 << 64) - 1, wall=1200)]
    res = deduplicate(s, DedupConfig(hash_distance=6))
    assert sorted(res.kept_ids) == [1, 2]
    assert res.report["near_duplicates"] == 0


def test_hard_examples_preserved():
    rep = _s(1, 0b0, bbox_area=900)
    hard = _s(2, 0b1, wall=1200, hard=True, bbox_area=400)
    kept_none = deduplicate([rep, hard], DedupConfig(keep_hard_fraction=0.0))
    assert kept_none.kept_ids == [1]
    kept_all = deduplicate([rep, hard], DedupConfig(keep_hard_fraction=1.0))
    assert sorted(kept_all.kept_ids) == [1, 2]
    assert kept_all.reason_by_id[2] == "kept_hard"


def test_max_samples_per_group_cap():
    s = [_s(i, (1 << i)) for i in range(6)]  # all distinct → all reps
    res = deduplicate(s, DedupConfig(hash_distance=0, max_samples_per_group=3))
    assert len(res.kept_ids) == 3
    assert list(res.report["reason_counts"].get("over_cap", 0) for _ in [0])[0] == 3


def test_disabled_passthrough():
    s = [_s(1, 0), _s(2, 0, wall=1100)]
    res = deduplicate(s, DedupConfig(enabled=False))
    assert sorted(res.kept_ids) == [1, 2]
    assert res.report["samples_after"] == 2


def test_deterministic():
    s = [_s(i, i % 3, wall=1000 + i * 10, bbox_area=100 + i) for i in range(30)]
    a = deduplicate(s, DedupConfig(hash_distance=1, keep_hard_fraction=0.25))
    b = deduplicate(s, DedupConfig(hash_distance=1, keep_hard_fraction=0.25))
    assert a.kept_ids == b.kept_ids
    assert a.report == b.report


def test_groups_do_not_cross_label_or_video():
    # Identical hash but different labels → never merged.
    s = [_s(1, 0, label="alisa"), _s(2, 0, label="chuzh", wall=1100)]
    res = deduplicate(s, DedupConfig(hash_distance=6))
    assert sorted(res.kept_ids) == [1, 2]
