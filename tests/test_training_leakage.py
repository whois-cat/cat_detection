"""Replay→eval leakage detection: exact, near-dup, content-hash, and clean."""
import numpy as np
import pytest

from training.leakage import (
    Identity,
    LeakageError,
    apply_leakage_policy,
    build_eval_identities,
    find_replay_leaks,
    perceptual_hash,
    replay_identities,
)


class _Replay:
    """Minimal stand-in for training.replay.ReplayItem."""
    def __init__(self, key, camera, wall_ms, phash=None):
        self.src_event_key = key
        self.camera = camera
        self.wall_ms = wall_ms
        self.phash = phash


def _eval_index(entries):
    # entries: (split, crop_index, rowid, camera, wall_ms[, phash])
    return build_eval_identities(
        (split, ci, Identity(rowid, cam, wall, *(rest or (None,))))
        for split, ci, rowid, cam, wall, *rest in entries
    )


def test_exact_duplicate_rowid():
    idx = _eval_index([("val", 0, 100, "grey", 5000)])
    leaks = find_replay_leaks(idx, replay_identities([_Replay(100, "grey", 999999)]),
                              window_ms=2000)
    assert len(leaks) == 1
    assert leaks[0].kind == "exact-rowid"
    assert leaks[0].split == "val"


def test_same_event_different_crop_same_frame():
    # Different rowid (a second box in the same frame), same camera+wall_ms.
    idx = _eval_index([("test", 3, 200, "grey", 5000)])
    leaks = find_replay_leaks(idx, replay_identities([_Replay(201, "grey", 5000)]),
                              window_ms=2000)
    assert len(leaks) == 1
    assert leaks[0].kind == "same-frame"
    assert leaks[0].split == "test"


def test_near_duplicate_timestamp_same_camera():
    idx = _eval_index([("val", 1, 300, "grey", 5000)])
    # 800 ms away on the same camera → same visit/episode.
    leaks = find_replay_leaks(idx, replay_identities([_Replay(999, "grey", 5800)]),
                              window_ms=2000)
    assert len(leaks) == 1
    assert leaks[0].kind == "near-timestamp"


def test_same_timestamp_different_camera_is_not_leakage():
    idx = _eval_index([("val", 1, 300, "grey", 5000)])
    leaks = find_replay_leaks(idx, replay_identities([_Replay(999, "brown", 5000)]),
                              window_ms=2000)
    assert leaks == []


def test_clean_non_overlapping_datasets():
    idx = _eval_index([
        ("val", 0, 10, "grey", 1000),
        ("test", 1, 11, "grey", 9000),
    ])
    replay = [_Replay(500, "grey", 5000), _Replay(501, "brown", 1000)]
    leaks = find_replay_leaks(idx, replay_identities(replay), window_ms=500)
    assert leaks == []


def test_near_duplicate_image_content_hash():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    base[8:24, 8:24] = 200
    near = base.copy()
    near[0, 0] = 5  # 1-pixel tweak → near-identical hash
    # Eval crop carries a pHash; replay crop (no shared metadata) carries one too.
    idx = build_eval_identities([
        ("val", 0, Identity(None, "camA", 1000, perceptual_hash(base))),
    ])
    leaks = find_replay_leaks(
        idx, [Identity(rowid=999, camera="other", wall_ms=999999,
                       phash=perceptual_hash(near))],
        window_ms=0,
    )
    assert len(leaks) == 1
    assert leaks[0].kind == "content-hash"


def test_policy_error_raises():
    leaks = find_replay_leaks(_eval_index([("val", 0, 1, "g", 1)]),
                              replay_identities([_Replay(1, "g", 1)]), window_ms=0)
    with pytest.raises(LeakageError):
        apply_leakage_policy("error", leaks, episodes=[[0]],
                             fresh_index_to_episode={0: 0},
                             train_idx=[], val_idx=[0], test_idx=[], n_replay=1)


def test_policy_drop_from_replay():
    leaks = find_replay_leaks(_eval_index([("val", 0, 1, "g", 1)]),
                              replay_identities([_Replay(1, "g", 1), _Replay(2, "g", 99)]),
                              window_ms=0)
    res = apply_leakage_policy("drop-from-replay", leaks, episodes=[[0]],
                               fresh_index_to_episode={0: 0},
                               train_idx=[], val_idx=[0], test_idx=[], n_replay=2)
    assert res.kept_replay == [1]      # replay 0 dropped, replay 1 kept
    assert res.dropped_replay == 1
    assert res.val_idx == [0]          # eval untouched


def test_policy_move_episode_to_train():
    # Episode 0 = crops [0,1] in val; a replay crop duplicates crop 0.
    episodes = [[0, 1], [2, 3]]
    leaks = find_replay_leaks(
        _eval_index([("val", 0, 10, "g", 1000), ("val", 1, 11, "g", 1100)]),
        replay_identities([_Replay(10, "g", 1000)]), window_ms=0)
    res = apply_leakage_policy(
        "move-related-episode-to-train", leaks, episodes=episodes,
        fresh_index_to_episode={0: 0, 1: 0, 2: 1, 3: 1},
        train_idx=[2, 3], val_idx=[0, 1], test_idx=[], n_replay=1)
    assert sorted(res.train_idx) == [0, 1, 2, 3]   # whole episode moved
    assert res.val_idx == []
    assert res.moved_eval_crops == 2
    assert res.kept_replay == [0]      # replay kept (eval cleaned instead)


def test_no_leaks_returns_inputs_unchanged():
    res = apply_leakage_policy("error", [], episodes=[[0]],
                               fresh_index_to_episode={0: 0},
                               train_idx=[1], val_idx=[0], test_idx=[], n_replay=2)
    assert res.train_idx == [1]
    assert res.kept_replay == [0, 1]
