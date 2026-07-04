"""Episode-grouped train/val/test split + leakage checks + in-episode sampling.

The unit of splitting is always an episode (a visit's worth of near-duplicate
frames), never an individual crop, so neighbours cannot leak across splits."""
from __future__ import annotations

import logging
import random
from collections import defaultdict

from training.classifier_types import Meta

log = logging.getLogger("training.classifier_split")


def build_episodes(metas: list[Meta], gap_ms: int) -> list[list[int]]:
    """Group crop indices into episodes: same camera, consecutive in wall_ms with
    no gap > gap_ms. Returns a list of episodes, each a list of crop indices."""
    by_cam: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, m in enumerate(metas):
        by_cam[m.camera].append((m.wall_ms, i))
    episodes: list[list[int]] = []
    for _cam, lst in by_cam.items():
        lst.sort()
        cur: list[int] = []
        prev: int | None = None
        for wall_ms, i in lst:
            if prev is not None and wall_ms - prev > gap_ms:
                episodes.append(cur)
                cur = []
            cur.append(i)
            prev = wall_ms
        if cur:
            episodes.append(cur)
    return episodes


def split_episodes(episodes: list[list[int]], metas: list[Meta], *,
                   val_frac: float, test_frac: float,
                   required: set[str], seed: int):
    """Assign whole episodes to train/val/test.

    The unit of splitting is an episode, never an individual crop, so adjacent
    near-duplicates cannot leak across splits. Val/test are nudged to contain
    every class when possible so per-class metrics are meaningful.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("--val-frac and --test-frac must be >=0 and sum to < 1")
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)

    total = sum(len(e) for e in episodes)
    test_target = test_frac * total
    val_target = val_frac * total
    test_eps: set[int] = set()
    val_eps: set[int] = set()

    n = 0
    for e in order:
        if n >= test_target:
            break
        test_eps.add(e)
        n += len(episodes[e])

    n = 0
    for e in order:
        if n >= val_target:
            break
        if e in test_eps:
            continue
        val_eps.add(e)
        n += len(episodes[e])

    def ep_labels(e: int) -> set[str]:
        return {metas[i].label for i in episodes[e]}

    def ensure_labels(split_name: str, target_eps: set[int], other_eps: set[int]) -> None:
        label_union = set().union(*(ep_labels(e) for e in target_eps)) if target_eps else set()
        for cat in required:
            if cat in label_union:
                continue
            for e in order:
                if e not in target_eps and e not in other_eps and cat in ep_labels(e):
                    target_eps.add(e)
                    label_union |= ep_labels(e)
                    break
            else:
                log.warning(
                    "class %r not present in %s — metric may be empty for that class",
                    cat, split_name,
                )

    if val_frac > 0:
        ensure_labels("val", val_eps, test_eps)
    if test_frac > 0:
        ensure_labels("test", test_eps, val_eps)

    train_idx, val_idx, test_idx = [], [], []
    for e in range(len(episodes)):
        if e in val_eps:
            val_idx.extend(episodes[e])
        elif e in test_eps:
            test_idx.extend(episodes[e])
        else:
            train_idx.extend(episodes[e])
    return train_idx, val_idx, test_idx


def check_split_leakage(episodes: list[list[int]],
                        train_idx: list[int], val_idx: list[int],
                        test_idx: list[int]) -> None:
    """Fail loudly if the train/val/test split leaks.

    Two independent guarantees: (1) the three index sets are pairwise disjoint —
    no crop is in two splits; (2) every episode (a whole visit's worth of
    near-duplicate neighbours) lands entirely in ONE split, so adjacent frames
    of the same visit can't straddle the boundary. Raises AssertionError on any
    violation; cheap enough to run unconditionally after every split."""
    train_s, val_s, test_s = set(train_idx), set(val_idx), set(test_idx)
    assert not (train_s & val_s), f"train∩val leak: {sorted(train_s & val_s)[:5]}"
    assert not (train_s & test_s), f"train∩test leak: {sorted(train_s & test_s)[:5]}"
    assert not (val_s & test_s), f"val∩test leak: {sorted(val_s & test_s)[:5]}"
    owner: dict[int, str] = {}
    for name, s in (("train", train_s), ("val", val_s), ("test", test_s)):
        for idx in s:
            owner[idx] = name
    for ep_n, episode in enumerate(episodes):
        homes = {owner[i] for i in episode if i in owner}
        assert len(homes) <= 1, (
            f"episode {ep_n} leaks across splits {sorted(homes)} "
            f"(crops {episode[:5]})"
        )


def summarize_split(episodes: list[list[int]],
                    train_idx, val_idx, test_idx,
                    metas: list,
                    *, identities=None) -> dict:
    """Compute leakage/distribution facts for the train/val/test split.

    ``identities[i]`` (optional) is a hashable per-crop image identity (e.g. the
    detector event rowid) used for the path/image-overlap check; falls back to
    the crop index itself, which still detects index reuse.
    """
    splits = {"train": list(train_idx), "val": list(val_idx), "test": list(test_idx)}
    idx_to_split: dict[int, str] = {}
    for name, idxs in splits.items():
        for i in idxs:
            idx_to_split[i] = name
    if identities is None:
        identities = list(range(len(metas)))

    group_counts = {name: set() for name in splits}
    group_overlap: list[tuple[int, list[str]]] = []
    for ep_n, ep in enumerate(episodes):
        homes = {idx_to_split[i] for i in ep if i in idx_to_split}
        for h in homes:
            group_counts[h].add(ep_n)
        if len(homes) > 1:
            group_overlap.append((ep_n, sorted(homes)))

    class_dist = {}
    for name, idxs in splits.items():
        d: dict[str, int] = defaultdict(int)
        for i in idxs:
            d[metas[i].label] += 1
        class_dist[name] = dict(sorted(d.items()))

    id_by_split = {name: {identities[i] for i in idxs} for name, idxs in splits.items()}
    path_overlap = 0
    path_examples: list[tuple[str, str, object]] = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = id_by_split[a] & id_by_split[b]
        path_overlap += len(inter)
        for x in list(inter)[:3]:
            path_examples.append((a, b, x))

    return {
        "sample_counts": {n: len(s) for n, s in splits.items()},
        "group_counts": {n: len(s) for n, s in group_counts.items()},
        "class_dist": class_dist,
        "group_overlap_count": len(group_overlap),
        "group_overlap_examples": group_overlap[:5],
        "path_overlap_count": path_overlap,
        "path_overlap_examples": path_examples,
    }


def print_split_report(summary: dict, *, episode_gap_sec: float) -> None:
    """Loud, explicit leakage report (acceptance: zero group/path overlap)."""
    print("\n=== train/val/test split report ===")
    print("split method: episode-grouped (whole visit → one split, never per-crop)")
    print(f"group key:    camera_id + consecutive wall_ms gap <= {episode_gap_sec:g}s")
    sc, gc = summary["sample_counts"], summary["group_counts"]
    for name in ("train", "val", "test"):
        print(f"  {name:<5} samples={sc[name]:<6} groups={gc[name]:<5} "
              f"classes={summary['class_dist'][name]}")
    print(f"group overlap (episodes spanning splits): {summary['group_overlap_count']}")
    if summary["group_overlap_examples"]:
        print(f"  examples: {summary['group_overlap_examples']}")
    print(f"path/image overlap (same crop in two splits): {summary['path_overlap_count']}")
    if summary["path_overlap_examples"]:
        print(f"  examples: {summary['path_overlap_examples']}")
    if summary["group_overlap_count"] or summary["path_overlap_count"]:
        raise SystemExit(
            "LEAKAGE DETECTED: train/val/test share groups or crops — refusing to "
            "train on a contaminated split (see report above)."
        )
    print("leakage check: OK (0 group overlap, 0 path overlap)\n")


def _evenly_spaced(indices: list[int], quota: int) -> list[int]:
    if quota <= 0:
        return []
    if len(indices) <= quota:
        return list(indices)
    if quota == 1:
        return [indices[0]]
    out = []
    step = (len(indices) - 1) / float(quota - 1)
    for n in range(quota):
        out.append(indices[round(n * step)])
    return list(dict.fromkeys(out))


def sample_indices_for_training(
    indices: list[int],
    episodes: list[list[int]],
    metas: list[Meta],
    *,
    max_per_episode: int = 0,
    max_per_duplicate_group: int = 0,
    keep_suspicious_per_episode: int = 4,
) -> list[int]:
    """Deterministically thin near-duplicate training examples within episodes.

    The split unit stays the episode/visit. This function only decides which
    crops inside already-assigned episodes are useful enough to decode/train on,
    so adjacent frames cannot leak across train/val/test.
    """
    allowed = set(indices)
    if not allowed:
        return []
    if max_per_episode <= 0 and max_per_duplicate_group <= 0:
        return list(indices)

    selected: set[int] = set()
    for episode in episodes:
        ep = [i for i in episode if i in allowed]
        if not ep:
            continue
        ep.sort(key=lambda i: (metas[i].wall_ms, metas[i].rowid or -1, i))

        keep: set[int] = set()
        # First/last preserve visit boundaries; suspicious keeps hard examples.
        keep.add(ep[0])
        keep.add(ep[-1])
        suspicious = sorted(
            ep,
            key=lambda i: (-float(metas[i].suspicious_score), metas[i].wall_ms, i),
        )
        for i in suspicious[:max(0, keep_suspicious_per_episode)]:
            if metas[i].suspicious_score > 0:
                keep.add(i)

        by_group: dict[str, list[int]] = defaultdict(list)
        for i in ep:
            gid = metas[i].duplicate_group_id or f"event:{i}"
            by_group[gid].append(i)
        if max_per_duplicate_group > 0:
            for group in by_group.values():
                keep.update(_evenly_spaced(group, max_per_duplicate_group))
        else:
            keep.update(ep)

        if max_per_episode > 0 and len(keep) < min(len(ep), max_per_episode):
            remaining = [i for i in ep if i not in keep]
            keep.update(_evenly_spaced(remaining, max_per_episode - len(keep)))

        if max_per_episode > 0 and len(keep) > max_per_episode:
            keep_order = sorted(
                keep,
                key=lambda i: (
                    i not in {ep[0], ep[-1]},
                    -float(metas[i].suspicious_score),
                    metas[i].wall_ms,
                    i,
                ),
            )
            keep = set(keep_order[:max_per_episode])
        selected.update(keep)

    return [i for i in indices if i in selected]
