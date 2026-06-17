"""Cross-source train/val/test leakage detection for replay memory.

The episode split (``train_classifier.split_episodes``) keeps neighbouring fresh
crops together, and ``check_split_leakage`` proves the fresh split is disjoint.
But replay crops are appended to TRAIN *after* that check, straight from the
replay manifest — and a replay crop is just an old fresh crop that survived video
pruning. If the same underlying event (or a near-duplicate of it) also landed in
val or test, the model trains on its eval data. That is silent leakage and
inflates every metric.

This module detects it, fail-closed by default. Identity is established from
source metadata first (it's exact and cheap), with a perceptual-hash fallback
for the rare case where metadata is missing:

  1. exact-rowid      replay.src_event_key == an eval crop's events rowid
  2. same-frame       same (camera, wall_ms) — same decoded frame, other box
  3. near-timestamp   same camera, |Δ wall_ms| <= window — same visit/episode
  4. content-hash     perceptual hash within Hamming threshold (metadata-light)

Nothing here decodes video; identities are built from the same metadata the
trainer already has in RAM.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------- identities --

@dataclass(frozen=True)
class Identity:
    """Stable identity of one crop, from source metadata (+ optional pHash).

    ``rowid`` is the events-table PK (== replay ``src_event_key``); ``camera`` is
    the RAW camera id (NOT the ``replay:`` namespaced one)."""
    rowid: int | None
    camera: str
    wall_ms: int
    phash: int | None = None


@dataclass(frozen=True)
class ReplayLeak:
    replay_index: int            # index into the replay list passed in
    split: str                   # "val" | "test"
    kind: str                    # exact-rowid | same-frame | near-timestamp | content-hash
    eval_crop_index: int         # fresh-crop index of the colliding eval sample
    detail: str


@dataclass
class EvalIndex:
    """Lookup structures over the val+test (fresh) crops."""
    phash_max_dist: int = 4
    _by_rowid: dict[int, tuple[str, int]] = field(default_factory=dict)
    _by_frame: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)
    # camera -> sorted list of (wall_ms, split, crop_index)
    _times: dict[str, list[tuple[int, str, int]]] = field(default_factory=lambda: defaultdict(list))
    _phashes: list[tuple[int, str, int]] = field(default_factory=list)

    def add(self, split: str, crop_index: int, ident: Identity) -> None:
        if ident.rowid is not None:
            self._by_rowid.setdefault(int(ident.rowid), (split, crop_index))
        self._by_frame.setdefault((ident.camera, int(ident.wall_ms)), (split, crop_index))
        self._times[ident.camera].append((int(ident.wall_ms), split, crop_index))
        if ident.phash is not None:
            self._phashes.append((int(ident.phash), split, crop_index))

    def finalize(self) -> "EvalIndex":
        for cam in self._times:
            self._times[cam].sort()
        return self

    def match(self, ident: Identity, window_ms: int) -> tuple[str, str, int] | None:
        """Return (split, kind, eval_crop_index) for the strongest collision, or
        None. Priority: exact rowid > same frame > near timestamp > content hash."""
        if ident.rowid is not None and int(ident.rowid) in self._by_rowid:
            split, ci = self._by_rowid[int(ident.rowid)]
            return split, "exact-rowid", ci
        frame_key = (ident.camera, int(ident.wall_ms))
        if frame_key in self._by_frame:
            split, ci = self._by_frame[frame_key]
            return split, "same-frame", ci
        if window_ms > 0:
            times = self._times.get(ident.camera)
            if times:
                near = _nearest_within(times, int(ident.wall_ms), window_ms)
                if near is not None:
                    split, ci = near
                    return split, "near-timestamp", ci
        if ident.phash is not None and self._phashes:
            for h, split, ci in self._phashes:
                if _hamming(h, int(ident.phash)) <= self.phash_max_dist:
                    return split, "content-hash", ci
        return None


def _nearest_within(times: list[tuple[int, str, int]], wall_ms: int,
                    window_ms: int) -> tuple[str, int] | None:
    """Closest (split, crop_index) whose wall_ms is within window of wall_ms."""
    keys = [t[0] for t in times]
    pos = bisect.bisect_left(keys, wall_ms)
    best: tuple[int, str, int] | None = None
    for j in (pos - 1, pos):
        if 0 <= j < len(times):
            d = abs(times[j][0] - wall_ms)
            if d <= window_ms and (best is None or d < best[0]):
                best = (d, times[j][1], times[j][2])
    return (best[1], best[2]) if best is not None else None


# ----------------------------------------------------------- perceptual hash --

def perceptual_hash(image: np.ndarray) -> int:
    """64-bit average hash (aHash). Pure numpy (no cv2) so this stays a light
    dependency; used only as a metadata-free fallback identity."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.size == 0:
        return 0
    rows = [r.mean(axis=0) for r in np.array_split(a, 8, axis=0)]
    by_row = np.stack(rows)                          # (8, W)
    cols = [c.mean(axis=1) for c in np.array_split(by_row, 8, axis=1)]
    pooled = np.stack(cols, axis=1)                  # (8, 8)
    bits = (pooled > pooled.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------- public API ----

def build_eval_identities(entries: Iterable[tuple[str, int, Identity]],
                          *, phash_max_dist: int = 4) -> EvalIndex:
    """Build an :class:`EvalIndex` from (split, crop_index, Identity) entries."""
    idx = EvalIndex(phash_max_dist=phash_max_dist)
    for split, crop_index, ident in entries:
        idx.add(split, crop_index, ident)
    return idx.finalize()


def replay_identities(replay_items) -> list[Identity]:
    """Identities for replay items (``training.replay.ReplayItem`` or anything
    with ``src_event_key``/``camera``/``wall_ms``[/``phash``])."""
    out: list[Identity] = []
    for it in replay_items:
        out.append(Identity(
            rowid=getattr(it, "src_event_key", None),
            camera=getattr(it, "camera", "unknown"),
            wall_ms=int(getattr(it, "wall_ms", 0)),
            phash=getattr(it, "phash", None),
        ))
    return out


def find_replay_leaks(eval_index: EvalIndex, replay_ids: list[Identity],
                      *, window_ms: int) -> list[ReplayLeak]:
    """Every replay identity that collides with a val/test crop."""
    leaks: list[ReplayLeak] = []
    for ri, ident in enumerate(replay_ids):
        hit = eval_index.match(ident, window_ms)
        if hit is None:
            continue
        split, kind, ci = hit
        leaks.append(ReplayLeak(
            replay_index=ri, split=split, kind=kind, eval_crop_index=ci,
            detail=f"replay[{ri}] cam={ident.camera} wall_ms={ident.wall_ms} "
                   f"rowid={ident.rowid} -> {split} crop#{ci} ({kind})",
        ))
    return leaks


def format_leak_report(leaks: list[ReplayLeak], *, max_examples: int = 8) -> str:
    """Counts by (split, kind) plus a few concrete examples — never thousands."""
    if not leaks:
        return "no replay→eval leakage detected"
    by_kind: dict[tuple[str, str], int] = defaultdict(int)
    for lk in leaks:
        by_kind[(lk.split, lk.kind)] += 1
    lines = [f"REPLAY LEAKAGE: {len(leaks)} replay crop(s) duplicate val/test data"]
    for (split, kind), n in sorted(by_kind.items()):
        lines.append(f"  {split:<4} {kind:<14} ×{n}")
    lines.append("examples:")
    for lk in leaks[:max_examples]:
        lines.append(f"  - {lk.detail}")
    if len(leaks) > max_examples:
        lines.append(f"  … and {len(leaks) - max_examples} more")
    return "\n".join(lines)


class LeakageError(RuntimeError):
    """Replay leakage detected under the default fail-closed policy."""


@dataclass
class LeakageResolution:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]
    kept_replay: list[int]       # replay-local indices to keep (train-only)
    dropped_replay: int = 0
    moved_eval_crops: int = 0


def apply_leakage_policy(policy: str, leaks: list[ReplayLeak], *,
                         episodes: list[list[int]],
                         fresh_index_to_episode: dict[int, int],
                         train_idx: list[int], val_idx: list[int],
                         test_idx: list[int], n_replay: int) -> LeakageResolution:
    """Resolve detected leaks per ``policy``.

    - ``error``: raise :class:`LeakageError` (fail closed; the default).
    - ``drop-from-replay``: drop the offending replay crops; eval split intact.
    - ``move-related-episode-to-train``: move the colliding eval crops' WHOLE
      episodes from val/test into train, so eval no longer contains duplicates.
    """
    keep_all = list(range(n_replay))
    if not leaks:
        return LeakageResolution(train_idx, val_idx, test_idx, keep_all)

    if policy == "error":
        raise LeakageError(format_leak_report(leaks))

    if policy == "drop-from-replay":
        bad = {lk.replay_index for lk in leaks}
        kept = [i for i in keep_all if i not in bad]
        return LeakageResolution(train_idx, val_idx, test_idx, kept,
                                 dropped_replay=len(bad))

    if policy == "move-related-episode-to-train":
        move_eps = set()
        for lk in leaks:
            ep = fresh_index_to_episode.get(lk.eval_crop_index)
            if ep is not None:
                move_eps.add(ep)
        move_crops: set[int] = set()
        for ep in move_eps:
            move_crops.update(episodes[ep])
        new_val = [i for i in val_idx if i not in move_crops]
        new_test = [i for i in test_idx if i not in move_crops]
        moved = [i for i in (val_idx + test_idx) if i in move_crops]
        new_train = list(train_idx) + sorted(move_crops)
        return LeakageResolution(new_train, new_val, new_test, keep_all,
                                 moved_eval_crops=len(moved))

    raise ValueError(f"unknown leakage policy: {policy!r}")
