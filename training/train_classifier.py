"""Train a cat-identity classifier from human-reviewed labels.

CPU-friendly. Reads crops straight from the recordings in memory (no JPEGs on
disk), applies the human label corrections, trains an EfficientNet-B0, and writes
the BEST-by-val model to a NEW path. The runtime is NOT touched — swapping the
model into production is a separate, later step.

Pipeline (all reused from this package):
  - training.db.iter_frames        — collect reviewed crop refs/metadata
  - training.reviews.load_reviews  — human corrections {rowid: label}
  - training.sources.decode_crop_batch — decode only the current batch in RAM
  - detector/classifier.py::_preprocess — the EXACT runtime eval transform

Label policy (flags; safe defaults):
  - cold start: ONLY human labels are trusted.
  - later active-learning passes may opt into classifier labels with
    --trust-classifier when events.cat_score >= --trust-conf (default 0.9).
    The detector gate is separate: --min-score filters events.score.
  - discard/unknown are dropped. Class names come from the surviving labels
    (sorted, unique) — never hardcoded — and are saved with the model.

Honest split: crops are grouped into episodes (same camera, wall_ms gaps >
--episode-gap-sec start a new one); a whole episode goes entirely to train, val,
or test, so near-duplicate neighbours never straddle the split. Val chooses the
best epoch; test is held out for the final honest report.

Run:  python -m training.train_classifier --db data/events/events.db \
          --recordings data/recordings --reviews-db data/review/reviews.db \
          --confuse alisa,felisis
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger("training.train_classifier")

ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DROP_LABELS = {"discard", "unknown"}

@dataclass(frozen=True)
class Meta:
    label: str
    camera: str
    wall_ms: int
    rowid: int | None = None
    duplicate_group_id: str | None = None
    suspicious_score: float = 0.0
    sampling_reason: str | None = None


CropRefLite = namedtuple("CropRefLite", ["camera_id", "wall_ms", "box", "rotate_deg"])
# A training item is decoded from exactly ONE of:
#   ref     — a fresh CropRefLite, decoded from recordings per batch, OR
#   replay  — a replay.ReplayItem, decoded from its .npz per batch, OR
#   image   — an already-decoded crop (legacy / pre-shrunk path).
TrainItem = namedtuple("TrainItem", ["meta", "ref", "image", "replay"])


# --------------------------------------------------------------- label policy --

def decide_label(det_label, det_conf, human, trust_classifier, trust_conf):
    """Final training label for one crop, or None to drop it.

    Default (trust_classifier=False): ONLY human labels are used. With
    --trust-classifier, an unreviewed crop may also use the existing classifier's
    label when its cat_score >= trust_conf. discard/unknown are always dropped.
    """
    if human is not None:
        return None if human in DROP_LABELS else human
    if not trust_classifier:
        return None                      # human-only by default
    if not det_label or det_label in DROP_LABELS:
        return None
    if det_conf is not None and det_conf >= trust_conf:
        return det_label
    return None


# ------------------------------------------------------------- episode split --

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


# ----------------------------------------------------------------- metrics ----

def confusion(y_true: list[int], y_pred: list[int], n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def per_class_pr(cm: np.ndarray):
    """Return (precision[], recall[]) per class from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    return precision, recall


def present_class_mask(cm: np.ndarray) -> np.ndarray:
    """Classes with at least one true sample in this eval split."""
    return cm.sum(axis=1) > 0


def supported_macro_recall(cm: np.ndarray) -> float:
    """Macro recall over classes that are actually present in this eval split."""
    _prec, rec = per_class_pr(cm)
    present = present_class_mask(cm)
    if not bool(present.any()):
        return 0.0
    return float(rec[present].mean())


def print_report(cm: np.ndarray, classes: list[str], confuse: set[str]) -> dict:
    prec, rec = per_class_pr(cm)
    support = cm.sum(axis=1)
    present = present_class_mask(cm)
    macro_recall = supported_macro_recall(cm)
    overall = float(np.diag(cm).sum() / max(1, cm.sum()))

    width = max(len(c) for c in classes) + 1
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(" " * width + "".join(f"{c[:7]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{width}}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(classes))))

    print("\nper-class precision / recall:")
    for i, c in enumerate(classes):
        flag = "  <-- confuse" if c in confuse else ""
        if support[i] == 0:
            print(f"  {c:<{width}} precision=NA     recall=NA   support=0{flag}")
        else:
            print(
                f"  {c:<{width}} precision={prec[i]:.3f}  "
                f"recall={rec[i]:.3f}  support={int(support[i])}{flag}"
            )
    print(
        f"\noverall accuracy = {overall:.3f}   "
        f"macro recall = {macro_recall:.3f} (present classes only)"
    )

    # The confusion cells we care about most.
    conf_list = sorted(confuse)
    if len(conf_list) == 2 and all(c in classes for c in conf_list):
        a, b = (classes.index(conf_list[0]), classes.index(conf_list[1]))
        print(f"alisa↔felisis cell: {classes[a]}→{classes[b]}={cm[a, b]}  "
              f"{classes[b]}→{classes[a]}={cm[b, a]}  "
              f"(cross-errors={int(cm[a, b] + cm[b, a])})")

    return {
        "classes": classes,
        "present_classes": [classes[i] for i, ok in enumerate(present) if ok],
        "missing_eval_classes": [classes[i] for i, ok in enumerate(present) if not ok],
        "precision": {c: float(prec[i]) for i, c in enumerate(classes)},
        "recall": {c: float(rec[i]) for i, c in enumerate(classes)},
        "support": {c: int(support[i]) for i, c in enumerate(classes)},
        "macro_recall": macro_recall,
        "overall_accuracy": overall,
        "confusion_matrix": cm.tolist(),
    }


def confuse_cross_errors(cm: np.ndarray, classes: list[str], confuse: set[str]) -> int:
    cs = [c for c in sorted(confuse) if c in classes]
    if len(cs) != 2:
        return 0
    a, b = classes.index(cs[0]), classes.index(cs[1])
    return int(cm[a, b] + cm[b, a])


def shrink_bgr_for_batch(img: np.ndarray, max_side: int) -> np.ndarray:
    """Cap a decoded crop before turning the current batch into tensors.

    Runtime preprocessing ultimately resizes classifier crops to 224px input.
    Keeping very large raw crops inside a batch only burns memory; a 384-512px
    cap preserves useful detail while making CPU training much harder to OOM.
    """
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    import cv2

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def index_batches(indices: list[int], batch_size: int, *,
                  rng: random.Random | None = None) -> Iterator[list[int]]:
    order = list(indices)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), max(1, batch_size)):
        yield order[start:start + max(1, batch_size)]


def load_checkpoint_remapped(model, checkpoint: dict, classes: list[str]) -> dict:
    """Load a previous classifier, remapping the head by class name.

    Weekly fine-tunes can add or temporarily miss classes. The backbone is still
    valuable, and classifier rows for overlapping class names should be reused.
    New class rows keep the freshly initialised weights.
    """
    state = checkpoint["state_dict"]
    checkpoint_classes = list(checkpoint.get("class_names", []))
    current = model.state_dict()

    loaded_body = 0
    skipped = []
    head_keys = {"classifier.1.weight", "classifier.1.bias"}
    for key, value in state.items():
        if key in head_keys:
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            loaded_body += 1
        else:
            skipped.append(key)

    overlap: list[str] = []
    if checkpoint_classes:
        old_index = {c: i for i, c in enumerate(checkpoint_classes)}
        weight_key = "classifier.1.weight"
        bias_key = "classifier.1.bias"
        if weight_key in state and bias_key in state:
            for new_i, cat in enumerate(classes):
                old_i = old_index.get(cat)
                if old_i is None or old_i >= state[weight_key].shape[0]:
                    continue
                current[weight_key][new_i].copy_(state[weight_key][old_i])
                current[bias_key][new_i].copy_(state[bias_key][old_i])
                overlap.append(cat)

    model.load_state_dict(current)
    return {
        "checkpoint_classes": checkpoint_classes,
        "overlap_classes": overlap,
        "new_classes": [c for c in classes if c not in overlap],
        "dropped_checkpoint_classes": [c for c in checkpoint_classes if c not in classes],
        "loaded_body_tensors": loaded_body,
        "skipped_tensors": skipped,
    }


def configure_finetune(model, *, head_only: bool, full_finetune: bool):
    """Set ``requires_grad`` for the chosen fine-tune mode and return
    ``(trainable_params, mode, n_trainable, n_frozen)``.

    Modes (mutually exclusive):
      - ``head``     — ONLY the classifier head trains (CPU-friendly; the backbone
                       is a frozen feature extractor). ``--head-only``.
      - ``partial``  — head + the last two feature blocks (the default).
      - ``full``     — the whole backbone (low LR). ``--full-finetune``.

    The optimizer must be built from the returned ``trainable_params`` so frozen
    tensors never get gradients/updates.
    """
    if head_only and full_finetune:
        raise ValueError("--head-only and --full-finetune are mutually exclusive")

    if full_finetune:
        for p in model.parameters():
            p.requires_grad = True
        mode = "full"
    else:
        # Both head-only and partial start from a fully frozen backbone.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        if head_only:
            mode = "head"
        else:
            for blk in (model.features[-1], model.features[-2]):
                for p in blk.parameters():
                    p.requires_grad = True
            mode = "partial"

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable))
    n_frozen = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))
    return trainable, mode, n_trainable, n_frozen


# ------------------------------------------------------------------- main -----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--reviews-db", type=Path, default=None,
                    help="reviews.db with human corrections (strongly recommended)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None, help="detector model filter (default: all)")
    ap.add_argument("--confuse", default="",
                    help="OPTIONAL pair to highlight in the confusion matrix "
                         "(e.g. alisa,felisis); does NOT affect trust or split")
    ap.add_argument("--trust-classifier", action="store_true",
                    help="also use classifier labels for unreviewed crops (OFF by "
                        "default — human labels only for cold start)")
    ap.add_argument("--trust-detector", dest="trust_classifier", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--trust-conf", type=float, default=0.9,
                    help="min identity cat_score required before --trust-classifier may "
                         "reuse an existing classifier label")
    ap.add_argument("--replay-set", type=Path, action="append", default=[],
                    help="compact replay set directory/manifest; added to train only")
    ap.add_argument("--replay-max-items", type=int, default=None,
                    help="cap total replay crops kept (balanced across classes); "
                         "deterministic with --seed. Default: keep all")
    ap.add_argument("--replay-leakage-policy",
                    choices=("error", "drop-from-replay", "move-related-episode-to-train"),
                    default="error",
                    help="what to do when a replay crop duplicates a val/test "
                         "sample: error (default, fail loudly), drop-from-replay, "
                         "or move-related-episode-to-train")
    ap.add_argument("--replay-leak-window-sec", type=float, default=2.0,
                    help="same-camera timestamp window (s) for near-duplicate "
                         "replay-vs-eval leakage detection")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="crop context padding — MUST match the detector "
                         "CLASSIFIER_PAD_FRAC and build_cluster_manifest --pad-frac")
    ap.add_argument("--default-rotate-deg", type=int, default=0,
                    help="rotation to assume for events recorded BEFORE rotate_deg "
                         "was persisted (set to the camera's rotate_deg then)")
    ap.add_argument("--min-score", type=float, default=0.7, help="drop low YOLO-score boxes")
    ap.add_argument("--episode-gap-sec", type=float, default=60.0)
    ap.add_argument("--max-crops-per-episode", type=int, default=24,
                    help="sample at most this many fresh reviewed crops per "
                         "episode/visit per split; 0 disables")
    ap.add_argument("--max-crops-per-duplicate-group", type=int, default=3,
                    help="sample at most this many crops from the same review "
                         "duplicate_group_id; 0 disables")
    ap.add_argument("--keep-suspicious-per-episode", type=int, default=4,
                    help="always try to retain this many suspicious/hard examples "
                         "per episode before thinning")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6, help="early stop on val macro-recall")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="conservative CPU default; raise if you have RAM headroom")
    ap.add_argument("--batch-max-side", type=int, default=384,
                    help="resize decoded crops to at most this many pixels on the "
                         "long side while building an in-RAM batch; 0 disables")
    ap.add_argument("--cache-max-side", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--torch-threads", type=int, default=min(4, os.cpu_count() or 1),
                    help="limit PyTorch CPU worker threads; set 0 to leave default")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 head / 1e-4 full")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="optional previous cat_classifier.pt checkpoint to fine-tune from")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train the whole backbone (low LR) instead of head + last block")
    ap.add_argument("--head-only", action="store_true",
                    help="CPU-friendly: train ONLY the classifier head; the backbone "
                         "is a frozen feature extractor. Recommended CPU combo: "
                         "--head-only --batch-size 4 --batch-max-side 320 --num-workers 0. "
                         "Mutually exclusive with --full-finetune")
    ap.add_argument("--num-workers", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--min-recall", type=float, default=0.9,
                    help="PASS/FAIL gate: every class (incl. confuse) needs val recall >= this")
    ap.add_argument("--out-root", type=Path, default=ROOT / "models/trained")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.cache_max_side is not None:
        args.batch_max_side = args.cache_max_side

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    try:
        from classifier import _preprocess  # exact runtime eval preprocessing
    except ImportError:
        sys.path.insert(0, str(ROOT / "detector"))
        from classifier import _preprocess
    from training import load_reviews
    from training.reviews import load_review_rows
    from training.ram import log_rss
    from training.db import iter_frames, open_db_ro
    from training.replay import decode_replay_image, load_replay_items
    from training.leakage import (
        apply_leakage_policy,
        build_eval_identities,
        find_replay_leaks,
        format_leak_report,
        replay_identities,
    )
    from training.segments import SegmentIndex
    from training.sources import decode_crop_batch

    # Determinism.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    confuse = {c.strip() for c in args.confuse.split(",") if c.strip()}

    # --- collect crop refs/metadata only; pixels are decoded per batch later ---
    review_rows = load_review_rows(args.reviews_db) if args.reviews_db else {}
    reviews = {key: row.label for key, row in review_rows.items()}
    if not review_rows and args.reviews_db:
        reviews = load_reviews(args.reviews_db)
    log.info("loaded %d human corrections", len(reviews))

    segment_indices: dict[str, SegmentIndex] = {}

    def segment_index(camera_id: str) -> SegmentIndex:
        index = segment_indices.get(camera_id)
        if index is None:
            index = SegmentIndex.from_dir(args.recordings / camera_id)
            segment_indices[camera_id] = index
        return index

    log_rss(log, "before-dataset")
    fresh_items: list[TrainItem] = []
    scanned = 0
    unavailable = 0
    warned_missing_rotate = False
    conn = open_db_ro(args.db)
    try:
        frames = iter_frames(
            conn,
            camera_id=args.camera,
            model=args.model,
            min_score=args.min_score,
        )
        for frame in frames:
            rot = frame.rotate_deg
            if rot is None:
                rot = args.default_rotate_deg
                if not warned_missing_rotate:
                    log.warning(
                        "event(s) without recorded rotate_deg — assuming %d°. "
                        "Set --default-rotate-deg to the camera's rotation at "
                        "capture time for pre-migration data.", rot,
                    )
                    warned_missing_rotate = True
            hit = segment_index(frame.camera_id).locate(frame.wall_ms)
            for box in frame.boxes:
                scanned += 1
                review_row = review_rows.get(box.rowid) if box.rowid is not None else None
                human = (
                    review_row.label if review_row is not None
                    else reviews.get(box.rowid) if box.rowid is not None
                    else None
                )
                label = decide_label(
                    box.cat, box.cat_score, human,
                    args.trust_classifier, args.trust_conf,
                )
                if label is None:
                    continue
                if hit is None:
                    unavailable += 1
                    continue
                meta = Meta(
                    label=label,
                    camera=frame.camera_id,
                    wall_ms=frame.wall_ms,
                    rowid=box.rowid,
                    duplicate_group_id=(
                        review_row.duplicate_group_id if review_row is not None else None
                    ),
                    suspicious_score=(
                        review_row.suspicious_score if review_row is not None else 0.0
                    ),
                    sampling_reason=(
                        review_row.sampling_reason if review_row is not None else None
                    ),
                )
                ref = CropRefLite(frame.camera_id, frame.wall_ms, box, int(rot or 0))
                fresh_items.append(TrainItem(meta=meta, ref=ref, image=None, replay=None))
    finally:
        conn.close()

    if not fresh_items:
        raise SystemExit("no usable crops after the label policy — loosen "
                         "--trust-classifier or add human reviews")
    if unavailable:
        log.warning("skipped %d reviewed crops whose recordings are unavailable", unavailable)
    log.info(
        "kept %d / %d event boxes after label policy; pixels will decode per batch",
        len(fresh_items), scanned,
    )

    fresh_metas = [item.meta for item in fresh_items]

    # Replay: load METADATA only (paths, not pixels). Crops decode per batch from
    # their .npz, so RAM stays flat no matter how big the replay set is.
    replay_loaded = []
    for replay_path in args.replay_set:
        replay_loaded.extend(
            load_replay_items(replay_path,
                              max_items=args.replay_max_items, seed=args.seed)
        )
    if replay_loaded:
        # ~ per-crop bytes are unknown until decode; report the cap and a rough
        # per-crop estimate at the batch-shrink cap so the config is auditable.
        side = args.batch_max_side if args.batch_max_side > 0 else 384
        approx_mb_per_crop = side * side * 3 / (1024 * 1024)
        log.info(
            "replay: %d crops (metadata only) from %d set(s); pixels decode per "
            "batch (≈%.2f MB/crop at <=%dpx, peak ≈ batch-size × that, not the set)",
            len(replay_loaded), len(args.replay_set), approx_mb_per_crop, side,
        )
    replay_items = [
        TrainItem(meta=it.meta, ref=None, image=None, replay=it)
        for it in replay_loaded
    ]
    replay_metas = [item.meta for item in replay_items]

    classes = sorted({m.label for m in [*fresh_metas, *replay_metas]})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    log.info("classes (%d): %s", len(classes), classes)
    for c in sorted(confuse):
        if c not in cls_to_idx:
            log.warning("confuse cat %r has NO crops — its confusion metric is empty", c)

    # --- honest, episode-grouped, confuse-stratified split ---
    episodes = build_episodes(fresh_metas, int(args.episode_gap_sec * 1000))
    train_idx, val_idx, test_idx = split_episodes(
        episodes, fresh_metas,
        val_frac=args.val_frac, test_frac=args.test_frac,
        required={m.label for m in fresh_metas}, seed=args.seed,
    )
    check_split_leakage(episodes, train_idx, val_idx, test_idx)
    raw_split_sizes = (len(train_idx), len(val_idx), len(test_idx))
    train_idx = sample_indices_for_training(
        train_idx,
        episodes,
        fresh_metas,
        max_per_episode=args.max_crops_per_episode,
        max_per_duplicate_group=args.max_crops_per_duplicate_group,
        keep_suspicious_per_episode=args.keep_suspicious_per_episode,
    )
    val_idx = sample_indices_for_training(
        val_idx,
        episodes,
        fresh_metas,
        max_per_episode=args.max_crops_per_episode,
        max_per_duplicate_group=args.max_crops_per_duplicate_group,
        keep_suspicious_per_episode=args.keep_suspicious_per_episode,
    )
    test_idx = sample_indices_for_training(
        test_idx,
        episodes,
        fresh_metas,
        max_per_episode=args.max_crops_per_episode,
        max_per_duplicate_group=args.max_crops_per_duplicate_group,
        keep_suspicious_per_episode=args.keep_suspicious_per_episode,
    )
    check_split_leakage(episodes, train_idx, val_idx, test_idx)
    if raw_split_sizes != (len(train_idx), len(val_idx), len(test_idx)):
        log.info(
            "sampled fresh crops after episode split: train %d/%d, val %d/%d, test %d/%d",
            len(train_idx), raw_split_sizes[0],
            len(val_idx), raw_split_sizes[1],
            len(test_idx), raw_split_sizes[2],
        )

    # --- cross-source leakage guard: replay crops must not duplicate val/test ---
    # Replay memory is train-only, but a replay crop is an old fresh crop; if the
    # same event (or a near-duplicate) is in val/test, training on it leaks. We
    # check BEFORE appending replay to train, fail closed by default.
    kept_replay = list(range(len(replay_items)))
    if replay_items:
        from training.leakage import Identity, LeakageError
        fresh_index_to_episode = {
            ci: ep for ep, crops in enumerate(episodes) for ci in crops
        }
        eval_entries = []
        for split_name, idxs in (("val", val_idx), ("test", test_idx)):
            for ci in idxs:
                box = fresh_items[ci].ref.box
                eval_entries.append((split_name, ci, Identity(
                    rowid=box.rowid,
                    camera=fresh_items[ci].meta.camera,
                    wall_ms=fresh_items[ci].meta.wall_ms,
                )))
        eval_index = build_eval_identities(eval_entries)
        replay_ids = replay_identities([it.replay for it in replay_items])
        leaks = find_replay_leaks(
            eval_index, replay_ids,
            window_ms=int(args.replay_leak_window_sec * 1000),
        )
        if leaks:
            print("\n" + format_leak_report(leaks))
        try:
            res = apply_leakage_policy(
                args.replay_leakage_policy, leaks,
                episodes=episodes, fresh_index_to_episode=fresh_index_to_episode,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                n_replay=len(replay_items),
            )
        except LeakageError as e:
            raise SystemExit(
                f"{e}\n\nReplay leakage is fatal by default. Rebuild the replay "
                "set to exclude these events, or pass --replay-leakage-policy "
                "drop-from-replay / move-related-episode-to-train."
            )
        train_idx, val_idx, test_idx = res.train_idx, res.val_idx, res.test_idx
        kept_replay = res.kept_replay
        if res.dropped_replay:
            log.warning("dropped %d leaking replay crop(s) from train", res.dropped_replay)
        if res.moved_eval_crops:
            log.warning("moved %d eval crop(s) to train to resolve replay leakage",
                        res.moved_eval_crops)
            train_idx = sample_indices_for_training(
                train_idx,
                episodes,
                fresh_metas,
                max_per_episode=args.max_crops_per_episode,
                max_per_duplicate_group=args.max_crops_per_duplicate_group,
                keep_suspicious_per_episode=args.keep_suspicious_per_episode,
            )
        # Re-verify the fresh split is still internally consistent after any move.
        check_split_leakage(episodes, train_idx, val_idx, test_idx)

    items = fresh_items + replay_items
    metas = fresh_metas + replay_metas
    if replay_metas:
        train_idx.extend(len(fresh_metas) + ri for ri in kept_replay)
    log.info("%d episodes -> train %d crops / val %d crops / test %d crops",
             len(episodes), len(train_idx), len(val_idx), len(test_idx))
    train_classes = {metas[j].label for j in train_idx}
    missing_train = sorted(set(classes) - train_classes)
    if missing_train:
        raise SystemExit(
            "train split has no crops for class(es): "
            + ", ".join(missing_train)
            + ". Add more reviewed episodes or reduce --val-frac/--test-frac."
        )
    if not val_idx:
        raise SystemExit("validation split is empty; reduce --test-frac or add more episodes")

    # --- transforms: train augments; val is byte-identical to runtime ---
    train_pipe = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),   # no hue/sat: IR is ~grayscale
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def train_tf(img_bgr):
        from PIL import Image
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        return train_pipe(Image.fromarray(rgb))

    def eval_tf(img_bgr):
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        return torch.from_numpy(_preprocess(rgb)[0]).float()  # == runtime preprocessing

    def make_batch(batch_indices: list[int], tf, *, max_side: int = 0):
        refs = []
        ref_slots = []
        batch_images: list[np.ndarray | None] = [None] * len(batch_indices)
        for slot, item_index in enumerate(batch_indices):
            item = items[item_index]
            if item.image is not None:
                batch_images[slot] = item.image
            elif item.replay is not None:
                img = decode_replay_image(item.replay, missing_ok=True)
                if img is not None:
                    batch_images[slot] = shrink_bgr_for_batch(img, max_side)
            else:
                refs.append(item.ref)
                ref_slots.append(slot)

        if refs:
            decoded = decode_crop_batch(
                refs,
                args.recordings,
                pad_frac=args.pad_frac,
                indices=segment_indices,
            )
            for slot, img in zip(ref_slots, decoded):
                if img is not None:
                    batch_images[slot] = shrink_bgr_for_batch(img, max_side)

        xs = []
        ys = []
        for item_index, img in zip(batch_indices, batch_images):
            if img is None:
                continue
            xs.append(tf(img))
            ys.append(cls_to_idx[metas[item_index].label])
        if not xs:
            return None
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def predict_indices(indices: list[int]):
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for batch_indices in index_batches(indices, args.batch_size):
                batch = make_batch(batch_indices, eval_tf)
                if batch is None:
                    continue
                x, y = batch
                pred = model(x).argmax(1)
                y_true.extend(y.tolist())
                y_pred.extend(pred.tolist())
        return y_true, y_pred

    # --- class-imbalance weighting (inverse frequency on TRAIN) ---
    counts = np.bincount([cls_to_idx[metas[j].label] for j in train_idx],
                         minlength=len(classes)).astype(np.float64)
    weights = len(train_idx) / (len(classes) * np.maximum(counts, 1))
    ce_weight = torch.tensor(weights, dtype=torch.float32)
    log.info("train per-class counts: %s", dict(zip(classes, counts.astype(int))))

    # --- model: EfficientNet-B0 (ImageNet), new head ---
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    if args.init_from:
        checkpoint = torch.load(str(args.init_from), map_location="cpu", weights_only=False)
        init_report = load_checkpoint_remapped(model, checkpoint, classes)
        if init_report["checkpoint_classes"] and not init_report["overlap_classes"]:
            raise SystemExit(
                "--init-from has no overlapping class_names with current labels: "
                f"{init_report['checkpoint_classes']} vs {classes}"
            )
        log.info(
            "initialized model from %s (reused head rows=%s, new=%s, dropped_old=%s)",
            args.init_from,
            init_report["overlap_classes"],
            init_report["new_classes"],
            init_report["dropped_checkpoint_classes"],
        )

    if args.head_only and args.full_finetune:
        raise SystemExit("--head-only and --full-finetune are mutually exclusive")
    params, finetune_mode, n_trainable, n_frozen = configure_finetune(
        model, head_only=args.head_only, full_finetune=args.full_finetune,
    )
    # Full backbone wants a low LR; head/partial can use the higher default.
    lr = args.lr if args.lr is not None else (1e-4 if finetune_mode == "full" else 1e-3)
    log.info(
        "finetune mode=%s  trainable params=%d (%.1f%%)  frozen=%d  lr=%g",
        finetune_mode, n_trainable,
        100.0 * n_trainable / max(1, n_trainable + n_frozen), n_frozen, lr,
    )

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    # --- train loop with best-by-(macro-recall, fewest confuse cross-errors) ---
    best = {"macro_recall": -1.0, "cross": 10**9, "state": None, "metrics": None, "epoch": -1}
    since_improved = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        batch_rng = random.Random(args.seed + epoch)
        first_batch = epoch == 1
        for batch_indices in index_batches(train_idx, args.batch_size, rng=batch_rng):
            batch = make_batch(batch_indices, train_tf, max_side=args.batch_max_side)
            if batch is None:
                continue
            x, y = batch
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
            if first_batch:
                log_rss(log, "after-first-batch")
                first_batch = False
        sched.step()
        log_rss(log, "after-epoch", epoch=epoch)

        model.eval()
        y_true, y_pred = predict_indices(val_idx)
        cm = confusion(y_true, y_pred, len(classes))
        macro = supported_macro_recall(cm)
        cross = confuse_cross_errors(cm, classes, confuse)
        log.info("epoch %02d  train_loss=%.4f  val_macro_recall=%.3f  confuse_cross=%d",
                 epoch, running / max(1, seen), macro, cross)

        improved = (macro > best["macro_recall"] + 1e-6 or
                    (abs(macro - best["macro_recall"]) <= 1e-6 and cross < best["cross"]))
        if improved:
            best.update(macro_recall=macro, cross=cross, epoch=epoch,
                        state={k: v.clone() for k, v in model.state_dict().items()})
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= args.patience:
                log.info("early stop at epoch %d (no val improvement for %d)",
                         epoch, args.patience)
                break

    # --- restore + report the BEST model (not the last epoch) ---
    assert best["state"] is not None
    model.load_state_dict(best["state"])
    model.eval()
    y_true, y_pred = predict_indices(val_idx)
    cm = confusion(y_true, y_pred, len(classes))
    print(f"\n===== BEST model (epoch {best['epoch']}) on val =====")
    metrics = print_report(cm, classes, confuse)

    if test_idx:
        y_true, y_pred = predict_indices(test_idx)
        test_cm = confusion(y_true, y_pred, len(classes))
        print("\n===== Held-out TEST set =====")
        test_metrics = print_report(test_cm, classes, confuse)
    else:
        test_metrics = None

    # --- save to a NEW path; export_classifier-compatible .pt ---
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "cat_classifier.pt"
    torch.save({"state_dict": model.state_dict(),
                "class_names": classes,
                "num_classes": len(classes)}, pt_path)
    meta = {
        "created": stamp,
        "class_names": classes,
        "pad_frac": args.pad_frac,
        "preprocessing": {
            "resize_short_side": 256, "center_crop": 224,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "interpolation": "bilinear",
            "note": "byte-identical to detector/classifier.py::_preprocess",
        },
        "trust_classifier": args.trust_classifier,
        "trust_conf": args.trust_conf, "confuse": sorted(confuse),
        "replay_sets": [str(p) for p in args.replay_set],
        "replay_count": len(replay_metas),
        "batch_max_side": args.batch_max_side,
        "max_crops_per_episode": args.max_crops_per_episode,
        "max_crops_per_duplicate_group": args.max_crops_per_duplicate_group,
        "keep_suspicious_per_episode": args.keep_suspicious_per_episode,
        "init_from": str(args.init_from) if args.init_from else None,
        "full_finetune": args.full_finetune, "head_only": args.head_only,
        "finetune_mode": finetune_mode, "best_epoch": best["epoch"],
        "val_metrics": metrics, "test_metrics": test_metrics,
        "counts_train": dict(zip(classes, counts.astype(int).tolist())),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved best model -> {pt_path}")
    print(f"metadata         -> {out_dir / 'metadata.json'}")

    # --- PASS/FAIL guard (loud warning on FAIL; do NOT crash) ---
    _, rec = per_class_pr(cm)
    support = cm.sum(axis=1)
    failing = {classes[i]: float(rec[i]) for i in range(len(classes))
               if support[i] > 0 and rec[i] < args.min_recall}
    missing_eval = [classes[i] for i in range(len(classes)) if support[i] == 0]
    # confuse cats must explicitly clear the bar too (already covered by per-class,
    # but call them out so an empty confuse class can't slip through silently).
    for c in sorted(confuse):
        if c not in classes:
            failing[c] = 0.0
        elif support[classes.index(c)] == 0:
            log.warning("confuse cat %r has no validation samples; not included in PASS/FAIL", c)
    if missing_eval:
        print(
            "\nWARN — validation has no samples for: "
            + ", ".join(missing_eval)
            + ". They are excluded from val macro recall/PASS gate."
        )
    if failing:
        print("\n" + "!" * 64)
        print(f"FAIL — NOT ready for production (need recall >= {args.min_recall} "
              "for every class incl. the confuse pair).")
        for c, r in sorted(failing.items()):
            print(f"   {c}: recall={r:.3f}")
        print("Collect/relabel more crops (especially the confuse pair) and re-train.")
        print("!" * 64)
    else:
        print(f"\nPASS — every class recall >= {args.min_recall} (incl. the confuse pair). "
              "Model is a production candidate (export + swap is a separate step).")


if __name__ == "__main__":
    main()
