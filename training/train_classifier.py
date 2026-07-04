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
          --confuse NAME_A,NAME_B      # optional: your look-alike pair
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from training.augment import (
    INPUT_SIZE,
    LEVELS as AUG_LEVELS,
    RESIZE_SHORT,
    augment_spec,
    build_eval_transform,
    build_train_transform,
    resize_short_for,
    save_augmentation_preview,
)
from training.mltracking import start_run

log = logging.getLogger("training.train_classifier")

ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# The training pipeline is split across focused modules (types, labels, split,
# metrics, artifacts, model, batching). These re-exports keep the historical
# `from training.train_classifier import X` surface working for callers/tests,
# and let main() below reference the helpers by their bare names.
from training.classifier_types import CropRefLite, Meta, TrainItem  # noqa: E402,F401
from training.classifier_labels import (  # noqa: E402,F401
    DROP_LABELS,
    decide_label,
    require_min_classes,
)
from training.classifier_split import (  # noqa: E402,F401
    build_episodes,
    check_split_leakage,
    print_split_report,
    sample_indices_for_training,
    split_episodes,
    summarize_split,
)
from training.classifier_metrics import (  # noqa: E402,F401
    DangerousConfusion,
    confusion,
    dangerous_confusion_report,
    dangerous_from_confuse,
    format_epoch_line,
    load_dangerous_confusions,
    per_camera_confusion,
    per_class_f1,
    per_class_pr,
    per_group_accuracy,
    present_class_mask,
    print_report,
    supported_macro_recall,
)
from training.classifier_artifacts import (  # noqa: E402,F401
    labels_payload,
    write_confusion_csv,
    write_confusion_png,
    write_labels_json,
)
from training.classifier_model import (  # noqa: E402,F401
    configure_finetune,
    load_checkpoint_remapped,
    prototype_distance_stats,
    prototypes_from_embeddings,
)
from training.classifier_batching import (  # noqa: E402,F401
    index_batches,
    mean_from_batch_means,
    shrink_bgr_for_batch,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--reviews-db", type=Path, default=None,
                    help="reviews.db with human corrections (strongly recommended)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None, help="detector model filter (default: all)")
    ap.add_argument("--confuse", default="",
                    help="OPTIONAL pair of labels (e.g. NAME_A,NAME_B) whose directional "
                         "confusion is reported and used to break ties in model selection "
                         "— set this to your safety-critical look-alike pair. Empty by "
                         "default; ignored if absent from labels. Does NOT affect trust or split")
    ap.add_argument("--dangerous-confusions", type=Path, default=None,
                    help="OPTIONAL YAML/JSON file of dangerous confusion pairs "
                         "(list of {predicted, actual, reason}). Their counts are "
                         "reported and summed into the model-selection tiebreak.")
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
    ap.add_argument("--augment", choices=list(AUG_LEVELS), default="light",
                    help="on-the-fly TRAIN augmentation level (val/test stay clean): "
                         "off = deterministic (no augmentation); light (default) / medium "
                         "= mild, identity-preserving jitter (no flips, no 90/180, no hard crop)")
    ap.add_argument("--preview-augmentations", type=Path, default=None,
                    help="debug only: save a grid of original + augmented versions of a few "
                         "train crops to this dir, then continue. Never written into the dataset")
    ap.add_argument("--image-size", type=int, default=INPUT_SIZE,
                    help=f"classifier input (center-crop) size; default {INPUT_SIZE}. "
                         "Short-side resize scales proportionally (256/224 ratio). "
                         "Used consistently across train/val/test transforms. NOTE: "
                         "the runtime detector still preprocesses to 256->224, so a "
                         "non-default size requires aligning serve-time preprocessing "
                         "before promoting the model")
    ap.add_argument("--out-root", type=Path, default=ROOT / "models/trained")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cache_max_side is not None:
        args.batch_max_side = args.cache_max_side

    # Center-crop input size + its proportional short-side resize. Both feed the
    # train/val/test transforms so framing is consistent at any --image-size.
    if args.image_size < 1:
        raise SystemExit(f"--image-size must be >= 1, got {args.image_size}")
    args.resize_short = resize_short_for(args.image_size)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.image_size != INPUT_SIZE:
        log.warning(
            "image_size=%d (resize_short=%d) differs from the runtime default %d: "
            "the detector still preprocesses to %d->%d, so align serve-time "
            "preprocessing before promoting this model.",
            args.image_size, args.resize_short, INPUT_SIZE, RESIZE_SHORT, INPUT_SIZE,
        )

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
    # Dangerous confusions: the --confuse pair (both directions) plus any from a
    # YAML/JSON config. No hardcoded label pairs.
    dangerous = dangerous_from_confuse(confuse)
    if args.dangerous_confusions:
        dangerous += load_dangerous_confusions(args.dangerous_confusions)
    if dangerous:
        log.info("dangerous confusions configured: %s",
                 ", ".join(f"{d.actual}->{d.predicted}" for d in dangerous))

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
    require_min_classes(classes)
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

    # Loud, explicit leakage/distribution report; raises if any overlap is found.
    def _crop_identity(item, i):
        box = getattr(getattr(item, "ref", None), "box", None)
        return getattr(box, "rowid", None) if box is not None else f"idx:{i}"
    split_summary = summarize_split(
        episodes, train_idx, val_idx, test_idx, fresh_metas,
        identities=[_crop_identity(it, i) for i, it in enumerate(fresh_items)],
    )
    print_split_report(split_summary, episode_gap_sec=args.episode_gap_sec)

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

    # --- transforms: TRAIN augments on-the-fly; VAL/TEST stay clean & deterministic ---
    # Augmentation is identity-preserving (no flips, no 90/180, no hard crop) so it
    # can't erase the left/right markings that separate look-alike cats. See
    # training/augment.py. Eval is byte-identical to the detector runtime.
    spec = augment_spec(args.augment)
    log.info(
        "augmentation: level=%s random=%s (rotation=±%.0f° translate=%.0f%% scale=%.2f-%.2f "
        "brightness=%.2f contrast=%.2f gamma=%.2f noise=%.3f blur_p=%.2f pad=%.0f%%; "
        "flips=OFF quarter_turns=OFF hard_crop=OFF)",
        spec.level, spec.is_random, spec.rotation_deg, spec.translate_frac * 100,
        spec.scale_min, spec.scale_max, spec.brightness, spec.contrast,
        spec.gamma, spec.noise_std, spec.blur_prob, spec.pad_frac * 100,
    )
    train_pipe = build_train_transform(
        args.augment, IMAGENET_MEAN, IMAGENET_STD,
        size=args.image_size, resize=args.resize_short)

    # At the default size, val/test use the EXACT runtime preprocessing (_preprocess)
    # so the reported metrics match serve time. At a non-default --image-size the
    # runtime path doesn't apply, so val/test use the same deterministic eval
    # transform at the chosen size — keeping train/val/test framing consistent.
    eval_pipe = (
        build_eval_transform(IMAGENET_MEAN, IMAGENET_STD,
                             size=args.image_size, resize=args.resize_short)
        if args.image_size != INPUT_SIZE else None
    )

    def train_tf(img_bgr):
        from PIL import Image
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        return train_pipe(Image.fromarray(rgb))

    def eval_tf(img_bgr):
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        if eval_pipe is not None:
            from PIL import Image
            return eval_pipe(Image.fromarray(rgb))
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
        kept: list[int] = []   # source item indices, aligned with xs/ys
        for item_index, img in zip(batch_indices, batch_images):
            if img is None:
                continue
            xs.append(tf(img))
            ys.append(cls_to_idx[metas[item_index].label])
            kept.append(item_index)
        if not xs:
            return None
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long), kept

    # --- optional augmentation preview (debug only; never written to dataset) ---
    if args.preview_augmentations:
        sample = train_idx[:6]
        bgr_by_pos: dict[int, np.ndarray] = {}
        pending_refs, pending_pos = [], []
        for pos, j in enumerate(sample):
            it = items[j]
            if it.image is not None:
                bgr_by_pos[pos] = it.image
            elif getattr(it, "replay", None) is not None:
                img = decode_replay_image(it.replay, missing_ok=True)
                if img is not None:
                    bgr_by_pos[pos] = img
            elif getattr(it, "ref", None) is not None:
                pending_refs.append(it.ref)
                pending_pos.append(pos)
        if pending_refs:
            for pos, img in zip(pending_pos, decode_crop_batch(
                    pending_refs, args.recordings, pad_frac=args.pad_frac,
                    indices=segment_indices)):
                if img is not None:
                    bgr_by_pos[pos] = img
        crops_rgb = [np.ascontiguousarray(bgr_by_pos[p][..., ::-1])
                     for p in range(len(sample)) if p in bgr_by_pos]
        if crops_rgb:
            written = save_augmentation_preview(
                args.preview_augmentations, crops_rgb, train_pipe,
                build_eval_transform(IMAGENET_MEAN, IMAGENET_STD,
                                     size=args.image_size, resize=args.resize_short),
            )
            log.info("wrote %d augmentation preview grid(s) to %s",
                     len(written), args.preview_augmentations)
        else:
            log.warning("preview-augmentations: could not decode any sample crops")

    def predict_indices(indices: list[int]):
        """Returns (y_true, y_pred, kept_indices, mean_loss). The reported loss is
        the UNWEIGHTED mean cross-entropy (criterion_report), so val/test loss is
        comparable to train_loss and free of the weighted-mean × batch-ordering
        asymmetry. The weighted criterion is used only for gradients."""
        y_true: list[int] = []
        y_pred: list[int] = []
        kept: list[int] = []   # source item indices aligned with y_true/y_pred
        batch_means: list[tuple[float, int]] = []
        with torch.no_grad():
            for batch_indices in index_batches(indices, args.batch_size):
                batch = make_batch(batch_indices, eval_tf)
                if batch is None:
                    continue
                x, y, batch_kept = batch
                logits = model(x)
                batch_means.append((float(criterion_report(logits, y)), x.size(0)))
                y_true.extend(y.tolist())
                y_pred.extend(logits.argmax(1).tolist())
                kept.extend(batch_kept)
        return y_true, y_pred, kept, mean_from_batch_means(batch_means)

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
    # Unweighted CE used ONLY for the logged loss curve (not gradients): keeps
    # train_loss and val_loss on the same, comparable scale. The weighted
    # criterion above still drives optimisation, so training behaviour is unchanged.
    criterion_report = nn.CrossEntropyLoss()

    # --- experiment tracking (MLflow if installed; transparent no-op otherwise) ---
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = start_run(
        "cat_classifier",
        run_name=run_stamp,
        params={
            # dataset source / reproducibility
            "events_db": str(args.db), "recordings": str(args.recordings),
            "reviews_db": str(args.reviews_db) if args.reviews_db else "",
            # labels/classes (dynamic — never hardcoded)
            "classes": ",".join(classes), "num_classes": len(classes),
            "label_to_index": json.dumps(cls_to_idx),
            # split
            "split_method": "episode-grouped (camera + wall_ms gap)",
            "group_key": f"camera_id + wall_ms gap <= {args.episode_gap_sec:g}s",
            "episode_gap_sec": args.episode_gap_sec,
            "train_crops": len(train_idx), "val_crops": len(val_idx),
            "test_crops": len(test_idx), "n_episodes": len(episodes),
            "train_groups": split_summary["group_counts"]["train"],
            "val_groups": split_summary["group_counts"]["val"],
            "test_groups": split_summary["group_counts"]["test"],
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "replay_count": len(replay_metas),
            # model / optimisation
            "model": "efficientnet_b0", "image_size": args.image_size,
            "resize_short_side": args.resize_short,
            "optimizer": "AdamW", "weight_decay": 1e-4,
            "lr": lr, "epochs": args.epochs, "batch_size": args.batch_size,
            "finetune_mode": finetune_mode, "seed": args.seed,
            # augmentation (mode + full config)
            "augment": args.augment,
            **{f"aug_{k}": v for k, v in vars(spec).items()},
            # confusion config
            "confuse": ",".join(sorted(confuse)),
            "dangerous_confusions": ";".join(f"{d.actual}->{d.predicted}" for d in dangerous),
            "trust_classifier": args.trust_classifier,
            "min_recall_gate": args.min_recall,
        },
        tags={
            "split_method": "episode-grouped (camera + wall_ms gap)",
            "group_key": f"camera_id + wall_ms gap <= {args.episode_gap_sec:g}s",
            "augment_random": spec.is_random,
        },
    )
    if run.active:
        log.info("mlflow ENABLED: uri=%s experiment=cat_classifier run_id=%s run_name=%s",
                 run.tracking_uri, run.run_id, run_stamp)
    else:
        log.info("mlflow DISABLED (package not installed / no tracking) — training continues")
    run.log_metrics({f"train_count_{c}": int(n) for c, n in zip(classes, counts)})

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
            x, y, _ = batch
            optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)          # weighted — drives gradients (unchanged)
            loss.backward()
            optim.step()
            # Logged train_loss uses the UNWEIGHTED mean CE (same as val), so the
            # two curves are comparable; gradients above are unaffected.
            running += float(criterion_report(out.detach(), y)) * x.size(0)
            seen += x.size(0)
            if first_batch:
                log_rss(log, "after-first-batch")
                first_batch = False
        sched.step()
        log_rss(log, "after-epoch", epoch=epoch)

        model.eval()
        y_true, y_pred, _, val_loss = predict_indices(val_idx)
        cm = confusion(y_true, y_pred, len(classes))
        macro = supported_macro_recall(cm)
        cross = dangerous_confusion_report(cm, classes, dangerous)[0]
        train_loss = running / max(1, seen)
        log.info("%s", format_epoch_line(
            epoch, train_loss, val_loss, macro, cross,
            dangerous_configured=bool(dangerous)))
        run.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_recall": macro,
            "val_dangerous_errors": cross,
        }, step=epoch)

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
    idx_to_episode = {i: ep for ep, crops in enumerate(episodes) for i in crops}

    def full_report(indices):
        yt, yp, kept, loss = predict_indices(indices)
        cmx = confusion(yt, yp, len(classes))
        m = print_report(cmx, classes, confuse, dangerous=dangerous)
        m["loss"] = loss
        # Per-camera / per-group breakdowns when that metadata exists.
        cams = [getattr(metas[i], "camera", None) for i in kept]
        grps = [idx_to_episode.get(i) for i in kept]
        m["per_camera"] = per_camera_confusion(yt, yp, classes, cams)
        m["per_group"] = per_group_accuracy(yt, yp, grps)
        return cmx, m

    print(f"\n===== BEST model (epoch {best['epoch']}) on val =====")
    cm, metrics = full_report(val_idx)

    if test_idx:
        print("\n===== Held-out TEST set =====")
        test_cm, test_metrics = full_report(test_idx)
    else:
        test_metrics = None

    # --- open-set prototypes: mean embedding per class over TRAIN crops ---
    # EfficientNet's pre-classifier 1280-d embedding (features → avgpool →
    # flatten) centroid per class. Shipped alongside the model so the runtime can
    # reject crops that are far from every known cat (open-set) instead of always
    # forcing a closed-set softmax label. Fail-closed: computed here from the SAME
    # deterministic eval transform the runtime uses, never from augmented crops.
    def embed_indices(indices: list[int]):
        embs: list[np.ndarray] = []
        labels: list[int] = []
        with torch.no_grad():
            for batch_indices in index_batches(indices, args.batch_size):
                batch = make_batch(batch_indices, eval_tf)
                if batch is None:
                    continue
                x, y, _ = batch
                feat = torch.flatten(model.avgpool(model.features(x)), 1)
                embs.append(feat.cpu().numpy())
                labels.extend(y.tolist())
        if not embs:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.concatenate(embs, axis=0).astype(np.float32), labels

    train_emb, train_emb_labels = embed_indices(train_idx)
    prototypes: dict[str, list[float]] = {}
    proto_dist_stats: dict[str, dict] = {}
    embedding_dim = int(train_emb.shape[1]) if train_emb.size else 0
    if train_emb.size:
        proto_matrix = prototypes_from_embeddings(train_emb, train_emb_labels, len(classes))
        prototypes = {
            classes[c]: proto_matrix[c].tolist()
            for c in range(len(classes))
            if float(np.linalg.norm(proto_matrix[c])) > 0
        }
        proto_dist_stats = prototype_distance_stats(
            train_emb, train_emb_labels, proto_matrix, classes)
        if proto_dist_stats:
            suggested = max(s["p95"] for s in proto_dist_stats.values())
            log.info(
                "open-set prototypes: %d/%d classes, embedding_dim=%d; "
                "train cosine-distance to own prototype per class: %s",
                len(prototypes), len(classes), embedding_dim,
                {c: round(s["p95"], 3) for c, s in proto_dist_stats.items()},
            )
            log.info(
                "suggested runtime ceiling: DETECTOR_MAX_PROTOTYPE_DISTANCE≈%.2f "
                "(worst-class p95; raise to accept more, lower to reject more)",
                suggested,
            )
    else:
        log.warning("could not decode any train crops for prototypes; the open-set "
                    "distance gate will be unavailable for this model")

    # --- save to a NEW path; export_classifier-compatible .pt ---
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "cat_classifier.pt"
    torch.save({"state_dict": model.state_dict(),
                "class_names": classes,
                "num_classes": len(classes),
                "prototypes": prototypes,
                "embedding_dim": embedding_dim}, pt_path)
    meta = {
        "created": stamp,
        "class_names": classes,
        "pad_frac": args.pad_frac,
        "preprocessing": {
            "resize_short_side": args.resize_short, "center_crop": args.image_size,
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "interpolation": "bilinear",
            "note": ("byte-identical to detector/classifier.py::_preprocess"
                     if args.image_size == INPUT_SIZE else
                     "WARNING: image_size != runtime default; detector/classifier.py "
                     "still preprocesses to 256->224 — align serve-time preprocessing "
                     "before promoting"),
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
        "open_set": {
            "prototype_classes": sorted(prototypes),
            "embedding_dim": embedding_dim,
            "train_distance_stats": proto_dist_stats,
            "suggested_max_prototype_distance": (
                round(max(s["p95"] for s in proto_dist_stats.values()), 3)
                if proto_dist_stats else None
            ),
            "note": ("per-class mean embedding over TRAIN crops; ship as "
                     "prototypes.json and set DETECTOR_MAX_PROTOTYPE_DISTANCE at "
                     "runtime to reject crops far from every known cat"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved best model -> {pt_path}")
    print(f"metadata         -> {out_dir / 'metadata.json'}")

    # --- log final metrics + artifacts to the experiment tracker ---
    def _log_eval(prefix: str, mtr: dict | None) -> None:
        if not mtr:
            return
        flat = {
            f"{prefix}_overall_accuracy": mtr["overall_accuracy"],
            f"{prefix}_macro_recall": mtr["macro_recall"],
            f"{prefix}_macro_f1": mtr.get("macro_f1", 0.0),
            f"{prefix}_loss": mtr.get("loss", 0.0),
            f"{prefix}_dangerous_errors": mtr.get("dangerous_errors", 0),
        }
        for c, v in mtr["recall"].items():
            flat[f"{prefix}_recall_{c}"] = v
        for c, v in mtr["precision"].items():
            flat[f"{prefix}_precision_{c}"] = v
        for c, v in mtr.get("f1", {}).items():
            flat[f"{prefix}_f1_{c}"] = v
        # Each configured dangerous confusion as its own metric.
        for d in mtr.get("dangerous_confusions", []):
            flat[f"{prefix}_danger_{d['actual']}_to_{d['predicted']}"] = d["count"]
        # Per-camera accuracy, when present.
        for cam, cstats in (mtr.get("per_camera") or {}).items():
            flat[f"{prefix}_acc_camera_{cam}"] = cstats["accuracy"]
        # Per-group/episode mean accuracy, when present.
        pg = mtr.get("per_group") or {}
        if "mean_group_accuracy" in pg:
            flat[f"{prefix}_mean_group_accuracy"] = pg["mean_group_accuracy"]
        run.log_metrics(flat)

    _log_eval("val", metrics)
    if test_metrics is not None:
        _log_eval("test", test_metrics)
    run.log_metric("best_epoch", best["epoch"])
    run.log_artifact(pt_path)
    run.log_artifact(out_dir / "metadata.json")
    if args.preview_augmentations and Path(args.preview_augmentations).exists():
        run.log_artifact(args.preview_augmentations, artifact_path="aug_preview")

    # --- dynamic metrics artifacts (sized/keyed from the class list, nothing fixed) ---
    try:
        write_labels_json(out_dir / "labels.json", classes)
        (out_dir / "classification_report.json").write_text(
            json.dumps({"val": metrics, "test": test_metrics}, indent=2), encoding="utf-8")
        (out_dir / "split_report.json").write_text(json.dumps(split_summary, indent=2),
                                                   encoding="utf-8")
        (out_dir / "augmentation_config.json").write_text(
            json.dumps({"level": args.augment, **vars(spec)}, indent=2), encoding="utf-8")
        (out_dir / "training_config.json").write_text(json.dumps({
            "model": "efficientnet_b0", "image_size": args.image_size,
            "resize_short_side": args.resize_short, "optimizer": "AdamW",
            "weight_decay": 1e-4, "lr": lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "seed": args.seed,
            "finetune_mode": finetune_mode, "augment": args.augment,
            "events_db": str(args.db), "recordings": str(args.recordings),
            "reviews_db": str(args.reviews_db) if args.reviews_db else None,
            "classes": classes, "num_classes": len(classes),
        }, indent=2), encoding="utf-8")
        write_confusion_csv(out_dir / "confusion_matrix.csv", cm, classes)
        write_confusion_png(out_dir / "confusion_matrix.png", cm, classes,
                            title="val confusion matrix")
        artifacts = ["labels.json", "classification_report.json", "split_report.json",
                     "augmentation_config.json", "training_config.json",
                     "confusion_matrix.csv", "confusion_matrix.png"]
        if test_metrics is not None:
            write_confusion_csv(out_dir / "test_confusion_matrix.csv", test_cm, classes)
            write_confusion_png(out_dir / "test_confusion_matrix.png", test_cm, classes,
                                title="test confusion matrix")
            artifacts += ["test_confusion_matrix.csv", "test_confusion_matrix.png"]
        # Only when dangerous confusions were actually configured.
        if dangerous:
            (out_dir / "dangerous_confusions.json").write_text(json.dumps({
                "configured": [{"actual": d.actual, "predicted": d.predicted, "action": d.action}
                               for d in dangerous],
                "val": metrics.get("dangerous_confusions", []),
                "test": (test_metrics or {}).get("dangerous_confusions", []),
            }, indent=2), encoding="utf-8")
            artifacts.append("dangerous_confusions.json")
        for name in artifacts:
            run.log_artifact(out_dir / name)
        print(f"metrics artifacts -> {out_dir} ({', '.join(artifacts)})")
    except Exception as e:  # never let artifact/plotting issues fail a finished run
        log.warning("failed to write/log metrics artifacts: %r", e)

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

    run.log_metric("val_pass", 0.0 if failing else 1.0)
    run.set_tags({"result": "FAIL" if failing else "PASS"})

    # --- concise run summary ---
    print("\n===== run summary =====")
    if run.active:
        print(f"  mlflow run_id : {run.run_id}  (uri {run.tracking_uri})")
    else:
        print("  mlflow        : disabled")
    print(f"  model         : {pt_path}")
    print(f"  val accuracy  : {metrics['overall_accuracy']:.3f}   "
          f"macro F1 : {metrics.get('macro_f1', 0.0):.3f}")
    print(f"  confusion     : {out_dir / 'confusion_matrix.png'}")
    if dangerous:
        dt = metrics.get("dangerous_errors", 0)
        print(f"  dangerous     : {dt} error(s) across {len(dangerous)} configured pair(s)")
    print(f"  result        : {'FAIL' if failing else 'PASS'}")

    run.end()


if __name__ == "__main__":
    main()
