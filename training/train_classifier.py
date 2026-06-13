"""Train a cat-identity classifier from human-reviewed labels.

CPU-friendly. Reads crops straight from the recordings in memory (no JPEGs on
disk), applies the human label corrections, trains an EfficientNet-B0, and writes
the BEST-by-val model to a NEW path. The runtime is NOT touched — swapping the
model into production is a separate, later step.

Pipeline (all reused from this package):
  - training.CropSource            — decode crops from recordings by events coords
  - training.reviews.load_reviews  — human corrections {rowid: label}
  - training.torch_dataset.TorchCachedDataset — decode-once RAM cache (fork-safe:
    materialise() before any DataLoader workers; we default num_workers=0 anyway)
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
import random
import sys
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("training.train_classifier")

ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DROP_LABELS = {"discard", "unknown"}

Meta = namedtuple("Meta", ["label", "camera", "wall_ms"])


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
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="crop context padding — MUST match the detector "
                         "CLASSIFIER_PAD_FRAC and build_cluster_manifest --pad-frac")
    ap.add_argument("--default-rotate-deg", type=int, default=0,
                    help="rotation to assume for events recorded BEFORE rotate_deg "
                         "was persisted (set to the camera's rotate_deg then)")
    ap.add_argument("--min-score", type=float, default=0.7, help="drop low YOLO-score boxes")
    ap.add_argument("--episode-gap-sec", type=float, default=60.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6, help="early stop on val macro-recall")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 head / 1e-4 full")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="optional previous cat_classifier.pt checkpoint to fine-tune from")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train the whole backbone (low LR) instead of head + last block")
    ap.add_argument("--num-workers", type=int, default=0, help="keep 0 on CPU (fork-safe)")
    ap.add_argument("--min-recall", type=float, default=0.9,
                    help="PASS/FAIL gate: every class (incl. confuse) needs val recall >= this")
    ap.add_argument("--out-root", type=Path, default=ROOT / "models/trained")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms as T

    try:
        from classifier import _preprocess  # exact runtime eval preprocessing
    except ImportError:
        sys.path.insert(0, str(ROOT / "detector"))
        from classifier import _preprocess
    from training import CropSource, load_reviews
    from training.replay import load_replay_set
    from training.torch_dataset import TorchCachedDataset

    # Determinism.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    confuse = {c.strip() for c in args.confuse.split(",") if c.strip()}

    # --- decode crops once into RAM, deciding the final label per crop ---
    reviews = load_reviews(args.reviews_db) if args.reviews_db else {}
    log.info("loaded %d human corrections", len(reviews))

    def meta_fn(sample) -> Meta:
        sb = sample.src_box
        human = reviews.get(sb.rowid) if sb and sb.rowid is not None else None
        label = decide_label(sb.cat if sb else None,
                             sb.cat_score if sb else None,
                             human, args.trust_classifier, args.trust_conf)
        return Meta(label=label, camera=sample.camera_id, wall_ms=sample.wall_ms)

    # CropSource WITHOUT reviews= (we need raw classifier labels/conf for policy).
    # default_rotate_deg only affects pre-migration events with NULL rotate_deg.
    src = CropSource(db_path=args.db, recordings_root=args.recordings,
                     camera_id=args.camera, model=args.model,
                     min_score=args.min_score, pad_frac=args.pad_frac,
                     default_rotate_deg=args.default_rotate_deg)
    cache = TorchCachedDataset(src, transform=lambda x: x, target_fn=meta_fn)
    cache.materialise()   # decode pass in the MAIN process (CropSource isn't fork-safe)

    items = [cache[i] for i in range(len(cache))]          # (img_bgr, Meta)
    kept = [(img, m) for (img, m) in items if m.label is not None]
    if not kept:
        raise SystemExit("no usable crops after the label policy — loosen "
                         "--trust-classifier or add human reviews")
    fresh_images = [img for img, _ in kept]
    fresh_metas = [m for _, m in kept]
    log.info("kept %d / %d crops after label policy", len(kept), len(items))

    replay_items = []
    for replay_path in args.replay_set:
        replay_items.extend(load_replay_set(replay_path))
    if replay_items:
        log.info("loaded %d replay crops from %d replay set(s)",
                 len(replay_items), len(args.replay_set))
    replay_images = [img for img, _ in replay_items]
    replay_metas = [m for _, m in replay_items]

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
    images = fresh_images + replay_images
    metas = fresh_metas + replay_metas
    if replay_metas:
        train_idx.extend(range(len(fresh_metas), len(metas)))
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

    class Crops(Dataset):
        def __init__(self, idxs, tf):
            self.idxs, self.tf = idxs, tf
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, k):
            j = self.idxs[k]
            return self.tf(images[j]), cls_to_idx[metas[j].label]

    train_ds = Crops(train_idx, train_tf)
    val_ds = Crops(val_idx, eval_tf)
    test_ds = Crops(test_idx, eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, generator=g)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

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

    if args.full_finetune:
        lr = args.lr if args.lr is not None else 1e-4
        params = model.parameters()
    else:
        # Freeze backbone; train head + last conv stage + final block.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for blk in (model.features[-1], model.features[-2]):
            for p in blk.parameters():
                p.requires_grad = True
        lr = args.lr if args.lr is not None else 1e-3
        params = [p for p in model.parameters() if p.requires_grad]

    optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    # --- train loop with best-by-(macro-recall, fewest confuse cross-errors) ---
    best = {"macro_recall": -1.0, "cross": 10**9, "state": None, "metrics": None, "epoch": -1}
    since_improved = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        sched.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x).argmax(1)
                y_true.extend(y.tolist())
                y_pred.extend(pred.tolist())
        cm = confusion(y_true, y_pred, len(classes))
        macro = supported_macro_recall(cm)
        cross = confuse_cross_errors(cm, classes, confuse)
        log.info("epoch %02d  train_loss=%.4f  val_macro_recall=%.3f  confuse_cross=%d",
                 epoch, running / max(1, len(train_idx)), macro, cross)

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
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_dl:
            y_true.extend(y.tolist())
            y_pred.extend(model(x).argmax(1).tolist())
    cm = confusion(y_true, y_pred, len(classes))
    print(f"\n===== BEST model (epoch {best['epoch']}) on val =====")
    metrics = print_report(cm, classes, confuse)

    if test_idx:
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_dl:
                y_true.extend(y.tolist())
                y_pred.extend(model(x).argmax(1).tolist())
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
        "init_from": str(args.init_from) if args.init_from else None,
        "full_finetune": args.full_finetune, "best_epoch": best["epoch"],
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
