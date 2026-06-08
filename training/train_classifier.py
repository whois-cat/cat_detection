"""Train OUR cat-identity classifier (replaces the donor that confuses alisa↔felisis).

CPU-friendly. Reads crops straight from the recordings in memory (no JPEGs on
disk), applies the human label corrections, trains an EfficientNet-B0, and writes
the BEST-by-val model to a NEW path. The donor model and the runtime are NOT
touched — swapping it into production is a separate, later step.

Pipeline (all reused from this package):
  - training.CropSource            — decode crops from recordings by events coords
  - training.reviews.load_reviews  — human corrections {rowid: label}
  - training.torch_dataset.TorchCachedDataset — decode-once RAM cache (fork-safe:
    materialise() before any DataLoader workers; we default num_workers=0 anyway)
  - detector/classifier.py::_preprocess — the EXACT runtime eval transform

Label policy (flags; safe defaults):
  - confuse pair (--confuse alisa,felisis): ONLY human labels are trusted.
  - other classes: human label, OR the detector label when its classifier
    confidence (events.cat_score) >= --trust-conf (default 0.9).
  - discard/unknown are dropped. Class names come from the surviving labels
    (sorted, unique) — never hardcoded — and are saved with the model.

Honest split: crops are grouped into episodes (same camera, wall_ms gaps >
--episode-gap-sec start a new one); a whole episode goes entirely to train OR
val, so near-duplicate neighbours never straddle the split. The val set is forced
to contain both confuse cats, else the confusion metric would be empty.

Run:  python -m training.train_classifier --db data/events/events.db \
          --recordings data/recordings --reviews-db data/review/reviews.db \
          --confuse alisa,felisis
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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

def decide_label(det_label, det_conf, human, confuse, trust_conf):
    """Final training label for one crop, or None to drop it (see module doc)."""
    if human is not None:
        return None if human in DROP_LABELS else human
    # No human label below here — fall back to the detector, carefully.
    if not det_label or det_label in DROP_LABELS:
        return None
    if det_label in confuse:
        return None                      # never trust the detector on the confuse pair
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
                   val_frac: float, confuse: set[str], seed: int):
    """Assign whole episodes to train/val. Stratify so val holds both confuse
    cats. Returns (train_idx, val_idx) as crop-index lists."""
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)

    total = sum(len(e) for e in episodes)
    target = val_frac * total
    val_eps: set[int] = set()
    n = 0
    for e in order:
        if n >= target:
            break
        val_eps.add(e)
        n += len(episodes[e])

    def ep_labels(e: int) -> set[str]:
        return {metas[i].label for i in episodes[e]}

    # Guarantee each confuse cat appears in val (move a train episode if needed).
    val_label_union = set().union(*(ep_labels(e) for e in val_eps)) if val_eps else set()
    for cat in confuse:
        if cat in val_label_union:
            continue
        for e in order:
            if e not in val_eps and cat in ep_labels(e):
                val_eps.add(e)
                val_label_union |= ep_labels(e)
                break
        else:
            log.warning("confuse cat %r not present in ANY episode — confusion "
                        "metric for it will be empty", cat)

    train_idx, val_idx = [], []
    for e in range(len(episodes)):
        (val_idx if e in val_eps else train_idx).extend(episodes[e])
    return train_idx, val_idx


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


def print_report(cm: np.ndarray, classes: list[str], confuse: set[str]) -> dict:
    prec, rec = per_class_pr(cm)
    macro_recall = float(rec.mean())
    overall = float(np.diag(cm).sum() / max(1, cm.sum()))

    width = max(len(c) for c in classes) + 1
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(" " * width + "".join(f"{c[:7]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{width}}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(classes))))

    print("\nper-class precision / recall:")
    for i, c in enumerate(classes):
        flag = "  <-- confuse" if c in confuse else ""
        print(f"  {c:<{width}} precision={prec[i]:.3f}  recall={rec[i]:.3f}{flag}")
    print(f"\noverall accuracy = {overall:.3f}   macro recall = {macro_recall:.3f}")

    # The confusion cells we care about most.
    conf_list = sorted(confuse)
    if len(conf_list) == 2 and all(c in classes for c in conf_list):
        a, b = (classes.index(conf_list[0]), classes.index(conf_list[1]))
        print(f"alisa↔felisis cell: {classes[a]}→{classes[b]}={cm[a, b]}  "
              f"{classes[b]}→{classes[a]}={cm[b, a]}  "
              f"(cross-errors={int(cm[a, b] + cm[b, a])})")

    return {
        "classes": classes,
        "precision": {c: float(prec[i]) for i, c in enumerate(classes)},
        "recall": {c: float(rec[i]) for i, c in enumerate(classes)},
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
    ap.add_argument("--confuse", default="", help="confusable pair, e.g. alisa,felisis")
    ap.add_argument("--trust-conf", type=float, default=0.9,
                    help="min detector cat_score to trust its label (non-confuse classes)")
    ap.add_argument("--pad-frac", type=float, default=0.15,
                    help="crop context padding — MUST match the detector "
                         "CLASSIFIER_PAD_FRAC and build_review_manifest --pad-frac")
    ap.add_argument("--min-score", type=float, default=None, help="drop low YOLO-score boxes")
    ap.add_argument("--episode-gap-sec", type=float, default=60.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6, help="early stop on val macro-recall")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 head / 1e-4 full")
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

    from classifier import _preprocess  # exact runtime eval preprocessing
    from training import CropSource, load_reviews
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
                             human, confuse, args.trust_conf)
        return Meta(label=label, camera=sample.camera_id, wall_ms=sample.wall_ms)

    # CropSource WITHOUT reviews= (we need the raw detector label/conf for policy).
    src = CropSource(db_path=args.db, recordings_root=args.recordings,
                     camera_id=args.camera, model=args.model,
                     min_score=args.min_score, pad_frac=args.pad_frac)
    cache = TorchCachedDataset(src, transform=lambda x: x, target_fn=meta_fn)
    cache.materialise()   # decode pass in the MAIN process (CropSource isn't fork-safe)

    items = [cache[i] for i in range(len(cache))]          # (img_bgr, Meta)
    kept = [(img, m) for (img, m) in items if m.label is not None]
    if not kept:
        raise SystemExit("no usable crops after the label policy — loosen "
                         "--trust-conf or add human reviews")
    images = [img for img, _ in kept]
    metas = [m for _, m in kept]
    log.info("kept %d / %d crops after label policy", len(kept), len(items))

    classes = sorted({m.label for m in metas})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    log.info("classes (%d): %s", len(classes), classes)
    for c in sorted(confuse):
        if c not in cls_to_idx:
            log.warning("confuse cat %r has NO crops — its confusion metric is empty", c)

    # --- honest, episode-grouped, confuse-stratified split ---
    episodes = build_episodes(metas, int(args.episode_gap_sec * 1000))
    train_idx, val_idx = split_episodes(episodes, metas, val_frac=args.val_frac,
                                        confuse=confuse, seed=args.seed)
    log.info("%d episodes -> train %d crops / val %d crops",
             len(episodes), len(train_idx), len(val_idx))

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
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, generator=g)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
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
        _, rec = per_class_pr(cm)
        macro = float(rec.mean())
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

    # --- save to a NEW path (donor untouched); export_classifier-compatible .pt ---
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
        "trust_conf": args.trust_conf, "confuse": sorted(confuse),
        "full_finetune": args.full_finetune, "best_epoch": best["epoch"],
        "val_metrics": metrics,
        "counts_train": dict(zip(classes, counts.astype(int).tolist())),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nsaved best model -> {pt_path}")
    print(f"metadata         -> {out_dir / 'metadata.json'}")

    # --- PASS/FAIL guard (loud warning on FAIL; do NOT crash) ---
    _, rec = per_class_pr(cm)
    failing = {classes[i]: float(rec[i]) for i in range(len(classes))
               if rec[i] < args.min_recall}
    # confuse cats must explicitly clear the bar too (already covered by per-class,
    # but call them out so an empty confuse class can't slip through silently).
    for c in sorted(confuse):
        if c not in classes:
            failing[c] = 0.0
    if failing:
        print("\n" + "!" * 64)
        print(f"FAIL — NOT ready to replace the donor (need recall >= {args.min_recall} "
              "for every class incl. alisa & felisis).")
        for c, r in sorted(failing.items()):
            print(f"   {c}: recall={r:.3f}")
        print("Collect/relabel more crops (especially the confuse pair) and re-train.")
        print("!" * 64)
    else:
        print(f"\nPASS — every class recall >= {args.min_recall} (incl. the confuse pair). "
              "Model is a candidate to replace the donor (export + swap is a separate step).")


if __name__ == "__main__":
    main()
