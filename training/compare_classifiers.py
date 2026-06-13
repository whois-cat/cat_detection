"""Compare cat identity classifiers on the same human-reviewed crop set.

Examples:

    uv run python -m training.compare_classifiers \
        --db data/events/events.db \
        --recordings data/recordings \
        --reviews-db data/review/reviews.db \
        --candidate current=/opt/models/cat_classifier_openvino \
        --candidate new=models/trained/20260612-120000/cat_classifier.pt

The evaluator trusts only reviews.db labels. It decodes crops from recordings in
memory and reports closed-set metrics plus thresholded runtime behavior. Replay
sets are reported separately as forgetting/regression memory, not blended into
the headline promotion metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DROP_LABELS = {"discard", "unknown"}
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Prediction:
    label: str
    conf: float


class TorchPtModel:
    def __init__(self, path: Path) -> None:
        import torch
        import torch.nn as nn
        from torchvision import models

        sys.path.insert(0, str(ROOT / "detector"))
        from classifier import _preprocess

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.class_names = list(checkpoint["class_names"])
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_names))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._torch = torch
        self._model = model
        self._preprocess = _preprocess

    def predict(self, crop_bgr: np.ndarray) -> Prediction:
        rgb = np.ascontiguousarray(crop_bgr[..., ::-1])
        x = self._torch.from_numpy(self._preprocess(rgb)).float()
        with self._torch.inference_mode():
            logits = self._model(x)[0]
            probs = self._torch.softmax(logits, dim=0).cpu().numpy()
        idx = int(np.argmax(probs))
        return Prediction(self.class_names[idx], float(probs[idx]))


class OpenVinoModel:
    def __init__(self, path: Path) -> None:
        sys.path.insert(0, str(ROOT / "detector"))
        from classifier import CatClassifier

        self._model = CatClassifier(path)
        self.class_names = list(self._model.class_names)

    def predict(self, crop_bgr: np.ndarray) -> Prediction:
        rgb = np.ascontiguousarray(crop_bgr[..., ::-1])
        label, conf = self._model.classify(rgb)
        return Prediction(label, conf)


def load_model(path: Path):
    if path.is_dir() and (path / "cat_classifier.xml").exists():
        return OpenVinoModel(path)
    if path.is_dir() and (path / "cat_classifier.pt").exists():
        return TorchPtModel(path / "cat_classifier.pt")
    if path.suffix == ".pt":
        return TorchPtModel(path)
    raise SystemExit(
        f"unsupported model path {path}; expected .pt or OpenVINO dir with cat_classifier.xml"
    )


def parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    return name, Path(path)


def metrics_for(y_true: list[str], preds: list[Prediction], thresholds: list[float]) -> dict:
    classes = sorted(set(y_true))
    total = len(y_true)
    correct = [t == p.label for t, p in zip(y_true, preds)]
    by_true = {c: {"tp": 0, "fn": 0, "fp": 0} for c in classes}
    confusion = {c: {} for c in classes}
    for truth, pred in zip(y_true, preds):
        confusion.setdefault(truth, {})
        confusion[truth][pred.label] = confusion[truth].get(pred.label, 0) + 1
        if truth == pred.label:
            by_true[truth]["tp"] += 1
        else:
            by_true[truth]["fn"] += 1
            if pred.label in by_true:
                by_true[pred.label]["fp"] += 1

    per_class = {}
    recalls = []
    precisions = []
    for c in classes:
        tp = by_true[c]["tp"]
        fn = by_true[c]["fn"]
        fp = by_true[c]["fp"]
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        recalls.append(recall)
        precisions.append(precision)
        per_class[c] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
        }

    thresholded = {}
    for thr in thresholds:
        confident = [p.conf >= thr for p in preds]
        n_conf = sum(confident)
        n_conf_correct = sum(ok and keep for ok, keep in zip(correct, confident))
        n_conf_wrong = sum((not ok) and keep for ok, keep in zip(correct, confident))
        thresholded[f"{thr:.3g}"] = {
            "coverage": n_conf / max(1, total),
            "accuracy_when_confident": n_conf_correct / max(1, n_conf),
            "recall_counting_abstain_as_miss": n_conf_correct / max(1, total),
            "high_conf_wrong": n_conf_wrong,
        }

    return {
        "n": total,
        "accuracy": sum(correct) / max(1, total),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "min_recall": min(recalls) if recalls else 0.0,
        "per_class": per_class,
        "thresholds": thresholded,
        "confusion": confusion,
    }


def verdict(results: dict, baseline: str | None, threshold_key: str) -> dict:
    if not baseline or baseline not in results or len(results) < 2:
        return {"status": "info", "message": "no baseline comparison requested"}
    base = results[baseline]
    out = {}
    for name, m in results.items():
        if name == baseline:
            continue
        delta_macro = m["macro_recall"] - base["macro_recall"]
        delta_min = m["min_recall"] - base["min_recall"]
        base_wrong = base["thresholds"].get(threshold_key, {}).get("high_conf_wrong", 0)
        wrong = m["thresholds"].get(threshold_key, {}).get("high_conf_wrong", 0)
        if delta_macro >= 0 and delta_min >= 0 and wrong <= base_wrong:
            status = "candidate_ok"
        elif wrong > base_wrong:
            status = "risk_high_conf_wrong"
        else:
            status = "needs_review"
        out[name] = {
            "status": status,
            "delta_macro_recall": delta_macro,
            "delta_min_recall": delta_min,
            f"delta_high_conf_wrong_at_{threshold_key}": wrong - base_wrong,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--reviews-db", type=Path, required=True)
    ap.add_argument("--candidate", action="append", type=parse_candidate, required=True,
                    help="NAME=PATH; PATH is trained .pt or OpenVINO classifier dir")
    ap.add_argument("--baseline", default=None, help="candidate name to compare against")
    ap.add_argument("--replay-set", type=Path, action="append", default=[],
                    help="optional replay set(s) to include as regression memory")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--event-model", default=None, help="filter events.model in events.db")
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--default-rotate-deg", type=int, default=0)
    ap.add_argument("--t-from", type=int, default=None)
    ap.add_argument("--t-to", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresholds", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from training import CropSource, load_reviews
    from training.replay import load_replay_set

    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    review_labels = load_reviews(args.reviews_db)
    candidates = [(name, load_model(path)) for name, path in args.candidate]

    src = CropSource(
        db_path=args.db,
        recordings_root=args.recordings,
        camera_id=args.camera,
        model=args.event_model,
        t_from=args.t_from,
        t_to=args.t_to,
        min_score=args.min_score,
        pad_frac=args.pad_frac,
        default_rotate_deg=args.default_rotate_deg,
    )

    primary_true: list[str] = []
    primary_pred: dict[str, list[Prediction]] = {name: [] for name, _model in candidates}
    for sample in src:
        sb = sample.src_box
        if sb is None or sb.rowid is None:
            continue
        label = review_labels.get(sb.rowid)
        if label is None or label in DROP_LABELS:
            continue
        primary_true.append(label)
        for name, model in candidates:
            primary_pred[name].append(model.predict(sample.image))
        if args.limit is not None and len(primary_true) >= args.limit:
            break

    replay_true: list[str] = []
    replay_pred: dict[str, list[Prediction]] = {name: [] for name, _model in candidates}
    for replay_path in args.replay_set:
        for image, meta in load_replay_set(replay_path):
            replay_true.append(meta.label)
            for name, model in candidates:
                replay_pred[name].append(model.predict(image))

    if not primary_true:
        raise SystemExit("no reviewed crops available for comparison")

    results = {
        name: metrics_for(primary_true, preds, thresholds)
        for name, preds in primary_pred.items()
    }
    replay_results = (
        {
            name: metrics_for(replay_true, preds, thresholds)
            for name, preds in replay_pred.items()
        }
        if replay_true
        else {}
    )
    threshold_key = f"{max(thresholds):.3g}" if thresholds else "0.9"
    report = {
        "n": len(primary_true),
        "classes": sorted(set(primary_true)),
        "baseline": args.baseline,
        "results": results,
        "replay_n": len(replay_true),
        "replay_classes": sorted(set(replay_true)),
        "replay_results": replay_results,
        "verdict": verdict(results, args.baseline, threshold_key),
    }

    print(f"primary reviewed crops: n={len(primary_true)} classes={sorted(set(primary_true))}")
    for name, m in results.items():
        print(
            f"{name}: n={m['n']} acc={m['accuracy']:.3f} "
            f"macro_recall={m['macro_recall']:.3f} min_recall={m['min_recall']:.3f}"
        )
        if threshold_key in m["thresholds"]:
            t = m["thresholds"][threshold_key]
            print(
                f"  @{threshold_key}: coverage={t['coverage']:.3f} "
                f"acc_when_conf={t['accuracy_when_confident']:.3f} "
                f"high_conf_wrong={t['high_conf_wrong']}"
            )
    if replay_results:
        print(f"\nreplay regression memory: n={len(replay_true)} classes={sorted(set(replay_true))}")
        for name, m in replay_results.items():
            print(
                f"{name}: replay_acc={m['accuracy']:.3f} "
                f"replay_macro_recall={m['macro_recall']:.3f} "
                f"replay_min_recall={m['min_recall']:.3f}"
            )
    if args.baseline:
        print("verdict:", json.dumps(report["verdict"], ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
