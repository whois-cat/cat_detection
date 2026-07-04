"""Classification metrics + reports (confusion matrix, per-class PR/F1, dangerous-confusion accounting, human-readable report printing)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def per_class_f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denom = precision + recall
    return np.divide(2 * precision * recall, denom,
                     out=np.zeros_like(precision), where=denom > 0)


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


@dataclass(frozen=True)
class DangerousConfusion:
    """A user-configured confusion that matters (e.g. a feeder safety risk).
    `predicted` mistaken for the model's output, `actual` the true label."""
    predicted: str
    actual: str
    reason: str = ""


def load_dangerous_confusions(path: Path) -> list[DangerousConfusion]:
    """Load dangerous confusion pairs from a YAML or JSON file. Accepts either a
    top-level list or a mapping with a ``dangerous_confusions`` key:

        dangerous_confusions:
          - predicted: cat_a
            actual: cat_b
            reason: "cat_a feeder must not open for cat_b"
    """
    import yaml  # available in the training env; parses JSON too
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("dangerous_confusions", [])
    out: list[DangerousConfusion] = []
    for item in data or []:
        out.append(DangerousConfusion(
            predicted=str(item["predicted"]),
            actual=str(item["actual"]),
            reason=str(item.get("reason", "")),
        ))
    return out


def dangerous_from_confuse(confuse) -> list[DangerousConfusion]:
    """Back-compat: treat a `--confuse a,b` pair as dangerous in BOTH directions."""
    cs = sorted(c for c in confuse if c)
    if len(cs) != 2:
        return []
    a, b = cs
    return [DangerousConfusion(a, b, "configured via --confuse"),
            DangerousConfusion(b, a, "configured via --confuse")]


def dangerous_confusion_report(cm: np.ndarray, classes: list[str],
                               dangerous) -> tuple[int, list[dict]]:
    """Count, per configured dangerous pair, how many true-`actual` crops were
    predicted `predicted`. Returns (total_errors, per-pair details)."""
    idx = {c: i for i, c in enumerate(classes)}
    total = 0
    details: list[dict] = []
    for dc in dangerous:
        if dc.predicted not in idx or dc.actual not in idx:
            continue
        a, p = idx[dc.actual], idx[dc.predicted]
        count = int(cm[a, p])
        support = int(cm[a].sum())
        total += count
        details.append({
            "predicted": dc.predicted, "actual": dc.actual,
            "count": count, "support": support,
            "rate": (count / support) if support else 0.0,
            "reason": dc.reason,
        })
    return total, details


def format_epoch_line(epoch: int, train_loss: float, val_loss: float,
                      macro_recall: float, dangerous_errors: int,
                      *, dangerous_configured: bool) -> str:
    """One-line epoch summary. ``dangerous_errors`` is shown only when dangerous
    confusions are configured; otherwise it is always 0 and would be noise, so the
    field is omitted entirely (no labels assumed either way)."""
    line = (f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_macro_recall={macro_recall:.3f}")
    if dangerous_configured:
        line += f"  dangerous_errors={dangerous_errors}"
    return line


def print_report(cm: np.ndarray, classes: list[str], confuse: set[str] = frozenset(),
                 *, dangerous=()) -> dict:
    prec, rec = per_class_pr(cm)
    f1 = per_class_f1(prec, rec)
    support = cm.sum(axis=1)
    present = present_class_mask(cm)
    macro_recall = supported_macro_recall(cm)
    macro_f1 = float(f1[present].mean()) if bool(present.any()) else 0.0
    overall = float(np.diag(cm).sum() / max(1, cm.sum()))

    width = max((len(c) for c in classes), default=1) + 1
    # Dynamic N×N confusion matrix — built from the class list, never a fixed size.
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(" " * width + "".join(f"{c[:7]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:<{width}}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(classes))))

    print("\nper-class precision / recall / F1:")
    for i, c in enumerate(classes):
        flag = "  <-- confuse" if c in confuse else ""
        if support[i] == 0:
            print(f"  {c:<{width}} precision=NA     recall=NA     F1=NA     support=0{flag}")
        else:
            print(
                f"  {c:<{width}} precision={prec[i]:.3f}  recall={rec[i]:.3f}  "
                f"F1={f1[i]:.3f}  support={int(support[i])}{flag}"
            )
    print(
        f"\noverall accuracy = {overall:.3f}   "
        f"macro recall = {macro_recall:.3f}   macro F1 = {macro_f1:.3f} "
        f"(present classes only)"
    )

    dangerous_total, dangerous_details = dangerous_confusion_report(cm, classes, dangerous)
    if dangerous_details:
        print("\ndangerous confusions (true → predicted):")
        for d in dangerous_details:
            tail = f"  — {d['reason']}" if d["reason"] else ""
            print(f"  {d['actual']} → {d['predicted']}: {d['count']}/{d['support']} "
                  f"({d['rate'] * 100:.1f}%){tail}")
        print(f"  total dangerous errors = {dangerous_total}")

    return {
        "classes": classes,
        "present_classes": [classes[i] for i, ok in enumerate(present) if ok],
        "missing_eval_classes": [classes[i] for i, ok in enumerate(present) if not ok],
        "precision": {c: float(prec[i]) for i, c in enumerate(classes)},
        "recall": {c: float(rec[i]) for i, c in enumerate(classes)},
        "f1": {c: float(f1[i]) for i, c in enumerate(classes)},
        "support": {c: int(support[i]) for i, c in enumerate(classes)},
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "overall_accuracy": overall,
        "confusion_matrix": cm.tolist(),
        "dangerous_confusions": dangerous_details,
        "dangerous_errors": dangerous_total,
    }


def per_camera_confusion(y_true, y_pred, classes, cameras) -> dict:
    """Per-camera confusion + accuracy, when camera metadata is available."""
    if not cameras:
        return {}
    by_cam: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for t, p, cam in zip(y_true, y_pred, cameras):
        by_cam[cam][0].append(t)
        by_cam[cam][1].append(p)
    out = {}
    print("\nper-camera accuracy:")
    for cam in sorted(by_cam, key=str):
        yt, yp = by_cam[cam]
        cmc = confusion(yt, yp, len(classes))
        acc = float(np.diag(cmc).sum() / max(1, cmc.sum()))
        print(f"  camera {cam}: n={len(yt)} accuracy={acc:.3f}")
        out[str(cam)] = {"n": len(yt), "accuracy": acc, "confusion_matrix": cmc.tolist()}
    return out


def per_group_accuracy(y_true, y_pred, groups, *, worst: int = 5) -> dict:
    """Per-group/episode accuracy summary, when group metadata is available.
    Prints the mean and the worst few groups rather than every group."""
    if not groups:
        return {}
    tally: dict[object, list[int]] = defaultdict(lambda: [0, 0])  # group -> [correct, total]
    for t, p, g in zip(y_true, y_pred, groups):
        if g is None:
            continue
        tally[g][1] += 1
        if t == p:
            tally[g][0] += 1
    accs = [(g, c / n, n) for g, (c, n) in tally.items() if n > 0]
    if not accs:
        return {}
    mean = sum(a for _, a, _ in accs) / len(accs)
    accs.sort(key=lambda x: (x[1], -x[2]))  # worst accuracy first
    print(f"\nper-group/episode accuracy: groups={len(accs)} mean={mean:.3f}")
    for g, a, n in accs[:worst]:
        print(f"  worst group {g}: accuracy={a:.3f} (n={n})")
    return {
        "n_groups": len(accs),
        "mean_group_accuracy": mean,
        "worst": [{"group": g, "accuracy": a, "n": n} for g, a, n in accs[:worst]],
    }
