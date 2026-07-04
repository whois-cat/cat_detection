"""Experiment artifacts: labels.json, confusion-matrix CSV/PNG. Everything is sized/keyed from the class list — nothing is a fixed shape."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def labels_payload(classes: list[str]) -> dict:
    """Run-level label metadata — class list, count, and both index mappings."""
    return {
        "classes": list(classes),
        "num_classes": len(classes),
        "label_to_index": {c: i for i, c in enumerate(classes)},
        "index_to_label": {str(i): c for i, c in enumerate(classes)},
    }


def write_labels_json(path: Path, classes: list[str]) -> None:
    Path(path).write_text(json.dumps(labels_payload(classes), indent=2), encoding="utf-8")


def write_confusion_csv(path: Path, cm, classes: list[str]) -> None:
    """Dynamic N×N confusion matrix as a labelled CSV (rows=true, cols=pred)."""
    import csv
    cm = np.asarray(cm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([r"true\pred", *classes])
        for i, c in enumerate(classes):
            w.writerow([c, *(int(cm[i, j]) for j in range(len(classes)))])


def write_confusion_png(path: Path, cm, classes: list[str], *, title: str = "confusion matrix") -> None:
    """Confusion-matrix heatmap; figure size scales with the number of classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    n = len(classes)
    side = max(4.0, 0.6 * n + 2.0)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = (cm.max() / 2) if cm.size else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
