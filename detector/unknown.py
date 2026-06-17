"""Open-set "unknown cat" decision + evaluation metrics.

Max-softmax alone is a poor open-set detector: an out-of-distribution cat can
still produce a confident-looking softmax. This module combines two signals and
is structured so stronger ones (temperature scaling, hard negatives, an explicit
unknown class) can be layered in later without changing call sites:

  1. softmax confidence (max probability), optionally with a per-class floor;
  2. distance from the crop embedding to the matched class PROTOTYPE (mean
     training embedding), with a distance ceiling.

Either signal failing → "unknown" (fail-closed: when in doubt, don't open).
Prototypes/embeddings are optional; with none supplied it degrades to a pure
softmax-threshold gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = "unknown"


@dataclass
class UnknownConfig:
    min_confidence: float = 0.9                 # global softmax floor
    per_class_min_confidence: dict | None = None  # overrides per class name
    max_prototype_distance: float | None = None   # cosine distance ceiling (None=off)

    def conf_floor(self, cat: str) -> float:
        if self.per_class_min_confidence and cat in self.per_class_min_confidence:
            return float(self.per_class_min_confidence[cat])
        return float(self.min_confidence)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - (a @ b) / (na * nb))


def decide_identity(
    probs: np.ndarray,
    class_names: list[str],
    config: UnknownConfig,
    *,
    embedding: np.ndarray | None = None,
    prototypes: dict | None = None,
) -> tuple[str, float]:
    """Return (label, confidence). label is a class name or UNKNOWN.

    Fail-closed: a low max-softmax OR a too-far prototype distance yields
    UNKNOWN, so the feeder never opens on an uncertain identity."""
    if probs is None or len(probs) == 0:
        return UNKNOWN, 0.0
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    cat = class_names[idx]

    if conf < config.conf_floor(cat):
        return UNKNOWN, conf

    if (config.max_prototype_distance is not None
            and embedding is not None and prototypes and cat in prototypes):
        dist = _cosine_distance(embedding, prototypes[cat])
        if dist > config.max_prototype_distance:
            return UNKNOWN, conf

    return cat, conf


# ---- evaluation metrics ------------------------------------------------------

def open_set_metrics(
    y_true: list[str],
    y_pred: list[str],
    known_classes: list[str],
) -> dict:
    """Open-set metrics. y_true uses UNKNOWN for genuine unknown/other cats.

    - false_accept_rate: unknown truths predicted as a known class.
    - false_reject_rate: known truths predicted UNKNOWN.
    - per_class precision/recall over known classes.
    - unknown_recall: unknowns correctly flagged UNKNOWN.
    """
    n_unknown_true = sum(1 for t in y_true if t == UNKNOWN)
    n_known_true = sum(1 for t in y_true if t != UNKNOWN)
    false_accepts = sum(1 for t, p in zip(y_true, y_pred) if t == UNKNOWN and p != UNKNOWN)
    false_rejects = sum(1 for t, p in zip(y_true, y_pred) if t != UNKNOWN and p == UNKNOWN)
    unknown_correct = sum(1 for t, p in zip(y_true, y_pred) if t == UNKNOWN and p == UNKNOWN)

    per_class = {}
    for c in known_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        per_class[c] = {
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        }

    return {
        "false_accept_rate": false_accepts / n_unknown_true if n_unknown_true else 0.0,
        "false_reject_rate": false_rejects / n_known_true if n_known_true else 0.0,
        "unknown_recall": unknown_correct / n_unknown_true if n_unknown_true else 0.0,
        "per_class": per_class,
    }


def feeder_false_accepts_per_1000_visits(
    visit_truths: list[str],
    visit_decisions: list[str],
) -> float:
    """How often the feeder opens (decision != UNKNOWN/closed) for a genuinely
    unknown cat, normalized per 1000 visits. `visit_decisions` holds the opened
    identity or UNKNOWN/"" when it stayed closed."""
    n = len(visit_truths)
    if n == 0:
        return 0.0
    bad = sum(
        1 for t, d in zip(visit_truths, visit_decisions)
        if t == UNKNOWN and d not in (UNKNOWN, "", None)
    )
    return 1000.0 * bad / n
