"""Small batching utilities shared by the train/eval loops."""
from __future__ import annotations

import random
from typing import Iterator

import numpy as np


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


def mean_from_batch_means(pairs: list[tuple[float, int]]) -> float:
    """Combine per-batch mean losses into a single dataset mean:
    ``sum(mean_b * size_b) / sum(size_b)``.

    With an UNWEIGHTED criterion (mean = (1/N)·ΣL) this equals the plain
    per-sample mean and is therefore independent of how samples are grouped into
    batches — so the logged train/val loss curves are comparable regardless of
    shuffling/episode ordering. (A weighted criterion's per-batch mean normalises
    by Σweight, so the same accumulation would NOT be grouping-invariant — which
    is exactly the asymmetry this logging path avoids.)
    """
    total = sum(n for _, n in pairs)
    return sum(m * n for m, n in pairs) / max(1, total)
