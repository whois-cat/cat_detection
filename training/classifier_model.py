"""Model helpers: checkpoint remap on class-name changes, fine-tune configuration, and open-set prototype computation. Torch is only touched via objects passed in — this module imports no torch itself."""
from __future__ import annotations

import numpy as np


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


def prototypes_from_embeddings(
    embeddings: np.ndarray,
    label_indices,
    num_classes: int,
) -> np.ndarray:
    """Per-class prototype = L2-normalized mean of L2-normalized embeddings.

    Returns an (num_classes, D) float32 array. Each prototype is the centroid of
    the unit embeddings for that class, itself re-normalized to unit length, so
    the runtime open-set gate can score a crop by cosine distance to it (see
    detector/unknown.py::decide_identity). Classes with no embeddings get an
    all-zero row; callers drop those (a zero prototype means cosine distance 1.0
    — 'always far' — which is the fail-closed default, never a false accept).

    This is what makes identity open-set without abandoning the softmax head: a
    stranger cat / raccoon / dog lands far from every prototype and is rejected
    as UNKNOWN even when the closed-set softmax looks confident.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D), got shape {emb.shape}")
    labels = np.asarray(list(label_indices))
    if labels.shape[0] != emb.shape[0]:
        raise ValueError("embeddings and label_indices length mismatch")
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    protos = np.zeros((num_classes, emb.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            continue
        mean = unit[mask].mean(axis=0)
        norm = np.linalg.norm(mean)
        protos[c] = mean / norm if norm > 1e-12 else mean
    return protos.astype(np.float32)


def prototype_distance_stats(
    embeddings: np.ndarray,
    label_indices,
    prototypes: np.ndarray,
    class_names: list[str],
) -> dict:
    """Per-class cosine-distance spread of TRAIN embeddings to their own
    prototype. Purely diagnostic: it tells the operator where to set the runtime
    ``DETECTOR_MAX_PROTOTYPE_DISTANCE`` ceiling (a good first guess is around the
    worst class's p95, so almost all genuine same-cat crops stay accepted)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.size == 0:
        return {}
    labels = np.asarray(list(label_indices))
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    stats: dict[str, dict] = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        if not mask.any() or np.linalg.norm(prototypes[c]) == 0:
            continue
        dist = 1.0 - (unit[mask] @ prototypes[c].astype(np.float64))
        stats[name] = {
            "mean": float(dist.mean()),
            "p95": float(np.percentile(dist, 95)),
            "max": float(dist.max()),
            "n": int(mask.sum()),
        }
    return stats
