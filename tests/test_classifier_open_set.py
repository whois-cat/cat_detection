"""Runtime open-set wiring in CatClassifier.decide().

We bypass OpenVINO by injecting a fake compiled model (as test_classify_batch
does). The decision logic itself lives in detector/unknown.py (already tested);
these tests prove the classifier feeds it embeddings + prototypes correctly and
degrades safely when either is missing."""
import numpy as np
import pytest

from classifier import CatClassifier, _load_prototypes
from unknown import UNKNOWN, UnknownConfig

CLASSES = ["alisa", "chuzh", "ellie"]
# logits whose softmax is very confident on class 0 ("alisa").
CONFIDENT_ALISA = np.array([12.0, -4.0, -4.0], dtype=np.float32)
LOW_CONF = np.array([0.6, 0.5, 0.4], dtype=np.float32)


class _FakeCompiled:
    """Mimics an OpenVINO compiled model. Returns [logits] or [logits, emb];
    each is a single fixed row broadcast to the batch size (tests pass 1 crop)."""

    def __init__(self, logits, emb=None):
        self._logits = np.asarray(logits, dtype=np.float32)
        self._emb = None if emb is None else np.asarray(emb, dtype=np.float32)

    def __call__(self, inp):
        n = inp.shape[0]
        rows = np.repeat(self._logits[None], n, axis=0)
        if self._emb is None:
            return [rows]
        return [rows, np.repeat(self._emb[None], n, axis=0)]


def _make(logits, emb=None, prototypes=None):
    clf = object.__new__(CatClassifier)
    clf.class_names = CLASSES
    clf.prototypes = prototypes
    clf._compiled = _FakeCompiled(logits, emb)
    return clf


CROP = np.zeros((40, 40, 3), np.uint8)
PROTOS = {"alisa": np.array([1.0, 0.0], dtype=np.float32)}


def test_confident_and_near_prototype_is_known():
    clf = _make(CONFIDENT_ALISA, emb=[0.99, 0.01], prototypes=PROTOS)
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=0.2)
    name, conf = clf.decide(CROP, cfg)
    assert name == "alisa"
    assert conf > 0.9


def test_confident_but_far_from_prototype_is_unknown():
    # The whole point: softmax is ~1.0 for alisa, but the embedding is orthogonal
    # to alisa's prototype → open-set rejection (a stranger cat / raccoon / dog).
    clf = _make(CONFIDENT_ALISA, emb=[0.0, 1.0], prototypes=PROTOS)
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=0.2)
    name, _ = clf.decide(CROP, cfg)
    assert name == UNKNOWN


def test_distance_gate_off_by_default_keeps_confident_label():
    # max_prototype_distance=None → softmax-only path (backward compatible).
    clf = _make(CONFIDENT_ALISA, emb=[0.0, 1.0], prototypes=PROTOS)
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=None)
    name, _ = clf.decide(CROP, cfg)
    assert name == "alisa"


def test_legacy_single_output_ir_falls_back_to_softmax():
    # No embedding output → cannot compute distance → gate is skipped, not fatal.
    clf = _make(CONFIDENT_ALISA, emb=None, prototypes=PROTOS)
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=0.2)
    name, _ = clf.decide(CROP, cfg)
    assert name == "alisa"


def test_low_softmax_is_unknown_regardless():
    clf = _make(LOW_CONF, emb=[1.0, 0.0], prototypes=PROTOS)
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=0.2)
    name, _ = clf.decide(CROP, cfg)
    assert name == UNKNOWN


def test_probs_emb_batch_empty():
    clf = _make(CONFIDENT_ALISA, emb=[1.0, 0.0])
    probs, emb = clf._probs_emb_batch([])
    assert probs.shape == (0, len(CLASSES))
    assert emb is None


# ---- prototypes.json loading ----

def test_load_prototypes_missing_file_returns_none(tmp_path):
    assert _load_prototypes(tmp_path / "prototypes.json", CLASSES) is None


def test_load_prototypes_normalizes_and_filters_unknown(tmp_path):
    p = tmp_path / "prototypes.json"
    p.write_text('{"alisa": [3.0, 4.0], "ghost": [1.0, 0.0]}', encoding="utf-8")
    protos = _load_prototypes(p, CLASSES)
    assert set(protos) == {"alisa"}                     # unknown class dropped
    np.testing.assert_allclose(np.linalg.norm(protos["alisa"]), 1.0, atol=1e-6)
    np.testing.assert_allclose(protos["alisa"], [0.6, 0.8], atol=1e-6)


def test_load_prototypes_rejects_zero_vector(tmp_path):
    p = tmp_path / "prototypes.json"
    p.write_text('{"alisa": [0.0, 0.0]}', encoding="utf-8")
    with pytest.raises(ValueError):
        _load_prototypes(p, CLASSES)
