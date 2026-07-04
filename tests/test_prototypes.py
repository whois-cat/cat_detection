"""Open-set prototype computation (training side, pure numpy).

Prototypes are the per-class centroid of L2-normalized embeddings, re-normalized
to unit length, so the runtime can reject crops far from every known cat by
cosine distance (detector/unknown.py::decide_identity)."""
import numpy as np

from training.train_classifier import (
    prototype_distance_stats,
    prototypes_from_embeddings,
)


def test_prototype_is_unit_normalized_class_mean():
    # Two classes, clearly separated directions.
    emb = np.array([
        [10.0, 0.0],   # class 0 (magnitude must not matter — normalized first)
        [2.0, 0.0],    # class 0
        [0.0, 5.0],    # class 1
    ], dtype=np.float32)
    protos = prototypes_from_embeddings(emb, [0, 0, 1], num_classes=2)

    assert protos.shape == (2, 2)
    # class 0 → unit vector along +x; class 1 → unit vector along +y.
    np.testing.assert_allclose(protos[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(protos[1], [0.0, 1.0], atol=1e-6)
    # Every non-empty prototype has unit norm.
    for p in protos:
        np.testing.assert_allclose(np.linalg.norm(p), 1.0, atol=1e-6)


def test_empty_class_gets_zero_prototype():
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    protos = prototypes_from_embeddings(emb, [0], num_classes=3)
    assert np.linalg.norm(protos[0]) > 0        # class 0 present
    assert np.linalg.norm(protos[1]) == 0       # absent → zero (fail-closed)
    assert np.linalg.norm(protos[2]) == 0


def test_shape_mismatch_raises():
    emb = np.zeros((3, 4), dtype=np.float32)
    try:
        prototypes_from_embeddings(emb, [0, 1], num_classes=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


def test_distance_stats_report_spread_and_skip_empty():
    emb = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float32)
    protos = prototypes_from_embeddings(emb, [0, 0, 1], num_classes=3)
    stats = prototype_distance_stats(emb, [0, 0, 1], protos, ["a", "b", "c"])

    assert set(stats) == {"a", "b"}             # class c had no samples → skipped
    assert stats["a"]["n"] == 2
    assert 0.0 <= stats["a"]["mean"] <= stats["a"]["max"]
    assert stats["b"]["mean"] < 1e-6            # single-sample class sits on its prototype
