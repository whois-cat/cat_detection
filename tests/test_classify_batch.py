"""classify_batch must run one inference and preserve input order. We bypass
OpenVINO by constructing the object and injecting a fake compiled model."""
import numpy as np

from classifier import CatClassifier


class _FakeCompiled:
    """Mimics an OpenVINO compiled model: callable, returns [logits] indexable
    by output port 0. Each row's argmax is its row index mod n_classes, so the
    predicted class encodes the input position — order errors are detectable."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.calls = 0

    def __call__(self, inp):
        self.calls += 1
        n = inp.shape[0]
        logits = np.full((n, self.n_classes), -5.0, dtype=np.float32)
        for i in range(n):
            logits[i, i % self.n_classes] = 10.0
        return [logits]


def _make(classes):
    clf = object.__new__(CatClassifier)
    clf.class_names = classes
    clf._compiled = _FakeCompiled(len(classes))
    return clf


def test_classify_batch_preserves_order_and_runs_once():
    clf = _make(["cat_a", "cat_b", "cat_c"])
    crops = [np.zeros((40, 40, 3), np.uint8) for _ in range(5)]
    out = clf.classify_batch(crops)
    assert [name for name, _ in out] == ["cat_a", "cat_b", "cat_c", "cat_a", "cat_b"]
    assert all(0.0 <= c <= 1.0 for _, c in out)
    assert clf._compiled.calls == 1                  # single batched inference


def test_classify_is_batch_wrapper():
    clf = _make(["cat_a", "cat_b"])
    name, conf = clf.classify(np.zeros((40, 40, 3), np.uint8))
    assert name == "cat_a"
    assert 0.0 <= conf <= 1.0


def test_classify_batch_empty():
    clf = _make(["cat_a", "cat_b"])
    assert clf.classify_batch([]) == []
