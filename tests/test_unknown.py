import numpy as np

from unknown import (
    UNKNOWN,
    UnknownConfig,
    decide_identity,
    feeder_false_accepts_per_1000_visits,
    open_set_metrics,
)

CLASSES = ["alisa", "chuzh", "ellie"]


def test_low_confidence_is_unknown():
    probs = np.array([0.5, 0.3, 0.2])
    label, conf = decide_identity(probs, CLASSES, UnknownConfig(min_confidence=0.9))
    assert label == UNKNOWN
    assert conf == 0.5


def test_confident_known_passes():
    probs = np.array([0.95, 0.03, 0.02])
    label, _ = decide_identity(probs, CLASSES, UnknownConfig(min_confidence=0.9))
    assert label == "alisa"


def test_per_class_threshold_override():
    probs = np.array([0.92, 0.05, 0.03])
    cfg = UnknownConfig(min_confidence=0.9, per_class_min_confidence={"alisa": 0.95})
    label, _ = decide_identity(probs, CLASSES, cfg)
    assert label == UNKNOWN  # alisa needs 0.95


def test_prototype_distance_gate():
    probs = np.array([0.99, 0.005, 0.005])
    cfg = UnknownConfig(min_confidence=0.9, max_prototype_distance=0.2)
    protos = {"alisa": np.array([1.0, 0.0])}
    near = decide_identity(probs, CLASSES, cfg, embedding=np.array([0.99, 0.01]), prototypes=protos)
    far = decide_identity(probs, CLASSES, cfg, embedding=np.array([0.0, 1.0]), prototypes=protos)
    assert near[0] == "alisa"
    assert far[0] == UNKNOWN  # too far from prototype despite high softmax


def test_open_set_metrics():
    y_true = ["alisa", "alisa", UNKNOWN, UNKNOWN, "chuzh"]
    y_pred = ["alisa", UNKNOWN, "alisa", UNKNOWN, "chuzh"]
    m = open_set_metrics(y_true, y_pred, CLASSES)
    assert m["false_accept_rate"] == 0.5    # 1 of 2 unknowns accepted as known
    assert m["false_reject_rate"] == 1 / 3  # 1 of 3 knowns rejected
    assert m["unknown_recall"] == 0.5
    assert m["per_class"]["chuzh"]["recall"] == 1.0


def test_feeder_false_accepts_per_1000():
    truths = [UNKNOWN, "alisa", UNKNOWN, "chuzh"]
    decisions = ["alisa", "alisa", UNKNOWN, "chuzh"]  # opened for 1 unknown
    assert feeder_false_accepts_per_1000_visits(truths, decisions) == 250.0
