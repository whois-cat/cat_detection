"""The OpenVINO parity gate judges recall at the runtime confidence threshold."""
from training.yolo_export import gate_summary, quality_gate


def _result(*, official_recall, fixed_recall, fp, map50_95=0.88):
    return {
        "official": {"precision": 0.99, "recall": official_recall, "map50": 0.99,
                     "map50_95": map50_95},
        "fixed_confidence": {"overall": {"recall": fixed_recall, "fp": fp, "fn": 0,
                                         "empty_image_errors": 0}},
    }


def _gate(base, cand):
    return quality_gate(base, cand, max_map_drop=0.01, max_recall_drop=0.01, max_fp_increase=0)


def test_max_f1_recall_shift_alone_does_not_fail():
    # The real INT8 export: Ultralytics' max-F1 recall 0.972 -> 0.944, but at the
    # runtime threshold recall rose and false positives fell.
    gate = _gate(_result(official_recall=0.9722, fixed_recall=0.9722, fp=6),
                 _result(official_recall=0.9444, fixed_recall=0.9792, fp=4))
    assert gate["passed"], gate["checks"]


def test_runtime_recall_drop_fails():
    gate = _gate(_result(official_recall=0.97, fixed_recall=0.97, fp=4),
                 _result(official_recall=0.97, fixed_recall=0.94, fp=4))
    assert not gate["checks"]["recall_drop_within_limit"]


def test_map_drop_and_extra_false_positives_still_fail():
    base = _result(official_recall=0.97, fixed_recall=0.97, fp=4)
    assert not _gate(base, _result(official_recall=0.97, fixed_recall=0.97, fp=4,
                                   map50_95=0.85))["passed"]
    assert not _gate(base, _result(official_recall=0.97, fixed_recall=0.97, fp=5))["passed"]


def test_summary_names_the_failing_check():
    base = _result(official_recall=0.97, fixed_recall=0.97, fp=4)
    cand = _result(official_recall=0.97, fixed_recall=0.97, fp=7)
    lines = gate_summary(base, cand, _gate(base, cand))
    failed = [line for line in lines if "FAIL" in line]
    assert len(failed) == 1 and "false positives 4 -> 7" in failed[0]
