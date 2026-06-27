"""Tests for the cluster-manifest validator."""
from __future__ import annotations

import json

from training.validate_cluster_manifest import (
    FORBIDDEN_ITEM_FIELDS,
    main,
    validate,
)


def _manifest(n_items=4, clusters=None):
    items = [{"crop_id": f"c{i}", "camera": "grey", "wall_ms": i} for i in range(n_items)]
    if clusters is None:
        clusters = [
            {"cluster_id": 0, "size": 2, "item_indices": [0, 1]},
            {"cluster_id": 1, "size": 2, "item_indices": [2, 3]},
        ]
    return {"items": items, "clusters": clusters}


def test_valid_manifest_passes_with_stats():
    errors, stats, warnings = validate(_manifest(), max_cluster_size=16)
    assert errors == []
    assert stats["clusters"] == 2
    assert stats["items"] == 4
    assert stats["max_cluster_size_seen"] == 2
    assert stats["avg_cluster_size"] == 2.0
    assert stats["suspicious_fields"] == {}
    assert warnings == []


def test_missing_items_or_clusters_fails():
    errors, _, _ = validate({"clusters": []}, max_cluster_size=16)
    assert any("items" in e for e in errors)
    errors, _, _ = validate({"items": []}, max_cluster_size=16)
    assert any("clusters" in e for e in errors)


def test_cluster_over_cap_fails():
    m = _manifest(n_items=20, clusters=[{"cluster_id": 0, "item_indices": list(range(20))}])
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("exceed --max-cluster-size" in e for e in errors)


def test_out_of_range_index_fails():
    m = _manifest(clusters=[{"cluster_id": 0, "item_indices": [0, 99]}])
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("out of range" in e for e in errors)


def test_duplicate_index_across_clusters_fails():
    m = _manifest(clusters=[
        {"cluster_id": 0, "item_indices": [0, 1]},
        {"cluster_id": 1, "item_indices": [1, 2]},   # index 1 reused
    ])
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("duplicated across clusters" in e for e in errors)


def test_size_mismatch_fails():
    m = _manifest(clusters=[{"cluster_id": 0, "size": 5, "item_indices": [0, 1]}])
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("size != len(item_indices)" in e for e in errors)


def test_hidden_fields_on_items_fail():
    for field in FORBIDDEN_ITEM_FIELDS:
        m = _manifest()
        m["items"][0][field] = True
        errors, stats, _ = validate(m, max_cluster_size=16)
        assert any("forbidden hidden/collapsed" in e for e in errors), field
        assert stats["suspicious_fields"].get(field) == 1


def test_unreferenced_item_is_warning_not_error():
    m = _manifest(n_items=4, clusters=[{"cluster_id": 0, "item_indices": [0, 1]}])
    errors, _, warnings = validate(m, max_cluster_size=16)
    assert errors == []
    assert any("not referenced" in w for w in warnings)


def test_bool_index_is_rejected():
    # True is an int subclass; must not be accepted as item index 1.
    m = _manifest(clusters=[{"cluster_id": 0, "item_indices": [0, True]}])
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("out of range" in e for e in errors)


# --- optional, config-driven label validation (no fixed class list) ----------

def test_no_label_assumptions_arbitrary_names_pass():
    # Any names, any count — valid without --labels.
    m = _manifest()
    m["labels"] = ["whiskers", "mittens", "id_7"]
    m["items"][0]["label"] = "whiskers"
    errors, stats, _ = validate(m, max_cluster_size=16)
    assert errors == []
    assert "whiskers" in stats["distinct_labels"]


def test_manifest_labels_must_be_strings():
    m = _manifest()
    m["labels"] = ["ok", 123]
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("'labels' must be a list of strings" in e for e in errors)


def test_labels_checked_against_configured_list_when_passed():
    m = _manifest()
    m["labels"] = ["cat_a", "cat_x"]
    m["items"][0]["label"] = "cat_z"
    errors, _, _ = validate(m, max_cluster_size=16, labels=["cat_a", "cat_b"])
    assert any("manifest 'labels' not in configured list" in e for e in errors)
    assert any("item label(s) not in configured list" in e for e in errors)


def test_item_label_optional_and_unlabeled_states_allowed():
    m = _manifest()
    m["items"][0]["label"] = "unknown"   # service state
    m["items"][1]["label"] = ""          # unlabeled
    m["items"][2]["label"] = None        # unlabeled
    # item 3 has no label field at all — fine
    errors, _, _ = validate(m, max_cluster_size=16, labels=["cat_a"])
    assert errors == []


def test_item_label_must_be_string():
    m = _manifest()
    m["items"][0]["label"] = 5
    errors, _, _ = validate(m, max_cluster_size=16)
    assert any("not strings" in e for e in errors)


def test_no_labels_anywhere_is_valid():
    # Manifest with no label fields at all → no label errors.
    errors, stats, _ = validate(_manifest(), max_cluster_size=16, labels=["cat_a"])
    assert errors == []
    assert stats["distinct_labels"] == []


# --- CLI exit codes ----------------------------------------------------------

def test_main_returns_0_on_valid(tmp_path):
    p = tmp_path / "clusters.json"
    p.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert main(["--manifest", str(p), "--max-cluster-size", "16"]) == 0


def test_main_returns_1_on_invalid(tmp_path):
    p = tmp_path / "clusters.json"
    p.write_text(json.dumps(_manifest(clusters=[{"cluster_id": 0, "item_indices": [0, 99]}])),
                 encoding="utf-8")
    assert main(["--manifest", str(p)]) == 1


def test_main_returns_2_on_missing_file(tmp_path):
    assert main(["--manifest", str(tmp_path / "nope.json")]) == 2


def test_main_returns_2_on_bad_json(tmp_path):
    p = tmp_path / "clusters.json"
    p.write_text("{not json", encoding="utf-8")
    assert main(["--manifest", str(p)]) == 2
