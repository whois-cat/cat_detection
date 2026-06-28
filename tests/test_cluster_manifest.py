import numpy as np

import pytest

from training.build_cluster_manifest import (
    REVIEW_ONLY_FIELDS,
    _strip_review_fields,
    assert_review_manifest_invariants,
    compact_items_to_clusters,
    select_review_crops,
    thin_cluster_members,
)


def test_invariants_pass_for_compliant_manifest():
    items = [{"crop_id": f"c{i}"} for i in range(3)]
    clusters = [{"cluster_id": 0, "size": 2, "item_indices": [0, 1]},
                {"cluster_id": 1, "size": 1, "item_indices": [2]}]
    assert_review_manifest_invariants(items, clusters, max_per_cluster=16)  # no raise


def test_invariants_reject_over_cap():
    items = [{"crop_id": f"c{i}"} for i in range(5)]
    clusters = [{"cluster_id": 0, "size": 5, "item_indices": [0, 1, 2, 3, 4]}]
    with pytest.raises(SystemExit, match="max-cluster-size"):
        assert_review_manifest_invariants(items, clusters, max_per_cluster=4)


def test_invariants_reject_hidden_item_field():
    items = [{"crop_id": "c0", "review_visible": True}]
    clusters = [{"cluster_id": 0, "size": 1, "item_indices": [0]}]
    with pytest.raises(SystemExit, match="hidden/collapsed"):
        assert_review_manifest_invariants(items, clusters, max_per_cluster=16)


def test_invariants_reject_hidden_cluster_field_and_bad_index():
    items = [{"crop_id": "c0"}]
    with pytest.raises(SystemExit, match="counter"):
        assert_review_manifest_invariants(
            items, [{"cluster_id": 0, "size": 1, "item_indices": [0], "visible_count": 1}], 16)
    with pytest.raises(SystemExit, match="out of range"):
        assert_review_manifest_invariants(
            items, [{"cluster_id": 0, "size": 1, "item_indices": [9]}], 16)


def _distinct_items(n, camera="grey", *, gap_ms=10_000):
    # Spread far apart in time so the dedupe window never collapses them.
    return [
        {
            "crop_id": f"{camera}:{i}",
            "camera": camera,
            "wall_ms": i * gap_ms,
            "score": 0.9,
            "distance": i / 100.0,
            "box": {"x": 20, "y": 20, "w": 40, "h": 40},
        }
        for i in range(n)
    ]


def _full_manifest(items, selected_by_cluster):
    """Mirror main(): build clusters from the selection, compact, strip."""
    clusters = [
        {"cluster_id": cid, "item_indices": sorted(sel)}
        for cid, sel in sorted(selected_by_cluster.items())
        if sel
    ]
    items, clusters = compact_items_to_clusters(items, clusters)
    for item in items:
        _strip_review_fields(item)
    return items, clusters


def test_select_review_crops_hard_cap_per_cluster():
    # 2 clusters of 40 distinct crops each, cap 16 → at most 16 per cluster.
    items = _distinct_items(80)
    members = {0: list(range(40)), 1: list(range(40, 80))}

    selected, stats = select_review_crops(
        items, members, None,
        duplicate_window_sec=0,           # dedupe disabled
        duplicate_threshold=0.0,
        max_per_cluster=16, seed=7,
    )

    assert len(selected[0]) == 16
    assert len(selected[1]) == 16
    assert stats["dropped_dedupe"] == 0
    assert stats["dropped_cap"] == 48          # (40-16) * 2


def test_select_review_crops_keeps_all_when_under_cap():
    items = _distinct_items(5)
    selected, stats = select_review_crops(
        items, {0: list(range(5))}, None,
        duplicate_window_sec=0, duplicate_threshold=0.0,
        max_per_cluster=16, seed=7,
    )
    assert sorted(selected[0]) == [0, 1, 2, 3, 4]
    assert stats == {"dropped_dedupe": 0, "dropped_cap": 0}


def test_select_review_crops_dedupe_drops_near_duplicates():
    # 5 crops, same camera, close in time, identical embeddings → 1 representative.
    items = [
        {"crop_id": f"grey:{i}", "camera": "grey", "wall_ms": i * 1_000,
         "score": 0.9, "distance": 0.0, "box": {"x": 0, "y": 0, "w": 40, "h": 40}}
        for i in range(5)
    ]
    x = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1))

    selected, stats = select_review_crops(
        items, {0: list(range(5))}, x,
        duplicate_window_sec=10, duplicate_threshold=0.99,
        max_per_cluster=16, seed=1,
    )

    assert len(selected[0]) == 1
    assert stats["dropped_dedupe"] == 4
    assert stats["dropped_cap"] == 0


def test_manifest_has_no_hidden_fields_and_compact_indices():
    items = _distinct_items(80)
    members = {0: list(range(40)), 1: list(range(40, 80))}
    selected, _ = select_review_crops(
        items, members, None,
        duplicate_window_sec=0, duplicate_threshold=0.0,
        max_per_cluster=16, seed=7,
    )
    out_items, out_clusters = _full_manifest(items, selected)

    # Hard cap holds across the whole manifest.
    assert len(out_items) <= len(out_clusters) * 16
    assert len(out_items) == 32

    # No hidden/collapsed fields survive on any crop record.
    for it in out_items:
        for field in REVIEW_ONLY_FIELDS:
            assert field not in it

    # No cluster-level hidden counters.
    for c in out_clusters:
        assert "visible_count" not in c
        assert "hidden_duplicate_count" not in c
        assert c["size"] == len(c["item_indices"])

    # Every item index is compact and in range — no out-of-range references.
    n = len(out_items)
    seen = set()
    for c in out_clusters:
        for idx in c["item_indices"]:
            assert 0 <= idx < n
            seen.add(idx)
    assert seen == set(range(n))          # every kept item is referenced exactly once


def test_strip_review_fields_removes_all_legacy_keys():
    item = {"crop_id": "a", "review_visible": True, "is_duplicate": False,
            "duplicate_group_id": "0:0", "representative_rank": 3,
            "sampling_reason": "review_sample", "suspicious_score": 0.4,
            "suspicious_reasons": ["x"], "distance": 0.1}
    _strip_review_fields(item)
    assert item == {"crop_id": "a", "distance": 0.1}


def test_thin_cluster_members_keeps_center_outlier_and_time_spread():
    items = [
        {
            "crop_id": f"grey:{i}",
            "camera": "grey",
            "wall_ms": i * 1_000,
            "distance": i / 100.0,
        }
        for i in range(40)
    ]

    kept, dropped = thin_cluster_members(
        list(range(40)),
        items,
        max_size=12,
        seed=7,
        cluster_id=3,
    )

    assert len(kept) == 12
    assert dropped == 28
    assert 0 in kept
    assert 39 in kept
    assert max(items[i]["wall_ms"] for i in kept) - min(items[i]["wall_ms"] for i in kept) > 20_000


def test_compact_items_to_clusters_remaps_indices_and_drops_unused_items():
    items = [{"crop_id": f"crop-{i}"} for i in range(6)]
    clusters = [
        {"cluster_id": 0, "size": 2, "item_indices": [4, 1], "representatives": []},
        {"cluster_id": 1, "size": 1, "item_indices": [5], "representatives": []},
    ]

    compacted_items, compacted_clusters = compact_items_to_clusters(items, clusters)

    assert [item["crop_id"] for item in compacted_items] == ["crop-1", "crop-4", "crop-5"]
    assert compacted_clusters[0]["item_indices"] == [1, 0]
    assert compacted_clusters[1]["item_indices"] == [2]
    assert compacted_clusters[0]["representatives"] == ["crop-4", "crop-1"]
