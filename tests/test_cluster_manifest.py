import numpy as np

from training.build_cluster_manifest import (
    annotate_review_metadata,
    compact_items_to_clusters,
    thin_cluster_members,
)


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


def test_review_metadata_groups_duplicates_without_dropping_raw_items():
    items = [
        {
            "crop_id": f"grey:{i}",
            "camera": "grey",
            "wall_ms": i * 1_000,
            "score": 0.9,
            "distance": 0.01 * i,
            "box": {"x": 20, "y": 20, "w": 40, "h": 40},
        }
        for i in range(6)
    ]
    x = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))

    stats = annotate_review_metadata(
        items,
        {0: list(range(6))},
        x,
        duplicate_window_sec=10,
        duplicate_threshold=0.99,
        max_visible_per_cluster=4,
        seed=1,
    )

    assert len(items) == 6
    assert stats["duplicate_crops"] == 5
    assert {item["duplicate_group_id"] for item in items} == {"0:0"}
    assert items[0]["is_duplicate"] is False
    assert all(item["is_duplicate"] for item in items[1:])
    assert sum(1 for item in items if item["review_visible"]) < len(items)
    assert items[0]["representative_rank"] == 0


def test_review_metadata_suspicious_order_is_deterministic_and_keeps_outlier():
    items = [
        {
            "crop_id": "grey:center",
            "camera": "grey",
            "wall_ms": 1_000,
            "score": 0.95,
            "distance": 0.01,
            "box": {"x": 20, "y": 20, "w": 40, "h": 40},
        },
        {
            "crop_id": "grey:low-score",
            "camera": "grey",
            "wall_ms": 2_000,
            "score": 0.55,
            "distance": 0.02,
            "box": {"x": 20, "y": 20, "w": 40, "h": 40},
        },
        {
            "crop_id": "grey:outlier",
            "camera": "grey",
            "wall_ms": 3_000,
            "score": 0.95,
            "distance": 2.0,
            "box": {"x": 0, "y": 20, "w": 40, "h": 40},
        },
    ]

    stats1 = annotate_review_metadata(
        items,
        {0: [0, 1, 2]},
        None,
        duplicate_window_sec=0,
        duplicate_threshold=0.0,
        max_visible_per_cluster=2,
        seed=9,
    )
    scores1 = [(it["crop_id"], it["suspicious_score"], it["suspicious_reasons"]) for it in items]

    items2 = [dict(item, suspicious_score=0.0, suspicious_reasons=[]) for item in items]
    stats2 = annotate_review_metadata(
        items2,
        {0: [0, 1, 2]},
        None,
        duplicate_window_sec=0,
        duplicate_threshold=0.0,
        max_visible_per_cluster=2,
        seed=9,
    )
    scores2 = [(it["crop_id"], it["suspicious_score"], it["suspicious_reasons"]) for it in items2]

    assert stats1 == stats2
    assert scores1 == scores2
    by_id = {it["crop_id"]: it for it in items}
    assert "far_from_centroid" in by_id["grey:outlier"]["suspicious_reasons"]
    assert "bbox_edge" in by_id["grey:outlier"]["suspicious_reasons"]
    assert "low_detector_score" in by_id["grey:low-score"]["suspicious_reasons"]
    ordered = sorted(items, key=lambda it: (-it["suspicious_score"], it["crop_id"]))
    assert ordered[0]["crop_id"] == "grey:outlier"
