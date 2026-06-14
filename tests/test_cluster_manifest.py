from training.build_cluster_manifest import compact_items_to_clusters, thin_cluster_members


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
