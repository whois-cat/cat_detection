"""Cluster-review app: idempotent re-split, recursive child split, per-crop
labels, and the live per-class crop counter.

The app loads its manifest + opens the reviews DB at import time, so we point the
env at a synthetic manifest / temp DB and import it via importlib before creating
a TestClient. Recordings are never touched here (split + counts + per-crop label
work purely off embeddings/metadata).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from starlette.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    # Two base clusters. Cluster 0 has six crops in two well-separated embedding
    # lobes (so kmeans(2) splits cleanly); cluster 1 has three crops.
    items = []

    def add(crop_id, key, emb, **extra):
        items.append({
            "crop_id": crop_id,
            "src_event_key": key,
            "camera": "grey",
            "wall_ms": 1000 + key,
            "score": 0.9,
            "rotate_deg": 0,
            "box": {"x": 0, "y": 0, "w": 10, "h": 10},
            "embedding": emb,
            **extra,
        })

    # cluster 0 members: indices 0..5  (lobe A near [0,0], lobe B near [9,9])
    for k in range(3):
        add(
            f"grey:{k}",
            k,
            [0.0 + 0.01 * k, 0.0],
            duplicate_group_id="0:a",
            is_duplicate=k > 0,
            review_visible=k == 0,
            suspicious_score=0.1 * k,
            suspicious_reasons=["low_detector_score"] if k == 2 else [],
        )
    for k in range(3, 6):
        add(
            f"grey:{k}",
            k,
            [9.0, 9.0 + 0.01 * k],
            duplicate_group_id="0:b",
            is_duplicate=k > 3,
            review_visible=k == 3,
            suspicious_score=0.9 if k == 5 else 0.0,
            suspicious_reasons=["far_from_centroid"] if k == 5 else [],
        )
    # cluster 1 members: indices 6..8
    for k in range(6, 9):
        add(f"grey:{k}", k, [4.0, 4.0])

    clusters = [
        {"cluster_id": 0, "size": 6, "item_indices": [0, 1, 2, 3, 4, 5],
         "representatives": [it["crop_id"] for it in items[:6]]},
        {"cluster_id": 1, "size": 3, "item_indices": [6, 7, 8],
         "representatives": [it["crop_id"] for it in items[6:9]]},
    ]
    return {
        "labels": ["alisa", "chuzh", "ellie", "felisis"],
        "items": items,
        "clusters": clusters,
        "params": {"min_score": 0.7, "pad_frac": 0.15},
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    man_path = tmp_path / "clusters.json"
    man_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    monkeypatch.setenv("CLUSTER_MANIFEST", str(man_path))
    monkeypatch.setenv("REVIEW_DB", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("RECORDINGS_ROOT", str(tmp_path / "recordings"))

    # Fresh import each test so module-level manifest/DB pick up this tmp env.
    spec = importlib.util.spec_from_file_location(
        f"cluster_app_{tmp_path.name}", REPO / "review" / "cluster_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with TestClient(mod.app) as c:
        c.mod = mod
        yield c


def _children_of(client, parent_id):
    q = client.get("/api/clusters").json()["queue"]
    return [c for c in q if c["parent_id"] == parent_id]


def test_split_is_idempotent_resplit_replaces_children_not_409(client):
    r1 = client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    assert r1.status_code == 200, r1.text
    assert r1.json()["replaced"] == 0
    assert len(_children_of(client, 0)) == 2

    # Re-split the SAME cluster: must succeed (NOT 409), replace the old children,
    # and NOT accumulate — the partition is replaced, not added to.
    r2 = client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    assert r2.status_code == 200, r2.text
    assert r2.json()["replaced"] == 2
    assert len(_children_of(client, 0)) == 2

    # A third re-split still replaces cleanly (regression guard for the 409 path).
    r3 = client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    assert r3.status_code == 200, r3.text
    assert r3.json()["replaced"] == 2
    assert len(_children_of(client, 0)) == 2


def test_resplit_into_more_parts(client):
    client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    r = client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 3})
    assert r.status_code == 200, r.text
    assert r.json()["replaced"] == 2
    assert len(_children_of(client, 0)) == 3


def test_split_child_recursively(client):
    client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    child = _children_of(client, 0)[0]
    child_id = child["cluster_id"]
    assert child["size"] >= 2

    r = client.post("/api/cluster_split", json={"cluster_id": child_id, "parts": 2})
    assert r.status_code == 200, r.text
    grandkids = _children_of(client, child_id)
    assert len(grandkids) == 2
    # Grandchildren appear in the flat queue with the child as their parent.
    assert all(g["parent_id"] == child_id for g in grandkids)


def test_resplit_removes_old_child_cluster_reviews(client):
    client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    old_ids = [c["cluster_id"] for c in _children_of(client, 0)]
    # Label one old child so it has a cluster_reviews row + per-crop reviews.
    client.post("/api/cluster_review", json={"cluster_id": old_ids[0], "label": "alisa"})

    client.post("/api/cluster_split", json={"cluster_id": 0, "parts": 2})
    conn = client.mod._conn
    leftover = conn.execute(
        "SELECT COUNT(*) FROM cluster_reviews WHERE cluster_id=?", (old_ids[0],)
    ).fetchone()[0]
    assert leftover == 0  # stale child status cleaned up
    # And the per-crop reviews written under cluster 0's members are cleared.
    assert client.get("/api/counts").json()["total"] == 0


def test_counts_endpoint_groups_crops_by_label(client):
    # Label whole clusters: cluster 1 (3 crops) → chuzh, cluster 0 (6) → alisa.
    client.post("/api/cluster_review", json={"cluster_id": 1, "label": "chuzh"})
    client.post("/api/cluster_review", json={"cluster_id": 0, "label": "alisa"})
    data = client.get("/api/counts").json()
    assert data["counts"]["alisa"] == 6
    assert data["counts"]["chuzh"] == 3
    assert data["total"] == 9
    assert data["order"][:4] == ["alisa", "chuzh", "ellie", "felisis"]


def test_per_crop_label_writes_one_row_each(client):
    crop_ids = ["grey:0", "grey:3", "grey:5"]
    r = client.post("/api/crop_label", json={"crop_ids": crop_ids, "label": "ellie"})
    assert r.status_code == 200, r.text
    assert r.json()["labeled"] == 3
    counts = client.get("/api/counts").json()["counts"]
    assert counts["ellie"] == 3

    # Re-labelling the same crop updates in place (no duplicate rows).
    client.post("/api/crop_label", json={"crop_ids": ["grey:0"], "label": "alisa"})
    counts = client.get("/api/counts").json()["counts"]
    assert counts["ellie"] == 2
    assert counts["alisa"] == 1


def test_per_crop_label_rejects_unknown_crop_and_label(client):
    assert client.post("/api/crop_label",
                       json={"crop_ids": ["nope:1"], "label": "alisa"}).status_code == 404
    assert client.post("/api/crop_label",
                       json={"crop_ids": ["grey:0"], "label": "banana"}).status_code == 400
    assert client.post("/api/crop_label",
                       json={"crop_ids": [], "label": "alisa"}).status_code == 400


def _review_rows(client):
    return client.mod._conn.execute(
        "SELECT crop_id, label FROM reviews ORDER BY src_event_key"
    ).fetchall()


def test_cluster_label_with_exceptions_labels_selected_and_rest(client):
    r = client.post("/api/cluster_label_with_exceptions", json={
        "cluster_id": 0,
        "majority_label": "felisis",
        "exceptions": {"discard": ["grey:0", "grey:5"], "chuzh": ["grey:3"]},
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total_cluster_crops"] == 6
    assert data["majority_labeled_count"] == 3
    assert data["exception_counts"] == {"discard": 2, "chuzh": 1}

    rows = dict(_review_rows(client))
    assert rows["grey:0"] == "discard"
    assert rows["grey:5"] == "discard"
    assert rows["grey:3"] == "chuzh"
    assert rows["grey:1"] == "felisis"
    assert rows["grey:2"] == "felisis"
    assert rows["grey:4"] == "felisis"
    status = client.get("/api/clusters").json()["queue"][0]["status"]
    assert status["status"] == "labeled"
    assert status["label"] == "felisis"


def test_cluster_label_with_exceptions_rejects_invalid_label_and_camera_names(client):
    assert client.post("/api/cluster_label_with_exceptions", json={
        "cluster_id": 0,
        "majority_label": "beige",
        "exceptions": {},
    }).status_code == 400
    assert client.post("/api/cluster_label_with_exceptions", json={
        "cluster_id": 0,
        "majority_label": "alisa",
        "exceptions": {"grey": ["grey:0"]},
    }).status_code == 400
    assert client.mod._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_cluster_label_with_exceptions_rejects_crop_from_other_cluster(client):
    r = client.post("/api/cluster_label_with_exceptions", json={
        "cluster_id": 0,
        "majority_label": "alisa",
        "exceptions": {"discard": ["grey:6"]},
    })
    assert r.status_code == 400
    assert "does not belong" in r.text
    assert client.mod._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0


def test_cluster_label_with_exceptions_rolls_back_on_write_failure(client, monkeypatch):
    calls = {"n": 0}
    original = client.mod._write_review_label

    def flaky(conn, item, label, now):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(conn, item, label, now)

    monkeypatch.setattr(client.mod, "_write_review_label", flaky)
    with pytest.raises(RuntimeError):
        client.post("/api/cluster_label_with_exceptions", json={
            "cluster_id": 0,
            "majority_label": "alisa",
            "exceptions": {"discard": ["grey:0"]},
        })
    assert client.mod._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    assert client.mod._conn.execute("SELECT COUNT(*) FROM cluster_reviews").fetchone()[0] == 0


def test_items_hide_duplicates_by_default_and_suspicious_mode_orders(client):
    data = client.get("/api/cluster/0/items?mode=representative").json()
    assert data["total"] == 6
    assert data["hidden_duplicate_count"] == 4
    assert [it["crop_id"] for it in data["items"]] == ["grey:0", "grey:3"]

    with_dups = client.get(
        "/api/cluster/0/items?mode=representative&show_duplicates=true"
    ).json()
    assert len(with_dups["items"]) == 6

    suspicious = client.get(
        "/api/cluster/0/items?mode=suspicious&show_duplicates=true"
    ).json()["items"]
    assert suspicious[0]["crop_id"] == "grey:5"
    assert suspicious[0]["suspicious_reasons"] == ["far_from_centroid"]


def test_hidden_duplicate_can_be_overridden_as_exception(client):
    r = client.post("/api/cluster_label_with_exceptions", json={
        "cluster_id": 0,
        "majority_label": "alisa",
        "exceptions": {"discard": ["grey:2"]},
    })
    assert r.status_code == 200, r.text
    rows = dict(_review_rows(client))
    assert rows["grey:2"] == "discard"
    assert rows["grey:1"] == "alisa"
