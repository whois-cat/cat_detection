"""Time-episode clustering mode for the cold-start review manifest.

Criterion: detections with a gap < EPISODE_GAP land in one cluster, a gap >= the
threshold splits into separate clusters, and clusters never mix cameras. The
manifest schema (items / clusters with item_indices, representatives, size) is
unchanged so the review UI opens exactly as before.
"""
import json
import sys

from training.build_cluster_manifest import build_time_episodes, dedupe_nearby


def _items(rows):
    return [{"crop_id": f"{c}:{w}", "camera": c, "wall_ms": w} for c, w in rows]


# ---- episode grouping (pure) ----

def test_gap_under_threshold_stays_one_cluster():
    gap_ms = 30_000
    items = _items([("grey", 0), ("grey", 10_000), ("grey", 25_000)])  # 10s, 15s < 30s
    eps = build_time_episodes(items, gap_ms)
    assert len(eps) == 1
    assert sorted(eps[0]) == [0, 1, 2]


def test_gap_at_threshold_splits():
    gap_ms = 30_000
    # exactly 30s gap → split (the >= boundary the review test pins)
    items = _items([("grey", 0), ("grey", 30_000)])
    eps = build_time_episodes(items, gap_ms)
    assert len(eps) == 2


def test_gap_over_threshold_splits():
    gap_ms = 30_000
    items = _items([("grey", 0), ("grey", 5_000), ("grey", 90_000), ("grey", 91_000)])
    eps = build_time_episodes(items, gap_ms)
    assert len(eps) == 2
    # chronological membership preserved within each episode
    assert sorted(eps[0]) == [0, 1]
    assert sorted(eps[1]) == [2, 3]


def test_episodes_never_cross_cameras():
    gap_ms = 30_000
    # interleaved cameras, all within the window — must NOT merge across cameras
    items = _items([("grey", 0), ("kitchen", 1_000), ("grey", 2_000), ("kitchen", 3_000)])
    eps = build_time_episodes(items, gap_ms)
    assert len(eps) == 2
    for members in eps.values():
        assert len({items[i]["camera"] for i in members}) == 1


def test_dedupe_time_only_drops_in_window_keeps_spaced_and_other_camera():
    items = _items([("grey", 0), ("grey", 2_000), ("grey", 20_000), ("kitchen", 1_000)])
    kept, x, dropped = dedupe_nearby(items, None, window_sec=15.0, threshold=0.0)
    assert x is None                       # no embeddings round-trips through
    assert dropped == 1                     # grey@2s collapses into grey@0
    ids = {it["crop_id"] for it in kept}
    assert ids == {"grey:0", "grey:20000", "kitchen:1000"}


# ---- end-to-end main() in time mode (no recordings / torch needed) ----

def test_time_mode_manifest_end_to_end(tmp_path, monkeypatch):
    from storage import init_db, insert_event  # detector storage (on sys.path)
    import training.build_cluster_manifest as bcm

    db = tmp_path / "events.db"
    conn = init_db(db)

    def ev(camera, wall_ms):
        insert_event(
            conn, camera_id=camera, model="yolov8n", wall_ms=wall_ms,
            pts=None, tb_num=None, tb_den=None, media_t=None,
            frame_w=320, frame_h=240, rotate_deg=0, cat=None, cat_score=None,
            box_x=10, box_y=20, box_w=40, box_h=40, score=0.9,
        )

    for w in (1_000, 3_000, 5_000):   # grey episode A
        ev("grey", w)
    for w in (70_000, 71_000):        # grey episode B (65s gap → split)
        ev("grey", w)
    for w in (2_000, 4_000):          # kitchen episode (separate camera)
        ev("kitchen", w)
    conn.close()

    out = tmp_path / "clusters.json"
    argv = [
        "prog", "--db", str(db), "--recordings", str(tmp_path / "rec"),
        "--out", str(out), "--mode", "time", "--episode-gap-sec", "30",
        "--no-ignore-config", "--dedupe-window-sec", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bcm.main()

    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["params"]["mode"] == "time"
    assert man["params"]["embedding"] == "time-episode"
    assert man["params"]["episode_gap_sec"] == 30.0
    # 3 episodes: grey(1-5s), grey(70-71s), kitchen(2-4s).
    assert len(man["clusters"]) == 3
    items = man["items"]
    for c in man["clusters"]:
        # schema unchanged → review UI opens as before
        assert {"cluster_id", "size", "item_indices", "representatives"} <= c.keys()
        assert c["size"] == len(c["item_indices"])
        assert len({items[i]["camera"] for i in c["item_indices"]}) == 1  # one camera per episode
    # No embeddings stored in time mode.
    assert man["params"]["stored_embeddings"] is False
    assert all("embedding" not in it for it in items)
