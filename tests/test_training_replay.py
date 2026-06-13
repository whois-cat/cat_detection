import json

import numpy as np

from training.replay import load_replay_set


def test_load_replay_set_from_npz_manifest(tmp_path):
    crop_dir = tmp_path / "crops" / "alisa"
    crop_dir.mkdir(parents=True)
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    np.savez_compressed(crop_dir / "1.npz", image=image)
    row = {
        "src_event_key": 1,
        "label": "alisa",
        "camera": "grey",
        "wall_ms": 1234,
        "path": "crops/alisa/1.npz",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_replay_set(tmp_path)

    assert len(loaded) == 1
    img, meta = loaded[0]
    assert img.shape == image.shape
    assert meta.label == "alisa"
    assert meta.camera == "replay:grey"
    assert meta.wall_ms == 1234
