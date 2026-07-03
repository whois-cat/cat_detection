"""DINOv2 embedding backend for cold-start cluster building (offline only).

These tests cover the argument surface and the backend-selection logic without
requiring transformers/torch or any model weights to be installed:

  * ``--embedding dinov2`` is accepted (and bad values rejected) by parsing;
  * ``--mode time`` never touches the embedding backend (DINO must not load);
  * missing DINO deps + an explicit ``--embedding dinov2`` fail with a clear,
    actionable error;
  * ``--embedding auto`` degrades gracefully when DINO is unavailable.
"""
import json
import sys

import pytest

import training.build_cluster_manifest as bcm
from training.build_cluster_manifest import build_arg_parser, build_extractor


_REQUIRED = ["--db", "db.sqlite", "--recordings", "rec", "--out", "clusters.json"]


# ---- argument parsing ----

def test_embedding_dinov2_is_accepted():
    args = build_arg_parser().parse_args(
        _REQUIRED + ["--mode", "embedding", "--embedding", "dinov2"])
    assert args.embedding == "dinov2"
    assert args.mode == "embedding"


def test_embedding_choices_still_include_legacy_backends():
    for backend in ("auto", "visual", "efficientnet", "dinov2"):
        args = build_arg_parser().parse_args(_REQUIRED + ["--embedding", backend])
        assert args.embedding == backend


def test_unknown_embedding_is_rejected():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(_REQUIRED + ["--embedding", "clip"])


# ---- backend selection (no torch/transformers needed) ----

def _boom_missing_deps(*_a, **_k):
    raise RuntimeError(
        "DINOv2 backend requires transformers and torch. "
        "Install optional labeling dependencies."
    )


def test_explicit_dinov2_missing_deps_gives_clear_error(monkeypatch):
    monkeypatch.setattr(bcm, "DinoV2Extractor", _boom_missing_deps)
    with pytest.raises(SystemExit) as exc:
        build_extractor("dinov2", batch_size=8, allow_download=False)
    msg = str(exc.value)
    assert "dinov2 embeddings unavailable" in msg
    assert "requires transformers and torch" in msg


def test_auto_falls_back_when_dino_and_efficientnet_missing(monkeypatch, capsys):
    # DINO first, then EfficientNet both unavailable → deterministic visual.
    monkeypatch.setattr(bcm, "DinoV2Extractor", _boom_missing_deps)
    monkeypatch.setattr(
        bcm, "EfficientNetExtractor",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no torch")))
    extractor = build_extractor("auto", batch_size=8, allow_download=False)
    assert extractor.name == "visual"
    warn = capsys.readouterr().out
    assert "dinov2 unavailable" in warn
    assert "falling back to visual" in warn


def test_auto_prefers_dino_when_available(monkeypatch):
    class _FakeDino:
        name = "dinov2:dinov2-small"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(bcm, "DinoV2Extractor", _FakeDino)
    extractor = build_extractor("auto", batch_size=8, allow_download=False)
    assert extractor.name == "dinov2:dinov2-small"


# ---- time mode must not initialize any embedding backend ----

def test_time_mode_does_not_build_any_extractor(tmp_path, monkeypatch):
    from storage import init_db, insert_event  # detector storage (on sys.path)

    def _fail_build(*_a, **_k):
        raise AssertionError("time mode must not build an embedding extractor")

    monkeypatch.setattr(bcm, "build_extractor", _fail_build)
    monkeypatch.setattr(
        bcm, "DinoV2Extractor",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("time mode must not load DINOv2")))

    db = tmp_path / "events.db"
    conn = init_db(db)
    for w in (1_000, 3_000, 5_000):
        insert_event(
            conn, camera_id="grey", model="yolov8n", wall_ms=w,
            pts=None, tb_num=None, tb_den=None, media_t=None,
            frame_w=320, frame_h=240, rotate_deg=0, cat=None, cat_score=None,
            box_x=10, box_y=20, box_w=40, box_h=40, score=0.9,
        )
    conn.close()

    out = tmp_path / "clusters.json"
    argv = [
        "prog", "--db", str(db), "--recordings", str(tmp_path / "rec"),
        "--out", str(out), "--mode", "time", "--episode-gap-sec", "30",
        "--no-ignore-config", "--dedupe-window-sec", "0",
        "--embedding", "dinov2",  # ignored in time mode — must NOT load DINO
    ]
    monkeypatch.setattr(sys, "argv", argv)
    bcm.main()  # would raise if build_extractor / DinoV2Extractor were touched

    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["params"]["mode"] == "time"
    assert man["params"]["embedding"] == "time-episode"
