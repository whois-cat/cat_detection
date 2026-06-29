"""Tests for volume-mounted classifier promote/rollback (versioned + symlinks).

Export is injected (fake) so these run without OpenVINO; validate_checkpoint
needs torch (present in the test env).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import promote_classifier as P  # noqa: E402

torch = pytest.importorskip("torch")


def _write_ckpt(path: Path, names=("cat_a", "cat_b"), *, num_classes=None, drop=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": {"w": torch.zeros(1)},
        "class_names": list(names),
        "num_classes": len(names) if num_classes is None else num_classes,
    }
    for k in (drop or []):
        ckpt.pop(k, None)
    torch.save(ckpt, path)
    return path


def _trained(tmp, stamp, names=("cat_a", "cat_b")):
    return _write_ckpt(tmp / "models" / "trained" / stamp / "cat_classifier.pt", names)


def _fake_export(src_pt, out_dir):
    """Stand in for the OpenVINO export: write the runtime artifact files, with
    classes.json mirroring the checkpoint's class_names so promote can validate."""
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    out_dir = Path(out_dir)
    (out_dir / "cat_classifier.xml").write_text("<net/>", encoding="utf-8")
    (out_dir / "cat_classifier.bin").write_bytes(b"\x00")
    (out_dir / "classes.json").write_text(json.dumps(ckpt["class_names"]), encoding="utf-8")


def _promote(tmp, src, **kw):
    return P.promote(src, trained_root=tmp / "models" / "trained",
                     classifier_root=tmp / "models" / "classifier",
                     export_fn=_fake_export, **kw)


def _active_version(root):
    return Path(os.readlink(root / "models" / "classifier" / "current")).name


# ---- checkpoint validation --------------------------------------------------

def test_validate_accepts_valid(tmp_path):
    assert P.validate_checkpoint(_write_ckpt(tmp_path / "c.pt", ("x", "y", "z")))["num_classes"] == 3


def test_validate_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        P.validate_checkpoint(tmp_path / "nope.pt")


@pytest.mark.parametrize("kwargs,msg", [
    ({"drop": ["state_dict"]}, "state_dict"),
    ({"drop": ["class_names"]}, "class_names"),
    ({"names": []}, "non-empty"),
    ({"names": ("a", "b"), "num_classes": 3}, "num_classes"),
])
def test_validate_refuses_malformed(tmp_path, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        P.validate_checkpoint(_write_ckpt(tmp_path / "bad.pt", **kwargs))


# ---- promote ----------------------------------------------------------------

def test_promote_explicit_src_writes_version_and_current(tmp_path):
    src = _trained(tmp_path, "20260101-000000")
    res = _promote(tmp_path, src)
    vdir = tmp_path / "models" / "classifier" / "versions" / "20260101-000000"
    assert vdir.is_dir()
    for f in ("cat_classifier.xml", "cat_classifier.bin", "classes.json", "metadata.json"):
        assert (vdir / f).exists()
    assert _active_version(tmp_path) == "20260101-000000"
    assert res["previous_version"] is None


def test_promote_default_picks_latest(tmp_path):
    old = _trained(tmp_path, "20260101-000000", ("cat_a",))
    new = _trained(tmp_path, "20260201-000000", ("cat_a", "cat_b"))
    os.utime(old, (1_000, 1_000)); os.utime(new, (2_000, 2_000))
    res = _promote(tmp_path, None)
    assert res["version_id"] == "20260201-000000"
    assert _active_version(tmp_path) == "20260201-000000"


def test_promote_no_checkpoints_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="No trained checkpoints found"):
        _promote(tmp_path, None)


def test_promote_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _promote(tmp_path, tmp_path / "nope.pt")


def test_promote_refuses_malformed_source(tmp_path):
    bad = _write_ckpt(tmp_path / "models" / "trained" / "s" / "cat_classifier.pt", drop=["class_names"])
    with pytest.raises(ValueError):
        _promote(tmp_path, bad)
    assert not (tmp_path / "models" / "classifier" / "current").exists()


def test_promote_sets_previous_and_keeps_old_version(tmp_path):
    a = _trained(tmp_path, "20260101-000000", ("cat_a",))
    b = _trained(tmp_path, "20260201-000000", ("cat_a", "cat_b"))
    _promote(tmp_path, a)
    res = _promote(tmp_path, b)
    assert _active_version(tmp_path) == "20260201-000000"
    assert res["previous_version"] == "20260101-000000"
    prev = tmp_path / "models" / "classifier" / "previous"
    assert Path(os.readlink(prev)).name == "20260101-000000"
    # Old version is NOT deleted.
    assert (tmp_path / "models" / "classifier" / "versions" / "20260101-000000").is_dir()


def test_promote_current_is_relative_symlink(tmp_path):
    _promote(tmp_path, _trained(tmp_path, "20260101-000000"))
    link = tmp_path / "models" / "classifier" / "current"
    assert link.is_symlink()
    assert os.readlink(link) == "versions/20260101-000000"   # relative → works in-container


# ---- rollback ---------------------------------------------------------------

def test_rollback_default_switches_to_previous(tmp_path):
    _promote(tmp_path, _trained(tmp_path, "20260101-000000", ("cat_a",)))
    _promote(tmp_path, _trained(tmp_path, "20260201-000000", ("cat_a", "cat_b")))
    P.rollback(None, classifier_root=tmp_path / "models" / "classifier")
    assert _active_version(tmp_path) == "20260101-000000"


def test_rollback_no_previous_fails_clearly(tmp_path):
    _promote(tmp_path, _trained(tmp_path, "20260101-000000"))   # only one version
    with pytest.raises(SystemExit, match="No previous version"):
        P.rollback(None, classifier_root=tmp_path / "models" / "classifier")


def test_rollback_explicit_version(tmp_path):
    _promote(tmp_path, _trained(tmp_path, "20260101-000000", ("cat_a",)))
    _promote(tmp_path, _trained(tmp_path, "20260201-000000", ("cat_a", "cat_b")))
    P.rollback("20260101-000000", classifier_root=tmp_path / "models" / "classifier")
    assert _active_version(tmp_path) == "20260101-000000"


def test_rollback_refuses_unknown_version(tmp_path):
    _promote(tmp_path, _trained(tmp_path, "20260101-000000"))
    with pytest.raises(SystemExit, match="version not found"):
        P.rollback("nope", classifier_root=tmp_path / "models" / "classifier")


# ---- restart service discovery ----------------------------------------------

def test_select_detector_services_excludes_infra():
    names = ["mediamtx", "pruner", "indexer", "webui", "mlflow",
             "detector-grey", "detector-beige", "feeder-feeder1"]
    assert P.select_detector_services(names) == ["detector-grey", "detector-beige"]


def test_select_detector_services_handles_generated_names():
    names = ["detector-cam_a", "detector-cam_b", "detector-kitchen_2"]
    assert P.select_detector_services(names) == names
