"""Tests for the classifier promote/rollback tool (build-time bake flow)."""
from __future__ import annotations

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


# ---- validation -------------------------------------------------------------

def test_validate_accepts_valid_checkpoint(tmp_path):
    p = _write_ckpt(tmp_path / "c.pt", ("x", "y", "z"))
    assert P.validate_checkpoint(p)["num_classes"] == 3


def test_validate_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        P.validate_checkpoint(tmp_path / "nope.pt")


@pytest.mark.parametrize("kwargs,msg", [
    ({"drop": ["state_dict"]}, "state_dict"),
    ({"names": []}, "non-empty"),
    ({"names": ("a", "b"), "num_classes": 3}, "num_classes"),
])
def test_validate_refuses_malformed(tmp_path, kwargs, msg):
    p = _write_ckpt(tmp_path / "bad.pt", **kwargs)
    with pytest.raises(ValueError, match=msg):
        P.validate_checkpoint(p)


# ---- promote ----------------------------------------------------------------

def test_promote_explicit_src(tmp_path):
    src = _trained(tmp_path, "20260101-000000")
    active = tmp_path / "models" / "cat_classifier.pt"
    res = P.promote(src, active=active, trained_root=tmp_path / "models" / "trained",
                    backup_dir=tmp_path / "models" / "backups")
    assert active.exists()
    assert active.read_bytes() == src.read_bytes()
    assert res["backup"] is None        # no previous active


def test_promote_default_picks_latest(tmp_path):
    troot = tmp_path / "models" / "trained"
    old = _trained(tmp_path, "20260101-000000", ("cat_a",))
    new = _trained(tmp_path, "20260201-000000", ("cat_a", "cat_b"))
    import os
    os.utime(old, (1_000, 1_000))       # make "new" unambiguously newer by mtime
    os.utime(new, (2_000, 2_000))
    active = tmp_path / "models" / "cat_classifier.pt"
    res = P.promote(None, active=active, trained_root=troot,
                    backup_dir=tmp_path / "models" / "backups")
    assert res["src"] == new
    assert active.read_bytes() == new.read_bytes()


def test_promote_no_checkpoints_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="No trained checkpoints found"):
        P.promote(None, active=tmp_path / "a.pt",
                  trained_root=tmp_path / "models" / "trained",
                  backup_dir=tmp_path / "models" / "backups")


def test_promote_refuses_missing_source(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        P.promote(tmp_path / "nope.pt", active=tmp_path / "a.pt",
                  backup_dir=tmp_path / "b")


def test_promote_refuses_malformed_source(tmp_path):
    bad = _write_ckpt(tmp_path / "bad.pt", drop=["class_names"])
    with pytest.raises(ValueError):
        P.promote(bad, active=tmp_path / "a.pt", backup_dir=tmp_path / "b")
    assert not (tmp_path / "a.pt").exists()   # active untouched on bad source


def test_promote_backs_up_existing_active(tmp_path):
    active = _write_ckpt(tmp_path / "models" / "cat_classifier.pt", ("old",))
    src = _trained(tmp_path, "20260301-000000", ("new_a", "new_b"))
    backup_dir = tmp_path / "models" / "backups"
    res = P.promote(src, active=active, trained_root=tmp_path / "models" / "trained",
                    backup_dir=backup_dir)
    assert res["backup"] is not None and res["backup"].exists()
    assert P.validate_checkpoint(res["backup"])["class_names"] == ["old"]   # the previous active
    assert active.read_bytes() == src.read_bytes()


# ---- rollback ---------------------------------------------------------------

def test_rollback_default_picks_latest_backup(tmp_path):
    active = _write_ckpt(tmp_path / "models" / "cat_classifier.pt", ("current",))
    backup_dir = tmp_path / "models" / "backups"
    b1 = _write_ckpt(backup_dir / "cat_classifier.20260101-000000-000000.pt", ("v1",))
    b2 = _write_ckpt(backup_dir / "cat_classifier.20260201-000000-000000.pt", ("v2",))
    import os
    os.utime(b1, (1_000, 1_000))
    os.utime(b2, (2_000, 2_000))
    res = P.rollback(None, active=active, backup_dir=backup_dir)
    assert res["backup"] == b2
    assert P.validate_checkpoint(active)["class_names"] == ["v2"]
    assert res["prev_backup"] is not None    # current was backed up before restore


def test_rollback_no_backup_fails_clearly(tmp_path):
    with pytest.raises(SystemExit, match="No backups found"):
        P.rollback(None, active=tmp_path / "a.pt", backup_dir=tmp_path / "models" / "backups")


def test_rollback_restores_previous(tmp_path):
    active = _write_ckpt(tmp_path / "models" / "cat_classifier.pt", ("now",))
    backup = _write_ckpt(tmp_path / "models" / "backups" / "cat_classifier.20260101-000000-000000.pt",
                         ("prev_a", "prev_b"))
    P.rollback(backup, active=active, backup_dir=tmp_path / "models" / "backups")
    assert P.validate_checkpoint(active)["class_names"] == ["prev_a", "prev_b"]
