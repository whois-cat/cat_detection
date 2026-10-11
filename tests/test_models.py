"""Deploy / rollback of the runtime models (tools/models.py): one versioned,
symlinked layout for both the detector and the classifier.

Exports are faked so these run without OpenVINO/ultralytics; the classifier's
checkpoint check needs torch (present in the test env).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import models as M  # noqa: E402

torch = pytest.importorskip("torch")


# ---- fixtures: trained runs + fake exports ---------------------------------------

def _classifier_run(tmp: Path, run: str, names=("cat_a", "cat_b"), **broken) -> Path:
    path = tmp / "trained" / run / "cat_classifier.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": {"w": torch.zeros(1)}, "class_names": list(names),
            "num_classes": broken.get("num_classes", len(names))}
    for key in broken.get("drop", ()):
        ckpt.pop(key)
    torch.save(ckpt, path)
    return path


def _detector_run(tmp: Path, run: str, dataset="v1") -> Path:
    path = tmp / "trained" / run / "weights" / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")
    (path.parents[1] / "report.json").write_text(json.dumps({"dataset": {"version_id": dataset}}))
    return path


def _fake_classifier_export(checkpoint: Path, out: Path, skip_gate=False) -> None:
    names = torch.load(checkpoint, weights_only=False)["class_names"]
    (out / "cat_classifier.xml").write_text("<net/>")
    (out / "cat_classifier.bin").write_bytes(b"\0")
    (out / "classes.json").write_text(json.dumps(names))


def _fake_detector_export(checkpoint: Path, out: Path, skip_gate=False, *, names=None) -> None:
    export = out / "best_int8_openvino_model"
    export.mkdir()
    (export / "best.xml").write_text("<net/>")
    (export / "best.bin").write_bytes(b"\0")
    (export / "metadata.yaml").write_text(yaml.safe_dump({"names": names or {15: "cat"}}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, export in (("classifier", _fake_classifier_export),
                         ("detector", _fake_detector_export)):
        monkeypatch.setitem(M.KINDS, name, M.KINDS[name].__class__(
            **{**M.KINDS[name].__dict__, "export": export}))
    monkeypatch.setattr(M, "YOLO_DATASET", tmp_path / "yolo_dataset")
    (tmp_path / "yolo_dataset" / "versions" / "v1").mkdir(parents=True)
    return tmp_path


def _deploy(tmp, kind, run=None, **kw):
    return M.deploy(kind, run, trained_root=tmp / "trained", models_root=tmp / "models", **kw)


def _rollback(tmp, kind, version=None):
    return M.rollback(kind, version, models_root=tmp / "models")


def _current(tmp, kind):
    link = tmp / "models" / kind / "current"
    return Path(os.readlink(link)).name if link.is_symlink() else None


# ---- both kinds share one layout -------------------------------------------------

@pytest.mark.parametrize("kind,make", [("classifier", _classifier_run), ("detector", _detector_run)])
def test_deploy_installs_version_and_switches_current(env, kind, make):
    make(env, "run-1")
    res = _deploy(env, kind, "run-1")
    assert res["version"] == "run-1" and res["previous"] is None
    assert os.readlink(env / "models" / kind / "current") == "versions/run-1"   # relative
    assert (env / "models" / kind / "versions" / "run-1" / "metadata.json").is_file()


@pytest.mark.parametrize("kind,make", [("classifier", _classifier_run), ("detector", _detector_run)])
def test_newest_run_by_default_then_rollback_and_back(env, kind, make):
    old, new = make(env, "run-1"), make(env, "run-2")
    os.utime(old, (1_000, 1_000)); os.utime(new, (2_000, 2_000))
    _deploy(env, kind, "run-1")
    assert _deploy(env, kind)["version"] == "run-2"
    assert _rollback(env, kind)["version"] == "run-1"
    assert _rollback(env, kind)["version"] == "run-2"           # reversible
    assert _rollback(env, kind, "run-1")["version"] == "run-1"  # explicit
    assert (env / "models" / kind / "versions" / "run-2").is_dir()   # never deleted


def test_redeploying_an_installed_run_points_at_rollback(env):
    _classifier_run(env, "run-1")
    _deploy(env, "classifier", "run-1")
    with pytest.raises(SystemExit, match="already installed"):
        _deploy(env, "classifier", "run-1")


def test_unknown_version_and_missing_runs_fail_clearly(env):
    with pytest.raises(SystemExit, match="no trained detector"):
        _deploy(env, "detector")
    _classifier_run(env, "run-1")
    _deploy(env, "classifier", "run-1")
    with pytest.raises(SystemExit, match="version not found"):
        _rollback(env, "classifier", "nope")


# ---- what differs per kind ---------------------------------------------------------

def test_classifier_without_previous_cannot_roll_back(env):
    _classifier_run(env, "run-1")
    _deploy(env, "classifier", "run-1")
    with pytest.raises(SystemExit, match="no previous classifier"):
        _rollback(env, "classifier")


def test_detector_rolls_back_to_the_builtin_model(env):
    _detector_run(env, "run-1")
    _deploy(env, "detector", "run-1")
    res = _rollback(env, "detector")
    assert res["version"] is None and _current(env, "detector") is None
    assert "yolov8n" in M.status(models_root=env / "models")[0]
    assert _rollback(env, "detector")["version"] == "run-1"     # and forward again


@pytest.mark.parametrize("broken,match", [
    ({"drop": ["class_names"]}, "class_names"),
    ({"num_classes": 3}, "num_classes"),
])
def test_bad_classifier_checkpoint_is_refused(env, broken, match):
    _classifier_run(env, "run-1", **broken)
    with pytest.raises(ValueError, match=match):
        _deploy(env, "classifier", "run-1")
    assert _current(env, "classifier") is None


def test_detector_needs_its_training_dataset(env):
    _detector_run(env, "run-1", dataset="gone")
    with pytest.raises(ValueError, match="dataset version"):
        _deploy(env, "detector", "run-1")


def test_detector_export_without_cat_is_refused(env, monkeypatch):
    monkeypatch.setitem(M.KINDS, "detector", M.KINDS["detector"].__class__(
        **{**M.KINDS["detector"].__dict__,
           "export": lambda c, o, s: _fake_detector_export(c, o, s, names={0: "dog"})}))
    _detector_run(env, "run-1")
    with pytest.raises(ValueError, match="no 'cat' class"):
        _deploy(env, "detector", "run-1")
    assert _current(env, "detector") is None
    assert not [p for p in (env / "models" / "detector" / "versions").iterdir()]   # temp cleaned


def test_status_marks_current_and_previous(env):
    for run in ("run-1", "run-2"):
        _classifier_run(env, run)
        _deploy(env, "classifier", run)
    lines = M.status(models_root=env / "models")
    assert "  * run-2" in lines and "  < run-1" in lines


# ---- the detector's quality gate (real export_detector, faked yolo_export) -----------

def _gate_exit(monkeypatch, code):
    """Fake `python -m training.yolo_export`: writes the export dir, exits `code`."""
    def run(cmd, cwd=None):
        model = Path(cmd[cmd.index("--model") + 1])
        _fake_detector_export(model, model.parent)
        return type("Done", (), {"returncode": code})()
    monkeypatch.setitem(M.KINDS, "detector", M.KINDS["detector"].__class__(
        **{**M.KINDS["detector"].__dict__, "export": M.export_detector}))
    monkeypatch.setattr(M.subprocess, "run", run)


def test_failed_detector_gate_switches_nothing(env, monkeypatch):
    _gate_exit(monkeypatch, 2)
    _detector_run(env, "run-1")
    with pytest.raises(RuntimeError, match="--skip-gate"):
        _deploy(env, "detector", "run-1")
    assert _current(env, "detector") is None


def test_skip_gate_installs_and_records_it(env, monkeypatch):
    _gate_exit(monkeypatch, 2)
    _detector_run(env, "run-1")
    _deploy(env, "detector", "run-1", skip_gate=True)
    assert _current(env, "detector") == "run-1"
    meta = json.loads((env / "models/detector/versions/run-1/metadata.json").read_text())
    assert meta["quality_gate"] == "failed, skipped"
    assert "yolo-reports" in meta["export_report"]


def test_skip_gate_never_installs_a_broken_export(env, monkeypatch):
    _gate_exit(monkeypatch, 1)
    _detector_run(env, "run-1")
    with pytest.raises(RuntimeError):
        _deploy(env, "detector", "run-1", skip_gate=True)
    assert _current(env, "detector") is None


def test_classifier_parity_cannot_be_skipped(env, monkeypatch):
    monkeypatch.setitem(M.KINDS, "classifier", M.KINDS["classifier"].__class__(
        **{**M.KINDS["classifier"].__dict__, "export": M.export_classifier}))
    _classifier_run(env, "run-1")
    with pytest.raises(SystemExit, match="cannot be skipped"):
        _deploy(env, "classifier", "run-1", skip_gate=True)
