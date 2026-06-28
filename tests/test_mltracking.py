"""Tests for the optional MLflow tracking shim.

mlflow need not be installed: the no-op path and a fake-mlflow path are both
covered, and every call must be exception-safe so logging can never break a
training run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from training import mltracking


# --- tracking URI resolution -------------------------------------------------

def test_default_tracking_uri_is_local_file_store(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = mltracking.resolve_tracking_uri()
    assert uri.startswith("file:")
    assert uri.endswith("data/mlflow")


def test_env_overrides_tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert mltracking.resolve_tracking_uri() == "http://localhost:5000"


# --- no-op path (mlflow unavailable) -----------------------------------------

def test_start_run_is_noop_without_mlflow(monkeypatch):
    monkeypatch.setattr(mltracking, "_import_mlflow", lambda: None)
    run = mltracking.start_run("exp", run_name="r", params={"a": 1})
    assert run.active is False
    # None of these may raise.
    run.log_params({"x": 1})
    run.log_metrics({"m": 1.0}, step=3)
    run.log_metric("m", 2.0)
    run.log_artifact("/nope")
    run.set_tags({"t": "v"})
    run.end()


# --- fake-mlflow path --------------------------------------------------------

class _FakeMlflow:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def set_tracking_uri(self, uri): self.calls.append(("uri", uri))
    def set_experiment(self, exp): self.calls.append(("exp", exp))

    def start_run(self, run_name=None, nested=False):
        self.calls.append(("start", run_name, nested))
        run = type("R", (), {})()
        run.info = type("I", (), {"run_id": "rid"})()
        return run

    def log_params(self, p): self.calls.append(("params", p))

    def log_metrics(self, m, step=None):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(("metrics", m, step))

    def log_artifact(self, p, artifact_path=None):
        self.calls.append(("artifact", p, artifact_path))

    def log_artifacts(self, p, artifact_path=None):
        self.calls.append(("artifacts", p, artifact_path))

    def set_tags(self, t): self.calls.append(("tags", t))
    def end_run(self, status="FINISHED"): self.calls.append(("end", status))


def _use_fake(monkeypatch, fake):
    monkeypatch.setattr(mltracking, "_import_mlflow", lambda: fake)


def test_run_exposes_run_id_and_tracking_uri(monkeypatch):
    fake = _FakeMlflow()
    fake.get_tracking_uri = lambda: "file:/tmp/mlruns"
    _use_fake(monkeypatch, fake)
    run = mltracking.start_run("e", run_name="r")
    assert run.run_id == "rid"
    assert run.tracking_uri == "file:/tmp/mlruns"


def test_noop_run_id_and_uri_are_none(monkeypatch):
    monkeypatch.setattr(mltracking, "_import_mlflow", lambda: None)
    run = mltracking.start_run("e")
    assert run.run_id is None and run.tracking_uri is None


def test_start_run_logs_params_and_starts(monkeypatch):
    fake = _FakeMlflow()
    _use_fake(monkeypatch, fake)
    run = mltracking.start_run("cat_classifier", run_name="run1",
                               params={"augment": "light"}, tags={"k": "v"})
    assert run.active is True
    kinds = [c[0] for c in fake.calls]
    assert kinds[:3] == ["uri", "exp", "start"]
    assert ("params", {"augment": "light"}) in fake.calls
    assert ("tags", {"k": "v"}) in fake.calls


def test_metrics_filtered_to_numbers_and_bool_dropped(monkeypatch):
    fake = _FakeMlflow()
    _use_fake(monkeypatch, fake)
    run = mltracking.start_run("e")
    run.log_metrics({"acc": 0.5, "name": "x", "flag": True, "n": 3}, step=2)
    logged = [c for c in fake.calls if c[0] == "metrics"][0]
    assert logged[1] == {"acc": 0.5, "n": 3.0}    # str + bool dropped
    assert logged[2] == 2


def test_long_param_value_is_truncated(monkeypatch):
    fake = _FakeMlflow()
    _use_fake(monkeypatch, fake)
    mltracking.start_run("e", params={"big": "x" * 600})
    params = [c for c in fake.calls if c[0] == "params"][0][1]
    assert len(params["big"]) <= 481 and params["big"].endswith("…")


def test_directory_artifact_uses_log_artifacts(monkeypatch, tmp_path):
    fake = _FakeMlflow()
    _use_fake(monkeypatch, fake)
    run = mltracking.start_run("e")
    a_file = tmp_path / "f.txt"
    a_file.write_text("hi")
    run.log_artifact(a_file)
    run.log_artifact(tmp_path, artifact_path="dir")
    kinds = [c[0] for c in fake.calls]
    assert "artifact" in kinds and "artifacts" in kinds


def test_logging_errors_are_swallowed(monkeypatch):
    fake = _FakeMlflow(fail=True)
    _use_fake(monkeypatch, fake)
    run = mltracking.start_run("e")
    # log_metrics raises inside the fake; the wrapper must swallow it.
    run.log_metrics({"acc": 1.0})
    run.end()


def test_init_failure_degrades_to_noop(monkeypatch):
    class Boom:
        def set_tracking_uri(self, uri): raise RuntimeError("nope")
    _use_fake(monkeypatch, Boom())
    run = mltracking.start_run("e", params={"a": 1})
    assert run.active is False           # fell back to no-op, no exception
