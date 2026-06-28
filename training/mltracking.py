"""Optional MLflow experiment tracking for the training scripts.

Always-on **when the ``mlflow`` package is installed**; a transparent no-op
otherwise, so training never hard-depends on MLflow being present. Any MLflow
error (server down, bad URI, …) also degrades to no-op rather than killing a
training run.

Tracking goes to a local file store at ``<repo>/data/mlflow`` by default
(``data/`` is gitignored). Override with the standard ``MLFLOW_TRACKING_URI``
env var — e.g. point it at the ``mlflow`` docker-compose service
(``http://localhost:5000``) to browse runs in the web UI.

Usage:

    from training.mltracking import start_run
    run = start_run("cat_classifier", run_name=stamp, params={...})
    try:
        run.log_metrics({"val_macro_recall": 0.97}, step=epoch)
        run.log_artifact(model_path)
    finally:
        run.end()
"""
from __future__ import annotations

import logging
import numbers
import os
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "data" / "mlflow"


def resolve_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, else a local file store under data/mlflow."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    DEFAULT_STORE.mkdir(parents=True, exist_ok=True)
    return DEFAULT_STORE.as_uri()  # file://…


def _is_number(v) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def _short(v) -> str:
    # MLflow rejects param values longer than 500 chars.
    s = str(v)
    return s if len(s) <= 480 else s[:477] + "…"


class _NoopRun:
    """Stand-in returned when MLflow is unavailable. Every method is a no-op."""

    active = False
    run_id = None
    tracking_uri = None

    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...
    def log_metric(self, key: str, value, step: int | None = None) -> None: ...
    def log_artifact(self, path, artifact_path: str | None = None) -> None: ...
    def set_tags(self, tags: dict) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...


class _MlflowRun:
    """Thin wrapper that swallows MLflow errors so logging never breaks training."""

    active = True

    def __init__(self, mlflow, run):
        self._mlflow = mlflow
        self._run = run

    @property
    def run_id(self):
        try:
            return self._run.info.run_id
        except Exception:  # pragma: no cover - defensive
            return None

    @property
    def tracking_uri(self):
        try:
            return self._mlflow.get_tracking_uri()
        except Exception:  # pragma: no cover - defensive
            return None

    def log_params(self, params: dict) -> None:
        try:
            self._mlflow.log_params({k: _short(v) for k, v in params.items()})
        except Exception as e:  # pragma: no cover - defensive
            log.warning("mlflow log_params failed: %r", e)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean, step=step)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("mlflow log_metrics failed: %r", e)

    def log_metric(self, key: str, value, step: int | None = None) -> None:
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        try:
            if os.path.isdir(path):
                self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            else:
                self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("mlflow log_artifact failed: %r", e)

    def set_tags(self, tags: dict) -> None:
        try:
            self._mlflow.set_tags({k: _short(v) for k, v in tags.items()})
        except Exception as e:  # pragma: no cover - defensive
            log.warning("mlflow set_tags failed: %r", e)

    def end(self, status: str = "FINISHED") -> None:
        try:
            self._mlflow.end_run(status=status)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("mlflow end_run failed: %r", e)


def _import_mlflow():
    try:
        import mlflow  # noqa: WPS433 (intentional lazy import)
        return mlflow
    except Exception:
        return None


def start_run(experiment: str, *, run_name: str | None = None,
              params: dict | None = None, tags: dict | None = None,
              nested: bool = False):
    """Begin a tracked run. Returns a run object (real or no-op) with a uniform
    API; always pair with ``run.end()`` (typically in a ``finally``)."""
    mlflow = _import_mlflow()
    if mlflow is None:
        log.info(
            "mlflow not installed — skipping experiment tracking "
            "(add 'mlflow' to the training env to enable)"
        )
        return _NoopRun()
    try:
        uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name, nested=nested)
        log.info("mlflow tracking → %s (experiment=%s, run_id=%s)",
                 uri, experiment, run.info.run_id)
        wrapper = _MlflowRun(mlflow, run)
        if params:
            wrapper.log_params(params)
        if tags:
            wrapper.set_tags(tags)
        return wrapper
    except Exception as e:
        log.warning("mlflow init failed (%r) — continuing without tracking", e)
        return _NoopRun()
