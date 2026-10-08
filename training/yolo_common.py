"""Shared, dependency-light helpers for the offline YOLO lifecycle.

This module deliberately does not import Ultralytics. Dataset validation,
hashing and report inspection therefore work without model dependencies.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


REPORT_SCHEMA = "cat-detection.yolo-run/v1"
ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_FILES = (
    "training/yolo_common.py",
    "training/yolo_train.py",
    "training/yolo_evaluate.py",
    "training/yolo_export.py",
    "training/compare_yolo_reports.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a file or directory independently of traversal order."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted((p for p in path.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(path).as_posix())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _hash_existing_files(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def code_identity(root: Path = ROOT) -> dict[str, Any]:
    lifecycle_hash = _hash_existing_files(root, LIFECYCLE_FILES)
    result: dict[str, Any] = {
        "lifecycle_files_sha256": lifecycle_hash,
        "git_commit": None,
        "git_dirty": None,
        "git_diff_sha256": None,
    }
    try:
        result["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout
        result["git_dirty"] = bool(status)
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout
        result["git_diff_sha256"] = hashlib.sha256(
            diff + lifecycle_hash.encode()
        ).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def link_or_copy(source: Path, dest: Path) -> None:
    """Hardlink ``source`` to ``dest`` (fall back to copy across filesystems).

    Lets dataset versions and merged training sets reference the catalog's images
    without a second copy of every frame.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        dest.write_bytes(source.read_bytes())


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        try:
            return {int(key): str(value) for key, value in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset/model class IDs must be integers") from exc
    raise ValueError("class names must be a list or mapping")


@dataclass(frozen=True)
class DatasetInfo:
    version_dir: Path
    data_yaml: Path
    manifest_path: Path
    manifest: dict[str, Any]
    dataset_sha256: str
    manifest_file_sha256: str
    names: dict[int, str]
    cat_id: int
    summary: dict[str, Any]

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("samples") or [])

    def split_samples(self, split: str) -> list[dict[str, Any]]:
        return [sample for sample in self.samples if sample.get("split") == split]

    def image_path(self, sample: dict[str, Any]) -> Path:
        return self.version_dir / "images" / str(sample["split"]) / f"{sample['sample_id']}.jpg"

    def label_path(self, sample: dict[str, Any]) -> Path:
        return self.version_dir / "labels" / str(sample["split"]) / f"{sample['sample_id']}.txt"


def resolve_dataset_path(path: Path) -> tuple[Path, Path]:
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        return candidate, candidate / "data.yaml"
    if candidate.name in {"data.yaml", "data.yml"}:
        return candidate.parent, candidate
    raise ValueError(f"dataset must be an immutable version directory or data.yaml: {path}")


def load_dataset(path: Path, *, require_splits: tuple[str, ...] = ()) -> DatasetInfo:
    version_dir, data_yaml = resolve_dataset_path(path)
    manifest_path = version_dir / "manifest.json"
    if not data_yaml.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"dataset version needs both {data_yaml} and {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("dataset manifest contains no samples")
    declared = manifest.get("manifest_sha256")
    computed = hashlib.sha256(
        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if not isinstance(declared, str) or declared != computed:
        raise ValueError("dataset manifest checksum mismatch; do not train from a modified version")
    raw_yaml = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = _parse_names(raw_yaml.get("names"))
    cat_ids = [class_id for class_id, name in names.items() if name.strip().casefold() == "cat"]
    if len(cat_ids) != 1 or len(names) != 1:
        raise ValueError(f"expected exactly one dataset class named cat, got {names}")
    if cat_ids[0] != 0:
        raise ValueError(f"immutable YOLO dataset must encode cat as class 0, got {cat_ids[0]}")

    per_split = {split: 0 for split in ("train", "val", "test")}
    for sample in samples:
        split = sample.get("split")
        if split not in per_split:
            raise ValueError(f"invalid sample split {split!r}")
        per_split[split] += 1
        image = version_dir / "images" / split / f"{sample.get('sample_id')}.jpg"
        label = version_dir / "labels" / split / f"{sample.get('sample_id')}.txt"
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"dataset sample is incomplete: {sample.get('sample_id')}")
    for split in require_splits:
        if per_split.get(split, 0) <= 0:
            raise ValueError(f"dataset {split} split is empty")

    summary = dict(manifest.get("summary") or {})
    summary.setdefault("images", len(samples))
    summary.setdefault("objects", sum(len(sample.get("boxes") or []) for sample in samples))
    summary.setdefault("negative", sum(not (sample.get("boxes") or []) for sample in samples))
    summary.setdefault("groups", len({sample.get("group_id") for sample in samples}))
    previous_splits = summary.get("splits") or {}
    summary["splits"] = {
        split: {**dict(previous_splits.get(split) or {}), "images": count}
        for split, count in per_split.items()
    }
    return DatasetInfo(
        version_dir=version_dir,
        data_yaml=data_yaml,
        manifest_path=manifest_path,
        manifest=manifest,
        dataset_sha256=declared,
        manifest_file_sha256=sha256_file(manifest_path),
        names=names,
        cat_id=cat_ids[0],
        summary=summary,
    )


def validate_training_weights(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"pretrained weights do not exist: {resolved}")
    if not resolved.is_file() or resolved.suffix.casefold() != ".pt":
        raise ValueError("training base must be an existing trainable .pt checkpoint, not an export directory")
    lowered = resolved.name.casefold()
    if "openvino" in lowered or "int8" in lowered:
        raise ValueError("OpenVINO/INT8 exports cannot be a training base; provide the source .pt")
    return resolved


def model_names(model: Any) -> dict[int, str]:
    raw = getattr(model, "names", None)
    if raw is None and getattr(model, "model", None) is not None:
        raw = getattr(model.model, "names", None)
    return _parse_names(raw)


def resolve_model_cat_id(model: Any) -> int:
    names = model_names(model)
    matches = [class_id for class_id, name in names.items() if name.strip().casefold() == "cat"]
    if len(matches) != 1:
        raise ValueError(f"model must expose exactly one class named cat, got {names}")
    return matches[0]


def package_versions() -> dict[str, str | None]:
    packages = ("ultralytics", "torch", "openvino", "numpy", "opencv-python-headless")
    out: dict[str, str | None] = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def _ram_total_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _gpu_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "devices": [], "metrics_measured": False,
                "reason": "nvidia-smi unavailable; GPU utilization not measured"}
    try:
        output = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
        ).stdout
        devices = []
        for line in output.splitlines():
            index, name, memory_mib = (value.strip() for value in line.split(",", 2))
            devices.append({"index": int(index), "name": name, "memory_total_mib": float(memory_mib)})
        return {"available": bool(devices), "devices": devices, "metrics_measured": False}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"available": False, "devices": [], "metrics_measured": False,
                "reason": f"nvidia-smi query failed: {exc}"}


def environment_info() -> dict[str, Any]:
    return {
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "python": platform.python_version(),
        "libraries": package_versions(),
        "cpu": {"logical_count": os.cpu_count(), "description": platform.processor() or None},
        "ram": {"total_bytes": _ram_total_bytes()},
        "gpu": _gpu_inventory(),
    }


class ResourceMonitor:
    """Sample whole-system and this job's process tree, when psutil is available."""

    def __init__(self, interval_sec: float = 2.0) -> None:
        if interval_sec <= 0:
            raise ValueError("resource sample interval must be positive")
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []
        self._started = 0.0
        self._psutil = None
        self._process = None
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            pass
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        self._started = time.monotonic()
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)
            self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name="yolo-resource-monitor", daemon=True)
        self._thread.start()

    def _gpu_sample(self) -> tuple[float | None, float | None]:
        if not self._nvidia_smi:
            return None, None
        try:
            output = subprocess.run(
                [self._nvidia_smi, "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=max(1.0, self.interval_sec / 2),
            ).stdout
            rows = [tuple(float(v.strip()) for v in line.split(",", 1)) for line in output.splitlines()]
            return ((sum(row[0] for row in rows) / len(rows), sum(row[1] for row in rows))
                    if rows else (None, None))
        except (OSError, subprocess.SubprocessError, ValueError):
            return None, None

    def sample(self) -> None:
        if self._psutil is None or self._process is None:
            return
        psutil = self._psutil
        try:
            processes = [self._process, *self._process.children(recursive=True)]
            job_cpu, job_rss = 0.0, 0
            for process in processes:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    job_cpu += process.cpu_percent(interval=None)
                    job_rss += process.memory_info().rss
            memory = psutil.virtual_memory()
            gpu_util, gpu_memory = self._gpu_sample()
            sample: dict[str, float] = {
                "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
                "system_ram_used_percent": float(memory.percent),
                "system_ram_available_bytes": float(memory.available),
                "job_cpu_percent": job_cpu,
                "job_rss_bytes": float(job_rss),
            }
            if gpu_util is not None:
                sample["gpu_utilization_percent"] = gpu_util
            if gpu_memory is not None:
                sample["gpu_memory_used_mib"] = gpu_memory
            self._samples.append(sample)
        except Exception:
            return

    def _run(self) -> None:
        self.sample()
        while not self._stop.wait(self.interval_sec):
            self.sample()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2))
        self.sample()
        summary: dict[str, Any] = {
            "sample_interval_seconds": self.interval_sec,
            "sample_count": len(self._samples),
            "elapsed_seconds": max(0.0, time.monotonic() - self._started),
            "units": {"cpu": "percent", "memory": "bytes", "gpu_memory": "MiB"},
            "scope": {
                "system_cpu": "whole machine, normalized to 0-100%",
                "system_ram": "whole machine",
                "job_cpu": "current process plus observable children; may exceed 100% on multicore",
                "job_rss": "sum of current process plus observable children",
            },
            "limitations": [], "system": {}, "job": {},
            "gpu": {"metrics_measured": False},
        }
        if not self._samples:
            summary["limitations"].append(
                "psutil unavailable or sampling failed; averages and process-tree peaks were not measured"
            )
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = int(peak_rss * 1024 if platform.system() != "Darwin" else peak_rss)
            summary["job"] = {"self_peak_rss_bytes": peak_bytes}
            summary["limitations"].append("self peak RSS excludes child processes")
            return summary

        def stats(key: str) -> dict[str, float] | None:
            values = [sample[key] for sample in self._samples if key in sample]
            return ({"average": sum(values) / len(values), "peak": max(values)} if values else None)

        summary["system"] = {
            "cpu_percent": stats("system_cpu_percent"),
            "ram_used_percent": stats("system_ram_used_percent"),
            "ram_available_bytes": stats("system_ram_available_bytes"),
        }
        summary["job"] = {
            "cpu_percent": stats("job_cpu_percent"), "rss_bytes": stats("job_rss_bytes")}
        gpu_util = stats("gpu_utilization_percent")
        summary["gpu"] = {
            "metrics_measured": gpu_util is not None,
            "utilization_percent": gpu_util,
            "memory_used_mib": stats("gpu_memory_used_mib"),
            "reason": None if gpu_util is not None else "GPU metrics unavailable; not measured",
        }
        return summary


class ReportSession:
    """Own an incrementally durable JSON report and resource sampler."""

    def __init__(self, path: Path, *, kind: str, parameters: dict[str, Any],
                 dataset: DatasetInfo | None = None, model: dict[str, Any] | None = None,
                 sample_interval_sec: float = 2.0) -> None:
        self.path = path.resolve()
        self.started_monotonic = time.monotonic()
        self.monitor = ResourceMonitor(sample_interval_sec)
        self.report: dict[str, Any] = {
            "schema": REPORT_SCHEMA, "kind": kind, "status": "running", "complete": False,
            "started_at": utc_now(), "finished_at": None, "report_path": str(self.path),
            "parameters": parameters,
            "timing_seconds": {"preparation": None, "training": None,
                               "evaluation": None, "total": None},
            "progress": {}, "code": code_identity(), "environment": environment_info(),
            "resources": {"sample_interval_seconds": sample_interval_sec, "sample_count": 0,
                          "limitations": ["run active; final resource aggregates unavailable"]},
            "model": model or {}, "dataset": dataset_report(dataset) if dataset else {},
            "metrics": {}, "artifacts": {}, "error": None,
            "notes": ["Inference timing is offline model-pipeline timing, not end-to-end feeder latency."],
        }
        atomic_write_json(self.path, self.report)
        self.monitor.start()

    def checkpoint(self, **updates: Any) -> None:
        self.report.update(updates)
        self.report["updated_at"] = utc_now()
        atomic_write_json(self.path, self.report)

    def progress(self, **values: Any) -> None:
        self.report.setdefault("progress", {}).update(values)
        self.checkpoint()

    def finish(self, status: str, *, error: BaseException | str | None = None) -> dict[str, Any]:
        self.report["status"] = status
        self.report["complete"] = status == "completed"
        self.report["finished_at"] = utc_now()
        self.report["timing_seconds"]["total"] = time.monotonic() - self.started_monotonic
        self.report["resources"] = self.monitor.stop()
        if error is not None:
            self.report["error"] = {
                "type": type(error).__name__ if isinstance(error, BaseException) else "error",
                "message": str(error),
            }
        atomic_write_json(self.path, self.report)
        return self.report


def dataset_report(dataset: DatasetInfo) -> dict[str, Any]:
    return {
        "version_id": dataset.manifest.get("version_id"), "path": str(dataset.version_dir),
        "data_yaml": str(dataset.data_yaml), "sha256": dataset.dataset_sha256,
        "manifest_file_sha256": dataset.manifest_file_sha256,
        "counts": dataset.summary, "class_names": dataset.names,
    }


@contextlib.contextmanager
def convert_termination_to_interrupt() -> Iterator[None]:
    """Turn SIGTERM into an exception so the final report can be flushed."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def handle_term(_signum, _frame):
        raise InterruptedError("received SIGTERM")

    signal.signal(signal.SIGTERM, handle_term)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def sample_subsets(sample: dict[str, Any]) -> list[str]:
    """Return generic subset memberships from subsets/tags/metadata fields."""
    values: set[str] = set()
    for field in ("subsets", "tags"):
        raw = sample.get(field)
        if isinstance(raw, str) and raw.strip():
            values.add(raw.strip())
        elif isinstance(raw, list):
            values.update(str(item).strip() for item in raw if str(item).strip())
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key, raw in metadata.items():
            sequence = raw if isinstance(raw, list) else [raw]
            for item in sequence:
                if isinstance(item, (str, int, float, bool)):
                    values.add(f"{key}={item}")
    return sorted(values)


def print_report_summary(report: dict[str, Any]) -> None:
    timing = report.get("timing_seconds") or {}
    dataset = report.get("dataset") or {}
    counts = dataset.get("counts") or {}
    metrics = report.get("metrics") or {}
    headline = metrics.get("official") or metrics.get("candidate") or metrics
    print(f"{report.get('kind', 'YOLO run')}: {report.get('status', 'unknown')}")
    print(f"report: {report.get('report_path', '-')}")
    if dataset:
        print("dataset: "
              f"{dataset.get('version_id') or '-'}; images={counts.get('images', '-')}, "
              f"groups={counts.get('groups', '-')}, objects={counts.get('objects', '-')}, "
              f"negative={counts.get('negative', '-')}")
    if headline:
        values = [f"{key}={headline[key]:.4f}" for key in
                  ("precision", "recall", "map50", "map50_95")
                  if isinstance(headline.get(key), (int, float))]
        if values:
            print("quality: " + ", ".join(values))
    rendered_times = [f"{key}={value:.2f}s" for key, value in timing.items()
                      if isinstance(value, (int, float))]
    if rendered_times:
        print("time: " + ", ".join(rendered_times))
    if report.get("error"):
        print(f"error: {report['error'].get('message')}")
