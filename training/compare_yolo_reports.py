"""Show one YOLO JSON report or compare two reports locally, without MLflow/jq."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.yolo_common import atomic_write_json, print_report_summary, utc_now


def _dig(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first(report: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _dig(report, path)
        if value is not None:
            return value
    return None


def _number(report: dict[str, Any], *paths: str) -> float | None:
    value = _first(report, *paths)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def headline(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize this lifecycle schema and older/ad-hoc report layouts."""
    return {
        "status": report.get("status"),
        "kind": report.get("kind") or report.get("type"),
        "dataset_sha256": _first(
            report, "dataset.sha256", "dataset.manifest_sha256", "dataset_hash",
            "dataset.version_hash"),
        "dataset_version": _first(report, "dataset.version_id", "dataset_version"),
        "parameters": report.get("parameters") or report.get("settings") or report.get("args") or {},
        "timing_seconds": {
            key: _number(report, f"timing_seconds.{key}", f"timing.{key}", f"durations.{key}")
            for key in ("preparation", "training", "export", "evaluation", "total")
        },
        "quality": {
            "precision": _number(report, "metrics.official.precision",
                                 "metrics.candidate.official.precision", "metrics.precision", "precision"),
            "recall": _number(report, "metrics.official.recall",
                              "metrics.candidate.official.recall", "metrics.recall", "recall"),
            "map50": _number(report, "metrics.official.map50",
                             "metrics.candidate.official.map50", "metrics.map50", "map50"),
            "map50_95": _number(report, "metrics.official.map50_95",
                                "metrics.candidate.official.map50_95", "metrics.map50_95",
                                "metrics.map", "map50_95"),
            "tp": _number(report, "metrics.fixed_confidence.overall.tp",
                          "metrics.candidate.fixed_confidence.overall.tp", "metrics.tp"),
            "fp": _number(report, "metrics.fixed_confidence.overall.fp",
                          "metrics.candidate.fixed_confidence.overall.fp", "metrics.fp"),
            "fn": _number(report, "metrics.fixed_confidence.overall.fn",
                          "metrics.candidate.fixed_confidence.overall.fn", "metrics.fn"),
            "empty_image_errors": _number(
                report, "metrics.fixed_confidence.overall.empty_image_errors",
                "metrics.candidate.fixed_confidence.overall.empty_image_errors",
                "metrics.empty_image_errors"),
        },
        "resources": {
            "job_cpu_average_percent": _number(report, "resources.job.cpu_percent.average",
                                               "resources.job_cpu_percent.average"),
            "job_rss_peak_bytes": _number(report, "resources.job.rss_bytes.peak",
                                          "resources.peak_job_rss_bytes"),
            "system_cpu_average_percent": _number(report, "resources.system.cpu_percent.average",
                                                  "resources.system_cpu_percent.average"),
            "gpu_utilization_average_percent": _number(
                report, "resources.gpu.utilization_percent.average",
                "resources.gpu_utilization_percent.average"),
            "gpu_metrics_measured": _first(report, "resources.gpu.metrics_measured",
                                           "environment.gpu.metrics_measured"),
        },
    }


def _delta(before: float | None, after: float | None) -> float | None:
    return after - before if before is not None and after is not None else None


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    left, right = headline(before), headline(after)
    left_parameters, right_parameters = left["parameters"], right["parameters"]
    keys = sorted(set(left_parameters) | set(right_parameters))
    changed = {
        key: {"before": left_parameters.get(key), "after": right_parameters.get(key)}
        for key in keys if left_parameters.get(key) != right_parameters.get(key)
    }
    dataset_known = bool(left["dataset_sha256"] and right["dataset_sha256"])
    return {
        "schema": "cat-detection.yolo-report-comparison/v1",
        "created_at": utc_now(),
        "same_dataset": ((left["dataset_sha256"] == right["dataset_sha256"])
                         if dataset_known else None),
        "dataset_comparison_known": dataset_known,
        "same_settings": not changed,
        "changed_settings": changed,
        "before": left,
        "after": right,
        "deltas": {
            "timing_seconds": {
                key: _delta(left["timing_seconds"][key], right["timing_seconds"][key])
                for key in left["timing_seconds"]
            },
            "quality": {
                key: _delta(left["quality"][key], right["quality"][key])
                for key in left["quality"]
            },
            "resources": {
                key: _delta(left["resources"][key], right["resources"][key])
                for key in left["resources"]
                if key != "gpu_metrics_measured"
            },
        },
        "limitations": [
            "A delta is null when either report did not record that field.",
            "Whole-system utilization is contextual and is not attributed to the YOLO job.",
            "Quality deltas are comparable only when same_dataset is true and settings are compatible.",
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return value


def _format_delta(value: float | None, unit: str = "") -> str:
    return "not recorded" if value is None else f"{value:+.4f}{unit}"


def print_comparison(comparison: dict[str, Any], before_path: Path, after_path: Path) -> None:
    same_dataset = comparison["same_dataset"]
    dataset_text = "unknown" if same_dataset is None else "yes" if same_dataset else "NO"
    print(f"before: {before_path}")
    print(f"after:  {after_path}")
    print(f"same dataset: {dataset_text}")
    print(f"same settings: {'yes' if comparison['same_settings'] else 'NO'}")
    if comparison["changed_settings"]:
        print("changed settings:")
        for key, values in comparison["changed_settings"].items():
            print(f"  {key}: {values['before']!r} -> {values['after']!r}")
    quality = comparison["deltas"]["quality"]
    print("quality delta (after - before): " + ", ".join(
        f"{key}={_format_delta(quality[key])}"
        for key in ("precision", "recall", "map50", "map50_95", "fp", "fn")
    ))
    timing = comparison["deltas"]["timing_seconds"]
    print("time delta: " + ", ".join(
        f"{key}={_format_delta(value, 's')}" for key, value in timing.items() if value is not None
    ) if any(value is not None for value in timing.values()) else "time delta: not recorded")
    resources = comparison["deltas"]["resources"]
    print("resource delta: " + ", ".join(
        f"{key}={_format_delta(value)}" for key, value in resources.items() if value is not None
    ) if any(value is not None for value in resources.values()) else "resource delta: not recorded")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", metavar="REPORT",
                        help="one report to summarize, or two to compare")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="write machine-readable comparison (two-report mode)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.reports) not in (1, 2):
        raise SystemExit("provide one report to view or two reports to compare")
    first = _load(args.reports[0])
    if len(args.reports) == 1:
        if args.json_out:
            raise SystemExit("--json-out requires two reports")
        print_report_summary(first)
        return 0
    second = _load(args.reports[1])
    comparison = compare_reports(first, second)
    print_comparison(comparison, args.reports[0], args.reports[1])
    if args.json_out:
        atomic_write_json(args.json_out, comparison)
        print(f"comparison JSON: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
