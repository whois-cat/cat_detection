"""Export a trained YOLO ``.pt`` to OpenVINO and enforce a parity gate.

Both artifacts are evaluated on exactly the same immutable held-out test split.
The command writes a report and returns non-zero when quality regresses beyond
the configured limits.  It never promotes, links, or reloads a runtime model.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.yolo_common import (
    ReportSession,
    convert_termination_to_interrupt,
    load_dataset,
    print_report_summary,
    sha256_file,
    sha256_tree,
    ultralytics_data_yaml,
    validate_training_weights,
)
from training.yolo_evaluate import evaluate_artifact, validate_model_artifact


def _load_yolo(path: Path):
    from ultralytics import YOLO
    return YOLO(str(path), task="detect")


def quality_gate(baseline: dict[str, Any], candidate: dict[str, Any], *,
                 max_map_drop: float, max_recall_drop: float,
                 max_fp_increase: int) -> dict[str, Any]:
    base_official, candidate_official = baseline["official"], candidate["official"]
    base_fixed = baseline["fixed_confidence"]["overall"]
    candidate_fixed = candidate["fixed_confidence"]["overall"]

    def delta(name: str) -> float | None:
        left, right = base_official.get(name), candidate_official.get(name)
        return (float(right) - float(left)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None)

    deltas = {
        "precision": delta("precision"), "recall": delta("recall"),
        "map50": delta("map50"), "map50_95": delta("map50_95"),
        "fixed_confidence_recall": candidate_fixed["recall"] - base_fixed["recall"],
        "fixed_confidence_fp": candidate_fixed["fp"] - base_fixed["fp"],
        "fixed_confidence_fn": candidate_fixed["fn"] - base_fixed["fn"],
        "empty_image_errors": (candidate_fixed["empty_image_errors"]
                               - base_fixed["empty_image_errors"]),
    }
    checks = {
        "map50_95_drop_within_limit": (deltas["map50_95"] is not None
                                        and deltas["map50_95"] >= -max_map_drop),
        # Recall at the fixed (runtime) confidence. Ultralytics' own recall is
        # taken at each model's max-F1 threshold, which moves between the .pt and
        # the export, so it trades recall for precision without either model
        # changing at the threshold that actually serves.
        "recall_drop_within_limit": deltas["fixed_confidence_recall"] >= -max_recall_drop,
        "false_positive_increase_within_limit": deltas["fixed_confidence_fp"] <= max_fp_increase,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "deltas": deltas,
        "limits": {"max_map50_95_drop": max_map_drop,
                   "max_recall_drop": max_recall_drop,
                   "max_fixed_confidence_fp_increase": max_fp_increase},
        "decision": "artifact accepted for manual review" if all(checks.values()) else "artifact rejected",
        "promotion_performed": False,
    }


def _default_report(source: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return source.resolve().parent / f"openvino-export-{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="trained source .pt")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="evaluation batch size")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--int8", action="store_true", help="request Ultralytics INT8 calibration/export")
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--max-map50-95-drop", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--max-fp-increase", type=int, default=0)
    parser.add_argument("--resource-interval", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preparation_started = time.monotonic()
    dataset = load_dataset(args.dataset, require_splits=("train", "test"))
    source = validate_training_weights(args.model)
    expected = source.with_name(f"{source.stem}_openvino_model")
    if expected.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {expected}")
    parameters = {
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "workers": args.workers, "int8": args.int8,
        "evaluation_confidence": args.eval_conf, "evaluation_iou": args.eval_iou,
        "max_map50_95_drop": args.max_map50_95_drop,
        "max_recall_drop": args.max_recall_drop, "max_fp_increase": args.max_fp_increase,
    }
    report_path = args.report or _default_report(source)
    session = ReportSession(
        report_path, kind="yolo_openvino_export",
        parameters=parameters, dataset=dataset,
        model={"source": {"path": str(source), "sha256": sha256_file(source)}},
        sample_interval_sec=args.resource_interval,
    )
    session.report["timing_seconds"]["preparation"] = time.monotonic() - preparation_started
    session.report["timing_seconds"]["export"] = None
    session.checkpoint()
    try:
        with convert_termination_to_interrupt():
            evaluation_started = time.monotonic()
            session.progress(stage="evaluating_source")
            baseline = evaluate_artifact(
                source, dataset, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, confidence=args.eval_conf, iou_threshold=args.eval_iou,
                plots_dir=report_path.with_suffix("").with_name(report_path.stem + "-source"),
            )

            session.progress(stage="exporting_openvino")
            export_started = time.monotonic()
            model = _load_yolo(source)
            export_kwargs: dict[str, Any] = {
                "format": "openvino", "imgsz": args.imgsz, "int8": args.int8,
                "half": False, "dynamic": False, "batch": 1,
            }
            if args.int8:
                # Ultralytics uses the training split for calibration.
                export_kwargs["data"] = str(ultralytics_data_yaml(dataset))
            exported_value = model.export(**export_kwargs)
            exported = validate_model_artifact(Path(str(exported_value)))
            session.report["timing_seconds"]["export"] = time.monotonic() - export_started
            session.report["model"]["openvino"] = {
                "path": str(exported), "sha256": sha256_tree(exported), "int8": args.int8}
            session.report["artifacts"]["openvino_directory"] = str(exported)
            session.checkpoint()

            session.progress(stage="evaluating_export")
            candidate = evaluate_artifact(
                exported, dataset, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, confidence=args.eval_conf, iou_threshold=args.eval_iou,
                plots_dir=report_path.with_suffix("").with_name(report_path.stem + "-export"),
            )
            session.report["timing_seconds"]["evaluation"] = time.monotonic() - evaluation_started
            gate = quality_gate(
                baseline, candidate, max_map_drop=args.max_map50_95_drop,
                max_recall_drop=args.max_recall_drop, max_fp_increase=args.max_fp_increase,
            )
            session.report["metrics"] = {
                "baseline": baseline, "candidate": candidate, "quality_gate": gate}
            session.report["progress"] = {"stage": "finished"}
            report = session.finish("completed")
            print_report_summary(report)
            print(f"OpenVINO parity gate: {'PASS' if gate['passed'] else 'FAIL'}")
            print("Runtime model was not changed.")
            return 0 if gate["passed"] else 2
    except (KeyboardInterrupt, InterruptedError) as exc:
        print_report_summary(session.finish("interrupted", error=exc))
        return 130
    except BaseException as exc:
        print_report_summary(session.finish("failed", error=exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
