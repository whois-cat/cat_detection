"""Evaluate a YOLO ``.pt`` or OpenVINO export on an immutable test split.

Ultralytics supplies the standard detection metrics.  A second, fixed-confidence
pass reports operational TP/FP/FN counts, errors on human-confirmed empty frames,
subset metrics, and offline inference timing.  It never changes runtime models.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2

from training.yolo_common import (
    DatasetInfo,
    ReportSession,
    convert_termination_to_interrupt,
    load_dataset,
    print_report_summary,
    resolve_model_cat_id,
    sample_subsets,
    sha256_tree,
)


def _load_yolo(path: Path):
    from ultralytics import YOLO
    return YOLO(str(path), task="detect")


def validate_model_artifact(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.casefold() == ".pt":
        return path
    if path.is_dir() and any(path.glob("*.xml")):
        return path
    raise ValueError(f"model must be a .pt file or OpenVINO directory containing an .xml file: {path}")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def official_metrics(result: Any) -> dict[str, float | None]:
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", None) or {}
    return {
        "precision": _number(getattr(box, "mp", None)),
        "recall": _number(getattr(box, "mr", None)),
        "map50": _number(getattr(box, "map50", None)),
        "map50_95": _number(getattr(box, "map", None)),
        "preprocess_ms_per_image": _number(speed.get("preprocess")),
        "inference_ms_per_image": _number(speed.get("inference")),
        "postprocess_ms_per_image": _number(speed.get("postprocess")),
        "source": "Ultralytics model.val on immutable test split",
    }


def _as_rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    with_cpu = value.cpu() if hasattr(value, "cpu") else value
    array = with_cpu.numpy() if hasattr(with_cpu, "numpy") else with_cpu
    as_list = array.tolist() if hasattr(array, "tolist") else list(array)
    if as_list and not isinstance(as_list[0], (list, tuple)):
        return [list(map(float, as_list))]
    return [list(map(float, row)) for row in as_list]


def _as_values(value: Any) -> list[float]:
    if value is None:
        return []
    with_cpu = value.cpu() if hasattr(value, "cpu") else value
    array = with_cpu.numpy() if hasattr(with_cpu, "numpy") else with_cpu
    values = array.tolist() if hasattr(array, "tolist") else list(array)
    return [float(item) for item in values]


def _ground_truth(path: Path, width: int, height: int, cat_id: int) -> list[list[float]]:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label {path}:{line_number}")
        class_id = int(parts[0])
        if class_id != cat_id:
            raise ValueError(f"unexpected class {class_id} in one-class dataset {path}:{line_number}")
        x_center, y_center, box_width, box_height = map(float, parts[1:])
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(truth: list[list[float]], predicted: list[list[float]], iou_threshold: float) -> tuple[int, int, int]:
    candidates = sorted(
        ((box_iou(gt, pred), gt_index, pred_index)
         for gt_index, gt in enumerate(truth)
         for pred_index, pred in enumerate(predicted)),
        reverse=True,
    )
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    for iou, gt_index, pred_index in candidates:
        if iou < iou_threshold:
            break
        if gt_index not in used_truth and pred_index not in used_predicted:
            used_truth.add(gt_index)
            used_predicted.add(pred_index)
    tp = len(used_truth)
    return tp, len(predicted) - tp, len(truth) - tp


def _counter() -> dict[str, int]:
    return {"images": 0, "objects": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0,
            "negative_images": 0, "empty_image_errors": 0, "missed_positive_images": 0}


def _finish_counter(counter: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
    return {
        **counter,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def _add_counts(counter: dict[str, int], *, truth: int, predicted: int,
                tp: int, fp: int, fn: int) -> None:
    counter["images"] += 1
    counter["objects"] += truth
    counter["predictions"] += predicted
    counter["tp"] += tp
    counter["fp"] += fp
    counter["fn"] += fn
    counter["negative_images"] += int(truth == 0)
    counter["empty_image_errors"] += int(truth == 0 and predicted > 0)
    counter["missed_positive_images"] += int(truth > 0 and tp == 0)


def fixed_confidence_evaluation(
    model: Any,
    dataset: DatasetInfo,
    *,
    model_cat_id: int,
    confidence: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    samples = dataset.split_samples("test")
    if not samples:
        raise ValueError("dataset test split is empty")
    totals = _counter()
    subsets: dict[str, dict[str, int]] = defaultdict(_counter)
    wall_ms: list[float] = []
    speed_components: dict[str, list[float]] = defaultdict(list)
    for index, sample in enumerate(samples, 1):
        image_path = dataset.image_path(sample)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode test image {image_path}")
        height, width = image.shape[:2]
        truth = _ground_truth(dataset.label_path(sample), width, height, dataset.cat_id)
        started = time.perf_counter()
        output = model.predict(
            source=str(image_path), conf=confidence, iou=iou_threshold, imgsz=imgsz,
            device=device, classes=[model_cat_id], verbose=False,
        )
        wall_ms.append((time.perf_counter() - started) * 1000)
        result = output[0] if isinstance(output, (list, tuple)) else output
        boxes_object = getattr(result, "boxes", None)
        predicted = _as_rows(getattr(boxes_object, "xyxy", None))
        classes = _as_values(getattr(boxes_object, "cls", None))
        if classes and len(classes) != len(predicted):
            raise ValueError("prediction boxes/classes length mismatch")
        if classes:
            predicted = [box for box, class_id in zip(predicted, classes)
                         if int(class_id) == model_cat_id]
        tp, fp, fn = match_boxes(truth, predicted, iou_threshold)
        _add_counts(totals, truth=len(truth), predicted=len(predicted), tp=tp, fp=fp, fn=fn)
        for subset in sample_subsets(sample):
            _add_counts(subsets[subset], truth=len(truth), predicted=len(predicted), tp=tp, fp=fp, fn=fn)
        for key, value in (getattr(result, "speed", None) or {}).items():
            numeric = _number(value)
            if numeric is not None:
                speed_components[str(key)].append(numeric)
        if progress:
            progress(index, len(samples))

    sorted_wall = sorted(wall_ms)
    p95_index = max(0, math.ceil(len(sorted_wall) * 0.95) - 1)
    timing = {
        "scope": "one image per predict call, including preprocess/inference/postprocess; excludes camera/feeder latency",
        "images": len(samples),
        "wall_ms_per_image_mean": statistics.fmean(wall_ms),
        "wall_ms_per_image_p95": sorted_wall[p95_index],
        "throughput_images_per_second": len(samples) / (sum(wall_ms) / 1000),
        "ultralytics_component_ms_per_image_mean": {
            key: statistics.fmean(values) for key, values in sorted(speed_components.items()) if values
        },
    }
    return {
        "confidence_threshold": confidence,
        "match_iou_threshold": iou_threshold,
        "overall": _finish_counter(totals),
        "subsets": {name: _finish_counter(counter) for name, counter in sorted(subsets.items())},
        "inference_timing": timing,
    }


def evaluate_artifact(
    model_path: Path,
    dataset: DatasetInfo,
    *,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    workers: int = 2,
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
    progress: Callable[[int, int], None] | None = None,
    model_factory: Callable[[Path], Any] = _load_yolo,
) -> dict[str, Any]:
    model_path = validate_model_artifact(model_path)
    model = model_factory(model_path)
    cat_id = resolve_model_cat_id(model)
    official_started = time.monotonic()
    validation = model.val(
        data=str(dataset.data_yaml), split="test", imgsz=imgsz, batch=batch,
        device=device, workers=workers, classes=[cat_id], single_cls=True, verbose=False,
    )
    official_elapsed = time.monotonic() - official_started
    custom_started = time.monotonic()
    custom = fixed_confidence_evaluation(
        model, dataset, model_cat_id=cat_id, confidence=confidence,
        iou_threshold=iou_threshold, imgsz=imgsz, device=device, progress=progress,
    )
    return {
        "artifact": {"path": str(model_path), "sha256": sha256_tree(model_path),
                     "format": "openvino" if model_path.is_dir() else "pytorch",
                     "cat_class_id": cat_id},
        "official": official_metrics(validation),
        "fixed_confidence": custom,
        "timing_seconds": {
            "official_validation": official_elapsed,
            "fixed_confidence": time.monotonic() - custom_started,
            "total": official_elapsed + (time.monotonic() - custom_started),
        },
    }


def _default_report_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("models/trained/yolo-reports") / f"evaluate-{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="immutable version directory (or its data.yaml)")
    parser.add_argument("--model", type=Path, required=True, help="trained .pt or OpenVINO directory")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.25, help="fixed-confidence operational pass")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS and TP matching IoU")
    parser.add_argument("--resource-interval", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preparation_started = time.monotonic()
    dataset = load_dataset(args.dataset, require_splits=("test",))
    model_path = validate_model_artifact(args.model)
    params = {"imgsz": args.imgsz, "batch": args.batch, "device": args.device,
              "workers": args.workers, "confidence": args.conf, "iou": args.iou}
    session = ReportSession(
        args.report or _default_report_path(), kind="yolo_evaluation", parameters=params,
        dataset=dataset, model={"path": str(model_path), "sha256": sha256_tree(model_path)},
        sample_interval_sec=args.resource_interval,
    )
    session.report["timing_seconds"]["preparation"] = time.monotonic() - preparation_started
    session.checkpoint()
    try:
        with convert_termination_to_interrupt():
            eval_started = time.monotonic()

            def progress(done: int, total: int) -> None:
                if done == total or done % 25 == 0:
                    session.progress(evaluation_images_done=done, evaluation_images_total=total)

            result = evaluate_artifact(
                model_path, dataset, imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, confidence=args.conf, iou_threshold=args.iou,
                progress=progress,
            )
            session.report["timing_seconds"]["evaluation"] = time.monotonic() - eval_started
            session.report["model"] = result.pop("artifact")
            session.report["metrics"] = result
            report = session.finish("completed")
            print_report_summary(report)
            return 0
    except (KeyboardInterrupt, InterruptedError) as exc:
        print_report_summary(session.finish("interrupted", error=exc))
        return 130
    except BaseException as exc:
        print_report_summary(session.finish("failed", error=exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
