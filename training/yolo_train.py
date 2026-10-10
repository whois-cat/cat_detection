"""Fine-tune YOLO from an existing ``.pt`` on an immutable reviewed dataset.

This is an offline/manual command.  It uses the standard Ultralytics trainer,
evaluates the best checkpoint on the held-out test split, writes a durable JSON
report, and never replaces or restarts the runtime model.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.mltracking import start_run
from training.yolo_common import (
    DatasetInfo,
    ReportSession,
    convert_termination_to_interrupt,
    link_or_copy,
    load_dataset,
    print_report_summary,
    sha256_file,
    ultralytics_data_yaml,
    validate_training_weights,
)
from training.yolo_evaluate import evaluate_artifact


def _load_yolo(path: Path):
    from ultralytics import YOLO
    return YOLO(str(path), task="detect")


def _run_name() -> str:
    return "yolo-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _finite_metrics(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values[str(key)] = number
    return values


def merge_training_dataset(
    current: DatasetInfo, replays: list[DatasetInfo], dest: Path
) -> dict[str, Any]:
    """Train on the current train split plus old versions' train splits.

    Evaluation stays on ``current`` (its own val/test). Replay samples that
    collide with the current val/test by ``sample_id``, image ``sha256`` or visit
    ``group_id`` are dropped, so mixing in old data never leaks into evaluation.
    Returns a merged ``data.yaml`` (``val`` = current val) and a mix summary.
    """
    held_ids: set[str] = set()
    held_sha: set[str] = set()
    held_groups: set[str] = set()
    for split in ("val", "test"):
        for sample in current.split_samples(split):
            held_ids.add(sample["sample_id"])
            held_sha.add(sample.get("sha256"))
            held_groups.add(sample.get("group_id"))

    def link_sample(dataset: DatasetInfo, sample: dict[str, Any], subdir: str) -> None:
        sid = sample["sample_id"]
        link_or_copy(dataset.image_path(sample), dest / "images" / subdir / f"{sid}.jpg")
        link_or_copy(dataset.label_path(sample), dest / "labels" / subdir / f"{sid}.txt")

    seen: set[str] = set()
    for sample in current.split_samples("train"):
        link_sample(current, sample, "train")
        seen.add(sample["sample_id"])
    current_train = len(seen)
    for sample in current.split_samples("val"):
        link_sample(current, sample, "val")

    added = skipped = 0
    for replay in replays:
        for sample in replay.split_samples("train"):
            sid = sample["sample_id"]
            if (sid in seen or sid in held_ids
                    or sample.get("sha256") in held_sha
                    or sample.get("group_id") in held_groups):
                skipped += 1
                continue
            link_sample(replay, sample, "train")
            seen.add(sid)
            added += 1

    (dest / "data.yaml").write_text(
        "\n".join([f"path: {dest.resolve()}", "train: images/train", "val: images/val",
                   "names:", "  0: cat", ""]),
        encoding="utf-8",
    )
    return {
        "data_yaml": dest / "data.yaml",
        "current_train_images": current_train,
        "replay_added": added,
        "replay_skipped_leakage": skipped,
        "merged_train_images": current_train + added,
        "replay_versions": [r.manifest.get("version_id") or str(r.version_dir)
                            for r in replays],
    }


def train(
    *,
    dataset_path: Path,
    weights_path: Path,
    replay_paths: list[Path] = (),
    output_root: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    seed: int,
    patience: int,
    optimizer: str,
    learning_rate: float | None,
    confidence: float,
    iou_threshold: float,
    resource_interval: float,
    model_factory=_load_yolo,
) -> tuple[int, dict[str, Any]]:
    if epochs <= 0 or imgsz <= 0 or batch == 0 or workers < 0:
        raise ValueError("epochs/imgsz must be positive, batch non-zero, and workers non-negative")
    preparation_started = time.monotonic()
    dataset = load_dataset(dataset_path, require_splits=("train", "val", "test"))
    weights = validate_training_weights(weights_path)
    output_root = output_root.expanduser().resolve()
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse training run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    # Mix in old dataset versions' TRAIN samples so fine-tuning does not forget
    # earlier appearances (e.g. the cat before the haircut). Evaluation stays on
    # the current version; colliding replay samples are dropped.
    replays = [load_dataset(path, require_splits=("train",)) for path in replay_paths]
    if replays:
        merge = merge_training_dataset(dataset, replays, run_dir / "merged_dataset")
        train_data_yaml = merge["data_yaml"]
        replay_summary = {key: value for key, value in merge.items() if key != "data_yaml"}
    else:
        train_data_yaml = ultralytics_data_yaml(dataset)
        replay_summary = None
    parameters: dict[str, Any] = {
        "epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device,
        "workers": workers, "seed": seed, "patience": patience,
        "optimizer": optimizer, "learning_rate": learning_rate,
        "evaluation_confidence": confidence, "evaluation_iou": iou_threshold,
        "augmentation": "Ultralytics train defaults; training split only",
        "architecture": "inherited from initial .pt checkpoint",
        "replay": replay_summary,
    }
    session = ReportSession(
        report_path, kind="yolo_training", parameters=parameters, dataset=dataset,
        model={"initial_weights": {"path": str(weights), "sha256": sha256_file(weights)}},
        sample_interval_sec=resource_interval,
    )
    session.report["timing_seconds"]["preparation"] = time.monotonic() - preparation_started
    session.report["artifacts"] = {"run_directory": str(run_dir), "report": str(report_path)}
    session.checkpoint()
    mlflow_run = start_run(
        "cat_detector_yolo", run_name=run_name,
        params={**parameters, "dataset_sha256": dataset.dataset_sha256,
                "initial_weights_sha256": sha256_file(weights)},
        tags={"dataset_version": dataset.manifest.get("version_id", "unknown")},
    )
    try:
        with convert_termination_to_interrupt():
            model = model_factory(weights)

            def epoch_finished(trainer) -> None:
                current = int(getattr(trainer, "epoch", -1)) + 1
                metrics = _finite_metrics(getattr(trainer, "metrics", {}))
                session.progress(epoch=current, epochs_total=epochs,
                                 latest_training_metrics=metrics)
                mlflow_run.log_metrics(metrics, step=current)

            if hasattr(model, "add_callback"):
                model.add_callback("on_fit_epoch_end", epoch_finished)
            train_started = time.monotonic()
            kwargs: dict[str, Any] = {
                "data": str(train_data_yaml), "project": str(output_root), "name": run_name,
                "exist_ok": True, "epochs": epochs, "imgsz": imgsz, "batch": batch,
                "device": device, "workers": workers, "seed": seed, "deterministic": True,
                "patience": patience, "optimizer": optimizer, "augment": True,
                "plots": True, "save": True, "verbose": True,
            }
            if learning_rate is not None:
                kwargs["lr0"] = learning_rate
            training_result = model.train(**kwargs)
            session.report["timing_seconds"]["training"] = time.monotonic() - train_started
            trainer = getattr(model, "trainer", None)
            best_value = getattr(trainer, "best", None)
            best = Path(str(best_value)).resolve() if best_value else (run_dir / "weights" / "best.pt")
            if not best.is_file():
                raise FileNotFoundError(f"Ultralytics did not produce best checkpoint: {best}")
            last_value = getattr(trainer, "last", None)
            last = Path(str(last_value)).resolve() if last_value else (run_dir / "weights" / "last.pt")
            session.report["model"]["best_checkpoint"] = {
                "path": str(best), "sha256": sha256_file(best)}
            session.report["artifacts"]["best_checkpoint"] = str(best)
            if last.is_file():
                session.report["model"]["last_checkpoint"] = {
                    "path": str(last), "sha256": sha256_file(last)}
                session.report["artifacts"]["last_checkpoint"] = str(last)
            training_metrics = _finite_metrics(getattr(training_result, "results_dict", {}))
            session.report["metrics"]["training_validation"] = training_metrics
            session.checkpoint()

            evaluation_started = time.monotonic()

            def evaluation_progress(done: int, total: int) -> None:
                if done == total or done % 25 == 0:
                    session.progress(evaluation_images_done=done, evaluation_images_total=total)

            evaluation = evaluate_artifact(
                best, dataset, imgsz=imgsz, batch=batch, device=device, workers=workers,
                confidence=confidence, iou_threshold=iou_threshold,
                progress=evaluation_progress, plots_dir=run_dir / "test_eval",
            )
            session.report["timing_seconds"]["evaluation"] = time.monotonic() - evaluation_started
            session.report["metrics"]["official"] = evaluation["official"]
            session.report["metrics"]["fixed_confidence"] = evaluation["fixed_confidence"]
            session.report["metrics"]["evaluation_internal_timing"] = evaluation["timing_seconds"]
            numeric = {f"test_{key}": value for key, value in evaluation["official"].items()
                       if isinstance(value, (int, float))}
            mlflow_run.log_metrics(numeric)
            report = session.finish("completed")
            mlflow_run.log_artifact(report_path)
            mlflow_run.end("FINISHED")
            print_report_summary(report)
            print(f"best checkpoint (not deployed): {best}")
            return 0, report
    except (KeyboardInterrupt, InterruptedError) as exc:
        report = session.finish("interrupted", error=exc)
        mlflow_run.log_artifact(report_path)
        mlflow_run.end("KILLED")
        print_report_summary(report)
        return 130, report
    except BaseException as exc:
        report = session.finish("failed", error=exc)
        mlflow_run.log_artifact(report_path)
        mlflow_run.end("FAILED")
        print_report_summary(report)
        return 1, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="immutable version directory (or its data.yaml)")
    parser.add_argument("--weights", type=Path, required=True,
                        help="existing pretrained trainable .pt checkpoint")
    parser.add_argument("--replay-version", action="append", dest="replay_versions",
                        type=Path, default=None, metavar="VERSION_DIR",
                        help="older dataset version to mix into training (repeatable); "
                             "only its train split is added, collisions with the current "
                             "val/test are dropped")
    parser.add_argument("--output-root", type=Path, default=Path("models/trained"))
    parser.add_argument("--name", default=None, help="new run directory name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--resource-interval", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, _report = train(
        dataset_path=args.dataset, weights_path=args.weights,
        replay_paths=args.replay_versions or [],
        output_root=args.output_root, run_name=args.name or _run_name(),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, seed=args.seed, patience=args.patience,
        optimizer=args.optimizer, learning_rate=args.lr0,
        confidence=args.eval_conf, iou_threshold=args.eval_iou,
        resource_interval=args.resource_interval,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
