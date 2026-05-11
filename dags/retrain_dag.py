from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "/app/scripts")

from airflow import DAG
from airflow.operators.python import PythonOperator


DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

VM_URL = os.environ.get("VM_URL", "http://victoriametrics:8428")
AUTO_LABEL_THRESHOLD = 0.8


def _dedup_crops() -> None:
    from deduplicate_frames import run_deduplicate_crops
    run_deduplicate_crops()


def _auto_label() -> None:
    from auto_label import run_auto_label
    run_auto_label(threshold=AUTO_LABEL_THRESHOLD)


def _train_and_log() -> None:
    import time
    import mlflow
    from train_classifier import (
        DEFAULT_OUTPUT,
        TrainHistory,
        build_model,
        build_optimizer,
        build_transforms,
        build_train_dir,
        compute_confusion_matrix,
        compute_per_class_metrics,
        run_epoch,
        run_train_classifier,
        select_device,
        set_backbone_frozen,
        stratified_split,
    )
    from pipeline_db import DEFAULT_CROPS_DIR, DEFAULT_TRAIN_DIR, DEFAULT_MODEL_PATH
    from metrics import push_metric
    from torchvision import datasets
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Subset
    import copy

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    UNSORTED_LABEL = "unsorted"
    RESERVED_LABELS = {"unsorted", "groups"}
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LR = 1e-3
    DEFAULT_PATIENCE = 7
    FREEZE_BACKBONE_EPOCHS = 5

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("cat_classifier")

    with mlflow.start_run():
        started = time.perf_counter()
        run_train_classifier()
        elapsed = time.perf_counter() - started

        mlflow.log_param("lr", DEFAULT_LR)
        mlflow.log_param("epochs", DEFAULT_EPOCHS)
        mlflow.log_param("batch_size", DEFAULT_BATCH_SIZE)

        device = select_device()
        build_train_dir(DEFAULT_CROPS_DIR, DEFAULT_TRAIN_DIR)

        full_dataset = datasets.ImageFolder(
            root=str(DEFAULT_TRAIN_DIR),
            is_valid_file=lambda p: Path(p).suffix.lower() in IMAGE_EXTENSIONS,
        )
        valid_classes = sorted(
            n for n in full_dataset.class_to_idx if n not in RESERVED_LABELS
        )
        if len(valid_classes) >= 2:
            class_to_idx = {n: i for i, n in enumerate(valid_classes)}
            full_dataset.samples = [
                (p, class_to_idx[full_dataset.classes[lbl]])
                for p, lbl in full_dataset.samples
                if full_dataset.classes[lbl] in class_to_idx
            ]
            full_dataset.targets = [lbl for _, lbl in full_dataset.samples]
            full_dataset.classes = valid_classes
            full_dataset.class_to_idx = class_to_idx

            _, val_transform = build_transforms()
            _, val_indices = stratified_split(full_dataset, 0.2)
            val_ds = Subset(full_dataset, val_indices)
            val_ds.dataset = copy.copy(full_dataset)
            val_ds.dataset.transform = val_transform
            val_loader = DataLoader(val_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

            import torch
            checkpoint = torch.load(str(DEFAULT_MODEL_PATH), map_location=device, weights_only=False)
            model = build_model(len(valid_classes))
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)

            confusion = compute_confusion_matrix(model, val_loader, device, len(valid_classes))
            per_class = compute_per_class_metrics(confusion, valid_classes)

            _, val_acc = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
            mlflow.log_metric("accuracy", val_acc)
            for class_name, precision, recall in per_class:
                mlflow.log_metric(f"precision_{class_name}", precision)
                mlflow.log_metric(f"recall_{class_name}", recall)

            mlflow.log_artifact(str(DEFAULT_MODEL_PATH))

            push_metric("model_accuracy", val_acc, vm_url=VM_URL)
            push_metric("retrain_duration_seconds", elapsed, vm_url=VM_URL)
            for class_name, precision, recall in per_class:
                push_metric("model_precision", precision, {"class": class_name}, VM_URL)
                push_metric("model_recall", recall, {"class": class_name}, VM_URL)

            from pipeline_db import DEFAULT_CROPS_DIR as CROPS_DIR
            for label_dir in CROPS_DIR.iterdir():
                if label_dir.is_dir() and label_dir.name not in RESERVED_LABELS:
                    count = sum(1 for f in label_dir.iterdir() if f.is_file())
                    push_metric("retrain_crops_total", float(count), {"cat_name": label_dir.name}, VM_URL)


with DAG(
    dag_id="cat_retrain",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["cat_detection"],
) as dag:

    t1 = PythonOperator(task_id="dedup_crops", python_callable=_dedup_crops)
    t2 = PythonOperator(task_id="auto_label", python_callable=_auto_label)
    t3 = PythonOperator(task_id="train_and_log", python_callable=_train_and_log)

    t1 >> t2 >> t3
