#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "pillow>=11.1.0",
#   "torch>=2.0.0",
#   "torchvision>=0.15.0",
# ]
# ///

from __future__ import annotations

import copy
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from pipeline_db import DEFAULT_CROPS_DIR, DEFAULT_MODEL_PATH, DEFAULT_TRAIN_DIR, PROJECT_ROOT


UNSORTED_LABEL = "unsorted"
RESERVED_LABELS = {"unsorted", "groups"}
PREVIEW_FILENAME = "_preview.jpg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 7
FREEZE_BACKBONE_EPOCHS = 5
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "cat_classifier_best.pt"


@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def is_valid_class_folder(folder: Path) -> bool:
    return folder.is_dir() and folder.name != UNSORTED_LABEL


def build_train_dir(crops_dir: Path, train_dir: Path) -> int:
    import shutil

    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)

    total_links = 0
    for label_dir in sorted(crops_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name in RESERVED_LABELS:
            continue

        dest_dir = train_dir / label_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for source in sorted(label_dir.iterdir()):
            if not source.is_file():
                continue
            if source.name == PREVIEW_FILENAME:
                continue
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            link_path = dest_dir / source.name
            link_path.symlink_to(source.resolve())
            total_links += 1

    return total_links


def stratified_split(
    dataset: datasets.ImageFolder,
    val_fraction: float,
) -> tuple[list[int], list[int]]:
    per_class: dict[int, list[int]] = {}
    for index, (_, label) in enumerate(dataset.samples):
        per_class.setdefault(label, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for label_indices in per_class.values():
        split_point = max(1, int(len(label_indices) * (1 - val_fraction)))
        train_indices.extend(label_indices[:split_point])
        val_indices.extend(label_indices[split_point:])

    return train_indices, val_indices


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = not frozen


def build_optimizer(model: nn.Module, lr: float) -> AdamW:
    classifier_params = [
        p for name, p in model.named_parameters() if name.startswith("classifier")
    ]
    backbone_params = [
        p for name, p in model.named_parameters() if not name.startswith("classifier")
    ]
    return AdamW(
        [
            {"params": classifier_params, "lr": lr},
            {"params": backbone_params, "lr": lr * 0.1},
        ]
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[list[int]]:
    model.eval()
    matrix = [[0] * num_classes for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for true_label, pred_label in zip(
                labels.cpu().tolist(), predicted.cpu().tolist()
            ):
                matrix[true_label][pred_label] += 1

    return matrix


def format_confusion_matrix(matrix: list[list[int]], class_names: list[str]) -> str:
    max_name_len = max(len(name) for name in class_names)
    header = " " * (max_name_len + 2) + "  ".join(f"{name:>6}" for name in class_names)
    lines = [header]
    for row_index, row in enumerate(matrix):
        label = class_names[row_index].ljust(max_name_len)
        row_str = "  ".join(f"{count:>6}" for count in row)
        lines.append(f"{label}  {row_str}")
    return "\n".join(lines)


def compute_per_class_metrics(
    matrix: list[list[int]],
    class_names: list[str],
) -> list[tuple[str, float, float]]:
    metrics: list[tuple[str, float, float]] = []
    num_classes = len(class_names)

    for class_index in range(num_classes):
        tp = matrix[class_index][class_index]
        fp = sum(matrix[r][class_index] for r in range(num_classes)) - tp
        fn = sum(matrix[class_index]) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics.append((class_names[class_index], precision, recall))

    return metrics


@click.command()
@click.option(
    "--crops-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=DEFAULT_CROPS_DIR,
    show_default=True,
    help="Root directory with per-class crop subfolders.",
)
@click.option(
    "--epochs",
    type=click.IntRange(min=1),
    default=DEFAULT_EPOCHS,
    show_default=True,
    help="Maximum number of training epochs.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--lr",
    type=float,
    default=DEFAULT_LR,
    show_default=True,
    help="Learning rate for the classifier head (backbone uses lr * 0.1).",
)
@click.option(
    "--patience",
    type=click.IntRange(min=1),
    default=DEFAULT_PATIENCE,
    show_default=True,
    help="Early stopping patience (epochs without val accuracy improvement).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Path to save the best model checkpoint.",
)
def main(
    crops_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    output: Path,
) -> None:
    """Train a cat identity classifier on labelled crops."""

    run_train_classifier(
        crops_dir=crops_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        output=output,
    )


def run_train_classifier(
    crops_dir: Path = DEFAULT_CROPS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    device = select_device()
    click.echo(f"train: device={device}")

    total_links = build_train_dir(crops_dir, DEFAULT_TRAIN_DIR)
    click.echo(f"train: linked {total_links} crops into {DEFAULT_TRAIN_DIR}")

    train_transform, val_transform = build_transforms()

    full_dataset = datasets.ImageFolder(
        root=str(DEFAULT_TRAIN_DIR),
        is_valid_file=lambda path: Path(path).suffix.lower() in IMAGE_EXTENSIONS,
    )
    full_dataset.samples = [
        (path, label)
        for path, label in full_dataset.samples
        if is_valid_class_folder(Path(path).parent)
    ]
    full_dataset.targets = [label for _, label in full_dataset.samples]

    valid_classes = sorted(
        name
        for name in full_dataset.class_to_idx
        if name != UNSORTED_LABEL
    )
    if len(valid_classes) < 2:
        raise click.ClickException(
            f"need at least 2 labelled class folders in {crops_dir} "
            f"(found: {valid_classes})"
        )

    class_to_idx = {name: idx for idx, name in enumerate(valid_classes)}
    full_dataset.samples = [
        (path, class_to_idx[full_dataset.classes[old_label]])
        for path, old_label in full_dataset.samples
        if full_dataset.classes[old_label] in class_to_idx
    ]
    full_dataset.targets = [label for _, label in full_dataset.samples]
    full_dataset.classes = valid_classes
    full_dataset.class_to_idx = class_to_idx

    class_counts = Counter(full_dataset.targets)
    click.echo(f"train: classes={valid_classes}")
    for class_name in valid_classes:
        idx = class_to_idx[class_name]
        click.echo(f"  {class_name}: {class_counts.get(idx, 0)} samples")

    train_indices, val_indices = stratified_split(full_dataset, val_fraction=0.2)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type != "cpu",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type != "cpu",
    )

    click.echo(f"train: train_samples={len(train_indices)} val_samples={len(val_indices)}")

    num_classes = len(valid_classes)
    model = build_model(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history = TrainHistory()

    set_backbone_frozen(model, frozen=True)
    click.echo(f"train: backbone frozen for first {FREEZE_BACKBONE_EPOCHS} epochs")

    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            set_backbone_frozen(model, frozen=False)
            click.echo("train: backbone unfrozen")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.train_losses.append(train_loss)
        history.train_accs.append(train_acc)
        history.val_losses.append(val_loss)
        history.val_accs.append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        click.echo(
            f"epoch {epoch:>3}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}"
        )

        if epochs_without_improvement >= patience:
            click.echo(
                f"train: early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    elapsed_seconds = time.perf_counter() - started_at
    click.echo(f"train: elapsed {elapsed_seconds:.1f}s")

    if best_state_dict is None:
        raise click.ClickException("no improvement observed during training")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state_dict,
            "class_names": valid_classes,
            "num_classes": num_classes,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "total_epochs": len(history.train_losses),
        },
        str(output),
    )
    click.echo(f"train: best model saved to {output} (epoch {best_epoch}, val_acc={best_val_acc:.4f})")

    model.load_state_dict(best_state_dict)
    confusion = compute_confusion_matrix(model, val_loader, device, num_classes)
    click.echo("\nconfusion matrix (rows=true, cols=predicted):")
    click.echo(format_confusion_matrix(confusion, valid_classes))

    click.echo("\nper-class metrics:")
    per_class = compute_per_class_metrics(confusion, valid_classes)
    for class_name, precision, recall in per_class:
        click.echo(f"  {class_name}: precision={precision:.4f} recall={recall:.4f}")


if __name__ == "__main__":
    main()
