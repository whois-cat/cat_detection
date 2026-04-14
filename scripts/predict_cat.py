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

from pathlib import Path

import click
import torch
from PIL import Image
from torchvision import models, transforms

from pipeline_db import DEFAULT_MODEL_PATH


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_classifier(
    model_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, list[tuple[str, float]]]:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    pil_image.close()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)

    sorted_indices = probabilities.argsort(descending=True).tolist()
    top_class = class_names[sorted_indices[0]]
    top_confidence = float(probabilities[sorted_indices[0]])

    all_predictions = [
        (class_names[idx], float(probabilities[idx])) for idx in sorted_indices
    ]

    return top_class, top_confidence, all_predictions


def collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        paths: list[Path] = []
        for child in sorted(input_path.iterdir()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(child)
        return paths

    return []


@click.command()
@click.option(
    "--model",
    type=click.Path(path_type=Path, exists=True, file_okay=True, dir_okay=False),
    default=DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the trained model checkpoint.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to an image or a directory of images.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum confidence to accept a prediction (below = unknown).",
)
def main(model: Path, input_path: Path, threshold: float) -> None:
    """Predict cat identity on image(s) using a trained classifier."""

    run_predict_cat(
        model_path=model,
        input_path=input_path,
        threshold=threshold,
    )


def run_predict_cat(
    model_path: Path = DEFAULT_MODEL_PATH,
    input_path: Path = Path("."),
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    device = select_device()
    classifier, class_names = load_classifier(model_path, device)
    click.echo(f"predict: model={model_path.name} device={device} classes={class_names}")

    image_paths = collect_image_paths(input_path)
    if not image_paths:
        raise click.ClickException(f"no images found at {input_path}")

    for image_path in image_paths:
        top_class, top_confidence, all_predictions = predict_image(
            classifier, image_path, class_names, device
        )

        label = top_class if top_confidence >= threshold else "unknown"
        click.echo(f"  {image_path.name}: {label} ({top_confidence:.4f})")

        if top_confidence < threshold:
            top_entries = all_predictions[:3]
            details = ", ".join(
                f"{name}={conf:.3f}" for name, conf in top_entries
            )
            click.echo(f"    top-3: {details}")

    click.echo(f"predict: {len(image_paths)} images processed")


if __name__ == "__main__":
    main()
