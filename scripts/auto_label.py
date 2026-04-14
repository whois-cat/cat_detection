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

import shutil
from pathlib import Path

import click

from pipeline_db import DEFAULT_CROPS_DIR, DEFAULT_MODEL_PATH, select_inference_device


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_classifier(model_path: Path, device):
    import torch
    from torchvision import models

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


def _build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@click.command()
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
    help="Minimum confidence to accept a prediction; below → delete crop.",
)
def main(threshold: float) -> None:
    """Auto-label unsorted crops using the trained classifier."""
    run_auto_label(threshold=threshold)


def run_auto_label(threshold: float = 0.8) -> None:
    import torch
    from PIL import Image

    device_str = select_inference_device()
    device = torch.device(device_str)

    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise click.ClickException(f"model not found: {model_path}")

    model, class_names = _load_classifier(model_path, device)
    transform = _build_transform()

    unsorted_dir = DEFAULT_CROPS_DIR / "unsorted"
    if not unsorted_dir.exists():
        click.echo("auto-label: unsorted dir not found, nothing to do")
        return

    crop_files = sorted(
        p for p in unsorted_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not crop_files:
        click.echo("auto-label: no crops in unsorted/")
        return

    click.echo(f"auto-label: device={device_str} model={model_path.name} crops={len(crop_files)}")

    moved = 0
    deleted = 0

    for crop_path in crop_files:
        try:
            pil_image = Image.open(crop_path).convert("RGB")
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            pil_image.close()
        except Exception as exc:
            click.echo(f"auto-label: skip {crop_path.name} ({exc})")
            continue

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        top_index = int(probabilities.argmax().item())
        top_confidence = float(probabilities[top_index].item())
        top_name = class_names[top_index]

        if top_confidence >= threshold:
            dest_dir = DEFAULT_CROPS_DIR / top_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(crop_path), str(dest_dir / crop_path.name))
            moved += 1
        else:
            crop_path.unlink()
            deleted += 1

    click.echo(f"auto-label: moved={moved} deleted={deleted} total={len(crop_files)}")


if __name__ == "__main__":
    main()
