"""Per-cat identity classifier (EfficientNet-B0, OpenVINO IR). No torch at runtime.

Preprocessing must be bit-identical to training (torchvision Resize(256) +
CenterCrop(224), PIL backend); export_classifier.py's parity gate checks it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB crop → (1, 3, 224, 224) float32, matching torchvision:
    Resize long edge = int(256 * long / short) (truncation); CenterCrop offset =
    round((dim - 224) / 2)."""
    from PIL import Image

    img = Image.fromarray(crop_rgb).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = 256, int(256 * h / w)
    else:
        nw, nh = int(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    w, h = img.size
    left = int(round((w - 224) / 2.0))
    top = int(round((h - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]


class CatClassifier:
    """Model dir holds cat_classifier.xml/.bin and classes.json (label list)."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        for name in ("cat_classifier.xml", "cat_classifier.bin", "classes.json"):
            if not (model_dir / name).exists():
                raise FileNotFoundError(
                    f"classifier model file missing: {model_dir / name} "
                    "(needs cat_classifier.xml, cat_classifier.bin, classes.json; is a model promoted and mounted?)")
        names = json.loads((model_dir / "classes.json").read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError(f"classes.json must be a non-empty list of labels ({model_dir})")
        self.class_names: list[str] = names
        # The `current` symlink's target names the version.
        self.version = model_dir.resolve().name

        import openvino as ov
        import openvino.properties.hint as hints

        core = ov.Core()
        # Force FP32: on bf16-capable CPUs the plugin would otherwise run FP32
        # IRs in bf16, shifting confidences away from the trained model.
        self._compiled = core.compile_model(
            core.read_model(str(model_dir / "cat_classifier.xml")), "CPU",
            {hints.inference_precision: ov.Type.f32})
        try:
            out_dim = self._compiled.output(0).partial_shape[-1].get_length()
        except Exception:
            out_dim = None  # dynamic
        if out_dim is not None and out_dim != len(names):
            raise ValueError(f"classifier outputs {out_dim} classes but classes.json lists {len(names)} ({model_dir})")

    def probs(self, crops_rgb: list[np.ndarray]) -> list[dict[str, float]]:
        """Probability per class for each crop, in one inference call."""
        if not crops_rgb:
            return []
        logits = np.asarray(self._compiled(np.concatenate([preprocess(c) for c in crops_rgb]))[0], dtype=np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return [{n: float(v) for n, v in zip(self.class_names, row)} for row in p]
