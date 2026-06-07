"""Runtime EfficientNet-B0 cat classifier via OpenVINO. No torch at runtime.

Preprocessing is identical to donor model/loader.py:preprocess_for_inference —
resize to 256 on short side → center-crop 224 → ImageNet normalise → NCHW float32.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    # Must be bit-identical to the training transform
    # (torchvision Resize(256) + CenterCrop(224), PIL backend). Two rounding
    # rules have to match torchvision exactly or the runtime crop drifts a
    # row/column from training (verified by export_classifier.py's parity gate):
    #   - Resize: long edge = int(256 * long / short)  — truncation, not round.
    #   - CenterCrop: offset = round((dim - 224) / 2)   — round, not floor.
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
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 224, 224) float32


class CatClassifier:
    """Thread-safe EfficientNet-B0 classifier backed by an OpenVINO IR."""

    def __init__(self, model_dir: str | Path) -> None:
        import openvino as ov
        import openvino.properties.hint as hints

        model_dir = Path(model_dir)
        core = ov.Core()
        # Force FP32 inference. The CPU plugin otherwise runs FP32 IRs in bf16 by
        # default on AVX512_BF16/AMX hardware, whose ~8-bit mantissa shifts logits
        # vs the trained torch model. argmax usually survives, but the cat_score
        # confidences would drift from training (and the export parity gate
        # enforces FP32). Typed property — string keys can be silently ignored.
        # The speed cost is negligible for one B0 crop.
        self._compiled = core.compile_model(
            core.read_model(str(model_dir / "cat_classifier.xml")),
            "CPU",
            {hints.inference_precision: ov.Type.f32},
        )
        self.class_names: list[str] = json.loads(
            (model_dir / "classes.json").read_text(encoding="utf-8")
        )

    def classify(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        """Return (cat_name, confidence). crop_rgb is HWC uint8 RGB."""
        inp = _preprocess(crop_rgb)
        logits = self._compiled(inp)[0]  # (1, num_classes)
        probs = logits[0].astype(np.float64)
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])
