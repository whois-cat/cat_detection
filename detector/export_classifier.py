"""Build-time script: export cat_classifier.pt → OpenVINO IR + classes.json.

Run once during Docker image build:
  python export_classifier.py \
    --pt /app/models/cat_classifier.pt \
    --out /opt/models/cat_classifier_openvino/

Requires torch + torchvision + openvino (build-time only).

Parity gate
-----------
Exporting to OpenVINO is a classic drift point: the runtime path
(detector/classifier.py::_preprocess + OpenVINO inference) must produce the
same logits the trained torch model would, or identity decisions silently
diverge from training. After export this script PROVES that, and fails the
build on any mismatch:

  1. Preprocessing parity — runtime `_preprocess` (PIL: resize-256 short side
     → center-crop 224 → ImageNet norm) is compared element-wise against the
     canonical torchvision inference transform the classifier was trained with
     (documented in training/torch_dataset.py; the donor loader.py reimplemented
     the same steps). max|Δ| must be < PARITY_TOL. This catches PIL-vs-tensor
     resize/antialias/rounding drift.
  2. Model parity — for several crops, the torch model and the exported
     OpenVINO model are both fed the *same* `_preprocess` output and their
     logits compared. max|Δlogits| must be < PARITY_TOL.

Crops: pass `--crops <dir>` to use real cat crops (preferred — exercises the
true input distribution); otherwise a deterministic set of synthetic crops of
varied sizes is generated so the gate always runs in CI/build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Logits are FP32→FP32 (no quantization on the classifier), and preprocessing
# is bit-identical in practice, so the tolerance is tight on purpose.
PARITY_TOL = 1e-3

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_crops(crops_dir: Path | None, n_synthetic: int = 6) -> list:
    """Return a list of HWC uint8 RGB ndarrays to validate on."""
    import numpy as np

    if crops_dir is not None:
        from PIL import Image

        paths = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
        if not paths:
            sys.exit(f"[parity] --crops {crops_dir} has no images ({sorted(_IMG_EXTS)})")
        crops = [np.array(Image.open(p).convert("RGB")) for p in paths]
        print(f"[parity] loaded {len(crops)} real crops from {crops_dir}", flush=True)
        return crops

    # Deterministic synthetic crops of varied shapes — still exercise the resize
    # / center-crop / normalize path and the full torch↔OpenVINO numeric path.
    rng = np.random.default_rng(0)
    shapes = [(300, 300), (480, 640), (640, 480), (257, 257), (224, 400), (512, 288)]
    crops = [
        rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        for (h, w) in shapes[:n_synthetic]
    ]
    print(f"[parity] no --crops given; using {len(crops)} synthetic crops", flush=True)
    return crops


def _check_preprocess_parity(crops: list) -> None:
    """Runtime _preprocess vs the donor torchvision inference transform."""
    import numpy as np
    import torch
    from torchvision import transforms

    from classifier import _preprocess  # the runtime preprocessing under test

    donor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    worst = 0.0
    for crop in crops:
        ours = _preprocess(crop)[0]              # (3, 224, 224) float32
        ref = donor(crop).numpy()                # (3, 224, 224) float32
        worst = max(worst, float(np.max(np.abs(ours - ref))))
    print(f"[parity] preprocessing max|Δ| vs torchvision donor = {worst:.2e}", flush=True)
    if worst >= PARITY_TOL:
        sys.exit(
            f"[parity] FAIL: runtime _preprocess drifted from the training "
            f"transform (max|Δ|={worst:.2e} >= {PARITY_TOL:.0e}). "
            "Reconcile detector/classifier.py::_preprocess with the torchvision "
            "Resize(256)+CenterCrop(224)+Normalize pipeline before shipping."
        )


def _check_model_parity(model, compiled, crops: list) -> None:
    """torch model vs exported OpenVINO model on identical _preprocess inputs."""
    import numpy as np
    import torch

    from classifier import _preprocess

    worst = 0.0
    for crop in crops:
        inp = _preprocess(crop)                       # (1, 3, 224, 224) float32
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(inp)).numpy()[0]
        ov_logits = compiled(inp)[0][0]               # first output, drop batch dim
        worst = max(worst, float(np.max(np.abs(torch_logits - ov_logits))))
    print(f"[parity] torch↔OpenVINO max|Δlogits| = {worst:.2e}", flush=True)
    if worst >= PARITY_TOL:
        sys.exit(
            f"[parity] FAIL: OpenVINO logits drifted from torch "
            f"(max|Δ|={worst:.2e} >= {PARITY_TOL:.0e}). The export is not "
            "faithful — do not ship this IR."
        )


def export(pt_path: Path, out_dir: Path, crops_dir: Path | None = None) -> None:
    import openvino as ov
    import torch
    import torch.nn as nn
    from torchvision import models

    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    ov_model = ov.convert_model(model, example_input=dummy)

    xml_path = out_dir / "cat_classifier.xml"
    ov.save_model(ov_model, str(xml_path))
    print(f"[export] OpenVINO IR → {xml_path}", flush=True)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(class_names), encoding="utf-8")
    print(f"[export] classes → {classes_path} : {class_names}", flush=True)

    # ---- parity gate (fails the build on any drift) ----
    crops = _load_crops(crops_dir)
    _check_preprocess_parity(crops)
    compiled = ov.Core().compile_model(ov_model, "CPU")
    _check_model_parity(model, compiled, crops)
    print("[parity] OK — preprocessing and logits match within tolerance", flush=True)


def main() -> None:
    import argparse

    # The script lives next to classifier.py; ensure it's importable when run
    # from any CWD during the build.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--crops", default=None,
                   help="optional dir of real cat crops for the parity check")
    args = p.parse_args()
    crops_dir = Path(args.crops) if args.crops else None
    export(Path(args.pt), Path(args.out), crops_dir)


if __name__ == "__main__":
    main()
