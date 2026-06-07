"""Build-time script: export cat_classifier.pt → OpenVINO IR + classes.json.

Run once during Docker image build:
  python export_classifier.py \
    --pt /app/models/cat_classifier.pt \
    --out /opt/models/cat_classifier_openvino/
  # optional, preferred: validate on real crops
  #   --crops /app/models/parity_crops/

Requires torch + torchvision + openvino (build-time only).

Parity gate
-----------
Exporting to OpenVINO is a classic drift point: the runtime path
(detector/classifier.py::_preprocess + OpenVINO inference) must produce the
same logits the trained torch model would, or identity decisions silently
diverge from training. After export this script PROVES that and FAILS the build
on any mismatch. Two independent gates:

  1. Preprocessing parity — runtime `_preprocess` (PIL: resize-256 short side →
     center-crop 224 → ImageNet norm) compared element-wise against the
     canonical torchvision inference transform the classifier was trained with
     (documented in training/torch_dataset.py). Catches PIL-vs-tensor
     resize/antialias/rounding drift.
  2. Model parity — the exported OpenVINO model (re-read from the saved .xml, so
     we test the *shipped artifact*, not an in-memory object) and the torch
     model are fed the *same* normalized NCHW float32 tensor; their logits and
     argmax must agree.

Both gates require max|Δ| < PARITY_TOL. The model gate additionally requires the
argmax (the actual identity decision) to match on every sample.

Correctness-by-construction notes (each kills a known cause of false/real drift):
  - FP32 inference is forced via INFERENCE_PRECISION_HINT=f32 on the CPU plugin.
    This is THE one that mattered: the plugin runs FP32 IRs in bf16 by default on
    AVX512_BF16/AMX CPUs, whose ~8-bit mantissa shifts logits by up to ~1e1
    (argmax usually survives, magnitudes don't). classifier.py sets the same hint
    at runtime, so the gate matches production.
  - The IR is saved with compress_to_fp16=False (the on-disk weights stay FP32).
  - The torch model is in eval() under torch.inference_mode() for the reference
    forward (BatchNorm in train mode would recompute batch stats and diverge);
    convert_model traces under no_grad (inference_mode breaks torch.jit.trace).
  - Both backends receive the byte-for-byte same float32 array, produced by the
    *runtime* classifier._preprocess on real crops (--crops) or on synthetic
    images — so inputs stay in the true ImageNet-normalized range, never raw
    N(0,1) (out-of-distribution, inflates logit diffs) and never zeros.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Logits are FP32→FP32 (no quantization on the classifier) and preprocessing is
# bit-identical, so the tolerance is tight on purpose. Do not raise it to "pass".
PARITY_TOL = 1e-3

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---- crop / input loading ----------------------------------------------------

def _load_real_crops(crops_dir: Path) -> list:
    """Return real crops as HWC uint8 RGB ndarrays (sys.exit if none found)."""
    import numpy as np
    from PIL import Image

    paths = sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not paths:
        sys.exit(f"[parity] --crops {crops_dir} has no images ({sorted(_IMG_EXTS)})")
    crops = [np.array(Image.open(p).convert("RGB")) for p in paths]
    print(f"[parity] loaded {len(crops)} real crops from {crops_dir}", flush=True)
    return crops


def _synthetic_uint8_crops() -> list:
    """Deterministic raw crops of varied shapes, for the preprocessing gate."""
    import numpy as np

    rng = np.random.default_rng(0)
    shapes = [(300, 300), (480, 640), (640, 480), (257, 257), (224, 400), (512, 288)]
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for (h, w) in shapes]


def _model_parity_inputs(crops_dir: Path | None) -> list:
    """Return preprocessed model inputs: list of (1,3,224,224) float32 arrays.

    Both branches run the *runtime* classifier._preprocess, so the gate
    exercises exactly the production input format and value range
    (ImageNet-normalized, ~[-2.1, 2.6]). Raw N(0,1) tensors were a mistake:
    they're out-of-distribution and unbounded, which drives activations into
    extreme regimes and inflates absolute logit diffs for reasons unrelated to
    export fidelity.
    """
    import numpy as np

    crops = _load_real_crops(crops_dir) if crops_dir is not None else _synthetic_uint8_crops()
    from classifier import _preprocess

    inputs = [np.ascontiguousarray(_preprocess(c), dtype=np.float32) for c in crops]
    kind = "real" if crops_dir is not None else "synthetic"
    print(f"[parity] model inputs: {len(inputs)} {kind} crops via _preprocess",
          flush=True)
    return inputs


# ---- gates -------------------------------------------------------------------

def _check_preprocess_parity(crops_dir: Path | None) -> None:
    """Runtime _preprocess vs the donor torchvision inference transform."""
    import numpy as np
    from torchvision import transforms

    from classifier import _preprocess  # the runtime preprocessing under test

    crops = _load_real_crops(crops_dir) if crops_dir is not None else _synthetic_uint8_crops()

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
    print(f"[parity] preprocessing max|Δ| vs torchvision donor = {worst:.3e}", flush=True)
    if worst >= PARITY_TOL:
        sys.exit(
            f"[parity] FAIL: runtime _preprocess drifted from the training "
            f"transform (max|Δ|={worst:.3e} >= {PARITY_TOL:.0e}). "
            "Reconcile detector/classifier.py::_preprocess with the torchvision "
            "Resize(256)+CenterCrop(224)+Normalize pipeline before shipping."
        )


def _check_model_parity(model, compiled, inputs: list) -> None:
    """Read-back OpenVINO model vs torch model on identical normalized inputs.

    Requires max|Δlogits| < PARITY_TOL AND matching argmax on every sample.
    """
    import numpy as np
    import torch

    # Re-assert eval() right before the reference forward: convert_model / tracing
    # can leave the module's training flag flipped, and BatchNorm in train mode
    # would recompute batch stats from these inputs → large structural drift.
    model.eval()
    assert not model.training, "torch reference must be in eval() for parity"

    out_port = compiled.output(0)
    worst = 0.0
    failures = 0
    with torch.inference_mode():
        for i, x in enumerate(inputs):
            x = np.ascontiguousarray(x, dtype=np.float32)   # (1,3,224,224)
            torch_logits = model(torch.from_numpy(x)).cpu().numpy()[0]
            ov_logits = np.asarray(compiled(x)[out_port])[0]
            d = float(np.max(np.abs(torch_logits - ov_logits)))
            t_arg = int(torch_logits.argmax())
            o_arg = int(ov_logits.argmax())
            ok = d < PARITY_TOL and t_arg == o_arg
            worst = max(worst, d)
            failures += 0 if ok else 1
            print(f"[parity] sample {i}: max|Δ|={d:.3e} "
                  f"argmax torch={t_arg} ov={o_arg} {'OK' if ok else 'FAIL'}",
                  flush=True)

    print(f"[parity] torch↔OpenVINO worst max|Δlogits| = {worst:.3e} "
          f"over {len(inputs)} samples", flush=True)
    if failures:
        sys.exit(
            f"[parity] FAIL: {failures}/{len(inputs)} sample(s) diverged "
            f"(need max|Δ| < {PARITY_TOL:.0e} AND matching argmax). The exported "
            "IR is not faithful to torch — do not ship it."
        )


# ---- export ------------------------------------------------------------------

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
    model.eval()                                      # once, before anything
    print(f"[export] model.training after eval() = {model.training}", flush=True)

    # Convert under no_grad (NOT inference_mode): convert_model traces via
    # torch.jit.trace, and tracing under inference_mode raises "inference tensors
    # cannot be saved for backward". no_grad is enough — we never backprop.
    # example_input is shape-only (eval BatchNorm uses running stats, not it).
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=example)
    # If convert flipped the flag, this print localizes it; the parity step
    # re-evals defensively regardless.
    print(f"[export] model.training after convert_model = {model.training}", flush=True)

    xml_path = out_dir / "cat_classifier.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)  # FP32, match torch
    print(f"[export] OpenVINO IR (FP32) → {xml_path}", flush=True)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(class_names), encoding="utf-8")
    print(f"[export] classes → {classes_path} : {class_names}", flush=True)

    # ---- parity gates (fail the build on any drift) ----
    _check_preprocess_parity(crops_dir)

    # Test the SHIPPED artifact: re-read the saved .xml rather than the in-memory
    # model, so what we validate is exactly what runtime loads. Force FP32 — the
    # CPU plugin defaults to bf16 on AVX512_BF16/AMX, which alone shifts logits by
    # up to ~1e1 (argmax survives, magnitudes don't). classifier.py uses the same
    # hint at runtime, so this matches production.
    core = ov.Core()
    compiled = core.compile_model(
        core.read_model(str(xml_path)), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    inputs = _model_parity_inputs(crops_dir)
    _check_model_parity(model, compiled, inputs)

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
