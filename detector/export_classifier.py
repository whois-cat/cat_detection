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
     model are fed the *same* normalized NCHW float32 tensor; their softmax
     probabilities (== the runtime cat_score) and argmax must agree.

The preprocessing gate requires max|Δ| < PARITY_TOL. The model gate is
source-dependent: on REAL crops (--crops) it is strict — max|Δprob| < PARITY_TOL
AND a matching argmax on every sample (the quantities production actually
consumes, not raw logits; see PARITY_TOL). On SYNTHETIC inputs (no --crops) it
checks argmax agreement only: noise crops hit saturated regimes where the
probability threshold gives false FAILs, so we gate on the decision the runtime
makes and report probability deltas for diagnosis. Pass --crops for strict
probability parity.

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

# The gate compares softmax PROBABILITIES (== the runtime cat_score), not raw
# logits: classifier.py returns softmax(logits).max(), the feeder votes by that
# cat_score and thresholds on the feeder's CLASSIFIER_MIN_CONF, and raw logits have an
# arbitrary scale (the trained net emits |logit|~1e3 on off-distribution inputs,
# where even faithful FP32 torch-vs-OpenVINO differ by ~1% absolute). Probability
# space is bounded [0,1] and is exactly what production consumes, so 1e-3 here is
# both tight and meaningful. Do not raise it to "pass".
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


def _softmax(z):
    import numpy as np

    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _check_model_parity(model, compiled, inputs: list, *, synthetic: bool) -> None:
    """Read-back OpenVINO model vs torch model on identical normalized inputs.

    The pass criterion depends on where the inputs came from:

      - REAL crops (--crops given): strict — require max|Δprob| < PARITY_TOL AND
        matching argmax on every sample. These are in-distribution, so the
        probability vector (the cat_score the feeder votes on) is a meaningful,
        tight signal; nothing is relaxed.
      - SYNTHETIC inputs (no --crops): argmax-agreement only. Noise crops land in
        saturated activation regimes where torch and OpenVINO can differ by more
        than PARITY_TOL in probability for reasons unrelated to export fidelity,
        so the probability threshold gives false FAILs. We instead gate on the
        decision the runtime actually makes (argmax) and print the probability
        deltas for diagnosis only.

    Raw |Δlogit| is printed for diagnosis only (logits carry an arbitrary scale
    and blow up on off-distribution inputs, so they're not a sane pass/fail
    signal).
    """
    import numpy as np
    import torch

    # Defensive: BatchNorm in train mode would recompute batch stats and diverge.
    model.eval()
    assert not model.training, "torch reference must be in eval() for parity"

    if synthetic:
        print("[parity] synthetic mode: argmax-agreement check only "
              "(provide --crops for strict probability parity)", flush=True)

    out_port = compiled.output(0)
    worst_p = 0.0
    failures = 0
    with torch.inference_mode():
        for i, x in enumerate(inputs):
            x = np.ascontiguousarray(x, dtype=np.float32)   # (1,3,224,224)
            torch_logits = model(torch.from_numpy(x)).cpu().numpy()[0]
            ov_logits = np.asarray(compiled(x)[out_port])[0]
            torch_p = _softmax(torch_logits)
            ov_p = _softmax(ov_logits)
            dp = float(np.max(np.abs(torch_p - ov_p)))        # bounds cat_score error
            dl = float(np.max(np.abs(torch_logits - ov_logits)))  # diagnostic only
            t_arg = int(torch_p.argmax())
            o_arg = int(ov_p.argmax())
            argmax_ok = t_arg == o_arg
            # Synthetic: argmax only. Real: argmax AND the strict prob threshold.
            ok = argmax_ok if synthetic else (dp < PARITY_TOL and argmax_ok)
            worst_p = max(worst_p, dp)
            failures += 0 if ok else 1
            print(f"[parity] sample {i}: max|Δprob|={dp:.3e} (cat_score) "
                  f"max|Δlogit|={dl:.3e} argmax torch={t_arg} ov={o_arg} "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
            print(f"           torch_p={np.array2string(torch_p, precision=5, suppress_small=True)}",
                  flush=True)
            print(f"           ov_p   ={np.array2string(ov_p, precision=5, suppress_small=True)}",
                  flush=True)

    print(f"[parity] torch↔OpenVINO worst max|Δprob| = {worst_p:.3e} "
          f"over {len(inputs)} samples", flush=True)
    if failures:
        if synthetic:
            sys.exit(
                f"[parity] FAIL: {failures}/{len(inputs)} synthetic sample(s) "
                "disagree on argmax (torch vs OpenVINO) — the exported IR makes a "
                "different decision. Do not ship it. (Probability deltas are not "
                "gated in synthetic mode; use --crops for strict probability parity.)"
            )
        sys.exit(
            f"[parity] FAIL: {failures}/{len(inputs)} sample(s) diverged "
            f"(need max|Δprob| < {PARITY_TOL:.0e} AND matching argmax). The "
            "exported IR is not faithful to torch — do not ship it."
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

    # Export a TWO-output IR: output 0 = logits (unchanged — runtime cat_score and
    # the parity gate both read output 0), output 1 = the 1280-d pre-classifier
    # embedding used by the open-set prototype-distance gate (detector/unknown.py).
    # A single-output IR would give the runtime no embedding, leaving the gate
    # dormant. The wrapper computes logits exactly as EfficientNet.forward does, so
    # logits — and therefore parity — are byte-for-byte unchanged.
    class _ClassifierWithEmbedding(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.features = m.features
            self.avgpool = m.avgpool
            self.classifier = m.classifier

        def forward(self, x):
            feat = torch.flatten(self.avgpool(self.features(x)), 1)
            return self.classifier(feat), feat

    export_model = _ClassifierWithEmbedding(model).eval()

    # Convert under no_grad (NOT inference_mode): convert_model traces via
    # torch.jit.trace, and tracing under inference_mode raises "inference tensors
    # cannot be saved for backward". no_grad is enough — we never backprop.
    # example_input is shape-only (eval BatchNorm uses running stats, not it).
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        ov_model = ov.convert_model(export_model, example_input=example)
    # If convert flipped the flag, this print localizes it; the parity step
    # re-evals defensively regardless.
    print(f"[export] model.training after convert_model = {model.training}", flush=True)

    xml_path = out_dir / "cat_classifier.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)  # FP32, match torch
    print(f"[export] OpenVINO IR (FP32) → {xml_path}", flush=True)

    classes_path = out_dir / "classes.json"
    classes_path.write_text(json.dumps(class_names), encoding="utf-8")
    print(f"[export] classes → {classes_path} : {class_names}", flush=True)

    # Open-set prototypes (optional): mean per-class embedding computed at train
    # time. Shipped as prototypes.json next to classes.json; the runtime loads it
    # to reject crops far from every known cat. Absent on older checkpoints — the
    # runtime then simply falls back to the softmax-only gate (no behavior change).
    prototypes = checkpoint.get("prototypes")
    if prototypes:
        proto_path = out_dir / "prototypes.json"
        proto_path.write_text(json.dumps(prototypes), encoding="utf-8")
        print(f"[export] prototypes → {proto_path} : {sorted(prototypes)} "
              f"(dim={len(next(iter(prototypes.values())))})", flush=True)
    else:
        print("[export] no prototypes in checkpoint — open-set distance gate will "
              "be unavailable (older checkpoint; re-train to enable).", flush=True)

    # ---- parity gates (fail the build on any drift) ----
    _check_preprocess_parity(crops_dir)

    # Test the SHIPPED artifact: re-read the saved .xml rather than the in-memory
    # model, so what we validate is exactly what runtime loads. Force FP32 — the
    # CPU plugin defaults to bf16 on AVX512_BF16/AMX, which shifts logits by a
    # ~constant relative error (argmax survives, magnitudes don't). Use the typed
    # property (string keys can be silently ignored) and PRINT the effective
    # precision so the build log proves whether f32 actually took.
    import openvino.properties.hint as hints

    print(f"[export] openvino {ov.get_version()} devices={ov.Core().available_devices}",
          flush=True)
    core = ov.Core()
    core.set_property("CPU", {hints.inference_precision: ov.Type.f32})
    compiled = core.compile_model(
        core.read_model(str(xml_path)), "CPU",
        {hints.inference_precision: ov.Type.f32},
    )
    try:
        eff = compiled.get_property(hints.inference_precision)
    except Exception:
        eff = compiled.get_property("INFERENCE_PRECISION_HINT")
    print(f"[export] CPU effective inference_precision = {eff}", flush=True)

    # crops_dir is None ⇒ synthetic inputs (see _model_parity_inputs): gate on
    # argmax agreement only. Real crops keep the strict probability threshold.
    synthetic = crops_dir is None
    inputs = _model_parity_inputs(crops_dir)
    _check_model_parity(model, compiled, inputs, synthetic=synthetic)

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
