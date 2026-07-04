"""Runtime EfficientNet-B0 cat classifier via OpenVINO. No torch at runtime.

Preprocessing is identical to donor model/loader.py:preprocess_for_inference —
resize to 256 on short side → center-crop 224 → ImageNet normalise → NCHW float32.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from unknown import UnknownConfig, decide_identity

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


# Shown in every "model not usable" error so the operator knows the next step.
# The runtime model is delivered via the shared read-only volume
# (CLASSIFIER_WEIGHTS=/opt/models/classifier/current); switch it with promote.
_PROMOTE_HINT = (
    "To (re)create / fix the runtime model (no image rebuild needed):\n"
    "    just classifier-promote     # export latest trained checkpoint + switch `current`\n"
    "    just classifier-restart     # restart detectors to load it\n"
    "Required files in the model dir: cat_classifier.xml + cat_classifier.bin + classes.json."
)


def _require_model_dir(model_dir: Path) -> None:
    # CLASSIFIER_WEIGHTS usually points at the `current` symlink; a missing dir
    # means nothing has been promoted yet (or the symlink dangles).
    if not model_dir.exists():
        raise FileNotFoundError(
            f"classifier model dir does not exist: {model_dir}\n"
            "(is a model promoted? is the volume mounted?)\n"
            f"{_PROMOTE_HINT}"
        )


def _require_runtime_file(path: Path, model_dir: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"runtime classifier file missing: {path}\n"
            f"(classifier model dir: {model_dir})\n{_PROMOTE_HINT}"
        )


def _load_class_names(classes_path: Path, model_dir: Path) -> list[str]:
    try:
        names = json.loads(classes_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(
            f"invalid classes.json ({classes_path}): {e}\n{_PROMOTE_HINT}"
        ) from e
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) for n in names)):
        raise ValueError(
            f"classes.json must be a non-empty list of label strings ({classes_path})\n"
            f"{_PROMOTE_HINT}"
        )
    return names


def _load_prototypes(proto_path: Path, class_names: list[str]) -> dict | None:
    """Load optional open-set prototypes.json → {class_name: unit np.ndarray}.

    Missing file → None (the open-set distance gate stays off; behavior is
    exactly the softmax-only path). A present-but-malformed file is a promotion
    error, so it raises rather than silently disabling the safety gate. Only
    prototypes for known class names are kept; each is L2-normalized so cosine
    distance is stable regardless of how it was stored."""
    if not proto_path.exists():
        return None
    try:
        raw = json.loads(proto_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"invalid prototypes.json ({proto_path}): {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"prototypes.json must be a non-empty object ({proto_path})")
    out: dict[str, np.ndarray] = {}
    known = set(class_names)
    for name, vec in raw.items():
        if name not in known:
            continue
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if arr.ndim != 1 or norm == 0.0:
            raise ValueError(
                f"prototypes.json[{name!r}] must be a non-zero 1-D vector ({proto_path})")
        out[name] = arr / norm
    return out or None


def resolve_model_version(model_dir: Path) -> str:
    """Concrete version the model resolves to — the target dir name of the
    `current` symlink (e.g. "20260629-001122"), or "?" if unresolvable. Used for
    startup logging and the /status endpoint so the active version is visible."""
    try:
        return Path(model_dir).resolve().name
    except Exception:
        return "?"


def _check_output_dim(out_dim: int, n_classes: int, model_dir: Path) -> None:
    """Pure guard (no OpenVINO) so it's unit-testable: the IR's output width must
    equal the number of labels in classes.json, else predictions mislabel."""
    if out_dim != n_classes:
        raise ValueError(
            f"classifier/classes mismatch in {model_dir}: the model outputs "
            f"{out_dim} classes but classes.json lists {n_classes}. The IR and "
            f"classes.json are out of sync.\n{_PROMOTE_HINT}"
        )


class CatClassifier:
    """Thread-safe EfficientNet-B0 classifier backed by an OpenVINO IR."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        xml = model_dir / "cat_classifier.xml"
        bin_ = model_dir / "cat_classifier.bin"
        classes_path = model_dir / "classes.json"
        # Validate required artifacts up front, BEFORE importing OpenVINO, so the
        # failure is a clear actionable message (and so this guard is testable
        # without an OpenVINO install).
        _require_model_dir(model_dir)
        _require_runtime_file(xml, model_dir)
        _require_runtime_file(bin_, model_dir)
        _require_runtime_file(classes_path, model_dir)
        self.class_names: list[str] = _load_class_names(classes_path, model_dir)
        # Optional open-set prototypes (mean per-class embedding). None → the
        # distance gate is unavailable and identity falls back to softmax only.
        self.prototypes = _load_prototypes(model_dir / "prototypes.json", self.class_names)

        # Status attributes (also surfaced by the /status endpoint).
        self.model_dir = str(model_dir)
        self.resolved_version = resolve_model_version(model_dir)
        self.format = "openvino"

        # Startup summary: model dir, resolved version (the `current` symlink's
        # target dir name), classes, artifact format. Helps debug which version
        # is live after a promote + restart.
        version = self.resolved_version
        print(
            f"[classifier] enabled: model_dir={model_dir} version={version} "
            f"classes_path={classes_path} num_classes={len(self.class_names)} "
            f"classes={self.class_names} format=openvino-ir",
            flush=True,
        )

        import openvino as ov
        import openvino.properties.hint as hints

        core = ov.Core()
        # Force FP32 inference. The CPU plugin otherwise runs FP32 IRs in bf16 by
        # default on AVX512_BF16/AMX hardware, whose ~8-bit mantissa shifts logits
        # vs the trained torch model. argmax usually survives, but the cat_score
        # confidences would drift from training (and the export parity gate
        # enforces FP32). Typed property — string keys can be silently ignored.
        # The speed cost is negligible for one B0 crop.
        self._compiled = core.compile_model(
            core.read_model(str(xml)),
            "CPU",
            {hints.inference_precision: ov.Type.f32},
        )
        # Output width must match the label count (when statically known).
        try:
            out_dim = self._compiled.output(0).partial_shape[-1].get_length()
        except Exception:
            out_dim = None     # dynamic shape — skip cleanly rather than crash
        if out_dim is not None:
            _check_output_dim(out_dim, len(self.class_names), model_dir)

        # A 2-output IR (logits + embedding) enables the open-set distance gate.
        # A legacy 1-output IR has no embedding, so the gate stays off regardless
        # of prototypes.json — the runtime falls back to softmax-only cleanly.
        try:
            self._has_embedding = len(self._compiled.outputs) > 1
        except Exception:
            self._has_embedding = False
        gate = (
            "on" if (self._has_embedding and self.prototypes) else
            "off (no embedding output)" if self.prototypes else
            "off (no prototypes)"
        )
        print(f"[classifier] open-set prototype gate: {gate}; "
              f"prototypes={sorted(self.prototypes) if self.prototypes else None}",
              flush=True)

    def status(self) -> dict:
        """Runtime status fragment for the /status endpoint: enabled flag, active
        model dir + resolved version, class count (not labels), artifact format."""
        return {
            "enabled": True,
            "model_dir": self.model_dir,
            "resolved_version": self.resolved_version,
            "classes_count": len(self.class_names),
            "format": self.format,
            "has_embedding": bool(getattr(self, "_has_embedding", False)),
            "prototype_classes": (
                len(self.prototypes) if getattr(self, "prototypes", None) else 0
            ),
        }

    def _probs(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Softmax probability vector for one crop. classify_all() goes through
        here so review/diagnostics use byte-identical pixels and math."""
        inp = _preprocess(crop_rgb)
        logits = self._compiled(inp)[0]  # (1, num_classes)
        z = logits[0].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum()

    def _probs_batch(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """Softmax matrix (N, num_classes) for N crops in ONE inference call.

        Crops are preprocessed and stacked so the compiled model runs once per
        batch instead of once per crop — the CPU win on multi-cat frames. Row
        order matches the input order."""
        if not crops_rgb:
            return np.empty((0, len(self.class_names)), dtype=np.float64)
        inp = np.concatenate([_preprocess(c) for c in crops_rgb], axis=0)
        logits = self._compiled(inp)[0]  # (N, num_classes)
        z = np.asarray(logits, dtype=np.float64)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _probs_emb_batch(
        self, crops_rgb: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Softmax matrix (N, C) plus the embedding matrix (N, D) in ONE inference.

        Returns ``(probs, embeddings)``; ``embeddings`` is None when the IR has no
        embedding output (legacy 1-output model), in which case the open-set
        distance gate cannot run and identity degrades to softmax only. Works with
        a plain-list compiled stub (tests) as well as an OpenVINO OVDict."""
        if not crops_rgb:
            return np.empty((0, len(self.class_names)), dtype=np.float64), None
        inp = np.concatenate([_preprocess(c) for c in crops_rgb], axis=0)
        result = self._compiled(inp)
        z = np.asarray(result[0], dtype=np.float64)  # (N, C) logits
        e = np.exp(z - z.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        emb = None
        try:
            if len(result) > 1:
                emb = np.asarray(result[1], dtype=np.float32)  # (N, D)
        except (TypeError, KeyError, IndexError):
            emb = None
        return probs, emb

    def decide(self, crop_rgb: np.ndarray, config: UnknownConfig) -> tuple[str, float]:
        """Open-set identity decision for one crop: (label_or_UNKNOWN, confidence).

        Combines the softmax floor with cosine distance to the matched class
        prototype (detector/unknown.py::decide_identity), so a crop that is far
        from every known cat is rejected as UNKNOWN even when softmax looks
        confident. Degrades safely: with no prototypes / no embedding / no
        configured distance ceiling it is exactly the softmax-threshold gate."""
        probs, emb = self._probs_emb_batch([crop_rgb])
        embedding = emb[0] if emb is not None else None
        return decide_identity(
            probs[0], self.class_names, config,
            embedding=embedding, prototypes=self.prototypes,
        )

    def classify_batch(self, crops_rgb: list[np.ndarray]) -> list[tuple[str, float]]:
        """Classify many crops at once. Returns [(cat_name, confidence), ...] in
        the SAME order as the input list (element type matches classify())."""
        probs = self._probs_batch(crops_rgb)
        out: list[tuple[str, float]] = []
        for row in probs:
            idx = int(np.argmax(row))
            out.append((self.class_names[idx], float(row[idx])))
        return out

    def classify(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        """Return (cat_name, confidence). crop_rgb is HWC uint8 RGB.
        Thin wrapper over classify_batch for a single crop."""
        return self.classify_batch([crop_rgb])[0]

    def classify_all(self, crop_rgb: np.ndarray) -> list[tuple[str, float]]:
        """Return [(cat_name, prob), ...] over ALL classes in class_names order.
        Same _preprocess + compiled model as classify() — for the label-review
        manifest where the full probability vector is needed."""
        probs = self._probs(crop_rgb)
        return [(name, float(p)) for name, p in zip(self.class_names, probs)]
