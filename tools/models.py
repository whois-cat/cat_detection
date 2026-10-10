"""Deploy / roll back the runtime models cv-worker serves.

Both models are delivered the same way, through read-only volumes:

    models/<kind>/
      versions/<id>/       exported runtime artifact + metadata.json
      current  -> versions/<id>     (relative symlink, what cv-worker loads)
      previous -> versions/<id>

``<kind>`` is ``detector`` (YOLO) or ``classifier`` (cat identity); ``<id>`` is
the training run name under models/trained/. Deploying exports the run's
checkpoint to OpenVINO with that kind's quality gate, installs it as a new
version and atomically switches ``current``; nothing switches if the gate fails.
cv-worker picks the change up on restart (the `just deploy` / `just rollback`
recipes do that). With no deployed detector, cv-worker serves the COCO yolov8n
baked into its image.

Export needs torch + openvino + ultralytics, so ``deploy`` runs in the
cv-worker image; ``rollback`` and ``status`` only move symlinks.

    python tools/models.py status
    python tools/models.py deploy   {detector,classifier} [RUN]   # default: newest run
    python tools/models.py rollback {detector,classifier} [VERSION]  # default: previous
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
TRAINED_ROOT = ROOT / "models" / "trained"
MODELS_ROOT = ROOT / "models"
YOLO_DATASET = Path(os.environ.get("YOLO_DATASET", ROOT / "data" / "yolo_dataset"))


# ---- classifier ----------------------------------------------------------------

CLASSIFIER_FILES = ("cat_classifier.xml", "cat_classifier.bin", "classes.json")


def check_classifier_checkpoint(path: Path) -> dict:
    """A torch checkpoint with a state_dict and a consistent, non-empty class list."""
    import torch  # lazy: only needed when actually validating a .pt
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"not a readable torch checkpoint ({path}): {e}") from e
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint is not a dict ({path})")
    for key in ("state_dict", "class_names", "num_classes"):
        if key not in ckpt:
            raise ValueError(f"checkpoint missing {key!r} ({path})")
    names = ckpt["class_names"]
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) for n in names)):
        raise ValueError(f"checkpoint class_names must be a non-empty list of strings ({path})")
    if len(names) != ckpt["num_classes"]:
        raise ValueError(
            f"checkpoint class_names ({len(names)}) != num_classes ({ckpt['num_classes']}) ({path})"
        )
    return {"classes": names}


def export_classifier(checkpoint: Path, out_dir: Path) -> None:
    """OpenVINO IR + classes.json via cv-worker/tools/export_classifier.py, which
    fails on any torch-vs-OpenVINO parity mismatch."""
    script = ROOT / "cv-worker" / "tools" / "export_classifier.py"
    cmd = [sys.executable, str(script), "--pt", str(checkpoint), "--out", str(out_dir)]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError(f"classifier export failed: {' '.join(cmd)}")


def check_classifier_artifact(version_dir: Path) -> dict:
    for name in CLASSIFIER_FILES:
        if not (version_dir / name).exists():
            raise ValueError(f"exported classifier missing {name} in {version_dir}")
    try:
        names = json.loads((version_dir / "classes.json").read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"invalid classes.json in {version_dir}: {e}") from e
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"classes.json must be a non-empty list of strings in {version_dir}")
    return {"classes": names}


# ---- detector ------------------------------------------------------------------

def _training_dataset(checkpoint: Path) -> Path:
    """The dataset version a YOLO run was trained on, from its report.json."""
    report_path = checkpoint.parent.parent / "report.json"
    if not report_path.is_file():
        raise ValueError(f"no training report next to {checkpoint} ({report_path})")
    dataset = json.loads(report_path.read_text(encoding="utf-8")).get("dataset") or {}
    if not dataset.get("version_id"):
        raise ValueError(f"training report has no dataset version: {report_path}")
    return YOLO_DATASET / "versions" / dataset["version_id"]


def check_detector_checkpoint(path: Path) -> dict:
    version = _training_dataset(path)
    if not version.is_dir():
        raise ValueError(f"dataset version the run was trained on is missing: {version}")
    return {"dataset": version.name}


def export_detector(checkpoint: Path, out_dir: Path) -> None:
    """INT8 OpenVINO export gated against the .pt on the run's own test split
    (training.yolo_export). Exported from a copy so the run dir is untouched."""
    work = out_dir / "export"
    work.mkdir()
    source = work / checkpoint.name
    shutil.copy2(checkpoint, source)
    cmd = [sys.executable, "-m", "training.yolo_export",
           "--dataset", str(_training_dataset(checkpoint)), "--model", str(source),
           "--int8", "--report", str(work / "report.json")]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise RuntimeError(f"detector export failed its quality gate (see {work / 'report.json'})")
    # Keep Ultralytics' *_openvino_model name: it recognises OpenVINO by it.
    shutil.move(str(work / f"{source.stem}_int8_openvino_model"), out_dir)
    source.unlink()


def check_detector_artifact(version_dir: Path) -> dict:
    """Exactly one Ultralytics OpenVINO export dir (what cv-worker loads) with a
    `cat` class."""
    import yaml

    exports = list(version_dir.glob("*_openvino_model"))
    if len(exports) != 1 or not any(exports[0].glob("*.xml")):
        raise ValueError(f"expected one *_openvino_model export with an .xml in {version_dir}")
    meta = exports[0] / "metadata.yaml"
    names = (yaml.safe_load(meta.read_text(encoding="utf-8")) or {}).get("names") if meta.is_file() else None
    if not names or "cat" not in {str(n).casefold() for n in dict(names).values()}:
        raise ValueError(f"exported detector has no 'cat' class in {meta}")
    return {"classes": len(names)}


# ---- kinds ---------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    name: str
    checkpoint: str                              # checkpoint path inside a training run
    check_checkpoint: Callable[[Path], dict]
    export: Callable[[Path, Path], None]         # (checkpoint, empty out dir)
    check_artifact: Callable[[Path], dict]
    builtin: str | None = None                   # what cv-worker serves with nothing deployed


KINDS = {
    "detector": Kind("detector", "weights/best.pt", check_detector_checkpoint,
                     export_detector, check_detector_artifact,
                     builtin="COCO yolov8n baked into the cv-worker image"),
    "classifier": Kind("classifier", "cat_classifier.pt", check_classifier_checkpoint,
                       export_classifier, check_classifier_artifact),
}


# ---- versions and symlinks ---------------------------------------------------------

def _switch(link: Path, version: str) -> None:
    """Point ``link`` at versions/<version> atomically (relative, so it resolves
    inside the container too)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.parent / f".{link.name}.tmp-{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(f"versions/{version}", tmp)
    os.replace(tmp, link)


def _target(link: Path) -> str | None:
    return Path(os.readlink(link)).name if link.is_symlink() else None


def run_dir(kind: Kind, checkpoint: Path) -> Path:
    """models/trained/<run> for a checkpoint at <run>/<kind.checkpoint>."""
    return checkpoint.parents[len(Path(kind.checkpoint).parts) - 1]


def find_checkpoint(kind: Kind, run: str | None, trained_root: Path) -> Path:
    if run:
        candidate = Path(run)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = trained_root / run
        checkpoint = candidate / kind.checkpoint if candidate.is_dir() else candidate
        if not checkpoint.is_file():
            raise SystemExit(f"no {kind.name} checkpoint at {checkpoint}")
        return checkpoint
    found = sorted(trained_root.glob(f"*/{kind.checkpoint}"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no trained {kind.name} under {trained_root}/*/{kind.checkpoint}")
    print(f"[deploy] newest {kind.name} run: {run_dir(kind, found[-1]).name}")
    return found[-1]


def deploy(kind_name: str, run: str | None = None, *, trained_root: Path = TRAINED_ROOT,
           models_root: Path = MODELS_ROOT) -> dict:
    kind = KINDS[kind_name]
    checkpoint = find_checkpoint(kind, run, trained_root)
    run = run_dir(kind, checkpoint)
    source = kind.check_checkpoint(checkpoint)

    root = models_root / kind.name
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    version = run.name
    if (versions / version).exists():
        raise SystemExit(f"{kind.name} {version} is already installed; "
                         f"`rollback {kind.name} {version}` switches back to it")

    # Export into a temp dir on the same filesystem, check, then move into place.
    tmp = Path(tempfile.mkdtemp(dir=versions, prefix=f".{version}.tmp-"))
    try:
        kind.export(checkpoint, tmp)
        artifact = kind.check_artifact(tmp)
        metadata = tmp / "metadata.json"
        if (run / "metadata.json").is_file():
            shutil.copy2(run / "metadata.json", metadata)
        else:
            metadata.write_text(json.dumps({
                "kind": kind.name, "version_id": version, "source_checkpoint": str(checkpoint),
                "deployed_at": datetime.now().isoformat(), **source,
            }, indent=2), encoding="utf-8")
        os.replace(tmp, versions / version)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    previous = _target(root / "current")
    if previous:
        _switch(root / "previous", previous)
    _switch(root / "current", version)
    return {"kind": kind.name, "version": version, "previous": previous,
            "source": checkpoint, **artifact}


def rollback(kind_name: str, version: str | None = None, *,
             models_root: Path = MODELS_ROOT) -> dict:
    kind = KINDS[kind_name]
    root = models_root / kind.name
    current, previous = root / "current", root / "previous"
    rolling_from = _target(current)
    target = version or _target(previous)
    if target is None:
        if kind.builtin and rolling_from:
            current.unlink()
            _switch(previous, rolling_from)
            return {"kind": kind.name, "version": None, "previous": rolling_from}
        raise SystemExit(f"no previous {kind.name} to roll back to; pass a version "
                         f"(see `python tools/models.py status`)")
    if not (root / "versions" / target).is_dir():
        raise SystemExit(f"{kind.name} version not found: {root / 'versions' / target}")
    kind.check_artifact(root / "versions" / target)
    if rolling_from and rolling_from != target:
        _switch(previous, rolling_from)
    _switch(current, target)
    return {"kind": kind.name, "version": target, "previous": rolling_from}


def status(*, models_root: Path = MODELS_ROOT) -> list[str]:
    lines = []
    for kind in KINDS.values():
        root = models_root / kind.name
        current, previous = _target(root / "current"), _target(root / "previous")
        versions = sorted(p.name for p in (root / "versions").glob("*")
                          if p.is_dir() and not p.name.startswith(".")) if root.is_dir() else []
        lines.append(f"{kind.name}: {current or kind.builtin or '(none)'}")
        for name in versions:
            mark = "*" if name == current else ("<" if name == previous else " ")
            lines.append(f"  {mark} {name}")
    lines.append("(* current, < previous)")
    return lines


# ---- CLI ---------------------------------------------------------------------------

def _describe(res: dict) -> str:
    serving = res["version"] or KINDS[res["kind"]].builtin
    return f"{res['kind']}: now {serving} (previous: {res['previous'] or 'none'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    d = sub.add_parser("deploy")
    d.add_argument("kind", choices=KINDS)
    d.add_argument("run", nargs="?", default="", help="training run name or path; default newest")
    r = sub.add_parser("rollback")
    r.add_argument("kind", choices=KINDS)
    r.add_argument("version", nargs="?", default="", help="version id; default previous")
    args = ap.parse_args(argv)

    if args.cmd == "status":
        print("\n".join(status()))
    elif args.cmd == "deploy":
        print(f"[deploy] {_describe(deploy(args.kind, args.run or None))}")
    else:
        print(f"[rollback] {_describe(rollback(args.kind, args.version or None))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
