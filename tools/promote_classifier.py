"""Promote / roll back the runtime classifier via a shared read-only volume.

Runtime model delivery is volume-mounted, not baked into the image:

    models/classifier/
      versions/<version_id>/{cat_classifier.xml,.bin, classes.json, metadata.json}
      current  -> versions/<active_version>     (symlink)
      previous -> versions/<previous_version>   (symlink)

Detector containers mount ``./models/classifier:/opt/models/classifier:ro`` and
read ``CLASSIFIER_WEIGHTS=/opt/models/classifier/current``. Promotion exports a
trained checkpoint to the OpenVINO runtime format into a NEW version dir and
atomically switches the ``current`` symlink. Containers pick it up only after an
explicit restart (``just classifier-restart``) — no image rebuild, no hot reload.

Training never touches this tree; promotion is explicit and validated.

    python tools/promote_classifier.py promote  [--src PATH]      # default: latest trained
    python tools/promote_classifier.py rollback [--version ID]    # default: -> previous
    python tools/promote_classifier.py services                   # stdin svc names -> detector ones
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINED_ROOT = ROOT / "models" / "trained"
CLASSIFIER_ROOT = ROOT / "models" / "classifier"

REQUIRED_RUNTIME_FILES = ("cat_classifier.xml", "cat_classifier.bin", "classes.json")
RESTART_HINT = "Next: restart the detector containers to load it:\n    just classifier-restart"


# ---- discovery / validation -------------------------------------------------

def find_latest_checkpoint(trained_root: Path = TRAINED_ROOT) -> Path | None:
    """Newest models/trained/*/cat_classifier.pt by mtime, or None."""
    cands = sorted(Path(trained_root).glob("*/cat_classifier.pt"),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def validate_checkpoint(path: Path) -> dict:
    """Confirm ``path`` is a torch checkpoint with a state_dict and a consistent,
    non-empty class list. Returns a summary; raises ValueError otherwise."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"checkpoint not found: {path}")
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
    return {"class_names": names, "num_classes": int(ckpt["num_classes"])}


def validate_runtime_artifact(version_dir: Path) -> list[str]:
    """Confirm an exported version dir has the runtime files and a non-empty,
    valid classes.json. Returns the class list; raises ValueError otherwise.
    (Output-dim/IR check needs OpenVINO; the export parity gate + the detector
    runtime guard cover that — this stays dependency-free.)"""
    version_dir = Path(version_dir)
    for name in REQUIRED_RUNTIME_FILES:
        if not (version_dir / name).exists():
            raise ValueError(f"exported runtime artifact missing {name} in {version_dir}")
    try:
        names = json.loads((version_dir / "classes.json").read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"invalid classes.json in {version_dir}: {e}") from e
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"classes.json must be a non-empty list of strings in {version_dir}")
    return names


# ---- export (pluggable so tests don't need OpenVINO) ------------------------

def default_export(src_pt: Path, out_dir: Path) -> None:
    """Export a torch checkpoint to OpenVINO IR + classes.json by reusing the
    existing detector/export_classifier.py (which includes the parity gate).
    Runs in-process via subprocess; requires torch + openvino (present in the
    detector image, which is where `just classifier-promote` runs)."""
    script = ROOT / "detector" / "export_classifier.py"
    cmd = [sys.executable, str(script), "--pt", str(src_pt), "--out", str(out_dir)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"export failed (exit {proc.returncode}): {' '.join(cmd)}. "
            "Run this inside the detector image (torch + openvino), e.g. via "
            "`just classifier-promote`."
        )


# ---- atomic symlink switching -----------------------------------------------

def _switch_symlink(link: Path, target_rel: str) -> None:
    """Point ``link`` at ``target_rel`` (relative) atomically: make a temp
    symlink in the same dir, then os.replace over the existing link."""
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.parent / f".{link.name}.tmp-{os.getpid()}-{datetime.now().strftime('%H%M%S%f')}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target_rel, tmp)
    os.replace(tmp, link)            # atomic within the same directory


def _link_target_name(link: Path) -> str | None:
    """Version id a symlink points at (basename of its target), or None."""
    if not link.is_symlink():
        return None
    return Path(os.readlink(link)).name


def _version_id(src: Path, versions_dir: Path) -> str:
    """Deterministic, unique version id: the training stamp (src parent dir
    name) when available, else a timestamp; suffixed if it already exists."""
    base = src.parent.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    vid = base
    while (versions_dir / vid).exists():
        vid = f"{base}-{datetime.now().strftime('%H%M%S%f')}"
    return vid


# ---- operations --------------------------------------------------------------

def promote(src: Path | None, *, trained_root: Path = TRAINED_ROOT,
            classifier_root: Path = CLASSIFIER_ROOT, export_fn=default_export,
            version_id: str | None = None) -> dict:
    if src is None:
        src = find_latest_checkpoint(trained_root)
        if src is None:
            raise SystemExit(
                f"No trained checkpoints found under {trained_root}/*/cat_classifier.pt. "
                "Run training first or pass SRC=..."
            )
        print(f"[promote] selected latest trained checkpoint: {src}")
    src = Path(src)
    summary = validate_checkpoint(src)

    versions_dir = classifier_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    vid = version_id or _version_id(src, versions_dir)

    # Export into a temp dir on the SAME filesystem, validate, then move in.
    tmp_dir = Path(tempfile.mkdtemp(dir=str(versions_dir), prefix=f".{vid}.tmp-"))
    try:
        export_fn(src, tmp_dir)
        validate_runtime_artifact(tmp_dir)
        # Carry provenance: training metadata.json if present, else a minimal one.
        meta_dst = tmp_dir / "metadata.json"
        train_meta = src.parent / "metadata.json"
        if train_meta.exists() and not meta_dst.exists():
            shutil.copy2(train_meta, meta_dst)
        if not meta_dst.exists():
            meta_dst.write_text(json.dumps({
                "version_id": vid, "source_checkpoint": str(src),
                "created": datetime.now().isoformat(), "class_names": summary["class_names"],
            }, indent=2), encoding="utf-8")
        version_dir = versions_dir / vid
        os.replace(tmp_dir, version_dir)        # atomic move into place
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Switch symlinks: previous := old current target, then current := new.
    current = classifier_root / "current"
    previous = classifier_root / "previous"
    prev_version = _link_target_name(current)
    if prev_version:
        _switch_symlink(previous, f"versions/{prev_version}")
    _switch_symlink(current, f"versions/{vid}")

    return {"src": src, "version_id": vid, "version_dir": version_dir,
            "previous_version": prev_version, "current": current,
            "runtime_path": current, "class_names": summary["class_names"]}


def rollback(version: str | None, *, classifier_root: Path = CLASSIFIER_ROOT) -> dict:
    current = classifier_root / "current"
    previous = classifier_root / "previous"
    versions_dir = classifier_root / "versions"

    if version is None:
        target = _link_target_name(previous)
        if not target:
            raise SystemExit(
                f"No previous version to roll back to ({previous} is unset). "
                "Promote at least twice, or pass VERSION=<version_id>."
            )
        print(f"[rollback] switching current -> previous version: {target}")
    else:
        target = version
    target_dir = versions_dir / target
    if not target_dir.exists():
        raise SystemExit(f"version not found: {target_dir}")
    validate_runtime_artifact(target_dir)

    # New previous := the version we are rolling back FROM, so it's reversible.
    rolling_from = _link_target_name(current)
    if rolling_from and rolling_from != target:
        _switch_symlink(previous, f"versions/{rolling_from}")
    _switch_symlink(current, f"versions/{target}")
    return {"version_id": target, "previous_version": rolling_from, "current": current}


def select_detector_services(names) -> list[str]:
    """Detector services use the classifier volume; restart only those. Names
    are generated per-camera (detector-<id>); never db/webui/mediamtx/etc."""
    return [n for n in names if n.startswith("detector")]


# ---- CLI ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--src", default="", help="checkpoint path; empty = latest trained")
    r = sub.add_parser("rollback")
    r.add_argument("--version", default="", help="version id; empty = previous")
    sub.add_parser("services", help="filter detector services from stdin (one name per line)")
    args = ap.parse_args(argv)

    if args.cmd == "promote":
        res = promote(Path(args.src) if args.src else None)
        print(f"[promote] source       : {res['src']}")
        print(f"[promote] new version  : {res['version_id']}  ({res['version_dir']})")
        print(f"[promote] previous     : {res['previous_version'] or '(none)'}")
        print(f"[promote] runtime path : {res['runtime_path']} -> versions/{res['version_id']}")
        print(f"[promote] classes ({len(res['class_names'])}): {res['class_names']}")
        print(RESTART_HINT)
        return 0
    if args.cmd == "rollback":
        res = rollback(args.version or None)
        print(f"[rollback] active version : {res['version_id']}")
        print(f"[rollback] previous       : {res['previous_version'] or '(none)'}")
        print(RESTART_HINT)
        return 0
    if args.cmd == "services":
        names = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]
        for s in select_detector_services(names):
            print(s)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
