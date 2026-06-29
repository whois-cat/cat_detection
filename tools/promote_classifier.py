"""Promote / roll back the active classifier checkpoint for the build-time bake.

The detector image is built from the repo-root ``models/cat_classifier.pt``
(Dockerfile → export_classifier.py → OpenVINO IR). Training never touches that
file; promotion is an explicit, validated copy:

    trained models/trained/<stamp>/cat_classifier.pt
        --(promote)-->  models/cat_classifier.pt   (active bake source)

Backups of the previous active file go to ``models/backups/`` so a promotion can
be rolled back. Nothing here exports or deploys — after promoting, rebuild the
detector (``just up``).

Usage:
    python tools/promote_classifier.py promote  [--src PATH]      # default: latest trained
    python tools/promote_classifier.py rollback [--backup PATH]   # default: latest backup
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINED_ROOT = ROOT / "models" / "trained"
ACTIVE = ROOT / "models" / "cat_classifier.pt"
BACKUP_DIR = ROOT / "models" / "backups"

REBUILD_HINT = "Next: rebuild + restart the detector to bake the new model:\n    just up"


# ---- checkpoint discovery / validation --------------------------------------

def find_latest_checkpoint(trained_root: Path = TRAINED_ROOT) -> Path | None:
    """Newest models/trained/*/cat_classifier.pt by mtime, or None."""
    cands = sorted(
        Path(trained_root).glob("*/cat_classifier.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def find_latest_backup(backup_dir: Path = BACKUP_DIR) -> Path | None:
    cands = sorted(
        Path(backup_dir).glob("cat_classifier.*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def validate_checkpoint(path: Path) -> dict:
    """Confirm ``path`` is a torch checkpoint with a state_dict and a consistent,
    non-empty class list. Returns a small summary; raises ValueError otherwise."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"checkpoint not found: {path}")
    import torch  # lazy: only needed when actually validating
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
            f"checkpoint class_names ({len(names)}) != num_classes "
            f"({ckpt['num_classes']}) ({path})"
        )
    return {"class_names": names, "num_classes": int(ckpt["num_classes"])}


# ---- atomic copy + backup ----------------------------------------------------

def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src → dst atomically: write a temp file in dst's dir, then rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix=".promote-", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)            # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _backup_active(active: Path, backup_dir: Path) -> Path | None:
    if not active.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = backup_dir / f"cat_classifier.{ts}.pt"
    shutil.copy2(active, backup)
    return backup


# ---- operations --------------------------------------------------------------

def promote(src: Path | None, *, active: Path = ACTIVE,
            trained_root: Path = TRAINED_ROOT, backup_dir: Path = BACKUP_DIR) -> dict:
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
    backup = _backup_active(active, backup_dir)
    _atomic_copy(src, active)
    return {"src": src, "active": active, "backup": backup,
            "class_names": summary["class_names"]}


def rollback(backup: Path | None, *, active: Path = ACTIVE,
             backup_dir: Path = BACKUP_DIR) -> dict:
    if backup is None:
        backup = find_latest_backup(backup_dir)
        if backup is None:
            raise SystemExit(
                f"No backups found under {backup_dir}/cat_classifier.*.pt. Nothing to roll back to."
            )
        print(f"[rollback] selected latest backup: {backup}")
    backup = Path(backup)
    validate_checkpoint(backup)
    # Back up the current active file too, so a rollback is itself reversible.
    pre = _backup_active(active, backup_dir)
    _atomic_copy(backup, active)
    return {"backup": backup, "active": active, "prev_backup": pre}


# ---- CLI ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote", help="promote a trained checkpoint to active")
    p.add_argument("--src", default="", help="checkpoint path; empty = latest trained")
    r = sub.add_parser("rollback", help="restore a previous active checkpoint")
    r.add_argument("--backup", default="", help="backup path; empty = latest backup")
    args = ap.parse_args(argv)

    if args.cmd == "promote":
        res = promote(Path(args.src) if args.src else None)
        print(f"[promote] source      : {res['src']}")
        print(f"[promote] destination : {res['active']}")
        print(f"[promote] backup      : {res['backup'] or '(none — no previous active)'}")
        print(f"[promote] classes ({len(res['class_names'])}): {res['class_names']}")
        print(REBUILD_HINT)
        return 0
    if args.cmd == "rollback":
        res = rollback(Path(args.backup) if args.backup else None)
        print(f"[rollback] restored from : {res['backup']}")
        print(f"[rollback] destination   : {res['active']}")
        print(f"[rollback] pre-rollback backup : {res['prev_backup'] or '(none)'}")
        print(REBUILD_HINT)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
