"""Remove local files git ignores, without touching config and state.

    python tools/clean.py junk     [--yes]   # build artifacts, caches, venvs, old-stack leftovers (OBSOLETE)
    python tools/clean.py history  [--yes]   # junk + the new stack's recordings and dry-run journal

Without --yes it only lists what it would remove. Everything under KEEP (and
any directory containing a KEEP path) is never removed; `history` then removes
HISTORY explicitly. The previous stack's data/recordings and data/events are
kept: training still reads them.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config and state: never removed.
KEEP = ("config.yaml", ".env", "secrets", "data", "models", "reviews.db")
# Previous-stack leftovers inside KEEP dirs (removed by `junk` too): mediamtx's
# generated camera URLs with credentials, and an older feeder's state.
OBSOLETE = ("secrets/cameras.env", "data/cooldowns")
# Recorded history of the new stack (removed by `history`).
HISTORY = ("data/streamhub/recordings", "data/decider/feed_journal/journal.dry-run.db",
           "data/decider/feed_journal/journal.dry-run.db-wal", "data/decider/feed_journal/journal.dry-run.db-shm")


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def junk(ignored: list[str], keep: tuple[str, ...] = KEEP) -> list[str]:
    """Ignored paths (as `git ls-files --directory` lists them, dirs ending in
    "/") that are neither kept nor contain a kept path."""
    out = []
    for entry in ignored:
        p = entry.rstrip("/")
        if any(_under(p, k) or _under(k, p) for k in keep):
            continue
        out.append(p)
    return out


def ignored_paths() -> list[str]:
    res = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"],
        cwd=ROOT, check=True, capture_output=True)
    return [p for p in res.stdout.decode().split("\0") if p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["junk", "history"])
    ap.add_argument("--yes", action="store_true", help="actually remove (default: only list)")
    args = ap.parse_args(argv)

    targets = junk(ignored_paths()) + [o for o in OBSOLETE if (ROOT / o).exists()]
    if args.what == "history":
        targets += [h for h in HISTORY if (ROOT / h).exists()]
    if not targets:
        print("nothing to remove")
        return 0
    for t in targets:
        print(("removing " if args.yes else "would remove ") + t)
        if args.yes:
            path = ROOT / t
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    if not args.yes:
        print("(dry run — add --yes to remove)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
