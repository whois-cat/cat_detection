"""Make the repo root and cv-worker importable for training/review tests
(training imports the runtime classifier from cv_worker)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "cv-worker"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
