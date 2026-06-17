"""Make the feeder package modules importable by their bare names.

feeder/ modules import each other flat (`from zone_state import ...`), matching
how they run inside the feeder container (WORKDIR=/app). Tests import them the
same way, so we prepend feeder/ to sys.path.
"""
import sys
from pathlib import Path

FEEDER = Path(__file__).resolve().parent.parent / "feeder"
if str(FEEDER) not in sys.path:
    sys.path.insert(0, str(FEEDER))

# detector/ modules also import each other flat (WORKDIR=/app in the container).
DETECTOR = Path(__file__).resolve().parent.parent / "detector"
if str(DETECTOR) not in sys.path:
    sys.path.insert(0, str(DETECTOR))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
