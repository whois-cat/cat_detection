import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import clean  # noqa: E402


def test_junk_never_touches_kept_paths_or_their_parents():
    ignored = [".venv/", "webui/node_modules/", "data/", "models/trained/", "models/cat.pt",
               "secrets/", ".env", "config.yaml", "reviews.db", "cameras.yaml", "cv-worker/.venv/"]
    assert clean.junk(ignored) == [".venv", "webui/node_modules", "cameras.yaml", "cv-worker/.venv"]
    # A directory that merely contains a kept path is kept too.
    assert clean.junk(["data/"], keep=("data/decider",)) == []
    assert clean.junk(["database/"], keep=("data",)) == ["database"]
