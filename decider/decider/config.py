"""The decider's part of config.yaml (see config.example.yaml). `${VAR}` is
expanded from the environment like in streamhub; an unset variable is an error.
Per-feeder tuning defaults are the previous feeder service's defaults."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


@dataclass
class FeedConfig:
    # "none": no automatic feeding (door control only); "scheduled": fixed times.
    mode: str = "none"
    grain_num: int = 1
    times: list[str] = field(default_factory=list)  # "HH:MM"
    tz: str = ""  # IANA name; empty = local
    catchup_max: int = 2


@dataclass
class FeederConfig:
    id: str
    camera: str
    api_base_url: str
    serial_number: str
    allowed_cats: list[str]
    door_close_timeout_sec: float = 30
    min_meal_sec: float = 10
    presence_window_sec: float = 5
    # Identity votes below this count as unknown (or are dropped) in ZoneState.
    classifier_min_conf: float = 0.9
    # A detection whose top identity probability is below this is labelled
    # "unknown" before ZoneState (what the old detector did).
    unknown_conf: float = 0.5
    open_min_confidence: float = 0
    open_min_margin: float = 0
    open_debounce_sec: float = 3
    multi_debounce_sec: float = 2
    display_text_interval: int = 2
    display_refresh_min_sec: float = 25
    stream_blip_grace_sec: float = 25
    close_backstop_max_attempts: int = 3
    close_backstop_max_sec: float = 30
    # Box centres must be inside (camera-frame fractions) to count.
    action_polygon: list[list[float]] = field(default_factory=lambda: [p[:] for p in FULL_FRAME])
    dangerous_confusions: list[dict[str, str]] = field(default_factory=list)
    # Open when any allowed identity got votes in the presence window, even if
    # another cat won the vote; all allowed identities then count as one (so
    # alternating between them neither delays opening nor closes the door).
    open_if_any_allowed: bool = False
    # Show a short live status ("<cat letter> <C/O>") on the feeder display
    # instead of only the cat name on open (see display.py).
    status_display: bool = False
    feed: FeedConfig = field(default_factory=FeedConfig)


@dataclass
class DeciderConfig:
    hub: str
    journal_db: str
    feeders: list[FeederConfig]
    # Decide and journal, but never call the feeder API (log instead).
    dry_run: bool = False


def _expand(node: Any, missing: set[str]) -> Any:
    """Replace ${VAR} in all string values (not keys or comments)."""
    if isinstance(node, str):
        missing.update(m for m in _ENV_REF.findall(node) if m not in os.environ)
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), node)
    if isinstance(node, dict):
        return {k: _expand(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v, missing) for v in node]
    return node


def _build(cls, raw: dict[str, Any], where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    try:
        return cls(**raw)
    except TypeError as e:
        raise ValueError(f"{where}: {e}") from None


def parse(text: str) -> DeciderConfig:
    missing: set[str] = set()
    doc = _expand(yaml.safe_load(text) or {}, missing)
    if missing:
        raise ValueError(f"config references unset environment variables: {', '.join(sorted(missing))}")
    data_dir = doc.get("data_dir", "data")
    cameras = {c["id"] for c in doc.get("cameras", [])}
    d = doc.get("decider") or {}
    feeders = []
    for i, raw in enumerate(d.get("feeders") or []):
        raw = dict(raw)
        feed = _build(FeedConfig, raw.pop("feed", None) or {}, f"decider.feeders[{i}].feed")
        f = _build(FeederConfig, {**raw, "feed": feed}, f"decider.feeders[{i}]")
        if f.camera not in cameras:
            raise ValueError(f"feeder {f.id}: unknown camera {f.camera!r}")
        if not f.allowed_cats:
            raise ValueError(f"feeder {f.id}: allowed_cats must not be empty")
        if feed.mode not in ("none", "scheduled"):
            raise ValueError(f"feeder {f.id}: feed.mode must be none or scheduled")
        if len(f.action_polygon) < 3:
            raise ValueError(f"feeder {f.id}: action_polygon needs at least 3 points")
        feeders.append(f)
    ids = [f.id for f in feeders]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate feeder ids")
    hub = d.get("hub") or "127.0.0.1:" + str(doc.get("streamhub", {}).get("hub_listen", ":9000")).rpartition(":")[2]
    # DECIDER_DRY_RUN=1 forces a dry run regardless of the config (e.g. a test
    # machine sharing the production config); it can never turn one off.
    dry_run = bool(d.get("dry_run", False)) or os.environ.get("DECIDER_DRY_RUN", "") not in ("", "0")
    # A dry run journals pretend feeds and door sessions: keep them out of the
    # real journal, which scheduled feeding trusts for "already fed today".
    default_journal = "journal.dry-run.db" if dry_run else "journal.db"
    return DeciderConfig(
        hub=hub,
        journal_db=d.get("journal_db") or str(Path(data_dir) / "feed_journal" / default_journal),
        feeders=feeders,
        dry_run=dry_run,
    )


def load(path: str) -> DeciderConfig:
    return parse(Path(path).read_text(encoding="utf-8"))
