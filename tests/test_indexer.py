"""Tests for the live indexer service glue (indexer/indexer.py).

The heavy lifting is in recordings_index.py (covered by test_recordings_index);
here we just pin the restart catch-up window math, which is the part unique to
the service.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest


@pytest.fixture
def indexer(monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    monkeypatch.setenv("STARTUP_LOOKBACK_SEC", str(24 * 3600))
    monkeypatch.setenv("SEGMENT_DURATION_SEC", "30")
    import indexer as mod

    return importlib.reload(mod)


def test_catchup_resumes_from_last_indexed(indexer, tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "events.db")
    from recordings_index import ensure_schema

    ensure_schema(conn)
    now = 2_000_000_000_000
    last_end = now - 5 * 60_000  # 5 min ago, well inside the 24h cap
    conn.execute(
        "INSERT INTO recording_segments "
        "(camera_id, start_ms, end_ms, path, created_at_ms, status) "
        "VALUES ('grey', ?, ?, '/x.mp4', 0, 'ready')",
        (last_end - 30_000, last_end),
    )
    conn.commit()

    since = indexer.catchup_since_ms(conn, now)

    # Resumes a bit before the last indexed end (overlap), not at the 24h floor.
    assert since == last_end - indexer.CATCHUP_OVERLAP_MS
    assert since > now - indexer.STARTUP_LOOKBACK_MS


def test_catchup_caps_at_startup_lookback(indexer, tmp_path):
    conn = sqlite3.connect(tmp_path / "events.db")
    from recordings_index import ensure_schema

    ensure_schema(conn)
    now = 2_000_000_000_000
    # Index has been down for days: last segment is way past the lookback cap.
    ancient_end = now - 10 * 24 * 3600 * 1000
    conn.execute(
        "INSERT INTO recording_segments "
        "(camera_id, start_ms, end_ms, path, created_at_ms, status) "
        "VALUES ('grey', ?, ?, '/y.mp4', 0, 'ready')",
        (ancient_end - 30_000, ancient_end),
    )
    conn.commit()

    since = indexer.catchup_since_ms(conn, now)

    assert since == now - indexer.STARTUP_LOOKBACK_MS


def test_catchup_empty_index_uses_floor(indexer, tmp_path):
    conn = sqlite3.connect(tmp_path / "events.db")
    from recordings_index import ensure_schema

    ensure_schema(conn)
    now = 2_000_000_000_000

    since = indexer.catchup_since_ms(conn, now)

    assert since == now - indexer.STARTUP_LOOKBACK_MS
