"""Tests for the live indexer service glue (indexer/indexer.py).

The heavy lifting is in recordings_index.py (covered by test_recordings_index);
here we pin the per-camera restart catch-up math and camera resolution, which
are the parts unique to the service.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest

from recordings_index import ensure_schema


@pytest.fixture
def indexer(monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    monkeypatch.setenv("STARTUP_LOOKBACK_SEC", str(24 * 3600))
    monkeypatch.setenv("SEGMENT_DURATION_SEC", "30")
    monkeypatch.delenv("RECORDINGS_CAMERAS", raising=False)
    import indexer as mod

    return importlib.reload(mod)


def _seg(conn, camera, end_ms, path):
    conn.execute(
        "INSERT INTO recording_segments "
        "(camera_id, start_ms, end_ms, path, created_at_ms, status) "
        "VALUES (?, ?, ?, ?, 0, 'ready')",
        (camera, end_ms - 30_000, end_ms, path),
    )
    conn.commit()


def test_catchup_resumes_from_that_cameras_last_indexed(indexer, tmp_path):
    conn = sqlite3.connect(tmp_path / "events.db")
    ensure_schema(conn)
    now = 2_000_000_000_000
    grey_end = now - 5 * 60_000          # grey is current (5 min ago)
    beige_end = now - 3 * 3600_000       # beige lagged (3h ago)
    _seg(conn, "grey", grey_end, "data/recordings/grey/a.mp4")
    _seg(conn, "beige", beige_end, "data/recordings/beige/b.mp4")

    # Each camera resumes from its OWN last end, not a shared global MAX.
    assert indexer.catchup_since_ms(conn, "grey", now) == grey_end - indexer.CATCHUP_OVERLAP_MS
    assert indexer.catchup_since_ms(conn, "beige", now) == beige_end - indexer.CATCHUP_OVERLAP_MS


def test_catchup_caps_at_startup_lookback(indexer, tmp_path):
    conn = sqlite3.connect(tmp_path / "events.db")
    ensure_schema(conn)
    now = 2_000_000_000_000
    ancient_end = now - 10 * 24 * 3600 * 1000  # down for days
    _seg(conn, "grey", ancient_end, "data/recordings/grey/y.mp4")

    assert indexer.catchup_since_ms(conn, "grey", now) == now - indexer.STARTUP_LOOKBACK_MS


def test_catchup_unknown_camera_uses_floor(indexer, tmp_path):
    conn = sqlite3.connect(tmp_path / "events.db")
    ensure_schema(conn)
    now = 2_000_000_000_000
    _seg(conn, "grey", now - 60_000, "data/recordings/grey/y.mp4")

    # A camera with nothing indexed yet starts at the lookback floor.
    assert indexer.catchup_since_ms(conn, "beige", now) == now - indexer.STARTUP_LOOKBACK_MS


def test_resolve_cameras_prefers_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    monkeypatch.setenv("RECORDINGS_CAMERAS", "beige,grey")
    import indexer as mod

    mod = importlib.reload(mod)
    conn = sqlite3.connect(tmp_path / "events.db")
    ensure_schema(conn)
    # A stale typo camera in the DB must NOT be picked up when an allow-list is set.
    _seg(conn, "beidge", 1_000, "data/recordings/beidge/z.mp4")

    assert mod.resolve_cameras(conn) == ["beige", "grey"]


def test_resolve_cameras_falls_back_to_disk_and_db(indexer, tmp_path, monkeypatch):
    # No allow-list: union of on-disk dirs and already-indexed cameras.
    (tmp_path / "data/recordings/grey").mkdir(parents=True)
    monkeypatch.setattr(indexer, "RECORDINGS_DIR", tmp_path / "data/recordings")
    conn = sqlite3.connect(tmp_path / "events.db")
    ensure_schema(conn)
    _seg(conn, "beige", 1_000, "data/recordings/beige/z.mp4")

    assert indexer.resolve_cameras(conn) == ["beige", "grey"]
