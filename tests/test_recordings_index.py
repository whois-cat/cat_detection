from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from recordings_index import (
    lookup_segment,
    merge_availability_ranges,
    query_ranges,
    refresh_index,
)
from storage import init_db


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _segment(root, camera: str, name: str, body: bytes = b"mp4"):
    path = root / camera / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_storage_init_creates_recording_segments_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    conn = init_db(tmp_path / "events.db")
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "events" in names
    assert "recording_segments" in names


def test_rebuild_indexes_fake_recording_files_and_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    p = _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = refresh_index(conn, rec, now_ms=1_000)

    assert stats == {"ready": 1, "marked_missing": 0, "cameras": ["grey"]}
    seg = lookup_segment(conn, camera_id="grey", wall_ms=_ms("2026-06-24T12:00:05"))
    assert seg is not None
    assert seg.path == str(p)
    assert seg.start_ms == _ms("2026-06-24T12:00:00")
    assert seg.end_ms == _ms("2026-06-24T12:00:30")


def test_query_ranges_returns_segments_overlapping_range(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    _segment(rec, "grey", "2026-06-24_12-00-30-000000.mp4")
    _segment(rec, "blue", "2026-06-24_12-00-00-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    rows = query_ranges(
        conn,
        camera_id="grey",
        from_ms=_ms("2026-06-24T12:00:20"),
        to_ms=_ms("2026-06-24T12:00:45"),
    )

    assert rows == [[
        _ms("2026-06-24T12:00:20"),
        _ms("2026-06-24T12:00:45"),
    ]]
    assert all("mp4" not in str(r) for r in rows)


def test_merge_availability_ranges_keeps_gaps_and_clamps_window():
    rows = [
        (_ms("2026-06-24T12:00:00"), _ms("2026-06-24T12:00:30")),
        (_ms("2026-06-24T12:00:30"), _ms("2026-06-24T12:01:00")),
        (_ms("2026-06-24T12:03:00"), _ms("2026-06-24T12:03:30")),
    ]

    merged = merge_availability_ranges(
        rows,
        from_ms=_ms("2026-06-24T12:00:10"),
        to_ms=_ms("2026-06-24T12:03:10"),
    )

    assert merged == [
        [_ms("2026-06-24T12:00:10"), _ms("2026-06-24T12:01:00")],
        [_ms("2026-06-24T12:03:00"), _ms("2026-06-24T12:03:10")],
    ]


def test_missing_file_is_marked_missing_and_not_returned(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    p = _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    p.unlink()
    stats = refresh_index(conn, rec)

    assert stats["ready"] == 0
    assert stats["marked_missing"] == 1
    assert lookup_segment(conn, camera_id="grey", wall_ms=_ms("2026-06-24T12:00:05")) is None
    status = conn.execute("SELECT status FROM recording_segments WHERE path=?", (str(p),)).fetchone()[0]
    assert status == "missing"


def test_duplicate_path_refresh_updates_existing_row(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    p = _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4", b"a")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    p.write_bytes(b"longer")
    refresh_index(conn, rec)

    count = conn.execute("SELECT COUNT(*) FROM recording_segments").fetchone()[0]
    size = conn.execute("SELECT size_bytes FROM recording_segments WHERE path=?", (str(p),)).fetchone()[0]
    assert count == 1
    assert size == len(b"longer")


def test_adjacent_boundary_prefers_newer_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    first = _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    second = _segment(rec, "grey", "2026-06-24_12-00-30-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    seg = lookup_segment(conn, camera_id="grey", wall_ms=_ms("2026-06-24T12:00:30"))

    assert seg is not None
    assert seg.path == str(second)
    assert seg.path != str(first)


def test_empty_index_returns_empty_fallback_values(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = refresh_index(conn, tmp_path / "missing-recordings")

    assert stats == {"ready": 0, "marked_missing": 0, "cameras": []}
    assert query_ranges(conn, camera_id="grey", from_ms=0, to_ms=1) == []
    assert lookup_segment(conn, camera_id="grey", wall_ms=1) is None
