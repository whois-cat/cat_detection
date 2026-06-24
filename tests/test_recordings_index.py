from __future__ import annotations

from datetime import datetime, timezone
import os
import sqlite3

from recordings_index import (
    delete_paths_from_index,
    incremental_update,
    last_indexed_to_ms,
    lookup_segment,
    merge_availability_ranges,
    query_ranges,
    reconcile_deletions,
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


def _age(path, *, seconds: float) -> None:
    """Backdate a file's mtime so the indexer treats it as a settled segment."""
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


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


# --- live (incremental) indexer ------------------------------------------------


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def test_incremental_indexes_new_file_appearing_after_rebuild(tmp_path, monkeypatch):
    """A fresh mp4 written after the initial rebuild becomes visible via the
    live indexer with no manual rebuild — the core bug we're fixing."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    conn = sqlite3.connect(tmp_path / "events.db")

    # Initial state already indexed (a segment from ~30 min ago).
    first_name = datetime.fromtimestamp((now - 1800_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    first = _segment(rec, "grey", first_name)
    _age(first, seconds=60)
    refresh_index(conn, rec, now_ms=now)
    assert query_ranges(conn, camera_id="grey", from_ms=now - 3600_000, to_ms=now + 3600_000)

    # A new segment lands later. No rebuild is run.
    fname = datetime.fromtimestamp((now - 60_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    fresh = _segment(rec, "grey", fname)
    _age(fresh, seconds=30)

    stats = incremental_update(conn, rec, now_ms=now)

    assert stats["inserted"] == 1
    seg = lookup_segment(conn, camera_id="grey", wall_ms=now - 45_000)
    assert seg is not None and seg.path == str(fresh)


def test_incremental_skips_file_still_being_written(tmp_path, monkeypatch):
    """A just-modified (partial) mp4 is ignored until it settles."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    fname = datetime.fromtimestamp((now - 5_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    partial = _segment(rec, "grey", fname, b"partial")
    _age(partial, seconds=2)  # younger than the 10s stable-age default
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = incremental_update(conn, rec, now_ms=now)
    assert stats["inserted"] == 0
    assert stats["skipped_unstable"] == 1
    assert lookup_segment(conn, camera_id="grey", wall_ms=now - 1_000) is None

    # Once it stops changing, the next cycle picks it up.
    _age(partial, seconds=30)
    stats = incremental_update(conn, rec, now_ms=now)
    assert stats["inserted"] == 1


def test_incremental_empty_size_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    fname = datetime.fromtimestamp((now - 60_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    empty = _segment(rec, "grey", fname, b"")
    _age(empty, seconds=60)
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = incremental_update(conn, rec, now_ms=now)
    assert stats["inserted"] == 0
    assert stats["skipped_unstable"] == 1


def test_incremental_only_scans_recent_window(tmp_path, monkeypatch):
    """Old files outside the window are not touched by an incremental cycle."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    old = _segment(rec, "grey", "2020-01-01_00-00-00-000000.mp4")
    _age(old, seconds=99999)
    recent_name = datetime.fromtimestamp((now - 60_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    recent = _segment(rec, "grey", recent_name)
    _age(recent, seconds=30)
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = incremental_update(conn, rec, now_ms=now, window_ms=10 * 60_000)

    assert stats["inserted"] == 1  # only the recent file
    assert conn.execute(
        "SELECT COUNT(*) FROM recording_segments WHERE path=?", (str(old),)
    ).fetchone()[0] == 0


def test_incremental_reconciles_recent_deletion(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    fname = datetime.fromtimestamp((now - 60_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    seg = _segment(rec, "grey", fname)
    _age(seg, seconds=30)
    conn = sqlite3.connect(tmp_path / "events.db")
    incremental_update(conn, rec, now_ms=now)

    seg.unlink()
    stats = incremental_update(conn, rec, now_ms=now)

    assert stats["marked_missing"] == 1
    assert lookup_segment(conn, camera_id="grey", wall_ms=now - 45_000) is None


def test_reconcile_deletions_drops_pruned_old_rows(tmp_path, monkeypatch):
    """Pruner deletes old files outside the incremental window; reconcile drops
    their rows so the timeline stops advertising them."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    old = _segment(rec, "grey", "2026-06-20_12-00-00-000000.mp4")
    keep = _segment(rec, "grey", "2026-06-20_12-30-00-000000.mp4")  # survives pruning
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)
    assert conn.execute("SELECT COUNT(*) FROM recording_segments").fetchone()[0] == 2

    old.unlink()  # pruner removes the aged-out segment
    stats = reconcile_deletions(conn)

    assert stats == {"checked": 2, "deleted": 1, "skipped_cameras": []}
    rows = conn.execute("SELECT path FROM recording_segments").fetchall()
    assert rows == [(str(keep),)]


def test_reconcile_deletions_safety_skips_when_all_of_a_camera_missing(tmp_path, monkeypatch):
    """If every file for a camera is gone we assume its dir is unmounted and
    change nothing for it."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    a = _segment(rec, "grey", "2026-06-20_12-00-00-000000.mp4")
    b = _segment(rec, "grey", "2026-06-20_12-00-30-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    a.unlink()
    b.unlink()
    stats = reconcile_deletions(conn)

    assert stats["skipped_cameras"] == ["grey"]
    assert stats["deleted"] == 0
    assert conn.execute("SELECT COUNT(*) FROM recording_segments").fetchone()[0] == 2


def test_reconcile_deletions_is_per_camera(tmp_path, monkeypatch):
    """One camera's whole dir being unmounted must not block reconciling a
    healthy sibling, and must not touch the unmounted camera's rows."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    # grey: 2 files, 1 pruned -> healthy, reconcile the gone one.
    g_old = _segment(rec, "grey", "2026-06-20_12-00-00-000000.mp4")
    _segment(rec, "grey", "2026-06-20_12-30-00-000000.mp4")
    # beige: both files vanish at once -> looks unmounted, leave alone.
    b1 = _segment(rec, "beige", "2026-06-20_12-00-00-000000.mp4")
    b2 = _segment(rec, "beige", "2026-06-20_12-00-30-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    g_old.unlink()
    b1.unlink()
    b2.unlink()
    stats = reconcile_deletions(conn)

    assert stats["deleted"] == 1
    assert stats["skipped_cameras"] == ["beige"]
    cams = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT camera_id, COUNT(*) FROM recording_segments GROUP BY camera_id"
        )
    }
    assert cams == {"grey": 1, "beige": 2}


def test_delete_paths_from_index_removes_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    p = _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    refresh_index(conn, rec)

    removed = delete_paths_from_index(conn, [str(p)])

    assert removed == 1
    assert conn.execute("SELECT COUNT(*) FROM recording_segments").fetchone()[0] == 0


def test_incremental_does_not_resurrect_deleted_typo_camera(tmp_path, monkeypatch):
    """An allow-list keeps a deleted/typo camera dir from being re-indexed even
    if files still linger on disk, and reconcile drops any stale rows."""
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    now = _now_ms()
    good_name = datetime.fromtimestamp((now - 60_000) / 1000, timezone.utc).strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    ) + ".mp4"
    good = _segment(rec, "beige", good_name)
    typo = _segment(rec, "beidge", good_name)
    _age(good, seconds=30)
    _age(typo, seconds=30)
    conn = sqlite3.connect(tmp_path / "events.db")

    stats = incremental_update(conn, rec, now_ms=now, cameras={"beige", "grey"})

    assert stats["cameras"] == ["beige"]
    cams = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT camera_id FROM recording_segments"
        ).fetchall()
    ]
    assert cams == ["beige"]
    assert lookup_segment(conn, camera_id="beidge", wall_ms=now - 45_000) is None


def test_last_indexed_to_ms_tracks_catchup_point(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_TZ", "UTC")
    rec = tmp_path / "recordings"
    _segment(rec, "grey", "2026-06-24_12-00-00-000000.mp4")
    _segment(rec, "grey", "2026-06-24_12-00-30-000000.mp4")
    conn = sqlite3.connect(tmp_path / "events.db")
    assert last_indexed_to_ms(conn) is None

    refresh_index(conn, rec)

    assert last_indexed_to_ms(conn) == _ms("2026-06-24T12:01:00")
    assert last_indexed_to_ms(conn, cameras={"grey"}) == _ms("2026-06-24T12:01:00")
    assert last_indexed_to_ms(conn, cameras={"nope"}) is None
