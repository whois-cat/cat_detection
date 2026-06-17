"""#10: WAL + busy_timeout, and inserts are committed immediately (autocommit),
so they are durable/visible without an explicit flush step."""
import sqlite3

from storage import init_db, insert_event, query_events


def _insert(conn, wall_ms):
    insert_event(conn, camera_id="grey", model="yolov8n", wall_ms=wall_ms,
                 pts=None, tb_num=None, tb_den=None, media_t=None,
                 frame_w=320, frame_h=240, rotate_deg=0, cat="alisa", cat_score=0.9,
                 box_x=1, box_y=2, box_w=3, box_h=4, score=0.8)


def test_wal_and_busy_timeout(tmp_path):
    conn = init_db(tmp_path / "events.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_insert_is_immediately_durable(tmp_path):
    db = tmp_path / "events.db"
    conn = init_db(db)
    _insert(conn, 1000)
    # A separate connection sees the row without any explicit commit/flush.
    other = sqlite3.connect(str(db))
    n = other.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert n == 1
    rows = query_events(conn, camera_id="grey", t_from=0, t_to=2000)
    assert rows[0]["cat"] == "alisa"
