"""SQLite-backed event store.

One row per detection. `camera_id` and `model` are mandatory tags so we can
support multi-camera and multi-model setups without schema changes later.

WAL mode + indices keep range queries fast while concurrent readers (pruner,
UI queries) run alongside the detector's writes.
"""
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY,
    camera_id TEXT    NOT NULL,
    model     TEXT    NOT NULL,
    wall_ms   INTEGER NOT NULL,
    pts       INTEGER,
    tb_num    INTEGER, tb_den INTEGER,
    media_t   REAL,
    frame_w   INTEGER, frame_h INTEGER,
    cat       TEXT,
    cat_score REAL,
    box_x     INTEGER, box_y INTEGER, box_w INTEGER, box_h INTEGER,
    score     REAL,
    track_id  INTEGER,
    source    TEXT    NOT NULL DEFAULT 'detector'
);
CREATE INDEX IF NOT EXISTS idx_events_cam_model_wall ON events(camera_id, model, wall_ms);
CREATE INDEX IF NOT EXISTS idx_events_cam_wall       ON events(camera_id, wall_ms);
CREATE INDEX IF NOT EXISTS idx_events_track          ON events(camera_id, model, track_id);
"""


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent, additive schema migrations for pre-existing databases.

    `cat_score` was added after the first events.db files were created in the
    field; ADD COLUMN is the only safe in-place change SQLite supports without
    a table rebuild, and a duplicate-column error just means we're already
    migrated."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if "cat_score" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN cat_score REAL")


def insert_event(conn: sqlite3.Connection, *,
                 camera_id: str, model: str, wall_ms: int,
                 pts: int | None, tb_num: int | None, tb_den: int | None,
                 media_t: float | None,
                 frame_w: int, frame_h: int,
                 cat: str | None, cat_score: float | None,
                 box_x: int, box_y: int, box_w: int, box_h: int, score: float,
                 track_id: int | None = None, source: str = "detector") -> None:
    conn.execute(
        """INSERT INTO events
           (camera_id, model, wall_ms, pts, tb_num, tb_den, media_t,
            frame_w, frame_h, cat, cat_score,
            box_x, box_y, box_w, box_h, score, track_id, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (camera_id, model, wall_ms, pts, tb_num, tb_den, media_t,
         frame_w, frame_h, cat, cat_score,
         box_x, box_y, box_w, box_h, score, track_id, source),
    )


def query_events(conn: sqlite3.Connection, *,
                 camera_id: str, t_from: int, t_to: int,
                 model: str | None = None) -> list[dict]:
    """Return events as one dict per row, in browser-compatible shape."""
    sql = (
        "SELECT wall_ms, pts, tb_num, tb_den, media_t,"
        "       frame_w, frame_h, cat, cat_score,"
        "       box_x, box_y, box_w, box_h, score, track_id, model "
        "FROM events WHERE camera_id=? AND wall_ms BETWEEN ? AND ?"
    )
    params: list = [camera_id, t_from, t_to]
    if model:
        sql += " AND model=?"
        params.append(model)
    sql += " ORDER BY wall_ms"
    out = []
    for r in conn.execute(sql, params):
        out.append({
            "wall_ms": r[0], "pts": r[1],
            "tb_num": r[2], "tb_den": r[3], "media_t": r[4],
            "w": r[5], "h": r[6], "cat": r[7], "cat_score": r[8],
            "boxes": [{"x": r[9], "y": r[10], "w": r[11], "h": r[12], "score": r[13]}],
            "track_id": r[14], "model": r[15],
        })
    return out


def query_event_times(conn: sqlite3.Connection, *,
                      camera_id: str | None = None,
                      t_from: int | None = None,
                      t_to: int | None = None) -> list[int]:
    """Lightweight: just wall_ms values, for the pruner."""
    sql = "SELECT wall_ms FROM events"
    where = []
    params: list = []
    if camera_id is not None:
        where.append("camera_id=?")
        params.append(camera_id)
    if t_from is not None:
        where.append("wall_ms >= ?")
        params.append(t_from)
    if t_to is not None:
        where.append("wall_ms <= ?")
        params.append(t_to)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY wall_ms"
    return [r[0] for r in conn.execute(sql, params)]


def list_models(conn: sqlite3.Connection, camera_id: str | None = None) -> list[str]:
    if camera_id:
        cur = conn.execute(
            "SELECT DISTINCT model FROM events WHERE camera_id=? ORDER BY model",
            (camera_id,),
        )
    else:
        cur = conn.execute("SELECT DISTINCT model FROM events ORDER BY model")
    return [r[0] for r in cur]
