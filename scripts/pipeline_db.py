from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "metadata" / "cat_pipeline.duckdb"
DEFAULT_VIDEOS_INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "videos_index.csv"
DEFAULT_RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_CVAT_EXPORTS_DIR = PROJECT_ROOT / "data" / "cvat_exports"
DEFAULT_CROPS_DIR = PROJECT_ROOT / "data" / "crops"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "train_data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "cat_classifier_best.pt"


@dataclass(frozen=True)
class VideoIndexRecord:
    video_name: str
    date: str
    time: str
    hour: int
    lighting: str
    split_group: str


@dataclass(frozen=True)
class VideoProbe:
    duration_seconds: float
    fps: float
    width: int
    height: int
    file_size_bytes: int


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_name VARCHAR PRIMARY KEY,
    video_path VARCHAR NOT NULL,
    date VARCHAR,
    time VARCHAR,
    hour INTEGER,
    lighting VARCHAR,
    split_group VARCHAR,
    duration_seconds DOUBLE,
    fps DOUBLE,
    width INTEGER,
    height INTEGER,
    file_size_bytes BIGINT,
    updated_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS detections (
    detection_uid VARCHAR PRIMARY KEY,
    video_name VARCHAR NOT NULL,
    timestamp_seconds DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    model_name VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    sampler_interval_seconds DOUBLE NOT NULL,
    frame_width INTEGER,
    frame_height INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS intervals (
    interval_uid VARCHAR PRIMARY KEY,
    video_name VARCHAR NOT NULL,
    start_seconds DOUBLE NOT NULL,
    end_seconds DOUBLE NOT NULL,
    detection_count INTEGER NOT NULL,
    max_confidence DOUBLE NOT NULL,
    interval_source VARCHAR NOT NULL,
    model_name VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS frames (
    frame_uid VARCHAR PRIMARY KEY,
    video_name VARCHAR NOT NULL,
    timestamp_seconds DOUBLE NOT NULL,
    frame_name VARCHAR NOT NULL UNIQUE,
    frame_path VARCHAR NOT NULL UNIQUE,
    width INTEGER,
    height INTEGER,
    cvat_task_name VARCHAR,
    is_duplicate BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT current_timestamp
);

ALTER TABLE frames ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT false;

CREATE TABLE IF NOT EXISTS frame_intervals (
    frame_uid VARCHAR NOT NULL,
    interval_uid VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE UNIQUE INDEX IF NOT EXISTS frame_intervals_unique_idx
ON frame_intervals(frame_uid, interval_uid);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_uid VARCHAR PRIMARY KEY,
    frame_uid VARCHAR NOT NULL,
    image_name VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    bbox_x DOUBLE,
    bbox_y DOUBLE,
    bbox_width DOUBLE,
    bbox_height DOUBLE,
    area DOUBLE,
    is_crowd BOOLEAN,
    cvat_task_name VARCHAR,
    export_path VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS crops (
    crop_uid VARCHAR PRIMARY KEY,
    frame_uid VARCHAR,
    frame_name VARCHAR NOT NULL,
    crop_path VARCHAR NOT NULL,
    label VARCHAR DEFAULT 'unsorted',
    confidence DOUBLE,
    bbox_x DOUBLE NOT NULL,
    bbox_y DOUBLE NOT NULL,
    bbox_width DOUBLE NOT NULL,
    bbox_height DOUBLE NOT NULL,
    padding DOUBLE NOT NULL,
    crop_width INTEGER,
    crop_height INTEGER,
    is_duplicate BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT current_timestamp
);

ALTER TABLE crops DROP COLUMN IF EXISTS annotation_uid;
ALTER TABLE crops ADD COLUMN IF NOT EXISTS frame_name VARCHAR;
ALTER TABLE crops ADD COLUMN IF NOT EXISTS confidence DOUBLE;
ALTER TABLE crops ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT false;

CREATE TABLE IF NOT EXISTS live_detections (
    detection_uid VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    cat_name VARCHAR NOT NULL,
    confidence DOUBLE NOT NULL,
    source VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp
);
"""


def select_inference_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def connect_db(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    connection.execute(SCHEMA_SQL)
    return connection


def make_uid(prefix: str, *parts: object) -> str:
    hasher = hashlib.sha1()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return f"{prefix}_{hasher.hexdigest()[:16]}"


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def frame_name_for_timestamp(video_name: str, timestamp_seconds: float) -> str:
    milliseconds = int(round(timestamp_seconds * 1000))
    return f"{Path(video_name).stem}__ts_{milliseconds:010d}.jpg"


def iter_timestamps(start_seconds: float, end_seconds: float, step_seconds: float) -> list[float]:
    timestamps: list[float] = []
    current = start_seconds
    while current <= end_seconds + 1e-9:
        timestamps.append(round(current, 3))
        current += step_seconds
    return timestamps


def load_video_index(index_csv_path: Path = DEFAULT_VIDEOS_INDEX_PATH) -> dict[str, VideoIndexRecord]:
    if not index_csv_path.exists():
        return {}

    with index_csv_path.open("r", encoding="utf-8", newline="") as input_file:
        rows = csv.DictReader(input_file)
        return {
            row["video_name"]: VideoIndexRecord(
                video_name=row["video_name"],
                date=row["date"],
                time=row["time"],
                hour=int(row["hour"]),
                lighting=row["lighting"],
                split_group=row["split_group"],
            )
            for row in rows
        }


def probe_video(video_path: Path) -> VideoProbe | None:
    import av
    import subprocess, json

    with av.open(str(video_path)) as container:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            return None

        duration_seconds = None
        if video_stream.duration is not None:
            duration_seconds = float(video_stream.duration * video_stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)

        if duration_seconds is None or duration_seconds <= 0:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    str(video_path),
                ],
                capture_output=True, text=True,
            )
            fmt = json.loads(result.stdout).get("format", {})
            if "duration" in fmt:
                duration_seconds = float(fmt["duration"])
            else:
                return None

        if video_stream.average_rate is None:
            return None

        return VideoProbe(
            duration_seconds=duration_seconds,
            fps=float(video_stream.average_rate),
            width=int(video_stream.width or 0),
            height=int(video_stream.height or 0),
            file_size_bytes=video_path.stat().st_size,
        )


def open_video_stream(video_path: Path) -> tuple[Any, Any]:
    import av

    container = av.open(str(video_path))
    video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
    if video_stream is None:
        container.close()
        raise RuntimeError(f"no video stream found in {video_path}")
    return container, video_stream


def load_frame_at_seconds(container: Any, video_stream: Any, timestamp_seconds: float) -> Any | None:
    import av

    seek_target = max(0, int(timestamp_seconds / float(video_stream.time_base)))
    try:
        container.seek(seek_target, stream=video_stream, backward=True)
    except av.error.PermissionError:
        # Some MKV files reject explicit seeks. Fall back to sequential decode
        # from the current cursor so the pipeline can continue instead of aborting.
        pass

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue

        frame_seconds = float(frame.pts * video_stream.time_base)
        if frame_seconds + 1e-6 >= timestamp_seconds:
            return frame

    return None


def upsert_video(
    connection: duckdb.DuckDBPyConnection,
    video_path: Path,
    index_record: VideoIndexRecord | None,
    probe: VideoProbe,
) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    connection.execute(
        """
        INSERT INTO videos (
            video_name,
            video_path,
            date,
            "time",
            hour,
            lighting,
            split_group,
            duration_seconds,
            fps,
            width,
            height,
            file_size_bytes,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (video_name) DO UPDATE SET
            video_path = EXCLUDED.video_path,
            date = EXCLUDED.date,
            "time" = EXCLUDED."time",
            hour = EXCLUDED.hour,
            lighting = EXCLUDED.lighting,
            split_group = EXCLUDED.split_group,
            duration_seconds = EXCLUDED.duration_seconds,
            fps = EXCLUDED.fps,
            width = EXCLUDED.width,
            height = EXCLUDED.height,
            file_size_bytes = EXCLUDED.file_size_bytes,
            updated_at = EXCLUDED.updated_at
        """,
        [
            video_path.name,
            relative_to_project(video_path),
            index_record.date if index_record else None,
            index_record.time if index_record else None,
            index_record.hour if index_record else None,
            index_record.lighting if index_record else None,
            index_record.split_group if index_record else None,
            probe.duration_seconds,
            probe.fps,
            probe.width,
            probe.height,
            probe.file_size_bytes,
            now,
        ],
    )


CROP_STATS_RESERVED = {"unsorted", "groups"}
CROP_STATS_EXTENSIONS = {".jpg", ".jpeg"}


def print_crop_stats(crops_dir: Path = DEFAULT_CROPS_DIR) -> None:
    if not crops_dir.exists():
        print(f"crop stats: directory not found: {crops_dir}")
        return

    label_dirs = sorted(
        child for child in crops_dir.iterdir()
        if child.is_dir() and child.name not in CROP_STATS_RESERVED
    )

    if not label_dirs:
        print(f"crop stats: no label folders in {crops_dir}")
        return

    counts: list[tuple[str, int]] = []
    for label_dir in label_dirs:
        count = sum(
            1 for child in label_dir.iterdir()
            if child.is_file() and child.suffix.lower() in CROP_STATS_EXTENSIONS
        )
        counts.append((label_dir.name, count))

    name_width = max(len(label) for label, _ in counts)
    count_width = max(len(str(count)) for _, count in counts)
    total = sum(count for _, count in counts)
    count_width = max(count_width, len(str(total)))

    print("crop stats:")
    for label, count in counts:
        print(f"  {label:<{name_width}}  {count:>{count_width}}")
    print(f"  {'-' * name_width}  {'-' * count_width}")
    print(f"  {'total':<{name_width}}  {total:>{count_width}}")


def prune_orphan_frames(connection: duckdb.DuckDBPyConnection) -> list[str]:
    orphan_rows = connection.execute(
        """
        SELECT frame_uid, frame_path
        FROM frames
        WHERE frame_uid NOT IN (SELECT DISTINCT frame_uid FROM frame_intervals)
        """
    ).fetchall()

    if not orphan_rows:
        return []

    orphan_paths = [str(frame_path) for _, frame_path in orphan_rows]

    for frame_path in orphan_paths:
        resolved_path = PROJECT_ROOT / frame_path
        if resolved_path.exists():
            resolved_path.unlink()

    connection.execute(
        """
        DELETE FROM annotations
        WHERE frame_uid IN (
            SELECT frame_uid
            FROM frames
            WHERE frame_uid NOT IN (SELECT DISTINCT frame_uid FROM frame_intervals)
        )
        """
    )
    connection.execute(
        """
        DELETE FROM frames
        WHERE frame_uid NOT IN (SELECT DISTINCT frame_uid FROM frame_intervals)
        """
    )

    return orphan_paths
