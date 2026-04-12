#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
# ]
# ///

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import click

from pipeline_db import connect_db, make_uid, prune_orphan_frames


@dataclass(frozen=True)
class DetectionPoint:
    video_name: str
    duration_seconds: float
    timestamp_seconds: float
    confidence: float
    model_name: str


@dataclass(frozen=True)
class IntervalRecord:
    interval_uid: str
    video_name: str
    start_seconds: float
    end_seconds: float
    detection_count: int
    max_confidence: float
    interval_source: str
    model_name: str


INTERVAL_SOURCE = "cat_detector_v1"
MIN_CONFIDENCE = 0.25
MERGE_GAP_SECONDS = 15.0
PAD_BEFORE_SECONDS = 3.0
PAD_AFTER_SECONDS = 7.0


def merge_detections_into_intervals(
    detections: list[DetectionPoint],
    interval_source: str,
    merge_gap_seconds: float,
    pad_before_seconds: float,
    pad_after_seconds: float,
) -> list[IntervalRecord]:
    if not detections:
        return []

    grouped: dict[tuple[str, str], list[DetectionPoint]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.video_name, detection.model_name)].append(detection)

    interval_records: list[IntervalRecord] = []

    for (video_name, model_name), video_detections in grouped.items():
        sorted_detections = sorted(video_detections, key=lambda item: item.timestamp_seconds)
        cluster: list[DetectionPoint] = [sorted_detections[0]]
        duration_seconds = sorted_detections[0].duration_seconds

        for detection in sorted_detections[1:]:
            previous = cluster[-1]
            if detection.timestamp_seconds - previous.timestamp_seconds <= merge_gap_seconds:
                cluster.append(detection)
                continue

            interval_records.append(
                build_interval_record(
                    cluster=cluster,
                    duration_seconds=duration_seconds,
                    interval_source=interval_source,
                    pad_before_seconds=pad_before_seconds,
                    pad_after_seconds=pad_after_seconds,
                    model_name=model_name,
                )
            )
            cluster = [detection]

        interval_records.append(
            build_interval_record(
                cluster=cluster,
                duration_seconds=duration_seconds,
                interval_source=interval_source,
                pad_before_seconds=pad_before_seconds,
                pad_after_seconds=pad_after_seconds,
                model_name=model_name,
            )
        )

    return interval_records


def build_interval_record(
    cluster: list[DetectionPoint],
    duration_seconds: float,
    interval_source: str,
    pad_before_seconds: float,
    pad_after_seconds: float,
    model_name: str,
) -> IntervalRecord:
    start_seconds = max(0.0, cluster[0].timestamp_seconds - pad_before_seconds)
    end_seconds = min(duration_seconds, cluster[-1].timestamp_seconds + pad_after_seconds)
    interval_uid = make_uid(
        "int",
        interval_source,
        cluster[0].video_name,
        round(start_seconds, 3),
        round(end_seconds, 3),
        model_name,
    )
    return IntervalRecord(
        interval_uid=interval_uid,
        video_name=cluster[0].video_name,
        start_seconds=round(start_seconds, 3),
        end_seconds=round(end_seconds, 3),
        detection_count=len(cluster),
        max_confidence=max(item.confidence for item in cluster),
        interval_source=interval_source,
        model_name=model_name,
    )


@click.command()
def main() -> None:
    """Build merged cat intervals from per-timestamp detections."""

    run_build_cat_intervals()


def run_build_cat_intervals() -> None:
    connection = connect_db()

    connection.execute(
        """
        DELETE FROM frame_intervals
        WHERE interval_uid IN (
            SELECT interval_uid FROM intervals WHERE interval_source = ?
        )
        """,
        [INTERVAL_SOURCE],
    )
    connection.execute("DELETE FROM intervals WHERE interval_source = ?", [INTERVAL_SOURCE])
    prune_orphan_frames(connection)

    query = """
        SELECT d.video_name, v.duration_seconds, d.timestamp_seconds, d.confidence, d.model_name
        FROM detections d
        JOIN videos v ON v.video_name = d.video_name
        WHERE d.confidence >= ?
    """
    parameters: list[object] = [MIN_CONFIDENCE]

    query += " ORDER BY d.video_name, d.timestamp_seconds"

    detections = [
        DetectionPoint(
            video_name=row[0],
            duration_seconds=float(row[1]),
            timestamp_seconds=float(row[2]),
            confidence=float(row[3]),
            model_name=row[4],
        )
        for row in connection.execute(query, parameters).fetchall()
    ]

    interval_records = merge_detections_into_intervals(
        detections=detections,
        interval_source=INTERVAL_SOURCE,
        merge_gap_seconds=MERGE_GAP_SECONDS,
        pad_before_seconds=PAD_BEFORE_SECONDS,
        pad_after_seconds=PAD_AFTER_SECONDS,
    )

    if interval_records:
        connection.executemany(
            """
            INSERT OR REPLACE INTO intervals (
                interval_uid,
                video_name,
                start_seconds,
                end_seconds,
                detection_count,
                max_confidence,
                interval_source,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    record.interval_uid,
                    record.video_name,
                    record.start_seconds,
                    record.end_seconds,
                    record.detection_count,
                    record.max_confidence,
                    record.interval_source,
                    record.model_name,
                ]
                for record in interval_records
            ],
        )

    click.echo(f"intervals created: {len(interval_records)}")
    click.echo(f"interval source: {INTERVAL_SOURCE}")


if __name__ == "__main__":
    main()
