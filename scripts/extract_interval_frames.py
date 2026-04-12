#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "av>=14.4.0",
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "pillow>=11.1.0",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import click

from pipeline_db import (
    DEFAULT_FRAMES_DIR,
    DEFAULT_RAW_VIDEOS_DIR,
    DEFAULT_VIDEOS_INDEX_PATH,
    PROJECT_ROOT,
    connect_db,
    frame_name_for_timestamp,
    iter_timestamps,
    make_uid,
    open_video_stream,
    prune_orphan_frames,
    relative_to_project,
    probe_video,
    upsert_video,
    load_video_index,
)

INTERVAL_SOURCE = "cat_detector_v1"
FRAME_INTERVAL_SECONDS = 1.0
SEEK_PADDING_SECONDS = 1.0


def seek_to_seconds(container: object, video_stream: object, timestamp_seconds: float) -> None:
    import av

    seek_seconds = max(0.0, timestamp_seconds)
    seek_target = max(0, int(seek_seconds / float(video_stream.time_base)))

    try:
        container.seek(seek_target, stream=video_stream, backward=True)
    except av.error.PermissionError:
        # Fall back to decoding forward from the start when the container rejects seeks.
        container.seek(0, stream=video_stream, backward=True)


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Re-extract frames for intervals that already have frames on disk.",
)
def main(force: bool) -> None:
    """Extract JPEG frames for CVAT from stored intervals."""

    run_extract_interval_frames(force=force)


def run_extract_interval_frames(force: bool = False) -> None:
    output_dir = DEFAULT_FRAMES_DIR
    connection = connect_db()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_records = load_video_index(DEFAULT_VIDEOS_INDEX_PATH)
    skipped_videos: list[str] = []
    for video_path in sorted(DEFAULT_RAW_VIDEOS_DIR.glob("*.mkv")):
        probe = probe_video(video_path)
        if probe is None:
            skipped_videos.append(video_path.name)
            click.echo(f"{video_path.name}: skipped (no duration)")
            continue
        upsert_video(connection, video_path, index_records.get(video_path.name), probe)

    if force:
        connection.execute(
            """
            DELETE FROM frame_intervals
            WHERE interval_uid IN (
                SELECT interval_uid FROM intervals WHERE interval_source = ?
            )
            """,
            [INTERVAL_SOURCE],
        )

    rows = connection.execute(
        """
        SELECT i.interval_uid, i.video_name, i.start_seconds, i.end_seconds, v.video_path
        FROM intervals i
        JOIN videos v ON v.video_name = i.video_name
        WHERE i.interval_source = ?
        ORDER BY i.video_name, i.start_seconds
        """,
        [INTERVAL_SOURCE],
    ).fetchall()

    if not rows:
        raise click.ClickException(f"no intervals found for source {INTERVAL_SOURCE}")

    already_extracted_uids: set[str] = set()
    if not force:
        already_extracted_uids = {
            interval_uid
            for (interval_uid,) in connection.execute(
                """
                SELECT DISTINCT fi.interval_uid
                FROM frame_intervals fi
                JOIN intervals i ON i.interval_uid = fi.interval_uid
                WHERE i.interval_source = ?
                """,
                [INTERVAL_SOURCE],
            ).fetchall()
        }

    total_frames = 0
    already_extracted_count = 0
    saved_frame_uids: dict[tuple[str, float], str] = {}

    for interval_uid, video_name, start_seconds, end_seconds, video_path in rows:
        if interval_uid in already_extracted_uids:
            already_extracted_count += 1
            click.echo(
                f"{video_name} {start_seconds:.1f}-{end_seconds:.1f}s already extracted, skipping"
            )
            continue

        resolved_video_path = Path(video_path)
        if not resolved_video_path.is_absolute():
            resolved_video_path = (PROJECT_ROOT / resolved_video_path).resolve()

        target_timestamps = iter_timestamps(float(start_seconds), float(end_seconds), FRAME_INTERVAL_SECONDS)
        if not target_timestamps:
            continue

        container, video_stream = open_video_stream(resolved_video_path)

        try:
            seek_to_seconds(
                container,
                video_stream,
                max(0.0, float(start_seconds) - SEEK_PADDING_SECONDS),
            )

            target_index = 0
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue

                frame_seconds = float(frame.pts * video_stream.time_base)
                if frame_seconds + 1e-6 < target_timestamps[target_index]:
                    continue

                while target_index < len(target_timestamps) and frame_seconds + 1e-6 >= target_timestamps[target_index]:
                    timestamp_seconds = target_timestamps[target_index]
                    frame_key = (video_name, timestamp_seconds)
                    frame_uid = saved_frame_uids.get(frame_key)

                    if frame_uid is None:
                        frame_uid = make_uid("frm", video_name, round(timestamp_seconds, 3))
                        frame_name = frame_name_for_timestamp(video_name, timestamp_seconds)
                        output_path = output_dir / frame_name

                        frame.to_image().save(output_path, format="JPEG", quality=95)
                        connection.execute(
                            """
                            INSERT INTO frames (
                                frame_uid, video_name, timestamp_seconds,
                                frame_name, frame_path, width, height
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT (frame_uid) DO UPDATE SET
                                video_name = EXCLUDED.video_name,
                                timestamp_seconds = EXCLUDED.timestamp_seconds,
                                frame_name = EXCLUDED.frame_name,
                                frame_path = EXCLUDED.frame_path,
                                width = EXCLUDED.width,
                                height = EXCLUDED.height
                            """,
                            [
                                frame_uid,
                                video_name,
                                round(timestamp_seconds, 3),
                                frame_name,
                                relative_to_project(output_path),
                                int(frame.width),
                                int(frame.height),
                            ],
                        )
                        saved_frame_uids[frame_key] = frame_uid
                        total_frames += 1

                    connection.execute(
                        "INSERT OR IGNORE INTO frame_intervals (frame_uid, interval_uid) VALUES (?, ?)",
                        [frame_uid, interval_uid],
                    )
                    target_index += 1

                    if target_index == len(target_timestamps):
                        break

                if target_index == len(target_timestamps):
                    break
        finally:
            container.close()

        click.echo(f"{video_name} {start_seconds:.1f}-{end_seconds:.1f}s extracted")

    pruned_paths = prune_orphan_frames(connection)

    click.echo(f"frames extracted: {total_frames}")
    click.echo(f"already extracted intervals: {already_extracted_count}")
    click.echo(f"orphan frames pruned: {len(pruned_paths)}")
    click.echo(f"output directory: {output_dir}")

    if skipped_videos:
        click.echo(f"skipped videos (no duration): {len(skipped_videos)}")
        for video_name in skipped_videos:
            click.echo(f"  {video_name}")


if __name__ == "__main__":
    main()
