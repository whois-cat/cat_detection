#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "numpy>=2.2.0",
#   "ultralytics>=8.3.0",
# ]
# ///

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import BinaryIO, Iterator

import click
import numpy as np
from ultralytics import YOLO

from pipeline_db import (
    DEFAULT_RAW_VIDEOS_DIR,
    connect_db,
    iter_timestamps,
    load_video_index,
    make_uid,
    probe_video,
    select_inference_device,
    upsert_video,
)


COCO_CAT_CLASS_ID = 15
MODEL_NAME = "yolov8n.pt"
SAMPLE_INTERVAL_SECONDS = 5.0
CONFIDENCE_THRESHOLD = 0.25
DEFAULT_BATCH_SIZE = 64
DEFAULT_DETECT_MAX_SIDE = 512


def resized_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise click.ClickException(f"invalid frame dimensions: {width}x{height}")

    scale = min(1.0, max_side / max(width, height))
    resized_width = max(2, int(round(width * scale)))
    resized_height = max(2, int(round(height * scale)))

    if resized_width % 2 == 1:
        resized_width -= 1
    if resized_height % 2 == 1:
        resized_height -= 1

    return max(2, resized_width), max(2, resized_height)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size

    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def iter_sampled_frame_batches(
    video_path: Path,
    duration_seconds: float,
    width: int,
    height: int,
    sample_interval_seconds: float,
    batch_size: int,
    detect_max_side: int,
    limit_duration: bool,
) -> Iterator[tuple[list[float], list[np.ndarray]]]:
    sample_timestamps = iter_timestamps(0.0, duration_seconds, sample_interval_seconds)
    if not sample_timestamps:
        return

    output_width, output_height = resized_dimensions(width, height, detect_max_side)
    filter_graph = f"fps=1/{sample_interval_seconds:g},scale={output_width}:{output_height}"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]
    if limit_duration:
        command += ["-t", f"{duration_seconds:.3f}"]
    command += [
        "-vsync",
        "cfr",
        "-vf",
        filter_graph,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise click.ClickException(f"failed to open ffmpeg pipes for {video_path}")

    frame_size_bytes = output_width * output_height * 3
    batch_timestamps: list[float] = []
    batch_frames: list[np.ndarray] = []
    sample_index = 0

    try:
        while sample_index < len(sample_timestamps):
            raw_frame = read_exact(process.stdout, frame_size_bytes)
            if not raw_frame:
                break
            if len(raw_frame) != frame_size_bytes:
                raise click.ClickException(
                    f"ffmpeg returned a partial frame for {video_path.name} "
                    f"({len(raw_frame)} of {frame_size_bytes} bytes)"
                )

            frame_array = np.frombuffer(raw_frame, dtype=np.uint8).reshape((output_height, output_width, 3))
            batch_timestamps.append(sample_timestamps[sample_index])
            batch_frames.append(frame_array)
            sample_index += 1

            if len(batch_frames) >= batch_size:
                yield batch_timestamps, batch_frames
                batch_timestamps = []
                batch_frames = []

        if batch_frames:
            yield batch_timestamps, batch_frames

        stderr_output = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise click.ClickException(f"ffmpeg failed for {video_path.name}: {stderr_output or return_code}")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def detect_cat_confidences(
    detector: YOLO,
    frames: list[np.ndarray],
    confidence_threshold: float,
    detect_max_side: int,
    device: str,
) -> list[float | None]:
    results = detector.predict(
        source=frames,
        verbose=False,
        conf=confidence_threshold,
        classes=[COCO_CAT_CLASS_ID],
        imgsz=detect_max_side,
        device=device,
    )

    confidences: list[float | None] = []
    for result in results:
        if len(result.boxes) == 0:
            confidences.append(None)
        else:
            confidences.append(float(max(result.boxes.conf.tolist())))
    return confidences


@click.command()
@click.option(
    "--start-at",
    type=str,
    default=None,
    help="Resume scanning from the specified video file name.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the next N videos after --start-at.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="YOLO inference batch size.",
)
@click.option(
    "--detect-max-side",
    type=click.IntRange(min=64),
    default=DEFAULT_DETECT_MAX_SIDE,
    show_default=True,
    help="Downscale frames so the longest side matches this value before inference.",
)
@click.option(
    "--max-seconds-per-video",
    type=float,
    default=None,
    help="Scan only the first N seconds of each video.",
)
@click.option(
    "--rescan",
    is_flag=True,
    help="Re-process videos that already have detections for this model.",
)
def main(
    start_at: str | None,
    limit: int | None,
    batch_size: int,
    detect_max_side: int,
    max_seconds_per_video: float | None,
    rescan: bool,
) -> None:
    """Scan videos and write cat detections into DuckDB."""

    run_scan_cat_detections(
        start_at=start_at,
        limit=limit,
        batch_size=batch_size,
        detect_max_side=detect_max_side,
        max_seconds_per_video=max_seconds_per_video,
        rescan=rescan,
    )


def run_scan_cat_detections(
    start_at: str | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    detect_max_side: int = DEFAULT_DETECT_MAX_SIDE,
    sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS,
    max_seconds_per_video: float | None = None,
    rescan: bool = False,
    video_names: list[str] | None = None,
) -> None:
    input_dir = DEFAULT_RAW_VIDEOS_DIR
    connection = connect_db()
    index_records = load_video_index()
    detector = YOLO(MODEL_NAME)
    inference_device = select_inference_device()
    detector.to(inference_device)

    video_paths = sorted(input_dir.glob("*.mkv"))
    if not video_paths:
        raise click.ClickException(f"no .mkv files found in {input_dir}")

    if video_names is not None:
        name_set = set(video_names)
        video_paths = [p for p in video_paths if p.name in name_set]

    if start_at is not None:
        matching_index = next((i for i, path in enumerate(video_paths) if path.name == start_at), None)
        if matching_index is None:
            raise click.ClickException(f"video {start_at} not found in {input_dir}")
        video_paths = video_paths[matching_index:]

    if limit is not None:
        video_paths = video_paths[:limit]

    click.echo(
        "scan config: "
        f"device={inference_device}, sample_every={sample_interval_seconds:g}s, "
        f"batch_size={batch_size}, detect_max_side={detect_max_side}, "
        f"max_seconds_per_video={max_seconds_per_video}"
    )

    total_detections = 0
    total_samples = 0
    skipped_videos: list[str] = []
    already_scanned_count = 0
    overall_started_at = time.perf_counter()

    for video_index, video_path in enumerate(video_paths, start=1):
        started_at = time.perf_counter()

        if not rescan:
            existing_detections = connection.execute(
                "SELECT 1 FROM detections WHERE video_name = ? AND model_name = ? LIMIT 1",
                [video_path.name, MODEL_NAME],
            ).fetchone()
            if existing_detections is not None:
                already_scanned_count += 1
                click.echo(
                    f"[{video_index}/{len(video_paths)}] {video_path.name}: "
                    f"already scanned, skipping"
                )
                continue

        probe = probe_video(video_path)
        if probe is None:
            skipped_videos.append(video_path.name)
            click.echo(
                f"[{video_index}/{len(video_paths)}] {video_path.name}: "
                f"skipped (no duration)"
            )
            continue
        upsert_video(connection, video_path, index_records.get(video_path.name), probe)

        connection.execute(
            "DELETE FROM detections WHERE video_name = ? AND model_name = ?",
            [video_path.name, MODEL_NAME],
        )

        detections_to_insert: list[list[object]] = []
        video_samples = 0
        scan_duration_seconds = probe.duration_seconds
        if max_seconds_per_video is not None:
            scan_duration_seconds = min(scan_duration_seconds, max_seconds_per_video)

        for batch_timestamps, batch_frames in iter_sampled_frame_batches(
            video_path=video_path,
            duration_seconds=scan_duration_seconds,
            width=probe.width,
            height=probe.height,
            sample_interval_seconds=sample_interval_seconds,
            batch_size=batch_size,
            detect_max_side=detect_max_side,
            limit_duration=max_seconds_per_video is not None,
        ):
            confidences = detect_cat_confidences(
                detector=detector,
                frames=batch_frames,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                detect_max_side=detect_max_side,
                device=inference_device,
            )

            for timestamp_seconds, confidence in zip(batch_timestamps, confidences, strict=True):
                video_samples += 1
                if confidence is None:
                    continue

                detections_to_insert.append(
                    [
                        make_uid("det", video_path.name, MODEL_NAME, round(timestamp_seconds, 3)),
                        video_path.name,
                        round(timestamp_seconds, 3),
                        confidence,
                        MODEL_NAME,
                        "cat",
                        sample_interval_seconds,
                        probe.width,
                        probe.height,
                    ]
                )

        if detections_to_insert:
            connection.executemany(
                """
                INSERT OR REPLACE INTO detections (
                    detection_uid,
                    video_name,
                    timestamp_seconds,
                    confidence,
                    model_name,
                    label,
                    sampler_interval_seconds,
                    frame_width,
                    frame_height
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                detections_to_insert,
            )

        elapsed_seconds = time.perf_counter() - started_at
        total_samples += video_samples
        total_detections += len(detections_to_insert)
        click.echo(
            f"[{video_index}/{len(video_paths)}] {video_path.name}: "
            f"{len(detections_to_insert)} detections from {video_samples} sampled frames "
            f"in {elapsed_seconds:.1f}s"
        )

    total_elapsed_seconds = time.perf_counter() - overall_started_at
    click.echo(f"processed videos: {len(video_paths)}")
    click.echo(f"already scanned: {already_scanned_count}")
    click.echo(f"sampled frames: {total_samples}")
    click.echo(f"total detections: {total_detections}")
    click.echo(f"scan elapsed seconds: {total_elapsed_seconds:.1f}")

    if skipped_videos:
        click.echo(f"skipped videos (no duration): {len(skipped_videos)}")
        for video_name in skipped_videos:
            click.echo(f"  {video_name}")


if __name__ == "__main__":
    main()
