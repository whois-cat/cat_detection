#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "click>=8.1.8",
#   "duckdb>=1.2.2",
#   "imagehash>=4.3.0",
#   "pillow>=11.1.0",
# ]
# ///

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import click

from pipeline_db import DEFAULT_CROPS_DIR, PROJECT_ROOT, connect_db


DEFAULT_HASH_THRESHOLD = 6
PHASH_SIZE = 8
UNSORTED_LABEL = "unsorted"


@dataclass(frozen=True)
class FrameRow:
    frame_uid: str
    frame_path: str
    timestamp_seconds: float


@dataclass(frozen=True)
class CropFileInfo:
    path: Path
    video_stem: str
    timestamp_ms: int


def compute_phash(image_path: Path) -> object:
    import imagehash
    from PIL import Image

    with Image.open(image_path) as image:
        return imagehash.phash(image, hash_size=PHASH_SIZE)


def parse_crop_filename(path: Path) -> CropFileInfo | None:
    stem = path.stem
    match = re.match(r"^(.+?)__ts_(\d+)__box_\d+$", stem)
    if match is None:
        return None
    return CropFileInfo(
        path=path,
        video_stem=match.group(1),
        timestamp_ms=int(match.group(2)),
    )


# ── Frames mode ──────────────────────────────────────────────


def find_duplicate_frames_in_interval(
    frames: list[FrameRow],
    threshold: int,
    phash_cache: dict[str, object],
) -> list[tuple[FrameRow, int]]:
    if len(frames) < 2:
        return []

    duplicates: list[tuple[FrameRow, int]] = []
    reference_hash: object | None = None

    for frame in frames:
        frame_hash = phash_cache.get(frame.frame_uid)
        if frame_hash is None:
            resolved_path = PROJECT_ROOT / frame.frame_path
            if not resolved_path.exists():
                continue
            frame_hash = compute_phash(resolved_path)
            phash_cache[frame.frame_uid] = frame_hash

        if reference_hash is None:
            reference_hash = frame_hash
            continue

        distance = int(frame_hash - reference_hash)
        if distance <= threshold:
            duplicates.append((frame, distance))
        else:
            reference_hash = frame_hash

    return duplicates


def run_deduplicate_frames(
    threshold: int = DEFAULT_HASH_THRESHOLD,
    dry_run: bool = False,
) -> None:
    connection = connect_db()

    if not dry_run:
        connection.execute("UPDATE frames SET is_duplicate = false")

    rows = connection.execute(
        """
        SELECT fi.interval_uid,
               f.frame_uid,
               f.frame_path,
               f.timestamp_seconds
        FROM frame_intervals fi
        JOIN frames f ON f.frame_uid = fi.frame_uid
        ORDER BY fi.interval_uid, f.timestamp_seconds
        """
    ).fetchall()

    interval_frames: dict[str, list[FrameRow]] = {}
    for interval_uid, frame_uid, frame_path, timestamp_seconds in rows:
        interval_frames.setdefault(interval_uid, []).append(
            FrameRow(
                frame_uid=frame_uid,
                frame_path=frame_path,
                timestamp_seconds=float(timestamp_seconds),
            )
        )

    if not interval_frames:
        click.echo("dedup: no frames to process")
        return

    phash_cache: dict[str, object] = {}
    duplicate_uids: set[str] = set()

    for interval_uid, frames in interval_frames.items():
        duplicates = find_duplicate_frames_in_interval(
            frames=frames,
            threshold=threshold,
            phash_cache=phash_cache,
        )
        for frame, distance in duplicates:
            duplicate_uids.add(frame.frame_uid)

    total_frames = sum(len(frames) for frames in interval_frames.values())
    click.echo(f"dedup: intervals={len(interval_frames)} frames={total_frames} threshold={threshold}")
    click.echo(f"dedup: duplicate frames detected: {len(duplicate_uids)}")

    if dry_run:
        click.echo("dedup: dry run, database not modified")
        return

    if duplicate_uids:
        connection.executemany(
            "UPDATE frames SET is_duplicate = true WHERE frame_uid = ?",
            [[frame_uid] for frame_uid in duplicate_uids],
        )

    click.echo(f"dedup: frames marked is_duplicate=true: {len(duplicate_uids)}")


# ── Crops mode ───────────────────────────────────────────────


def run_deduplicate_crops(
    threshold: int = DEFAULT_HASH_THRESHOLD,
    dry_run: bool = False,
) -> None:
    connection = connect_db()
    unsorted_dir = DEFAULT_CROPS_DIR / UNSORTED_LABEL

    if not unsorted_dir.exists():
        click.echo(f"dedup-crops: directory not found: {unsorted_dir}")
        return

    all_files = sorted(
        path for path in unsorted_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not all_files:
        click.echo("dedup-crops: no crop files found")
        return

    parsed: list[CropFileInfo] = []
    skipped_parse = 0
    for path in all_files:
        info = parse_crop_filename(path)
        if info is None:
            skipped_parse += 1
            continue
        parsed.append(info)

    by_video: dict[str, list[CropFileInfo]] = {}
    for info in parsed:
        by_video.setdefault(info.video_stem, []).append(info)
    for crops in by_video.values():
        crops.sort(key=lambda c: c.timestamp_ms)

    crop_uid_by_basename: dict[str, str] = {}
    existing = connection.execute("SELECT crop_uid, crop_path FROM crops").fetchall()
    for crop_uid, crop_path in existing:
        crop_uid_by_basename[Path(crop_path).name] = crop_uid

    total_crops = sum(len(crops) for crops in by_video.values())
    duplicate_paths: list[Path] = []
    duplicate_crop_uids: list[str] = []

    for video_stem, crops in by_video.items():
        if len(crops) < 2:
            continue

        reference_hash: object | None = None
        for crop_info in crops:
            crop_hash = compute_phash(crop_info.path)

            if reference_hash is None:
                reference_hash = crop_hash
                continue

            distance = int(crop_hash - reference_hash)
            if distance <= threshold:
                duplicate_paths.append(crop_info.path)
                uid = crop_uid_by_basename.get(crop_info.path.name)
                if uid is not None:
                    duplicate_crop_uids.append(uid)
            else:
                reference_hash = crop_hash

    click.echo(
        f"dedup-crops: videos={len(by_video)} crops={total_crops} "
        f"threshold={threshold}"
    )
    if skipped_parse:
        click.echo(f"dedup-crops: skipped unparseable filenames: {skipped_parse}")
    click.echo(f"dedup-crops: duplicate crops detected: {len(duplicate_paths)}")

    if dry_run:
        click.echo("dedup-crops: dry run, no files deleted")
        return

    for path in duplicate_paths:
        path.unlink(missing_ok=True)

    if duplicate_crop_uids:
        connection.executemany(
            "UPDATE crops SET is_duplicate = true WHERE crop_uid = ?",
            [[uid] for uid in duplicate_crop_uids],
        )

    click.echo(f"dedup-crops: deleted {len(duplicate_paths)} files")
    click.echo(f"dedup-crops: marked {len(duplicate_crop_uids)} rows is_duplicate=true")


# ── CLI ──────────────────────────────────────────────────────


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["frames", "crops"]),
    default="frames",
    show_default=True,
    help="What to deduplicate: extracted frames or unsorted crops.",
)
@click.option(
    "--threshold",
    type=click.IntRange(min=0),
    default=DEFAULT_HASH_THRESHOLD,
    show_default=True,
    help="Maximum Hamming distance between neighbouring pHashes to treat as duplicates.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only report counts without modifying files or database.",
)
def main(mode: str, threshold: int, dry_run: bool) -> None:
    """Deduplicate frames or crops using perceptual hashing."""

    if mode == "frames":
        run_deduplicate_frames(threshold=threshold, dry_run=dry_run)
    else:
        run_deduplicate_crops(threshold=threshold, dry_run=dry_run)


if __name__ == "__main__":
    main()
