from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VIDEO_FILENAME_PATTERN = re.compile(
    r"^video_(?P<date>\d{8})_(?P<time>\d{6})\.mkv$"
)


@dataclass(frozen=True)
class VideoMetadata:
    video_name: str
    date: str
    time: str
    hour: int
    lighting: str
    split_group: str


def detect_lighting_by_hour(video_hour: int) -> str:
    if 7 <= video_hour <= 18:
        return "day"

    if video_hour >= 22 or video_hour <= 5:
        return "night"

    return "unknown"


def parse_video_filename(video_file_path: Path) -> VideoMetadata | None:
    filename_match = VIDEO_FILENAME_PATTERN.match(video_file_path.name)

    if filename_match is None:
        return None

    raw_date = filename_match.group("date")
    raw_time = filename_match.group("time")
    parsed_datetime = datetime.strptime(f"{raw_date}{raw_time}", "%Y%m%d%H%M%S")

    formatted_date = parsed_datetime.strftime("%Y-%m-%d")
    formatted_time = parsed_datetime.strftime("%H:%M:%S")
    parsed_hour = parsed_datetime.hour
    detected_lighting = detect_lighting_by_hour(parsed_hour)

    return VideoMetadata(
        video_name=video_file_path.name,
        date=formatted_date,
        time=formatted_time,
        hour=parsed_hour,
        lighting=detected_lighting,
        split_group=formatted_date,
    )


def collect_video_metadata(raw_videos_directory: Path) -> tuple[list[VideoMetadata], list[str]]:
    collected_metadata: list[VideoMetadata] = []
    skipped_video_names: list[str] = []

    for video_file_path in sorted(raw_videos_directory.glob("*.mkv")):
        parsed_video_metadata = parse_video_filename(video_file_path)

        if parsed_video_metadata is None:
            skipped_video_names.append(video_file_path.name)
            continue

        collected_metadata.append(parsed_video_metadata)

    return collected_metadata, skipped_video_names


def save_videos_index_csv(
    output_csv_path: Path,
    video_metadata_rows: list[VideoMetadata],
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as output_file:
        csv_writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "video_name",
                "date",
                "time",
                "hour",
                "lighting",
                "split_group",
            ],
        )

        csv_writer.writeheader()

        for video_metadata_row in video_metadata_rows:
            csv_writer.writerow(
                {
                    "video_name": video_metadata_row.video_name,
                    "date": video_metadata_row.date,
                    "time": video_metadata_row.time,
                    "hour": video_metadata_row.hour,
                    "lighting": video_metadata_row.lighting,
                    "split_group": video_metadata_row.split_group,
                }
            )


def main() -> None:
    project_root_directory = Path(__file__).resolve().parents[1]
    raw_videos_directory = project_root_directory / "data" / "raw_videos"
    output_csv_path = project_root_directory / "data" / "metadata" / "videos_index.csv"

    if not raw_videos_directory.exists():
        raise FileNotFoundError(f"directory not found: {raw_videos_directory}")

    collected_metadata, skipped_video_names = collect_video_metadata(raw_videos_directory)

    if not collected_metadata:
        raise ValueError(f"no valid .mkv files found in {raw_videos_directory}")

    save_videos_index_csv(output_csv_path, collected_metadata)

    print(f"saved csv: {output_csv_path}")
    print(f"total videos: {len(collected_metadata)}")

    if skipped_video_names:
        print(f"skipped files: {len(skipped_video_names)}")
        for skipped_video_name in skipped_video_names:
            print(f"skipped: {skipped_video_name}")


if __name__ == "__main__":
    main()