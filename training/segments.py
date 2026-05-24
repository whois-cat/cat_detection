"""Map wall-clock milliseconds → recording segment file + seek offset.

WHY WALL-CLOCK AND NOT PTS — within one stream, PTS is the canonical frame
identity. But the detector and mediamtx's recorder are two *independent*
consumers of the same RTSP source: each segment fMP4 has its own PTS
origin (starting near 0), unrelated to the detector's camera-stream PTS
epoch. mediamtx doesn't publish a (camera_pts ↔ segment_pts) anchor, so
PTS can't be used to cross-reference. The thing both sides DO share is
wall-clock: the detector stamps `event.wall_ms`, mediamtx encodes each
segment's start time in its filename. So:

    event_wall_ms − segment_start_ms = segment-local time of the detected
                                       frame (in milliseconds)

and PyAV can seek inside the segment by that media offset. Precision is
"under one frame" in practice — the two wall-clock readings drift by a
few tens of ms at most, and PyAV's keyframe-aligned seek + forward decode
lands within a single GOP of the target. Fine for training data.

(For sub-frame precision in some future UI feature, the path would be:
peek the first PTS of each segment at open time, build an anchor
`(segment.start_ms ↔ first_pts)`, and convert detector `event.media_t` to
segment-local PTS — but that needs mediamtx to be confirmed PTS-preserving
through the record path.)

mediamtx writes segments named  YYYY-MM-DD_HH-MM-SS-ffffff.mp4  where the
timestamp is the segment START in *local* time (the container TZ). With
default config segments are 30 seconds long.

Caveats:
- mediamtx writes segments in local time. The detector and the trainer
  must agree on TZ. If you mount the segments on a host with a different
  TZ, set $TZ to match the recording container's TZ before running.
- The very last segment of a recording session can be shorter than the
  configured duration; we accept any media_t up to the next segment's start.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class Segment:
    path: Path
    start_ms: int     # local wall-clock at segment start


def _parse_filename(name: str) -> int | None:
    stem = name.rsplit(".", 1)[0]
    try:
        dt = datetime.strptime(stem, "%Y-%m-%d_%H-%M-%S-%f")
    except ValueError:
        return None
    # `.timestamp()` interprets a naive datetime as local time.
    return int(dt.timestamp() * 1000)


class SegmentIndex:
    """In-memory bisect-friendly index of segments for one camera."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._starts = [s.start_ms for s in segments]

    @classmethod
    def from_dir(cls, root: Path) -> "SegmentIndex":
        """Build from a per-camera dir like data/recordings/<camera>/."""
        segments: list[Segment] = []
        for p in sorted(root.rglob("*.mp4")):
            start_ms = _parse_filename(p.name)
            if start_ms is None:
                continue
            segments.append(Segment(path=p, start_ms=start_ms))
        segments.sort(key=lambda s: s.start_ms)
        return cls(segments)

    def locate(self, wall_ms: int) -> tuple[Segment, float] | None:
        """Return (segment, media_t_seconds) for the segment covering wall_ms.

        Returns None if the timestamp falls in a recording gap (segment was
        pruned, or recording was off).
        """
        if not self.segments:
            return None
        idx = bisect_right(self._starts, wall_ms) - 1
        if idx < 0:
            return None
        seg = self.segments[idx]
        # If the next segment starts AFTER wall_ms, we're between segments
        # which means inside `seg` (mediamtx writes contiguous segments). If
        # the gap to the next segment is larger than a reasonable segment
        # duration, treat as a gap.
        offset_ms = wall_ms - seg.start_ms
        next_start_ms = self._starts[idx + 1] if idx + 1 < len(self._starts) else None
        if next_start_ms is not None and wall_ms >= next_start_ms:
            return None
        # Generous max — covers default 30 s segments plus any final short one.
        if offset_ms > 60_000:
            return None
        return seg, offset_ms / 1000.0
