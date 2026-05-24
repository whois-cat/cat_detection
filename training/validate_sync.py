"""Validate the event-frame join across three different extraction paths.

For N randomly-sampled events from events.db we decode the corresponding
frame three ways:

  A. Wall-clock seek inside the raw segment FILE (what segments.py +
     sources.py do today):
        media_offset = (event.wall_ms - segment.start_ms) / 1000
        PyAV.seek(media_offset) → first frame at-or-after.
  B. PTS-primary lookup inside the same segment FILE:
        scan every frame's PTS, find the closest by media_t to
        event.media_t. Tests whether mediamtx preserved camera-stream
        PTS through the recording.
  C. mediamtx PLAYBACK API (what the browser sees):
        GET /get?path=<...>&start=<ISO-8601 of event.wall_ms>&duration=2
        Decode the first frame and read its PTS. If it equals event.pts,
        the playback API is PTS-synchronised with the detector's labels.

Per-event we print PTS values and a bit-for-bit ndarray comparison so we
can see which paths land on the same source frame. Summary at the end
tells us whether PTS-primary or wall-clock-primary is the right policy.

Usage (from live2/ root):

    uv run python -m training.validate_sync \
        --db data/events/events.db \
        --recordings data/recordings \
        --recording-subdir camera \
        --playback-url http://localhost:9996 \
        --playback-path camera \
        --n 5 --seed 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path
from urllib.parse import urlencode

import av
import numpy as np

from .db import iter_frames, open_db_ro, FrameRecord
from .segments import SegmentIndex


def _frame_at_media_t(seg_path: Path, target_sec: float):
    """Method A — PyAV seek inside a segment file by media time."""
    container = av.open(str(seg_path))
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)  # type: ignore[arg-type]
        container.seek(int(target_sec / tb), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if frame.pts * tb < target_sec:
                continue
            return frame.pts, frame.pts * tb, frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    return None


def _frame_closest_pts_in_segment(seg_path: Path, target_media_t: float):
    """Method B — scan all frames, return the one closest to target_media_t."""
    container = av.open(str(seg_path))
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)  # type: ignore[arg-type]
        best = None
        best_gap = float("inf")
        first_pts = None
        last_pts = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if first_pts is None:
                first_pts = frame.pts
            last_pts = frame.pts
            gap = abs(frame.pts * tb - target_media_t)
            if gap < best_gap:
                best_gap = gap
                best = (frame.pts, frame.pts * tb, frame.to_ndarray(format="bgr24"))
        return best, best_gap, first_pts, last_pts
    finally:
        container.close()


def _frame_via_playback(playback_url: str, path: str, wall_ms: int):
    """Method C — fetch fMP4 from mediamtx /get, decode first frame.

    mediamtx accepts ISO 8601 in `start`. We use UTC; mediamtx documents
    that any RFC3339 timestamp works.
    """
    iso = dt.datetime.fromtimestamp(wall_ms / 1000, dt.timezone.utc) \
            .isoformat(timespec="milliseconds").replace("+00:00", "Z")
    qs = urlencode({"path": path, "start": iso, "duration": "2"})
    url = f"{playback_url.rstrip('/')}/get?{qs}"
    try:
        container = av.open(url)
    except Exception as e:
        return None, url, repr(e)
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)  # type: ignore[arg-type]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            return (frame.pts, frame.pts * tb, frame.to_ndarray(format="bgr24")), url, None
    finally:
        container.close()
    return None, url, "no frames decoded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--recording-subdir", default=None,
                    help="override the per-camera subdir under --recordings "
                         "(defaults to event's camera_id). Use this when the "
                         "mediamtx path name and the detector's CAMERA_ID "
                         "are not the same string.")
    ap.add_argument("--playback-url", default=None,
                    help="mediamtx playback base URL, e.g. http://localhost:9996. "
                         "Omit to skip Method C.")
    ap.add_argument("--playback-path", default=None,
                    help="mediamtx path name to play (defaults to "
                         "--recording-subdir, then event.camera_id).")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    conn = open_db_ro(args.db)
    try:
        all_frames: list[FrameRecord] = list(iter_frames(
            conn, camera_id=args.camera, model=args.model,
        ))
    finally:
        conn.close()

    candidates = [f for f in all_frames if f.pts is not None and f.media_t is not None]
    if not candidates:
        print("no events with pts/media_t in DB — nothing to validate")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.n, len(candidates)))

    indices: dict[str, SegmentIndex] = {}
    def index_for(subdir: str) -> SegmentIndex:
        if subdir not in indices:
            indices[subdir] = SegmentIndex.from_dir(args.recordings / subdir)
        return indices[subdir]

    n_done = 0
    ab_same = 0
    ac_same = 0
    bc_same = 0
    c_pts_matches_event = 0

    print(f"\nvalidating {len(sample)} sampled events:\n")
    for ev in sample:
        subdir = args.recording_subdir or ev.camera_id
        playback_path = args.playback_path or args.recording_subdir or ev.camera_id
        hit = index_for(subdir).locate(ev.wall_ms)
        if hit is None:
            print(f"  [{ev.wall_ms}] camera={ev.camera_id} model={ev.model} — "
                  f"no segment in {args.recordings/subdir} covers this wall_ms; "
                  "skipping")
            continue
        seg, target_sec = hit

        a = _frame_at_media_t(seg.path, target_sec)
        b, b_gap, b_first_pts, b_last_pts = _frame_closest_pts_in_segment(
            seg.path, ev.media_t or 0.0,
        )
        c = c_url = c_err = None
        if args.playback_url:
            c, c_url, c_err = _frame_via_playback(
                args.playback_url, playback_path, ev.wall_ms,
            )

        n_done += 1
        print(f"  event wall_ms={ev.wall_ms}  segment={seg.path.name}")
        cats = ", ".join(b.cat or "?" for b in ev.boxes) or "(none)"
        print(f"    event.pts={ev.pts}  event.media_t={ev.media_t:.6f}  cats=[{cats}]")
        print(f"    segment PTS range in file: [{b_first_pts} .. {b_last_pts}]")

        def _fmt(label, x):
            if x is None:
                print(f"    {label}: NO FRAME")
                return
            pts, mt, img = x
            print(f"    {label}: pts={pts}  media_t={mt:.6f}  shape={img.shape}")

        _fmt("A (file wall-clock seek)", a)
        _fmt("B (file PTS scan)       ", b)
        if args.playback_url:
            if c_err:
                print(f"    C (playback URL)        : FAILED → {c_err}")
                print(f"      url: {c_url}")
            else:
                _fmt("C (playback URL)        ", c)

        # Pixel comparisons
        if a and b:
            same = a[2].shape == b[2].shape and np.array_equal(a[2], b[2])
            print(f"    pixel-identical A==B: {same}")
            if same: ab_same += 1
        if a and c:
            same = a[2].shape == c[2].shape and np.array_equal(a[2], c[2])
            print(f"    pixel-identical A==C: {same}")
            if same: ac_same += 1
        if b and c:
            same = b[2].shape == c[2].shape and np.array_equal(b[2], c[2])
            print(f"    pixel-identical B==C: {same}")
            if same: bc_same += 1
        if c and ev.pts is not None:
            c_pts = c[0]
            match = (c_pts == ev.pts)
            print(f"    C.pts == event.pts: {match}  (Δ = {c_pts - ev.pts})")
            if match: c_pts_matches_event += 1
        print()

    print("---- summary ----")
    print(f"  events validated:                  {n_done} / {len(sample)}")
    print(f"  pixel-identical A==B (file methods): {ab_same} / {n_done}")
    if args.playback_url:
        print(f"  pixel-identical A==C (file vs playback): {ac_same} / {n_done}")
        print(f"  pixel-identical B==C:                  {bc_same} / {n_done}")
        print(f"  Method-C PTS matches event.pts:        {c_pts_matches_event} / {n_done}")
        if c_pts_matches_event == n_done and n_done:
            print("\n  ⇒ mediamtx playback API preserves the detector-seen PTS. "
                  "PTS can be the primary join key for the BROWSER path. The "
                  "extractor's wall-clock join is still the only option for "
                  "direct-file access (because segment-file PTS uses a different "
                  "epoch — see B vs event.pts above).")
        elif c_pts_matches_event == 0:
            print("\n  ⇒ mediamtx playback rebases PTS too. Wall-clock is the "
                  "only cross-system join everywhere; current design stands.")


if __name__ == "__main__":
    main()
