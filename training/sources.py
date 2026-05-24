"""Sample sources for training-data extraction.

A *Sample* is a (BGR ndarray, list[Box]) pair plus minimal metadata. Two
concrete sources cover the two common training shapes:

  - FullFrameSource — yields one Sample per FRAME, with every detected box
    in that frame. Right for fine-tuning an object detector (YOLO etc.),
    where the model wants the whole image and learns to localise.

  - CropSource — yields one Sample per BOX, image cropped to the box (with
    optional padding). Right for training a classifier (per-cat identity),
    where the model has already been handed a crop and just decides "which
    cat is this".

Both share the same `SampleSource` ABC and a single underlying scan over
events grouped by recording segment, so seeks are minimised: each segment
is opened once, frames are decoded in PTS-monotonic order, and the
container is closed before moving on.

Dependencies: PyAV (`av`) for decode. cv2 (optional) only for crop padding
math — we use numpy here to keep the dep light.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .db import Box, FrameRecord, iter_frames, open_db_ro
from .segments import Segment, SegmentIndex


log = logging.getLogger(__name__)


@dataclass(slots=True)
class Sample:
    """One training sample.

    image: BGR uint8 ndarray. For FullFrameSource this is the whole frame
      (camera orientation). For CropSource it's the cropped region.
    boxes: detection boxes. For FullFrameSource all boxes in the frame, in
      camera coords. For CropSource a single box, coords expressed relative
      to the crop (so (0,0,w,h) covers the crop).
    wall_ms / camera_id / model: pass-through metadata for filenames, splits,
      provenance.
    """
    image: np.ndarray
    boxes: list[Box]
    wall_ms: int
    camera_id: str
    model: str


# ---------------------------------------------------------------- internals --

def _group_by_segment(frames: Iterator[FrameRecord],
                      index: SegmentIndex) -> Iterator[tuple[Segment, list[tuple[float, FrameRecord]]]]:
    """Bucket frames by the segment that covers their wall_ms, sorted by
    media offset within each segment. Frames in gaps are dropped (logged)."""
    current_seg: Segment | None = None
    bucket: list[tuple[float, FrameRecord]] = []
    for fr in frames:
        hit = index.locate(fr.wall_ms)
        if hit is None:
            log.debug("frame at wall_ms=%d falls in a recording gap; skipping", fr.wall_ms)
            continue
        seg, media_t = hit
        if current_seg is None or seg.path != current_seg.path:
            if current_seg is not None and bucket:
                bucket.sort(key=lambda x: x[0])
                yield current_seg, bucket
            current_seg = seg
            bucket = []
        bucket.append((media_t, fr))
    if current_seg is not None and bucket:
        bucket.sort(key=lambda x: x[0])
        yield current_seg, bucket


def _decode_frames_at(seg_path: Path, offsets_sec: list[float]) -> Iterator[tuple[float, np.ndarray]]:
    """Open a segment once and yield (requested_offset, BGR ndarray) in order.

    Seeks per requested offset. PyAV seek is keyframe-aligned, so the first
    decoded frame ≥ the requested PTS is what we return (small drift, OK for
    training data — typically <1 frame).
    """
    container = av.open(str(seg_path))
    try:
        stream = container.streams.video[0]
        tb = stream.time_base
        assert tb is not None, f"{seg_path}: stream has no time_base"
        for off in offsets_sec:
            target_pts = int(off / float(tb))
            try:
                container.seek(target_pts, stream=stream, any_frame=False, backward=True)
            except Exception as e:
                log.warning("seek failed in %s at offset %.3fs: %r", seg_path, off, e)
                continue
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts * float(tb) < off:
                    continue
                yield off, frame.to_ndarray(format="bgr24")
                break
    finally:
        container.close()


# ----------------------------------------------------------------- public API --

class SampleSource(ABC):
    """Abstract iterable of training samples.

    Subclass and implement `_emit()` to translate one decoded
    (FrameRecord, image) pair into N Samples. The base class handles
    SQLite query, segment grouping, and decode batching.
    """

    def __init__(self, db_path: Path, recordings_root: Path, *,
                 camera_id: str | None = None,
                 model: str | None = None,
                 cat: str | None = None,
                 t_from: int | None = None,
                 t_to: int | None = None,
                 min_score: float | None = None):
        self.db_path = Path(db_path)
        self.recordings_root = Path(recordings_root)
        self.camera_id = camera_id
        self.model = model
        self.cat = cat
        self.t_from = t_from
        self.t_to = t_to
        self.min_score = min_score
        self._segment_indices: dict[str, SegmentIndex] = {}

    def _index_for(self, camera_id: str) -> SegmentIndex:
        if camera_id not in self._segment_indices:
            cam_dir = self.recordings_root / camera_id
            self._segment_indices[camera_id] = SegmentIndex.from_dir(cam_dir)
        return self._segment_indices[camera_id]

    def __iter__(self) -> Iterator[Sample]:
        conn = open_db_ro(self.db_path)
        try:
            frames = iter_frames(
                conn,
                camera_id=self.camera_id, model=self.model, cat=self.cat,
                t_from=self.t_from, t_to=self.t_to, min_score=self.min_score,
            )
            # Frames come out sorted by (camera_id, wall_ms). We bucket
            # per-camera so each camera gets its own SegmentIndex (their
            # recording dirs are separate).
            by_camera: dict[str, list[FrameRecord]] = {}
            for fr in frames:
                by_camera.setdefault(fr.camera_id, []).append(fr)
            for camera_id, fr_list in by_camera.items():
                idx = self._index_for(camera_id)
                for seg, items in _group_by_segment(iter(fr_list), idx):
                    offsets = [t for (t, _) in items]
                    fr_by_off = {t: fr for (t, fr) in items}
                    for off, img in _decode_frames_at(seg.path, offsets):
                        fr = fr_by_off[off]
                        yield from self._emit(fr, img)
        finally:
            conn.close()

    @abstractmethod
    def _emit(self, frame: FrameRecord, img: np.ndarray) -> Iterator[Sample]:
        ...


class FullFrameSource(SampleSource):
    """One Sample per frame; all boxes for that frame attached."""

    def _emit(self, frame: FrameRecord, img: np.ndarray) -> Iterator[Sample]:
        yield Sample(
            image=img, boxes=list(frame.boxes),
            wall_ms=frame.wall_ms,
            camera_id=frame.camera_id, model=frame.model,
        )


class CropSource(SampleSource):
    """One Sample per BOX, image = the cropped region.

    pad_frac: extend the crop on each side by `pad_frac * max(box_w, box_h)`.
    Defaults to 0.15 — a little context helps the classifier and tolerates
    box jitter. Clamped to frame edges.
    """

    def __init__(self, *args, pad_frac: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_frac = pad_frac

    def _emit(self, frame: FrameRecord, img: np.ndarray) -> Iterator[Sample]:
        H, W = img.shape[:2]
        for box in frame.boxes:
            pad = int(self.pad_frac * max(box.w, box.h))
            x0 = max(0, box.x - pad)
            y0 = max(0, box.y - pad)
            x1 = min(W, box.x + box.w + pad)
            y1 = min(H, box.y + box.h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            # Box coords relative to the crop, clipped to its bounds.
            local = Box(
                x=box.x - x0, y=box.y - y0, w=box.w, h=box.h,
                cat=box.cat, score=box.score, track_id=box.track_id,
            )
            yield Sample(
                image=crop, boxes=[local],
                wall_ms=frame.wall_ms,
                camera_id=frame.camera_id, model=frame.model,
            )
