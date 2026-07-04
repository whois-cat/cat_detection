"""Offline YOLO rescan of recorded videos into events.db.

This is the cold-start escape hatch for a bad live detector pool: instead of
trusting already-recorded detector events, sample full frames from the mediamtx
recordings, run COCO cat detection, and write fresh event rows under a separate
model/source. The review manifest can then use `--model offline-yolo26n`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import yaml

from detector.storage import init_db, insert_event
from training.db import Box
from training.regions import (
    box_in_ignore_region,
    load_ignore_regions_from_camera_config,
)
from training.segments import Segment, SegmentIndex


ROOT = Path(__file__).resolve().parents[1]
COCO_CAT_CLASS_ID = 15


def _parse_rect(value) -> tuple[float, float, float, float]:
    if value is None:
        return (0.0, 0.0, 1.0, 1.0)
    if isinstance(value, str):
        vals = [float(v.strip()) for v in value.split(",") if v.strip()]
    else:
        vals = [float(v) for v in value]
    if len(vals) != 4:
        raise ValueError(f"rect must be x0,y0,x1,y1: {value!r}")
    return (vals[0], vals[1], vals[2], vals[3])


def _load_camera_config(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for cam in cfg.get("cameras", []) or []:
        cid = cam.get("id")
        if cid:
            out[cid] = cam
    return out


def _rot90_k(rotate_deg: int) -> int:
    return (-rotate_deg // 90) % 4


def _unrotate_box_to_camera(
    rx: int,
    ry: int,
    rw: int,
    rh: int,
    rot_w: int,
    rot_h: int,
    rotate_deg: int,
) -> tuple[int, int, int, int]:
    if rotate_deg == 0:
        return (rx, ry, rw, rh)
    if rotate_deg == 90:
        return (ry, rot_w - rx - rw, rh, rw)
    if rotate_deg == 180:
        return (rot_w - rx - rw, rot_h - ry - rh, rw, rh)
    if rotate_deg == 270:
        return (rot_h - ry - rh, rx, rh, rw)
    raise ValueError("rotate_deg must be 0/90/180/270")


def _iter_sampled_frames(seg: Segment, interval_sec: float) -> Iterable[tuple[object, np.ndarray, float, object]]:
    container = av.open(str(seg.path))
    try:
        stream = container.streams.video[0]
        tb = stream.time_base
        if tb is None:
            return
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            media_t = frame.pts * float(tb)
            if media_t + 1e-6 < next_t:
                continue
            yield frame, frame.to_ndarray(format="bgr24"), media_t, tb
            while next_t <= media_t + 1e-6:
                next_t += interval_sec
    finally:
        container.close()


def _detect_boxes(model, img_bgr: np.ndarray, *, conf: float, imgsz: int) -> list[dict]:
    results = model.predict(
        source=img_bgr,
        verbose=False,
        conf=conf,
        classes=[COCO_CAT_CLASS_ID],
        imgsz=imgsz,
    )
    out = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            out.append({
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "score": float(box.conf[0]),
            })
    return out


def _selected_segments(
    segments: list[Segment],
    *,
    t_from: int | None,
    t_to: int | None,
    limit_segments: int | None,
) -> list[Segment]:
    out = []
    for seg in segments:
        if t_from is not None and seg.start_ms + 60_000 < t_from:
            continue
        if t_to is not None and seg.start_ms > t_to:
            continue
        out.append(seg)
        if limit_segments is not None and len(out) >= limit_segments:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "cameras.yaml")
    ap.add_argument("--camera", action="append", default=[],
                    help="camera id to scan; repeatable; default scans all recording dirs")
    ap.add_argument("--weights", default="/opt/models/yolo26n.pt",
                    help="YOLO weights; default matches the non-quantized model from main")
    ap.add_argument("--model-name", default="offline-yolo26n")
    ap.add_argument("--source", default="offline_rescan")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--sample-interval-sec", type=float, default=1.0)
    ap.add_argument("--min-box-size", type=int, default=50)
    ap.add_argument("--ignore-region-min-coverage", type=float, default=None)
    ap.add_argument("--t-from", type=int, default=None)
    ap.add_argument("--t-to", type=int, default=None)
    ap.add_argument("--limit-segments", type=int, default=None)
    ap.add_argument("--replace", action=argparse.BooleanOptionalAction, default=True,
                    help="delete previous rows for this model/source before inserting")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = _load_camera_config(args.config)
    ignore_regions = load_ignore_regions_from_camera_config(args.config)
    cameras = args.camera or sorted(p.name for p in args.recordings.iterdir() if p.is_dir())
    if not cameras:
        raise SystemExit(f"no camera recording dirs found in {args.recordings}")

    conn = init_db(args.db)
    if args.replace:
        sql = "DELETE FROM events WHERE model=? AND source=?"
        params: list = [args.model_name, args.source]
        if args.t_from is not None:
            sql += " AND wall_ms >= ?"
            params.append(args.t_from)
        if args.t_to is not None:
            sql += " AND wall_ms <= ?"
            params.append(args.t_to)
        if args.camera:
            sql += " AND camera_id IN (%s)" % ",".join("?" for _ in args.camera)
            params.extend(args.camera)
        conn.execute(sql, params)

    model = YOLO(args.weights)
    total_frames = 0
    total_boxes = 0
    total_ignored = 0
    total_small = 0

    for camera_id in cameras:
        cam_cfg = cfg.get(camera_id, {})
        rotate_deg = int(cam_cfg.get("rotate_deg", 0))
        if rotate_deg not in (0, 90, 180, 270):
            raise SystemExit(f"camera {camera_id}: rotate_deg must be 0/90/180/270")
        detect_roi = _parse_rect(cam_cfg.get("detect_roi", "0,0,1,1"))
        min_coverage = (
            args.ignore_region_min_coverage
            if args.ignore_region_min_coverage is not None
            else float(cam_cfg.get("ignore_region_min_coverage", 0.8))
        )

        index = SegmentIndex.from_dir(args.recordings / camera_id)
        segments = _selected_segments(
            index.segments,
            t_from=args.t_from,
            t_to=args.t_to,
            limit_segments=args.limit_segments,
        )
        print(f"[offline-rescan] camera={camera_id} segments={len(segments)}")

        for seg_i, seg in enumerate(segments, start=1):
            for frame, img_cam, media_t, stream_tb in _iter_sampled_frames(seg, args.sample_interval_sec):
                cam_h, cam_w = img_cam.shape[:2]
                x0 = int(detect_roi[0] * cam_w)
                y0 = int(detect_roi[1] * cam_h)
                x1 = int(detect_roi[2] * cam_w)
                y1 = int(detect_roi[3] * cam_h)
                crop_cam = img_cam[y0:y1, x0:x1]
                if crop_cam.size == 0:
                    continue
                k = _rot90_k(rotate_deg)
                img_inf = np.ascontiguousarray(np.rot90(crop_cam, k=k)) if k else crop_cam
                rot_h, rot_w = img_inf.shape[:2]

                boxes = []
                for b in _detect_boxes(model, img_inf, conf=args.conf, imgsz=args.imgsz):
                    ux, uy, uw, uh = _unrotate_box_to_camera(
                        b["x"], b["y"], b["w"], b["h"], rot_w, rot_h, rotate_deg,
                    )
                    if uw < args.min_box_size or uh < args.min_box_size:
                        total_small += 1
                        continue
                    cam_box = Box(
                        x=ux + x0, y=uy + y0, w=uw, h=uh,
                        cat="cat", score=b["score"], track_id=None,
                    )
                    if box_in_ignore_region(
                        camera_id,
                        cam_w,
                        cam_h,
                        cam_box,
                        ignore_regions,
                        min_coverage=min_coverage,
                    ):
                        total_ignored += 1
                        continue
                    boxes.append(cam_box)

                wall_ms = int(seg.start_ms + media_t * 1000)
                pts = int(frame.pts) if frame.pts is not None else None
                tb_num = stream_tb.numerator if stream_tb is not None else None
                tb_den = stream_tb.denominator if stream_tb is not None else None
                for box in boxes:
                    insert_event(
                        conn,
                        camera_id=camera_id,
                        model=args.model_name,
                        wall_ms=wall_ms,
                        pts=pts,
                        tb_num=tb_num,
                        tb_den=tb_den,
                        media_t=float(media_t),
                        frame_w=cam_w,
                        frame_h=cam_h,
                        rotate_deg=rotate_deg,
                        cat="cat",
                        cat_score=None,
                        box_x=box.x,
                        box_y=box.y,
                        box_w=box.w,
                        box_h=box.h,
                        score=box.score,
                        source=args.source,
                    )
                total_frames += 1
                total_boxes += len(boxes)
            if seg_i % 100 == 0:
                print(
                    f"[offline-rescan] camera={camera_id} "
                    f"{seg_i}/{len(segments)} segments frames={total_frames} boxes={total_boxes}",
                    flush=True,
                )

    conn.commit()
    conn.close()
    print(
        "[offline-rescan] done "
        f"frames={total_frames} boxes={total_boxes} "
        f"ignored={total_ignored} small={total_small} "
        f"model={args.model_name!r}"
    )


if __name__ == "__main__":
    main()
