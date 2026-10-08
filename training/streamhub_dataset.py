"""Build a resumable manual-review queue from streamhub recordings.

The collector deliberately treats sidecar detections as selection hints, never
as labels. Saved images are full camera-orientation frames without overlays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat as stat_module
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import sys

import av
import cv2
import numpy as np
import yaml

from training.frame_geometry import apply_frame_geometry, camera_box_to_model


CLOCK_RATE = 90_000


def _log(message: str) -> None:
    """Progress to stderr (flushed), so a long scan is never silent."""
    print(message, file=sys.stderr, flush=True)
SEGMENT_RE = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{3}Z)_(?P<duration>\d+)ms\.mp4$"
)
SCHEMA_VERSION = 3


class BudgetExceeded(RuntimeError):
    """The image budget cannot be met without deleting reviewed/protected data."""


@dataclass(frozen=True, slots=True)
class Segment:
    camera: str
    path: Path
    relpath: str
    start_ms: int
    duration_ms: int

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms

    @property
    def sidecar(self) -> Path:
        stem = self.path.name.rsplit("_", 1)[0]
        return self.path.with_name(stem + ".labels.jsonl")


@dataclass(slots=True)
class CameraState:
    last_seen_ms: int | None = None
    last_hash: int | None = None
    last_saved_hash: int | None = None
    last_regular_ms: int | None = None
    visit_start_ms: int | None = None
    last_activity_ms: int | None = None
    last_visit_save_ms: int | None = None
    visit_saved: int = 0


def parse_utc(value: str) -> int:
    """Parse an ISO-8601 value or YYYY-MM-DD as inclusive UTC milliseconds."""
    text = value.strip()
    if len(text) == 10:
        text += "T00:00:00+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_segment(root: Path, path: Path) -> Segment | None:
    match = SEGMENT_RE.match(path.name)
    if not match:
        return None
    rel = path.relative_to(root)
    if len(rel.parts) < 4:
        return None
    camera = rel.parts[0]
    start = datetime.strptime(match.group("start"), "%Y-%m-%dT%H-%M-%S.%fZ").replace(tzinfo=timezone.utc)
    return Segment(
        camera=camera,
        path=path,
        relpath=rel.as_posix(),
        start_ms=int(start.timestamp() * 1000),
        duration_ms=int(match.group("duration")),
    )


def iter_segments(root: Path, cameras: set[str], start_ms: int, end_ms: int) -> Iterator[Segment]:
    found: list[Segment] = []
    for camera in sorted(cameras):
        cam_root = root / camera
        if not cam_root.is_dir():
            continue
        for path in cam_root.rglob("*.mp4"):
            seg = parse_segment(root, path)
            if seg and seg.end_ms >= start_ms and seg.start_ms < end_ms:
                found.append(seg)
    yield from sorted(found, key=lambda item: (item.camera, item.start_ms, item.relpath))


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("t") == "cv" and isinstance(value.get("pts"), int):
                    rows.append(value)
    except FileNotFoundError:
        pass
    rows.sort(key=lambda row: row["pts"])
    return rows


def nearest_prediction(rows: list[dict], pts: int, tolerance_ticks: int) -> dict | None:
    """Return the closest CV row without assuming every video frame was inferred."""
    if not rows:
        return None
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["pts"] < pts:
            lo = mid + 1
        else:
            hi = mid
    candidates = rows[max(0, lo - 1): min(len(rows), lo + 1)]
    best = min(candidates, key=lambda row: abs(row["pts"] - pts))
    return best if abs(best["pts"] - pts) <= tolerance_ticks else None


def project_prediction(
    prediction: dict | None, orig_w: int, orig_h: int,
    rotate_deg: int, detect_roi: list | None,
) -> dict | None:
    """Re-express a sidecar CV row's boxes in the saved model-input geometry.

    Sidecar boxes are fractions of the full camera frame; saved images are the
    model input (ROI-cropped, rotated). Projecting keeps the stored prediction
    hints aligned with the saved pixels so CVAT suggestions land correctly.
    Boxes outside the detection area are dropped.
    """
    if prediction is None:
        return None
    projected: list[dict] = []
    for det in prediction.get("dets") or []:
        try:
            x, y, w, h = (float(value) for value in det["box"])
        except (KeyError, TypeError, ValueError):
            continue
        box = camera_box_to_model((x, y, w, h), orig_w, orig_h, rotate_deg, detect_roi)
        if box is None:
            continue
        projected.append({**det, "box": list(box)})
    return {**prediction, "dets": projected}


def difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(left: int | None, right: int) -> int:
    return 64 if left is None else (left ^ right).bit_count()


def stable_id(camera: str, pts: int) -> str:
    raw = f"streamhub-v1\0{camera}\0{pts}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def read_camera_config(path: Path) -> dict[str, dict]:
    value = yaml.safe_load(path.read_text()) or {}
    return {str(row["id"]): dict(row.get("cv") or {}) for row in value.get("cameras", [])}


def open_catalog(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info(version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS segments(
          relpath TEXT PRIMARY KEY, camera TEXT NOT NULL, start_ms INTEGER NOT NULL,
          duration_ms INTEGER NOT NULL, size_bytes INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL, processed_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS segment_scans(
          relpath TEXT NOT NULL, scan_from_ms INTEGER NOT NULL,
          scan_to_ms INTEGER NOT NULL, source_fingerprint TEXT NOT NULL,
          selection_fingerprint TEXT NOT NULL, processed_at_ms INTEGER NOT NULL,
          PRIMARY KEY(
            relpath,scan_from_ms,scan_to_ms,source_fingerprint,selection_fingerprint
          )
        );
        CREATE INDEX IF NOT EXISTS segment_scans_lookup_idx ON segment_scans(
          relpath,source_fingerprint,selection_fingerprint,scan_from_ms,scan_to_ms
        );
        CREATE TABLE IF NOT EXISTS samples(
          sample_id TEXT PRIMARY KEY, camera TEXT NOT NULL, segment_relpath TEXT NOT NULL,
          pts INTEGER NOT NULL, wall_ms INTEGER NOT NULL, visit_group TEXT,
          image_relpath TEXT NOT NULL UNIQUE, width INTEGER NOT NULL, height INTEGER NOT NULL,
          jpeg_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, dhash TEXT NOT NULL,
          reasons_json TEXT NOT NULL, rotate_deg INTEGER NOT NULL,
          detect_roi_json TEXT, status TEXT NOT NULL DEFAULT 'unreviewed',
          protected INTEGER NOT NULL DEFAULT 0, source_missing INTEGER NOT NULL DEFAULT 0,
          tags_json TEXT NOT NULL DEFAULT '[]',
          created_at_ms INTEGER NOT NULL,
          UNIQUE(camera, pts)
        );
        CREATE INDEX IF NOT EXISTS samples_wall_idx ON samples(camera, wall_ms);
        CREATE INDEX IF NOT EXISTS samples_status_idx ON samples(status, protected);
        CREATE TABLE IF NOT EXISTS predictions(
          sample_id TEXT PRIMARY KEY REFERENCES samples(sample_id) ON DELETE CASCADE,
          source_pts INTEGER, model TEXT, worker TEXT, infer_ms REAL,
          detections_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collector_state(
          camera TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_batches(
          batch_id TEXT PRIMARY KEY, created_at_ms INTEGER NOT NULL,
          manifest_sha256 TEXT NOT NULL, status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS review_batch_samples(
          batch_id TEXT NOT NULL REFERENCES review_batches(batch_id),
          sample_id TEXT NOT NULL REFERENCES samples(sample_id),
          was_protected INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(batch_id,sample_id)
        );
        CREATE TABLE IF NOT EXISTS annotations(
          sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
          annotation_id INTEGER NOT NULL, class_name TEXT NOT NULL,
          x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
          source TEXT NOT NULL, updated_at_ms INTEGER NOT NULL,
          PRIMARY KEY(sample_id,annotation_id)
        );
        CREATE TABLE IF NOT EXISTS dataset_versions(
          version_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL, path TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dataset_version_samples(
          version_id TEXT NOT NULL REFERENCES dataset_versions(version_id),
          sample_id TEXT NOT NULL REFERENCES samples(sample_id), split TEXT NOT NULL,
          group_id TEXT NOT NULL, PRIMARY KEY(version_id,sample_id)
        );
        """
    )
    sample_columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    if "tags_json" not in sample_columns:
        conn.execute("ALTER TABLE samples ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'")
    batch_columns = {row[1] for row in conn.execute("PRAGMA table_info(review_batch_samples)")}
    if "was_protected" not in batch_columns:
        conn.execute(
            "ALTER TABLE review_batch_samples ADD COLUMN was_protected INTEGER NOT NULL DEFAULT 0"
        )
    row = conn.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] in (1, 2):
        conn.execute("UPDATE schema_info SET version=?", (SCHEMA_VERSION,))
    elif row[0] != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported catalog schema {row[0]}")
    conn.commit()
    return conn


def _file_signature(path: Path) -> dict[str, int] | None:
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return None
    return {"size": int(file_stat.st_size), "mtime_ns": int(file_stat.st_mtime_ns)}


def segment_source_fingerprint(seg: Segment) -> str:
    """Fingerprint both the immutable video and the possibly-late sidecar."""
    payload = {
        "schema": 1,
        "video": _file_signature(seg.path),
        "labels": _file_signature(seg.sidecar),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def selection_fingerprint(args, cv_config: dict) -> str:
    """Identify every input that can change selection or saved image bytes."""
    payload = {
        "schema": 1,
        "candidate_interval_sec": args.candidate_interval_sec,
        "regular_interval_sec": args.regular_interval_sec,
        "visit_gap_sec": args.visit_gap_sec,
        "visit_sample_interval_sec": args.visit_sample_interval_sec,
        "max_visit_samples": args.max_visit_samples,
        "motion_threshold": args.motion_threshold,
        "dedupe_threshold": args.dedupe_threshold,
        "low_confidence": args.low_confidence,
        "prediction_tolerance_sec": args.prediction_tolerance_sec,
        "jpeg_quality": args.jpeg_quality,
        "tags": sorted(args.tags),
        "camera_cv": cv_config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scan_is_complete(
    conn: sqlite3.Connection, seg: Segment, scan_from_ms: int, scan_to_ms: int,
    source_fingerprint: str, selection_fingerprint_value: str,
) -> bool:
    """Return true only when a completed scan covers the requested intersection."""
    return conn.execute(
        """SELECT 1 FROM segment_scans
           WHERE relpath=? AND source_fingerprint=? AND selection_fingerprint=?
             AND scan_from_ms<=? AND scan_to_ms>=?
           LIMIT 1""",
        (seg.relpath, source_fingerprint, selection_fingerprint_value,
         scan_from_ms, scan_to_ms),
    ).fetchone() is not None


def load_state(conn: sqlite3.Connection, camera: str, range_start_ms: int) -> CameraState:
    row = conn.execute("SELECT state_json FROM collector_state WHERE camera=?", (camera,)).fetchone()
    if row is None:
        return CameraState()
    state = CameraState(**json.loads(row[0]))
    # Operators may collect an older date after a newer one. Future state would
    # suppress regular samples and create invalid negative time gaps.
    if state.last_seen_ms is not None and state.last_seen_ms > range_start_ms:
        return CameraState()
    return state


def save_state(conn: sqlite3.Connection, camera: str, state: CameraState) -> None:
    payload = json.dumps({name: getattr(state, name) for name in state.__dataclass_fields__}, sort_keys=True)
    conn.execute(
        "INSERT INTO collector_state(camera,state_json,updated_at_ms) VALUES(?,?,?) "
        "ON CONFLICT(camera) DO UPDATE SET state_json=excluded.state_json,updated_at_ms=excluded.updated_at_ms",
        (camera, payload, int(time.time() * 1000)),
    )


def frame_reasons(
    *, wall_ms: int, prediction: dict | None, image_hash: int, state: CameraState,
    regular_interval_ms: int, motion_threshold: int, low_conf: float,
    visit_gap_ms: int, visit_sample_ms: int, max_visit_samples: int,
) -> tuple[list[str], str | None]:
    reasons: list[str] = []
    state.last_seen_ms = wall_ms
    motion = hamming(state.last_hash, image_hash) >= motion_threshold
    state.last_hash = image_hash
    detections = list((prediction or {}).get("dets") or [])
    low = any(float(det.get("score", 1.0)) < low_conf for det in detections)
    multi = len(detections) > 1
    activity = bool(detections) or motion or low or multi
    if motion:
        reasons.append("visual_change")
    if low:
        reasons.append("low_confidence")
    if multi:
        reasons.append("multiple_boxes")
    if state.last_regular_ms is None or wall_ms - state.last_regular_ms >= regular_interval_ms:
        reasons.append("regular")
        state.last_regular_ms = wall_ms
    if activity:
        if state.last_activity_ms is None or wall_ms - state.last_activity_ms > visit_gap_ms:
            state.visit_start_ms = wall_ms
            state.visit_saved = 0
            state.last_visit_save_ms = None
        state.last_activity_ms = wall_ms
    in_visit = state.last_activity_ms is not None and wall_ms - state.last_activity_ms <= visit_gap_ms
    visit_group = None
    if in_visit and state.visit_start_ms is not None:
        visit_group = f"{state.visit_start_ms:013d}-{state.visit_start_ms // 86_400_000:08x}"
        due = state.last_visit_save_ms is None or wall_ms - state.last_visit_save_ms >= visit_sample_ms
        if due and state.visit_saved < max_visit_samples:
            reasons.append("visit_sample")
            state.last_visit_save_ms = wall_ms
            state.visit_saved += 1
    return sorted(set(reasons)), visit_group


def write_sample(
    conn: sqlite3.Connection, out_root: Path, seg: Segment, pts: int, image: np.ndarray,
    reasons: list[str], visit_group: str | None, prediction: dict | None,
    cv_config: dict, jpeg_quality: int, tags: list[str],
) -> bool:
    sid = stable_id(seg.camera, pts)
    existing = conn.execute("SELECT tags_json FROM samples WHERE sample_id=?", (sid,)).fetchone()
    if existing:
        merged_tags = sorted(set(json.loads(existing[0]) or []) | set(tags))
        conn.execute(
            "UPDATE samples SET tags_json=? WHERE sample_id=?",
            (json.dumps(merged_tags, separators=(",", ":")), sid),
        )
        return False
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"could not encode frame {seg.camera}:{pts}")
    payload = encoded.tobytes()
    rel = Path("images") / seg.camera / f"{sid}.jpg"
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jpg.tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dest)
    height, width = image.shape[:2]
    wall_ms = pts * 1000 // CLOCK_RATE
    digest = hashlib.sha256(payload).hexdigest()
    dhash = f"{difference_hash(image):016x}"
    conn.execute(
        """INSERT INTO samples(
          sample_id,camera,segment_relpath,pts,wall_ms,visit_group,image_relpath,
          width,height,jpeg_bytes,sha256,dhash,reasons_json,rotate_deg,
          detect_roi_json,tags_json,created_at_ms
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, seg.camera, seg.relpath, pts, wall_ms, visit_group, rel.as_posix(),
         width, height, len(payload), digest, dhash, json.dumps(reasons),
         int(cv_config.get("rotate_deg") or 0),
         json.dumps(cv_config.get("detect_roi")) if cv_config.get("detect_roi") else None,
         json.dumps(sorted(set(tags)), separators=(",", ":")),
         int(time.time() * 1000)),
    )
    if prediction is not None:
        conn.execute(
            "INSERT INTO predictions(sample_id,source_pts,model,worker,infer_ms,detections_json) VALUES(?,?,?,?,?,?)",
            (sid, prediction.get("pts"), prediction.get("model"), prediction.get("worker"),
             prediction.get("infer_ms"), json.dumps(prediction.get("dets") or [], separators=(",", ":"))),
        )
    return True


def collect_segment(args, conn: sqlite3.Connection, out_root: Path, seg: Segment,
                    state: CameraState, cv_config: dict) -> tuple[int, int]:
    stat = seg.path.stat()
    scan_from_ms = max(args.from_ms, seg.start_ms)
    scan_to_ms = min(args.to_ms, seg.end_ms)
    source_fingerprint = segment_source_fingerprint(seg)
    selection_fingerprint_value = selection_fingerprint(args, cv_config)
    if scan_is_complete(
        conn, seg, scan_from_ms, scan_to_ms, source_fingerprint,
        selection_fingerprint_value,
    ):
        return 0, 0
    predictions = load_predictions(seg.sidecar)
    tolerance = int(args.prediction_tolerance_sec * CLOCK_RATE)
    sample_ticks = max(1, int(args.candidate_interval_sec * CLOCK_RATE))
    rotate_deg = int(cv_config.get("rotate_deg") or 0)
    detect_roi = cv_config.get("detect_roi")
    decoded = saved = 0
    next_pts: int | None = None
    with av.open(str(seg.path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded += 1
            pts = int(frame.pts)
            if next_pts is None:
                next_pts = pts
            if pts < next_pts:
                continue
            while next_pts <= pts:
                next_pts += sample_ticks
            wall_ms = pts * 1000 // CLOCK_RATE
            if wall_ms < args.from_ms or wall_ms >= args.to_ms:
                continue
            raw = frame.to_ndarray(format="bgr24")
            orig_h, orig_w = raw.shape[:2]
            # Save the frame as the detector sees it (ROI crop + rotation), so
            # labelling, training and runtime share one geometry.
            image = apply_frame_geometry(raw, rotate_deg, detect_roi)
            image_hash = difference_hash(image)
            pred = project_prediction(
                nearest_prediction(predictions, pts, tolerance),
                orig_w, orig_h, rotate_deg, detect_roi,
            )
            reasons, visit_group = frame_reasons(
                wall_ms=wall_ms, prediction=pred, image_hash=image_hash, state=state,
                regular_interval_ms=int(args.regular_interval_sec * 1000),
                motion_threshold=args.motion_threshold, low_conf=args.low_confidence,
                visit_gap_ms=int(args.visit_gap_sec * 1000),
                visit_sample_ms=int(args.visit_sample_interval_sec * 1000),
                max_visit_samples=args.max_visit_samples,
            )
            if visit_group is not None:
                visit_group = f"{seg.camera}:{visit_group}"
            if not reasons:
                continue
            # Near-identical visit frames are skipped unless this is the periodic
            # sample that guarantees coverage independent of YOLO.
            if "regular" not in reasons and hamming(state.last_saved_hash, image_hash) <= args.dedupe_threshold:
                continue
            wrote = write_sample(
                conn, out_root, seg, pts, image, reasons, visit_group, pred,
                cv_config, args.jpeg_quality, args.tags,
            )
            saved += int(wrote)
            if wrote:
                state.last_saved_hash = image_hash
    if segment_source_fingerprint(seg) != source_fingerprint:
        conn.rollback()
        raise RuntimeError(f"recording or labels changed while scanning {seg.relpath}; retry")
    processed_at_ms = int(time.time() * 1000)
    conn.execute(
        "INSERT OR REPLACE INTO segments(relpath,camera,start_ms,duration_ms,size_bytes,mtime_ns,processed_at_ms) "
        "VALUES(?,?,?,?,?,?,?)",
        (seg.relpath, seg.camera, seg.start_ms, seg.duration_ms, stat.st_size, stat.st_mtime_ns,
         processed_at_ms),
    )
    conn.execute(
        """INSERT OR IGNORE INTO segment_scans(
             relpath,scan_from_ms,scan_to_ms,source_fingerprint,
             selection_fingerprint,processed_at_ms
           ) VALUES(?,?,?,?,?,?)""",
        (seg.relpath, scan_from_ms, scan_to_ms, source_fingerprint,
         selection_fingerprint_value, processed_at_ms),
    )
    save_state(conn, seg.camera, state)
    conn.commit()
    return decoded, saved


def reconcile_missing(conn: sqlite3.Connection, recordings: Path) -> int:
    rows = conn.execute("SELECT sample_id,segment_relpath FROM samples WHERE source_missing=0").fetchall()
    missing = [row[0] for row in rows if not (recordings / row[1]).exists()]
    conn.executemany("UPDATE samples SET source_missing=1 WHERE sample_id=?", ((sid,) for sid in missing))
    conn.commit()
    return len(missing)


def _eviction_priority(row: sqlite3.Row) -> tuple[int, int]:
    """Lower value is evicted first; age breaks ties."""
    reasons = set(json.loads(row["reasons_json"]))
    value = 0
    if "visit_sample" in reasons:
        value += 10
    if "visual_change" in reasons:
        value += 20
    if "low_confidence" in reasons:
        value += 30
    if "multiple_boxes" in reasons:
        value += 40
    return value, int(row["wall_ms"])


def derived_storage_bytes(out_root: Path) -> int:
    """Count persistent model-image/version files once, without following symlinks."""
    total = 0
    seen: set[tuple[int, int]] = set()
    for directory in (out_root / "model_images", out_root / "versions"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            try:
                file_stat = path.lstat()
            except FileNotFoundError:
                continue
            if not stat_module.S_ISREG(file_stat.st_mode):
                continue
            identity = (int(file_stat.st_dev), int(file_stat.st_ino))
            if identity in seen:
                continue
            seen.add(identity)
            total += int(file_stat.st_size)
    return total


def enforce_budget(conn: sqlite3.Connection, out_root: Path, budget_bytes: int,
                   min_per_camera_day: int = 2) -> tuple[int, int]:
    """Evict replaceable review candidates until the image store fits.

    Only status=unreviewed and protected=0 rows are replaceable. The per-day
    reserve is soft: it is evicted only after every non-reserved candidate, so
    the hard byte limit still holds. Reviewed/versioned/protected rows are a
    hard boundary and cause BudgetExceeded instead of silent deletion.
    """
    raw_total = int(conn.execute("SELECT coalesce(sum(jpeg_bytes),0) FROM samples").fetchone()[0])
    derived_bytes = derived_storage_bytes(out_root)
    total = raw_total + derived_bytes
    if total <= budget_bytes:
        return 0, 0
    rows = conn.execute(
        "SELECT sample_id,camera,wall_ms,image_relpath,jpeg_bytes,reasons_json "
        "FROM samples WHERE status='unreviewed' AND protected=0"
    ).fetchall()
    replaceable = sum(int(row["jpeg_bytes"]) for row in rows)
    protected_bytes = total - replaceable
    if protected_bytes > budget_bytes:
        raise BudgetExceeded(
            f"protected/reviewed images use {protected_bytes} bytes, above budget {budget_bytes}; "
            "increase --budget-gb or archive a dataset version elsewhere"
        )

    by_day: dict[tuple[str, int], list[sqlite3.Row]] = {}
    for row in rows:
        day = int(row["wall_ms"]) // 86_400_000
        by_day.setdefault((row["camera"], day), []).append(row)
    reserved: set[str] = set()
    for group in by_day.values():
        # Reserve the most valuable/recent examples from each camera-day.
        keep = sorted(group, key=_eviction_priority, reverse=True)[:max(0, min_per_camera_day)]
        reserved.update(row["sample_id"] for row in keep)
    ordered = sorted(rows, key=lambda row: (row["sample_id"] in reserved, *_eviction_priority(row)))

    trash = out_root / ".trash"
    deleted = freed = 0
    for row in ordered:
        if total <= budget_bytes:
            break
        src = out_root / row["image_relpath"]
        parked = trash / f"{row['sample_id']}.jpg"
        trash.mkdir(parents=True, exist_ok=True)
        parked.unlink(missing_ok=True)
        if src.exists():
            os.replace(src, parked)
        try:
            conn.execute("DELETE FROM samples WHERE sample_id=?", (row["sample_id"],))
            conn.commit()
        except Exception:
            if parked.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                os.replace(parked, src)
            raise
        parked.unlink(missing_ok=True)
        size = int(row["jpeg_bytes"])
        total -= size
        freed += size
        deleted += 1
    try:
        trash.rmdir()
    except OSError:
        pass
    if total > budget_bytes:
        raise BudgetExceeded(
            f"cannot reduce image store to {budget_bytes} bytes: {total} bytes remain and "
            "all remaining images are reviewed/protected"
        )
    return deleted, freed


TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def validate_collection_args(args) -> None:
    positive = {
        "--candidate-interval-sec": args.candidate_interval_sec,
        "--regular-interval-sec": args.regular_interval_sec,
        "--visit-gap-sec": args.visit_gap_sec,
        "--visit-sample-interval-sec": args.visit_sample_interval_sec,
        "--prediction-tolerance-sec": args.prediction_tolerance_sec,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite value greater than zero")
    if args.max_visit_samples <= 0:
        raise ValueError("--max-visit-samples must be greater than zero")
    if not 1 <= args.motion_threshold <= 64:
        raise ValueError("--motion-threshold must be between 1 and 64")
    if not 0 <= args.dedupe_threshold <= 64:
        raise ValueError("--dedupe-threshold must be between 0 and 64")
    if not np.isfinite(args.low_confidence) or not 0 <= args.low_confidence <= 1:
        raise ValueError("--low-confidence must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if not np.isfinite(args.budget_gb) or args.budget_gb < 0:
        raise ValueError("--budget-gb must be finite and non-negative")
    if args.min_per_camera_day < 0:
        raise ValueError("--min-per-camera-day must be non-negative")
    if not args.cameras:
        raise ValueError("at least one non-empty --camera is required")
    invalid_tags = [tag for tag in args.tags if not TAG_RE.fullmatch(tag)]
    if invalid_tags:
        raise ValueError(
            "--tag must be 1-64 ASCII letters/digits or . _ : -; invalid: "
            + ", ".join(invalid_tags)
        )


def write_report_atomic(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("data/streamhub/recordings"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--camera", action="append", dest="cameras", required=True)
    parser.add_argument("--from", dest="from_time", required=True, help="UTC ISO timestamp or date")
    parser.add_argument("--to", dest="to_time", required=True, help="exclusive UTC ISO timestamp or date")
    parser.add_argument("--candidate-interval-sec", type=float, default=5.0)
    parser.add_argument("--regular-interval-sec", type=float, default=300.0)
    parser.add_argument("--visit-gap-sec", type=float, default=90.0)
    parser.add_argument("--visit-sample-interval-sec", type=float, default=20.0)
    parser.add_argument("--max-visit-samples", type=int, default=6)
    parser.add_argument("--motion-threshold", type=int, default=10)
    parser.add_argument("--dedupe-threshold", type=int, default=3)
    parser.add_argument("--low-confidence", type=float, default=0.35)
    parser.add_argument("--prediction-tolerance-sec", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--tag", action="append", dest="tags", default=[],
        help="repeatable range tag saved on selected samples (for example: shaved)",
    )
    parser.add_argument("--budget-gb", type=float, default=5.0,
                        help="hard image budget in decimal GB (default: 5.0); 0 disables")
    parser.add_argument("--min-per-camera-day", type=int, default=2,
                        help="soft reserve of unreviewed images per camera/day")
    parser.add_argument("--report-json", type=Path,
                        help="atomically write a small machine-readable run report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()
    metrics = {
        "segments": 0,
        "decoded_frames": 0,
        "new_images": 0,
        "evicted_images": 0,
        "evicted_bytes": 0,
        "catalog_images": 0,
        "catalog_bytes": 0,
        "derived_bytes": 0,
        "managed_bytes": 0,
        "source_missing": 0,
    }
    report = {
        "schema": 1,
        "status": "failed",
        "from": args.from_time,
        "to": args.to_time,
        "cameras": [],
        "tags": [],
        **metrics,
    }
    try:
        args.from_ms, args.to_ms = parse_utc(args.from_time), parse_utc(args.to_time)
        if args.to_ms <= args.from_ms:
            raise ValueError("--to must be after --from")
        args.cameras = [
            name.strip() for value in args.cameras for name in value.split(",") if name.strip()
        ]
        args.tags = sorted({tag.strip() for tag in args.tags})
        validate_collection_args(args)
        report["cameras"] = args.cameras
        report["tags"] = args.tags
        configs = read_camera_config(args.config)
        unknown = sorted(set(args.cameras) - set(configs))
        if unknown:
            raise ValueError(f"unknown camera(s): {', '.join(unknown)}")
        args.out.mkdir(parents=True, exist_ok=True)
        if args.report_json is not None:
            _log(f"run report will be written on completion to {args.report_json.resolve()}")
        conn = open_catalog(args.out / "catalog.sqlite3")
        try:
            states = {camera: load_state(conn, camera, args.from_ms) for camera in args.cameras}
            segments = list(
                iter_segments(args.recordings, set(args.cameras), args.from_ms, args.to_ms)
            )
            _log(
                f"scanning {len(segments)} segment(s): cameras={args.cameras} "
                f"range={args.from_time}..{args.to_time} recordings={args.recordings} "
                f"-> catalog {args.out / 'catalog.sqlite3'}"
            )
            if not segments:
                searched = [str(args.recordings / camera) for camera in args.cameras]
                _log(f"no segments matched; check that these camera dirs exist: {searched}")
            last_log = time.monotonic()
            for index, seg in enumerate(segments, 1):
                n_decoded, n_saved = collect_segment(
                    args, conn, args.out, seg, states[seg.camera], configs[seg.camera]
                )
                metrics["decoded_frames"] += n_decoded
                metrics["new_images"] += n_saved
                metrics["segments"] += int(n_decoded > 0)
                if args.budget_gb > 0:
                    n_evicted, n_freed = enforce_budget(
                        conn, args.out, int(args.budget_gb * 1_000_000_000),
                        args.min_per_camera_day,
                    )
                    metrics["evicted_images"] += n_evicted
                    metrics["evicted_bytes"] += n_freed
                now = time.monotonic()
                if index == len(segments) or now - last_log >= 2.0:
                    _log(f"[{index}/{len(segments)}] {seg.relpath} "
                         f"decoded={n_decoded} saved={n_saved} new_total={metrics['new_images']}")
                    last_log = now
            # Enforce once even when there were no source segments: a dataset
            # version/cache may have consumed the remaining budget meanwhile.
            if args.budget_gb > 0:
                n_evicted, n_freed = enforce_budget(
                    conn, args.out, int(args.budget_gb * 1_000_000_000),
                    args.min_per_camera_day,
                )
                metrics["evicted_images"] += n_evicted
                metrics["evicted_bytes"] += n_freed
            metrics["source_missing"] = reconcile_missing(conn, args.recordings)
            total = conn.execute(
                "SELECT count(*),coalesce(sum(jpeg_bytes),0) FROM samples"
            ).fetchone()
            metrics["catalog_images"] = int(total[0])
            metrics["catalog_bytes"] = int(total[1])
            metrics["derived_bytes"] = derived_storage_bytes(args.out)
            metrics["managed_bytes"] = metrics["catalog_bytes"] + metrics["derived_bytes"]
        finally:
            conn.close()
        report.update(metrics)
        report["status"] = "completed"
        return_code = 0
    except KeyboardInterrupt:
        report.update(metrics)
        report["status"] = "interrupted"
        report["error"] = "KeyboardInterrupt"
        return_code = 130
    except Exception as exc:
        report.update(metrics)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_sec"] = round(time.monotonic() - started, 6)
        if args.report_json is not None:
            write_report_atomic(args.report_json, report)
    if return_code == 0:
        print(
            f"segments={metrics['segments']} decoded_frames={metrics['decoded_frames']} "
            f"new_images={metrics['new_images']} evicted_images={metrics['evicted_images']} "
            f"evicted_bytes={metrics['evicted_bytes']} "
            f"catalog_images={metrics['catalog_images']} catalog_bytes={metrics['catalog_bytes']} "
            f"derived_bytes={metrics['derived_bytes']} managed_bytes={metrics['managed_bytes']} "
            f"source_missing={metrics['source_missing']} elapsed_sec={report['elapsed_sec']:.3f}"
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
