"""Training-set deduplication — filter a MANIFEST, never the source data.

Nothing here deletes events, labels, recordings, or SQLite rows. It takes a list
of lightweight per-crop descriptors, decides which crops a training manifest
should keep, and emits a separate report with aggregates + IDs. The actual
pixels/DB are untouched; the trainer simply reads fewer ids.

Two stages, both cheap:
  1. Temporal grouping by (label, camera, video, time-window bucket) so dozens of
     near-identical consecutive frames collapse to a few.
  2. Visual near-duplicate detection inside each group via a perceptual hash
     (dHash) + Hamming distance, comparing each crop only against the group's
     kept representatives — never a global O(N²) pairwise scan.

Representative selection prefers, in order: human-corrected label, larger bbox,
sharper (higher variance-of-Laplacian), normal exposure, higher detector
confidence. A configurable fraction of HARD examples (blur/dark/odd-pose/low-conf
/disagreement/rare-class — flagged by the caller) is retained even when they look
like duplicates, so hard cases aren't silently thinned away.

Deterministic for a fixed seed: ordering is a total sort by quality then id; no
unseeded randomness.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field


@dataclass
class DedupConfig:
    enabled: bool = True
    temporal_window_sec: float = 2.0
    hash_distance: int = 6          # max Hamming distance counted as near-dup
    keep_hard_fraction: float = 0.1  # fraction of HARD near-dups to retain
    max_samples_per_group: int = 0   # 0 = unlimited
    max_representatives: int = 128   # cap the per-group compare set (bounds cost)


@dataclass(slots=True)
class DedupSample:
    sample_id: int | str
    label: str
    camera: str
    video: str            # segment/video/episode key; groups never cross videos
    wall_ms: int
    dhash: int            # 64-bit perceptual hash of the crop
    bbox_area: int = 0
    blur: float = 0.0     # variance-of-Laplacian; higher = sharper
    exposure: float = 128.0  # mean brightness 0..255; 128 ~ ideal
    confidence: float = 0.0
    manual: bool = False  # human-corrected label present
    hard: bool = False    # caller-flagged hard/rare example


@dataclass
class DedupResult:
    kept_ids: list = field(default_factory=list)
    dropped_ids: list = field(default_factory=list)
    reason_by_id: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def _quality_key(s: DedupSample):
    """Ascending sort key → best representative first. Deterministic via id."""
    return (
        0 if s.manual else 1,
        -int(s.bbox_area),
        -float(s.blur),
        abs(float(s.exposure) - 128.0),
        -float(s.confidence),
        str(s.sample_id),
    )


def _group_key(s: DedupSample, window_ms: int) -> tuple:
    bucket = s.wall_ms // window_ms if window_ms > 0 else 0
    return (s.label, s.camera, s.video, bucket)


def _est_bytes(s: DedupSample) -> int:
    return int(max(0, s.bbox_area)) * 3  # uint8 BGR crop


def deduplicate(samples: list[DedupSample], config: DedupConfig) -> DedupResult:
    """Filter `samples` down to a deduplicated manifest. Pure / deterministic."""
    res = DedupResult()
    samples_before = len(samples)
    per_class_before = Counter(s.label for s in samples)
    ram_before = sum(_est_bytes(s) for s in samples)

    if not config.enabled:
        res.kept_ids = [s.sample_id for s in samples]
        for s in samples:
            res.reason_by_id[s.sample_id] = "kept_disabled"
        res.report = _build_report(
            samples_before, samples_before, 0, 0,
            per_class_before, Counter({"kept_disabled": samples_before}),
            ram_before, ram_before,
        )
        return res

    window_ms = int(config.temporal_window_sec * 1000)
    groups: dict[tuple, list[DedupSample]] = defaultdict(list)
    for s in samples:
        groups[_group_key(s, window_ms)].append(s)

    exact = 0
    near = 0
    for _gk in sorted(groups):  # deterministic group order
        g = sorted(groups[_gk], key=_quality_key)
        reps: list[DedupSample] = []
        hard_dups: list[DedupSample] = []
        for s in g:
            dup = False
            is_exact = False
            for r in reps:
                d = hamming(s.dhash, r.dhash)
                if d <= config.hash_distance:
                    dup = True
                    is_exact = d == 0
                    break
            if not dup:
                reps.append(s)
                res.reason_by_id[s.sample_id] = "representative"
                continue
            if is_exact:
                exact += 1
            else:
                near += 1
            if s.hard:
                hard_dups.append(s)
            else:
                res.reason_by_id[s.sample_id] = "exact_duplicate" if is_exact else "near_duplicate"

        # Retain a configurable fraction of HARD near-duplicates (best first).
        n_hard_keep = (
            math.ceil(config.keep_hard_fraction * len(hard_dups))
            if config.keep_hard_fraction > 0 else 0
        )
        hard_sorted = sorted(hard_dups, key=_quality_key)
        for s in hard_sorted[:n_hard_keep]:
            reps.append(s)
            res.reason_by_id[s.sample_id] = "kept_hard"
        for s in hard_sorted[n_hard_keep:]:
            res.reason_by_id[s.sample_id] = "near_duplicate"

        # Per-group cap: keep the best-ranked, drop the overflow.
        if config.max_samples_per_group > 0 and len(reps) > config.max_samples_per_group:
            reps_sorted = sorted(reps, key=_quality_key)
            for s in reps_sorted[config.max_samples_per_group:]:
                res.reason_by_id[s.sample_id] = "over_cap"
            reps = reps_sorted[: config.max_samples_per_group]

    kept_set = {
        sid for sid, why in res.reason_by_id.items()
        if why in {"representative", "kept_hard"}
    }
    # Preserve original order for stable manifests.
    res.kept_ids = [s.sample_id for s in samples if s.sample_id in kept_set]
    res.dropped_ids = [s.sample_id for s in samples if s.sample_id not in kept_set]
    kept_samples = [s for s in samples if s.sample_id in kept_set]
    per_class_after = Counter(s.label for s in kept_samples)
    ram_after = sum(_est_bytes(s) for s in kept_samples)

    res.report = _build_report(
        samples_before, len(res.kept_ids), exact, near,
        per_class_after, Counter(res.reason_by_id.values()),
        ram_before, ram_after,
    )
    return res


def _build_report(before, after, exact, near, per_class, reason_counts,
                  ram_before, ram_after) -> dict:
    return {
        "samples_before": before,
        "samples_after": after,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "samples_per_class": dict(sorted(per_class.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "estimated_ram_before": ram_before,
        "estimated_ram_after": ram_after,
    }


# ---- perceptual hash (used by the CLI; tests pass precomputed ints) ----------

def dhash(gray) -> int:
    """64-bit difference hash of a 2D uint8 grayscale image. Requires cv2."""
    import cv2
    import numpy as np

    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(bool(v))
    return bits & ((1 << 64) - 1)


# ---- CLI: build a filtered training manifest + a dedup report ----------------

def main() -> None:
    import argparse
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import cv2
    import numpy as np

    ap = argparse.ArgumentParser(description="Deduplicate a training manifest "
                                             "without touching source data.")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="filtered manifest JSON")
    ap.add_argument("--report", type=Path, default=None, help="dedup report JSON")
    ap.add_argument("--reviews-db", type=Path, default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--default-rotate-deg", type=int, default=0)
    ap.add_argument("--episode-gap-sec", type=float, default=60.0,
                    help="wall_ms gap that starts a new visit/video group")
    ap.add_argument("--dedup-temporal-window-sec", type=float, default=2.0)
    ap.add_argument("--dedup-hash-distance", type=int, default=6)
    ap.add_argument("--dedup-keep-hard-fraction", type=float, default=0.1)
    ap.add_argument("--dedup-max-samples-per-group", type=int, default=0)
    ap.add_argument("--hard-conf", type=float, default=0.5,
                    help="detector score below this flags a HARD sample")
    ap.add_argument("--rare-class-max", type=int, default=50,
                    help="classes with <= this many crops are kept as HARD/rare")
    ap.add_argument("--no-dedup", action="store_true", help="disable (passthrough)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report only; write nothing")
    args = ap.parse_args()

    from training import CropSource
    from training import load_reviews

    reviews = load_reviews(args.reviews_db) if args.reviews_db else {}
    gap_ms = int(args.episode_gap_sec * 1000)

    src = CropSource(
        db_path=args.db, recordings_root=args.recordings,
        camera_id=args.camera, model=args.model,
        min_score=args.min_score, pad_frac=args.pad_frac,
        default_rotate_deg=args.default_rotate_deg,
    )

    samples: list[DedupSample] = []
    refs: dict = {}
    last_wall: dict[str, int] = {}
    episode_n: dict[str, int] = {}
    for sample in src:
        sb = sample.src_box
        if sb is None or sb.rowid is None:
            continue
        cam = sample.camera_id
        prev = last_wall.get(cam)
        if prev is None or sample.wall_ms - prev > gap_ms:
            episode_n[cam] = episode_n.get(cam, -1) + 1
        last_wall[cam] = sample.wall_ms
        video = f"{cam}:{episode_n[cam]}"

        gray = cv2.cvtColor(sample.image, cv2.COLOR_BGR2GRAY)
        human = reviews.get(sb.rowid)
        label = human or sb.cat or "unlabeled"
        sid = f"{cam}:{sb.rowid}"
        samples.append(DedupSample(
            sample_id=sid, label=label, camera=cam, video=video,
            wall_ms=int(sample.wall_ms), dhash=dhash(gray),
            bbox_area=int(sb.w) * int(sb.h),
            blur=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            exposure=float(gray.mean()),
            confidence=float(sb.score),
            manual=human is not None,
            hard=(sb.score < args.hard_conf),
        ))
        refs[sid] = {
            "crop_id": sid, "src_event_key": int(sb.rowid), "camera": cam,
            "wall_ms": int(sample.wall_ms), "label": label,
            "box": {"x": int(sb.x), "y": int(sb.y), "w": int(sb.w), "h": int(sb.h)},
            "rotate_deg": int(sample.rotate_deg), "pad_frac": float(args.pad_frac),
        }

    # Rare classes are HARD: flag every sample of an under-represented class.
    from collections import Counter
    class_counts = Counter(s.label for s in samples)
    for s in samples:
        if class_counts[s.label] <= args.rare_class_max:
            s.hard = True

    cfg = DedupConfig(
        enabled=not args.no_dedup,
        temporal_window_sec=args.dedup_temporal_window_sec,
        hash_distance=args.dedup_hash_distance,
        keep_hard_fraction=args.dedup_keep_hard_fraction,
        max_samples_per_group=args.dedup_max_samples_per_group,
    )
    result = deduplicate(samples, cfg)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        **result.report,
        "dropped_ids": result.dropped_ids,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_ids"}, indent=2))

    if args.dry_run:
        print(f"[dedup] dry-run: would keep {len(result.kept_ids)}/{len(samples)} "
              f"crops; nothing written")
        return

    manifest = {
        "version": 1, "kind": "dedup_training_manifest",
        "created_at": report["created_at"], "config": asdict(cfg),
        "items": [refs[sid] for sid in result.kept_ids],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(result.kept_ids)} crops -> {args.out}")
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
