"""Materialise an immutable YOLO dataset version from reviewed catalog samples.

This is the bridge between human review (CVAT -> ``annotations`` in the catalog)
and training. It is the only writer of the version layout that
``training.yolo_common.load_dataset`` reads:

    <out>/<version_id>/
        data.yaml                       names: {0: cat}
        manifest.json                   samples + checksum + summary
        images/{train,val,test}/<id>.jpg   hardlinked from the catalog store
        labels/{train,val,test}/<id>.txt   "0 cx cy w h" (normalised, model space)

Key properties:

- **Group-level split.** Samples are clustered into visits by wall-clock gap
  across ALL cameras, so related observations of one visit (even on different
  cameras) never straddle the train/val/test boundary.
- **Stable assignment.** A group's split comes from a hash of its id, so adding
  data later does not reshuffle existing groups.
- **Honest about small data.** Too few groups, or an empty required split, is
  reported loudly instead of faking a 70/15/15 split.
- **Leakage guard.** Exact-duplicate images are de-duplicated before splitting;
  near-duplicates (dhash) across splits are counted and reported.
- **No image copies.** Images are hardlinked from the catalog store, so a
  version costs kilobytes of labels/metadata, not a second copy of every frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.streamhub_dataset import hamming, open_catalog
from training.yolo_common import atomic_write_json, link_or_copy, sha256_tree

SPLITS = ("train", "val", "test")


class BuildError(RuntimeError):
    """The reviewed data cannot produce a usable dataset version."""


@dataclass
class SampleRow:
    sample_id: str
    camera: str
    wall_ms: int
    pts: int
    width: int
    height: int
    image_relpath: str
    sha256: str
    dhash: int
    tags: list[str]
    rotate_deg: int
    detect_roi: list[float] | None
    boxes: list[list[float]] = field(default_factory=list)  # normalised cx,cy,w,h
    group_id: str = ""
    split: str = ""


def _normalise_box(x: float, y: float, w: float, h: float,
                   width: int, height: int) -> list[float] | None:
    """Pixel xywh (model-orientation image) -> clipped YOLO cx,cy,w,h."""
    left = max(0.0, min(float(x), width))
    top = max(0.0, min(float(y), height))
    right = max(0.0, min(float(x) + float(w), width))
    bottom = max(0.0, min(float(y) + float(h), height))
    bw, bh = right - left, bottom - top
    if bw <= 0 or bh <= 0:
        return None
    return [
        (left + bw / 2) / width,
        (top + bh / 2) / height,
        bw / width,
        bh / height,
    ]


def load_reviewed_samples(conn) -> list[SampleRow]:
    """Verified catalog samples (incl. confirmed-empty negatives) with their boxes."""
    rows = conn.execute(
        "SELECT * FROM samples WHERE status='verified' ORDER BY wall_ms,camera,sample_id"
    ).fetchall()
    samples: list[SampleRow] = []
    for row in rows:
        sample = SampleRow(
            sample_id=row["sample_id"], camera=row["camera"], wall_ms=int(row["wall_ms"]),
            pts=int(row["pts"]), width=int(row["width"]), height=int(row["height"]),
            image_relpath=row["image_relpath"], sha256=row["sha256"],
            dhash=int(row["dhash"], 16), tags=list(json.loads(row["tags_json"] or "[]")),
            rotate_deg=int(row["rotate_deg"] or 0),
            detect_roi=json.loads(row["detect_roi_json"]) if row["detect_roi_json"] else None,
        )
        for ann in conn.execute(
            "SELECT x,y,w,h FROM annotations WHERE sample_id=? ORDER BY annotation_id",
            (sample.sample_id,),
        ):
            box = _normalise_box(ann["x"], ann["y"], ann["w"], ann["h"],
                                 sample.width, sample.height)
            if box is not None:
                sample.boxes.append(box)
        samples.append(sample)
    return samples


def dedupe_exact(samples: list[SampleRow]) -> tuple[list[SampleRow], int]:
    """Drop later samples whose pixels are byte-identical, so no exact image can
    appear in two splits. Order is preserved; the earliest id wins."""
    seen: set[str] = set()
    kept: list[SampleRow] = []
    dropped = 0
    for sample in samples:
        if sample.sha256 in seen:
            dropped += 1
            continue
        seen.add(sample.sha256)
        kept.append(sample)
    return kept, dropped


def assign_groups(samples: list[SampleRow], gap_ms: int) -> None:
    """Cluster samples into visits by wall-clock gap across all cameras."""
    ordered = sorted(samples, key=lambda s: (s.wall_ms, s.camera, s.sample_id))
    group_start: int | None = None
    previous_wall: int | None = None
    for sample in ordered:
        if previous_wall is None or sample.wall_ms - previous_wall > gap_ms:
            group_start = sample.wall_ms
        sample.group_id = f"g{group_start:013d}"
        previous_wall = sample.wall_ms


def _group_fraction(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def assign_splits(samples: list[SampleRow], *, val_frac: float, test_frac: float,
                  seed: int) -> dict[str, str]:
    """Stable per-group split: a group's bucket depends only on its id and seed."""
    group_ids = sorted({sample.group_id for sample in samples})
    assignment: dict[str, str] = {}
    for group_id in group_ids:
        u = _group_fraction(group_id, seed)
        if u < test_frac:
            assignment[group_id] = "test"
        elif u < test_frac + val_frac:
            assignment[group_id] = "val"
        else:
            assignment[group_id] = "train"
    for sample in samples:
        sample.split = assignment[sample.group_id]
    return assignment


def cross_split_near_duplicates(samples: list[SampleRow], threshold: int) -> int:
    """Count near-duplicate image pairs (dhash) that landed in different splits."""
    pairs = 0
    for i, left in enumerate(samples):
        for right in samples[i + 1:]:
            if left.split != right.split and hamming(left.dhash, right.dhash) <= threshold:
                pairs += 1
    return pairs


def build_version(
    catalog: Path,
    root: Path,
    out: Path,
    *,
    val_frac: float,
    test_frac: float,
    group_gap_sec: float,
    dup_threshold: int,
    min_groups: int,
    seed: int,
    version_id: str | None = None,
) -> dict[str, Any]:
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val/test fractions must be in [0,1) and sum below 1")
    conn = open_catalog(catalog)
    try:
        samples = load_reviewed_samples(conn)
        if not samples:
            raise BuildError("no verified samples in the catalog; import a reviewed batch first")

        # Only samples whose stored image is still on disk can be materialised.
        present, missing = [], []
        for sample in samples:
            (present if (root / sample.image_relpath).is_file() else missing).append(sample)
        if not present:
            raise BuildError("every verified sample is missing its stored image")

        present, exact_dropped = dedupe_exact(present)
        assign_groups(present, int(group_gap_sec * 1000))
        assign_splits(present, val_frac=val_frac, test_frac=test_frac, seed=seed)
        near_dups = cross_split_near_duplicates(present, dup_threshold)

        per_split = {split: [s for s in present if s.split == split] for split in SPLITS}
        group_count = len({s.group_id for s in present})
        warnings: list[str] = []
        if missing:
            warnings.append(f"{len(missing)} verified samples skipped: stored image missing")
        if exact_dropped:
            warnings.append(f"{exact_dropped} exact-duplicate images de-duplicated before split")
        if near_dups:
            warnings.append(f"{near_dups} near-duplicate image pairs span different splits")
        empty = [split for split in SPLITS if not per_split[split]]
        if empty:
            warnings.append(
                f"splits with no samples: {', '.join(empty)} — honest evaluation is not "
                "possible yet; collect/review more independent visits"
            )
        if group_count < min_groups:
            warnings.append(
                f"only {group_count} independent visit groups (< {min_groups}); "
                "train/val/test overlap in appearance and metrics will be optimistic"
            )

        version = version_id or (
            datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
            + "-" + hashlib.sha256(
                "".join(sorted(s.sample_id for s in present)).encode()
            ).hexdigest()[:8]
        )
        version_dir = (out / version).resolve()
        if version_dir.exists():
            raise BuildError(f"refusing to overwrite existing version: {version_dir}")

        manifest_samples: list[dict[str, Any]] = []
        for sample in sorted(present, key=lambda s: s.sample_id):
            image_dest = version_dir / "images" / sample.split / f"{sample.sample_id}.jpg"
            label_dest = version_dir / "labels" / sample.split / f"{sample.sample_id}.txt"
            link_or_copy(root / sample.image_relpath, image_dest)
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            label_dest.write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        for cx, cy, w, h in sample.boxes),
                encoding="utf-8",
            )
            manifest_samples.append({
                "sample_id": sample.sample_id,
                "split": sample.split,
                "group_id": sample.group_id,
                "camera": sample.camera,
                "pts": sample.pts,
                "wall_ms": sample.wall_ms,
                "width": sample.width,
                "height": sample.height,
                "boxes": sample.boxes,
                "tags": sorted(sample.tags),
                "sha256": sample.sha256,
                "geometry": {"rotate_deg": sample.rotate_deg, "detect_roi": sample.detect_roi},
            })

        manifest_hash = hashlib.sha256(
            json.dumps(manifest_samples, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        summary = {
            "images": len(manifest_samples),
            "objects": sum(len(s["boxes"]) for s in manifest_samples),
            "negative": sum(1 for s in manifest_samples if not s["boxes"]),
            "groups": group_count,
            "splits": {
                split: {
                    "images": len(per_split[split]),
                    "objects": sum(len(s.boxes) for s in per_split[split]),
                    "negative": sum(1 for s in per_split[split] if not s.boxes),
                    "groups": len({s.group_id for s in per_split[split]}),
                }
                for split in SPLITS
            },
            "exact_duplicates_removed": exact_dropped,
            "cross_split_near_duplicate_pairs": near_dups,
            "warnings": warnings,
        }
        manifest = {
            "version_id": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": manifest_hash,
            "parameters": {
                "val_frac": val_frac, "test_frac": test_frac,
                "group_gap_sec": group_gap_sec, "dup_threshold": dup_threshold,
                "seed": seed,
            },
            "summary": summary,
            "samples": manifest_samples,
        }
        atomic_write_json(version_dir / "manifest.json", manifest)
        _write_data_yaml(version_dir)

        files_sha256 = sha256_tree(version_dir / "labels")
        record_version(conn, version, version_dir, manifest_hash, present)

        print(f"built dataset version {version} at {version_dir}")
        print(f"  images={summary['images']} objects={summary['objects']} "
              f"negatives={summary['negative']} groups={summary['groups']}")
        print("  splits: " + ", ".join(
            f"{split}={summary['splits'][split]['images']}" for split in SPLITS))
        for warning in warnings:
            print(f"  WARNING: {warning}")
        return {"version_id": version, "path": str(version_dir),
                "manifest_sha256": manifest_hash, "labels_sha256": files_sha256,
                "summary": summary}
    finally:
        conn.close()


def _write_data_yaml(version_dir: Path) -> None:
    (version_dir / "data.yaml").write_text(
        "\n".join([
            f"path: {version_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: cat",
            "",
        ]),
        encoding="utf-8",
    )


def record_version(conn, version_id: str, version_dir: Path, manifest_hash: str,
                   samples: list[SampleRow]) -> None:
    now = int(time.time() * 1000)
    conn.execute(
        "INSERT INTO dataset_versions(version_id,manifest_sha256,created_at_ms,path) "
        "VALUES(?,?,?,?)",
        (version_id, manifest_hash, now, str(version_dir)),
    )
    conn.executemany(
        "INSERT INTO dataset_version_samples(version_id,sample_id,split,group_id) "
        "VALUES(?,?,?,?)",
        ((version_id, s.sample_id, s.split, s.group_id) for s in samples),
    )
    conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/yolo_dataset/catalog.sqlite3"))
    parser.add_argument("--root", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--out", type=Path, default=None,
                        help="versions directory (default: <root>/versions)")
    parser.add_argument("--version-id", default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--group-gap-sec", type=float, default=120.0,
                        help="wall-clock gap that starts a new visit group")
    parser.add_argument("--dup-threshold", type=int, default=3,
                        help="dhash Hamming distance counted as a near-duplicate")
    parser.add_argument("--min-groups", type=int, default=6,
                        help="warn below this many independent visit groups")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or (args.root / "versions")
    build_version(
        args.catalog, args.root, out,
        val_frac=args.val_frac, test_frac=args.test_frac,
        group_gap_sec=args.group_gap_sec, dup_threshold=args.dup_threshold,
        min_groups=args.min_groups, seed=args.seed, version_id=args.version_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
