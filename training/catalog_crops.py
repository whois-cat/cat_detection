"""Identity-classifier crops cut from the YOLO catalog's human-verified boxes.

Replaces the previous stack's events.db + recordings as the classifier's crop
source: every box confirmed in Label Studio (``just detector-sync``) becomes one
crop, labelled with the cat name picked for that box (``annotations.cat_name``). Catalog frames are already stored in the detector's input geometry (ROI
crop + rotation, see ``streamhub_dataset``), so a crop is cut straight from the
frame JPEG with the runtime padding and needs no further rotation — the same
framing cv-worker hands the classifier.

A crop's ``key`` (the training code's ``src_event_key``) is a stable hash of
``(sample_id, annotation_id)``.

    python -m training.catalog_crops --catalog data/yolo_dataset/catalog.sqlite3
prints how many boxes each cat has (``just classifier-stats``).
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .db import Box
from .sources import CropUnavailable, _pad_crop

CAT_CLASS = "cat"


def crop_key(sample_id: str, annotation_id: int) -> int:
    """Stable positive 60-bit int for one verified box (fits SQLite INTEGER)."""
    digest = hashlib.sha1(f"{sample_id}:{annotation_id}".encode()).hexdigest()
    return int(digest[:15], 16)


@dataclass(frozen=True, slots=True)
class CatalogCrop:
    key: int
    sample_id: str
    annotation_id: int
    camera: str
    wall_ms: int
    visit_group: str | None
    image_relpath: str
    box: tuple[int, int, int, int]   # x, y, w, h in frame pixels
    name: str | None = None          # cat name picked in Label Studio; None = not named

    @property
    def crop_id(self) -> str:
        return f"{self.sample_id}:{self.annotation_id}"


@dataclass(frozen=True, slots=True)
class CatalogCropRef:
    """Training-time crop ref (same ``camera_id/wall_ms/box`` surface as the
    recordings ``CropRefLite``, so split/leakage code reads either)."""
    camera_id: str
    wall_ms: int
    box: Box          # frame pixels; rowid == src_event_key
    image_relpath: str
    rotate_deg: int = 0


def open_catalog_ro(path: Path) -> sqlite3.Connection:
    if not Path(path).is_file():
        raise SystemExit(f"YOLO catalog not found: {path} (run `just detector-collect` first)")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_catalog_crops(catalog: Path, *, camera: str | None = None) -> list[CatalogCrop]:
    """Every human-verified cat box, ordered by camera then time."""
    conn = open_catalog_ro(catalog)
    try:
        # A catalog not yet opened read-write since names were added has no names.
        named = "cat_name" in {r[1] for r in conn.execute("PRAGMA table_info(annotations)")}
        rows = conn.execute(
            f"""SELECT s.sample_id, s.camera, s.wall_ms, s.visit_group, s.image_relpath,
                      a.annotation_id, a.x, a.y, a.w, a.h,
                      {'a.cat_name' if named else 'NULL'} AS cat_name
                 FROM annotations a JOIN samples s USING(sample_id)
                WHERE s.status='verified' AND a.class_name=?
                  AND (? IS NULL OR s.camera=?)
                ORDER BY s.camera, s.wall_ms, a.annotation_id""",
            (CAT_CLASS, camera, camera),
        ).fetchall()
    finally:
        conn.close()
    crops = []
    for row in rows:
        x, y = int(round(row["x"])), int(round(row["y"]))
        w, h = int(round(row["w"])), int(round(row["h"]))
        if w <= 0 or h <= 0:
            continue
        crops.append(CatalogCrop(
            key=crop_key(row["sample_id"], int(row["annotation_id"])),
            sample_id=row["sample_id"], annotation_id=int(row["annotation_id"]),
            camera=row["camera"], wall_ms=int(row["wall_ms"]),
            visit_group=row["visit_group"], image_relpath=row["image_relpath"],
            box=(x, y, w, h), name=row["cat_name"],
        ))
    return crops


def cut_crop(image_bgr: np.ndarray, box: tuple[int, int, int, int],
             pad_frac: float) -> np.ndarray:
    x, y, w, h = box
    crop, _local = _pad_crop(
        image_bgr, Box(x=x, y=y, w=w, h=h, cat=None, score=1.0, track_id=None), pad_frac,
    )
    if crop is None:
        raise CropUnavailable(f"degenerate crop for box {box}")
    return crop


def read_catalog_crop(root: Path, image_relpath: str, box: tuple[int, int, int, int],
                      pad_frac: float) -> np.ndarray:
    """BGR crop from a catalog frame. Raises CropUnavailable if the JPEG is gone."""
    from PIL import Image   # Pillow, not cv2: the review app has no OpenCV

    try:
        with Image.open(Path(root) / image_relpath) as img:
            rgb = np.asarray(img.convert("RGB"))
    except OSError as exc:
        raise CropUnavailable(f"catalog frame missing or unreadable: {image_relpath}") from exc
    return cut_crop(np.ascontiguousarray(rgb[..., ::-1]), box, pad_frac)


def name_counts(crops: list[CatalogCrop]) -> str:
    """Per-camera box counts for each cat name (unnamed boxes shown as '-')."""
    counts = Counter((c.name or "-", c.camera) for c in crops)
    cameras = sorted({c.camera for c in crops})
    names = sorted({n for n, _ in counts}, key=lambda n: (n == "-", n == "unknown", n))
    lines = [f"{'name':<10}{'total':>7}" + "".join(f"{cam:>9}" for cam in cameras)]
    for name in names:
        per = [counts[(name, cam)] for cam in cameras]
        lines.append(f"{name:<10}{sum(per):>7}" + "".join(f"{n:>9}" for n in per))
    lines.append("('-' = verified box without a name yet; 'unknown' is not trained on)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cat-name counts of the verified catalog boxes.")
    ap.add_argument("--catalog", type=Path, default=Path("data/yolo_dataset/catalog.sqlite3"))
    args = ap.parse_args(argv)
    print(name_counts(load_catalog_crops(args.catalog)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
