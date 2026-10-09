"""COCO/CVAT export and validated import for the YOLO review queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

from .streamhub_dataset import derived_storage_bytes, open_catalog


def coco_image_id(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest()[:12], 16)


def _manifest_entry(row) -> dict:
    return {
        "sample_id": row["sample_id"],
        "file_name": f"{row['sample_id']}.jpg",
        "sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
        "camera": row["camera"],
        "pts": row["pts"],
        "wall_ms": row["wall_ms"],
    }


def _manifest_hash(samples: list[dict]) -> str:
    payload = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _publish_zip_without_overwrite(temporary: Path, output: Path) -> None:
    """Atomically publish a complete archive while refusing an existing path."""
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing export: {output}") from exc


def export_batch(catalog: Path, root: Path, output: Path | None, limit: int,
                 include_suggestions: bool = True) -> dict:
    if limit <= 0:
        raise ValueError("export limit must be greater than zero")
    # When no name is given, auto-name later as <root>/exports/<batch_id>.zip so
    # operators never have to invent a unique filename per batch.
    export_dir = output.parent if output is not None else (root / "exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    if output is not None and output.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".export.", suffix=".tmp", dir=export_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    conn = open_catalog(catalog)
    published = False
    transaction_started = False
    try:
        rows = conn.execute(
            "SELECT * FROM samples WHERE status='unreviewed' "
            "ORDER BY wall_ms,camera,sample_id LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            raise RuntimeError("no unreviewed samples available")
        images, annotations = [], []
        manifest_samples = [_manifest_entry(row) for row in rows]
        ann_id = 1
        for row in rows:
            sid = row["sample_id"]
            image_id = coco_image_id(sid)
            images.append({
                "id": image_id, "file_name": f"{sid}.jpg",
                "width": row["width"], "height": row["height"],
            })
            if include_suggestions:
                pred = conn.execute(
                    "SELECT detections_json FROM predictions WHERE sample_id=?", (sid,)
                ).fetchone()
                for det in json.loads(pred[0]) if pred else []:
                    try:
                        x, y, w, h = (float(v) for v in det["box"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    box = [x * row["width"], y * row["height"],
                           w * row["width"], h * row["height"]]
                    if not all(math.isfinite(value) for value in box) or w <= 0 or h <= 0:
                        continue
                    annotations.append({
                        "id": ann_id, "image_id": image_id, "category_id": 1,
                        "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                        "attributes": {"source": "model_suggestion"},
                    })
                    ann_id += 1
        manifest_hash = _manifest_hash(manifest_samples)
        batch_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + manifest_hash[:8]
        if output is None:
            output = export_dir / f"{batch_id}.zip"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing export: {output}")
        manifest = {
            "schema": 1, "batch_id": batch_id,
            "manifest_sha256": manifest_hash, "samples": manifest_samples,
        }
        coco = {
            "info": {"description": "cat_detection review; suggestions are not truth"},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
        }
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            archive.writestr(
                "annotations/instances_default.json", json.dumps(coco, indent=2) + "\n"
            )
            for row in rows:
                source = root / row["image_relpath"]
                if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                    raise RuntimeError(f"missing or changed source image for {row['sample_id']}")
                archive.write(source, f"images/{row['sample_id']}.jpg")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())

        conn.execute("BEGIN IMMEDIATE")
        transaction_started = True
        current = {
            row["sample_id"]: row
            for row in conn.execute(
                f"SELECT sample_id,status,protected FROM samples WHERE sample_id IN "
                f"({','.join('?' for _ in rows)})",
                tuple(row["sample_id"] for row in rows),
            )
        }
        if len(current) != len(rows) or any(
            current[row["sample_id"]]["status"] != "unreviewed" for row in rows
        ):
            raise RuntimeError("review queue changed during export; retry with a new output path")
        now = int(time.time() * 1000)
        conn.execute(
            "INSERT INTO review_batches(batch_id,created_at_ms,manifest_sha256,status) "
            "VALUES(?,?,?,?)",
            (batch_id, now, manifest_hash, "exported"),
        )
        conn.executemany(
            "INSERT INTO review_batch_samples(batch_id,sample_id,was_protected) VALUES(?,?,?)",
            ((batch_id, row["sample_id"], int(current[row["sample_id"]]["protected"]))
             for row in rows),
        )
        conn.executemany(
            "UPDATE samples SET status='exported',protected=1 WHERE sample_id=?",
            ((row["sample_id"],) for row in rows),
        )
        _publish_zip_without_overwrite(temporary, output)
        published = True
        conn.commit()
        transaction_started = False
        return {
            "batch_id": batch_id, "images": len(rows), "suggestions": len(annotations),
            "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    except BaseException:
        if transaction_started:
            conn.rollback()
        if published:
            output.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
        temporary.unlink(missing_ok=True)


def _read_package(path: Path) -> tuple[dict | None, dict]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in names else None
            annotation_names = [name for name in names if name.endswith(".json") and "instances" in name]
            if not annotation_names:
                raise ValueError("CVAT zip has no COCO instances JSON")
            coco = json.loads(archive.read(annotation_names[0]))
        return manifest, coco
    raise ValueError("CVAT import must be the exported COCO zip containing manifest.json")


def _index_coco_images(coco: dict) -> tuple[dict[str, dict], set[int]]:
    images: dict[str, dict] = {}
    image_ids: set[int] = set()
    for image in coco.get("images", []):
        try:
            sample_id = Path(str(image["file_name"])).stem
            image_id = int(image["id"])
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid COCO image entry") from exc
        if not sample_id or sample_id in images:
            raise ValueError(f"duplicate or empty COCO sample id: {sample_id!r}")
        if image_id in image_ids:
            raise ValueError(f"duplicate COCO image id: {image_id}")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions for {sample_id}")
        images[sample_id] = image
        image_ids.add(image_id)
    return images, image_ids


def _validate_manifest(manifest: dict) -> tuple[str, list[dict], str]:
    try:
        batch_id = str(manifest["batch_id"])
        samples = list(manifest["samples"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid review manifest") from exc
    sample_ids = [str(row.get("sample_id", "")) for row in samples]
    if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample IDs must be non-empty and unique")
    actual_hash = _manifest_hash(samples)
    declared_hash = manifest.get("manifest_sha256")
    if declared_hash is not None and declared_hash != actual_hash:
        raise ValueError("review manifest hash does not match its samples")
    return batch_id, samples, actual_hash


def import_batch(catalog: Path, package: Path, confirm_empty: bool,
                 requested_batch_id: str | None = None) -> dict:
    manifest, coco = _read_package(package)
    images, known_image_ids = _index_coco_images(coco)
    manifest_samples: list[dict] | None = None
    manifest_hash: str | None = None
    if manifest is not None:
        batch_id, manifest_samples, manifest_hash = _validate_manifest(manifest)
        if requested_batch_id and requested_batch_id != batch_id:
            raise ValueError("--batch-id does not match manifest batch_id")
    elif requested_batch_id:
        batch_id = requested_batch_id
    else:
        batch_id = ""

    categories: dict[int, str] = {}
    for category in coco.get("categories", []):
        try:
            category_id = int(category["id"])
            category_name = str(category["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid COCO category entry") from exc
        categories[category_id] = category_name
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        try:
            image_id = int(ann["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("annotation has invalid image_id") from exc
        if image_id not in known_image_ids:
            raise ValueError(f"annotation refers to unknown image id {image_id}")
        by_image.setdefault(image_id, []).append(ann)

    conn = open_catalog(catalog)
    try:
        if not batch_id:
            matches = []
            for row in conn.execute(
                "SELECT batch_id FROM review_batches WHERE status='exported'"
            ):
                ids = {item[0] for item in conn.execute(
                    "SELECT sample_id FROM review_batch_samples WHERE batch_id=?", (row[0],)
                )}
                if ids == set(images):
                    matches.append(row[0])
            if len(matches) != 1:
                raise ValueError("cannot uniquely match CVAT files to a batch; pass --batch-id")
            batch_id = matches[0]
        batch = conn.execute(
            "SELECT manifest_sha256,status FROM review_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError(f"unknown batch {batch_id}")
        if batch["status"] != "exported":
            raise ValueError(f"batch {batch_id} is {batch['status']}, not exported")
        expected_rows = conn.execute(
            """SELECT s.* FROM samples s JOIN review_batch_samples b
                 ON b.sample_id=s.sample_id
               WHERE b.batch_id=? ORDER BY s.wall_ms,s.camera,s.sample_id""",
            (batch_id,),
        ).fetchall()
        expected = {row["sample_id"]: row for row in expected_rows}
        if set(images) != set(expected):
            raise ValueError("COCO image set does not exactly match the exported batch")

        expected_manifest = [_manifest_entry(row) for row in expected_rows]
        expected_hash = _manifest_hash(expected_manifest)
        if expected_hash != batch["manifest_sha256"]:
            raise ValueError("stored batch manifest hash no longer matches catalog samples")
        if manifest_samples is not None:
            if manifest_hash != batch["manifest_sha256"]:
                raise ValueError("review manifest hash does not match the exported batch")
            if manifest_samples != expected_manifest:
                raise ValueError("review manifest content does not match catalog samples")

        now = int(time.time() * 1000)
        object_count = empty_count = 0
        for sid, expected_row in expected.items():
            image = images[sid]
            if (int(image["width"]), int(image["height"])) != (
                int(expected_row["width"]), int(expected_row["height"])
            ):
                raise ValueError(f"dimension mismatch for {sid}")
            anns = by_image.get(int(image["id"]), [])
            if not anns and not confirm_empty:
                raise ValueError(
                    f"{sid} has no boxes; pass --confirm-empty only after a human reviewed all empty frames"
                )
            conn.execute("DELETE FROM annotations WHERE sample_id=?", (sid,))
            for index, ann in enumerate(anns, 1):
                try:
                    category_id = int(ann["category_id"])
                    bbox = ann["bbox"]
                    if len(bbox) != 4:
                        raise ValueError
                    x, y, w, h = (float(v) for v in bbox)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid bbox for {sid}") from exc
                if str(categories.get(category_id, "")).lower() != "cat":
                    raise ValueError(f"unsupported category for {sid}; only cat is allowed")
                if (
                    not all(math.isfinite(value) for value in (x, y, w, h))
                    or w <= 0 or h <= 0 or x < 0 or y < 0
                    or x + w > image["width"] + 0.01
                    or y + h > image["height"] + 0.01
                ):
                    raise ValueError(f"invalid bbox for {sid}: {bbox}")
                conn.execute(
                    "INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?)",
                    (sid, index, "cat", x, y, w, h, "human_cvat", now),
                )
                object_count += 1
            empty_count += int(not anns)
            conn.execute("UPDATE samples SET status='verified',protected=1 WHERE sample_id=?", (sid,))
        conn.execute("UPDATE review_batches SET status='imported' WHERE batch_id=?", (batch_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"batch_id": batch_id, "images": len(expected), "objects": object_count,
            "confirmed_empty": empty_count}


def requeue_batch(catalog: Path, batch_id: str) -> dict:
    """Safely return a lost/abandoned exported batch to the review queue."""
    conn = open_catalog(catalog)
    try:
        conn.execute("BEGIN IMMEDIATE")
        batch = conn.execute(
            "SELECT status FROM review_batches WHERE batch_id=?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError(f"unknown batch {batch_id}")
        if batch["status"] != "exported":
            raise ValueError(f"batch {batch_id} is {batch['status']}, not exported")
        rows = conn.execute(
            """SELECT s.sample_id,s.status,b.was_protected
                 FROM review_batch_samples b JOIN samples s ON s.sample_id=b.sample_id
                WHERE b.batch_id=?""",
            (batch_id,),
        ).fetchall()
        if not rows or any(row["status"] != "exported" for row in rows):
            raise ValueError("batch samples are no longer all in exported state")
        sample_ids = tuple(row["sample_id"] for row in rows)
        placeholders = ",".join("?" for _ in sample_ids)
        if conn.execute(
            f"SELECT 1 FROM annotations WHERE sample_id IN ({placeholders}) LIMIT 1",
            sample_ids,
        ).fetchone():
            raise ValueError("cannot requeue a batch that already has imported annotations")
        if conn.execute(
            f"SELECT 1 FROM dataset_version_samples WHERE sample_id IN ({placeholders}) LIMIT 1",
            sample_ids,
        ).fetchone():
            raise ValueError("cannot requeue samples used by a dataset version")
        for row in rows:
            conn.execute(
                "UPDATE samples SET status='unreviewed',protected=? WHERE sample_id=?",
                (int(row["was_protected"]), row["sample_id"]),
            )
        conn.execute("UPDATE review_batches SET status='requeued' WHERE batch_id=?", (batch_id,))
        conn.commit()
        return {"batch_id": batch_id, "status": "requeued", "images": len(rows)}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def queue_status(catalog: Path, root: Path, budget_gb: float = 5.0) -> dict:
    """Read queue composition and storage usage without changing the catalog."""
    if not math.isfinite(budget_gb) or budget_gb <= 0:
        raise ValueError("status budget must be a finite value greater than zero")
    uri = f"file:{catalog.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        by_status = {
            row["status"]: int(row["count"])
            for row in conn.execute(
                "SELECT status,count(*) count FROM samples GROUP BY status ORDER BY status"
            )
        }
        by_camera = {
            row["camera"]: int(row["count"])
            for row in conn.execute(
                "SELECT camera,count(*) count FROM samples GROUP BY camera ORDER BY camera"
            )
        }
        reasons: dict[str, int] = {}
        for row in conn.execute("SELECT reasons_json FROM samples"):
            for reason in set(json.loads(row[0]) or []):
                reasons[reason] = reasons.get(reason, 0) + 1
        summary = conn.execute(
            """SELECT count(*) count,coalesce(sum(jpeg_bytes),0) raw_bytes,
                      coalesce(sum(CASE WHEN protected=1 OR status!='unreviewed'
                                        THEN jpeg_bytes ELSE 0 END),0) protected_raw_bytes,
                      min(wall_ms) oldest_ms,max(wall_ms) newest_ms
                 FROM samples"""
        ).fetchone()
    finally:
        conn.close()
    cache_bytes = derived_storage_bytes(root)
    total_bytes = int(summary["raw_bytes"]) + cache_bytes
    budget_bytes = int(budget_gb * 1_000_000_000)
    return {
        "images": int(summary["count"]),
        "by_status": by_status,
        "by_camera": by_camera,
        "by_reason": dict(sorted(reasons.items())),
        "raw_bytes": int(summary["raw_bytes"]),
        "cache_version_bytes": cache_bytes,
        "total_bytes": total_bytes,
        "protected_bytes": int(summary["protected_raw_bytes"]) + cache_bytes,
        "budget_bytes": budget_bytes,
        "remaining_bytes": max(0, budget_bytes - total_bytes),
        "over_budget": total_bytes > budget_bytes,
        "oldest_wall_ms": summary["oldest_ms"],
        "newest_wall_ms": summary["newest_ms"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/yolo_dataset/catalog.sqlite3"))
    parser.add_argument("--root", type=Path, default=Path("data/yolo_dataset"))
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, default=None,
                        help="output zip (default: <root>/exports/<batch_id>.zip)")
    export.add_argument("--limit", type=int, default=100)
    export.add_argument("--no-suggestions", action="store_true")
    imp = sub.add_parser("import")
    imp.add_argument("package", type=Path)
    imp.add_argument("--confirm-empty", action="store_true")
    imp.add_argument("--batch-id")
    requeue = sub.add_parser("requeue")
    requeue.add_argument("--batch-id", required=True)
    status = sub.add_parser("status")
    status.add_argument("--budget-gb", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_batch(args.catalog, args.root, args.out, args.limit, not args.no_suggestions)
    elif args.command == "import":
        result = import_batch(args.catalog, args.package, args.confirm_empty, args.batch_id)
    elif args.command == "requeue":
        result = requeue_batch(args.catalog, args.batch_id)
    else:
        result = queue_status(args.catalog, args.root, args.budget_gb)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
