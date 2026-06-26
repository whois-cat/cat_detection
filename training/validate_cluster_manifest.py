"""Validate a review cluster manifest (data/review/clusters.json).

Fails (exit 1) unless the manifest is the compact, review-exact shape that
``build_cluster_manifest`` produces: ``items`` + ``clusters`` present, every
cluster hard-capped, all ``item_indices`` valid, and no hidden/collapsed
sampling metadata leaking onto crop records.

Duplicate item indices across clusters are NOT allowed: a crop belongs to
exactly one cluster (k-means assigns one cluster per crop; time mode puts each
detection in one episode), so an index must be referenced by at most one
cluster. The validator therefore flags any cross-cluster duplicate reference.

Usage:
    python -m training.validate_cluster_manifest \
        --manifest data/review/clusters.json --max-cluster-size 16
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Fields that must NOT appear on crop records in a new manifest — the manifest
# is exactly what the review UI shows and labels, with nothing hidden/collapsed.
FORBIDDEN_ITEM_FIELDS = (
    "review_visible",
    "is_duplicate",
    "duplicate_group_id",
    "representative_rank",
    "sampling_reason",
    "hidden",
    "collapsed",
)

# Cluster-level counters that implied a hidden/collapsed model — not in new ones.
LEGACY_CLUSTER_FIELDS = ("visible_count", "hidden_duplicate_count")


def validate(manifest, *, max_cluster_size: int) -> tuple[list[str], dict, list[str]]:
    """Return (errors, stats, warnings). Empty errors == valid."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"], {}, warnings

    items = manifest.get("items")
    clusters = manifest.get("clusters")
    if not isinstance(items, list):
        errors.append("manifest missing 'items' list")
    if not isinstance(clusters, list):
        errors.append("manifest missing 'clusters' list")
    if errors:
        return errors, {}, warnings

    n_items = len(items)
    index_owner: dict[int, object] = {}     # item index -> first cluster that used it
    dup_refs = 0
    bad_index = 0
    oversized: list[tuple] = []
    size_mismatch: list[tuple] = []

    for ci, cluster in enumerate(clusters):
        cid = cluster.get("cluster_id", ci) if isinstance(cluster, dict) else ci
        if not isinstance(cluster, dict):
            errors.append(f"cluster #{ci}: not an object")
            continue
        idxs = cluster.get("item_indices")
        if not isinstance(idxs, list):
            errors.append(f"cluster {cid}: missing 'item_indices' list")
            continue

        if max_cluster_size and len(idxs) > max_cluster_size:
            oversized.append((cid, len(idxs)))

        size = cluster.get("size")
        if size is not None and size != len(idxs):
            size_mismatch.append((cid, size, len(idxs)))

        for idx in idxs:
            if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx >= n_items:
                bad_index += 1
                continue
            if idx in index_owner:
                dup_refs += 1
            else:
                index_owner[idx] = cid

        for field in LEGACY_CLUSTER_FIELDS:
            if field in cluster:
                warnings.append(f"cluster {cid}: legacy field {field!r} present")

    if oversized:
        errors.append(
            f"{len(oversized)} cluster(s) exceed --max-cluster-size "
            f"{max_cluster_size}: {oversized[:5]}"
        )
    if size_mismatch:
        errors.append(
            f"{len(size_mismatch)} cluster(s) where size != len(item_indices) "
            f"(cluster, size, actual): {size_mismatch[:5]}"
        )
    if bad_index:
        errors.append(f"{bad_index} item index reference(s) out of range [0,{n_items})")
    if dup_refs:
        errors.append(
            f"{dup_refs} item index reference(s) duplicated across clusters "
            "(each crop must belong to exactly one cluster)"
        )

    field_hits: Counter = Counter()
    for it in items:
        if isinstance(it, dict):
            for field in FORBIDDEN_ITEM_FIELDS:
                if field in it:
                    field_hits[field] += 1
    if field_hits:
        errors.append(
            "items contain forbidden hidden/collapsed field(s): "
            + ", ".join(f"{f}×{n}" for f, n in sorted(field_hits.items()))
        )

    unreferenced = n_items - len(index_owner)
    if unreferenced:
        warnings.append(f"{unreferenced} item(s) not referenced by any cluster")

    sizes = [len(c["item_indices"]) for c in clusters
             if isinstance(c, dict) and isinstance(c.get("item_indices"), list)]
    stats = {
        "clusters": len(clusters),
        "items": n_items,
        "max_cluster_size_seen": max(sizes) if sizes else 0,
        "avg_cluster_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "suspicious_fields": dict(field_hits),
    }
    return errors, stats, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/review/clusters.json"))
    ap.add_argument("--max-cluster-size", type=int, default=16,
                    help="hard cap each cluster must respect (default 16)")
    args = ap.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"FAIL: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"FAIL: manifest is not valid JSON: {exc}", file=sys.stderr)
        return 2

    errors, stats, warnings = validate(manifest, max_cluster_size=args.max_cluster_size)

    print(f"manifest: {args.manifest}")
    if stats:
        print(f"clusters:            {stats['clusters']}")
        print(f"items:               {stats['items']}")
        print(f"max cluster size:    {stats['max_cluster_size_seen']} (cap {args.max_cluster_size})")
        print(f"average cluster size:{stats['avg_cluster_size']:>6}")
        print(f"suspicious fields:   {stats['suspicious_fields'] or 'none'}")
    for w in warnings:
        print(f"WARN: {w}")

    if errors:
        print(f"\nFAIL ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK: manifest is compact, hard-capped, and free of hidden/collapsed fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
