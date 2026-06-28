"""Build a cold-start cluster manifest for bulk cat-label review.

It does NOT use an existing cat identity classifier. It only trusts the
detector box score ("there is probably a cat here"), decodes the crops, builds
visual or generic ImageNet embeddings, clusters them, and writes metadata for
the cluster-review UI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class VisualExtractor:
    name = "visual"

    def encode_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [visual_feature(img) for img in images]


class EfficientNetExtractor:
    name = "efficientnet-b0-imagenet"

    def __init__(self, *, batch_size: int, allow_download: bool) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError("torch/torchvision/Pillow are not installed") from exc

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        if not allow_download and not _torchvision_weights_cached(weights):
            raise RuntimeError(
                "EfficientNet ImageNet weights are not cached; re-run with "
                "--allow-download or use --embedding visual"
            )

        model = efficientnet_b0(weights=weights, progress=False)
        self._torch = torch
        self._image = Image
        self._tf = weights.transforms()
        self._model = torch.nn.Sequential(model.features, model.avgpool)
        self._model.eval()

    def encode_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        tensors = []
        for img_bgr in images:
            rgb = np.ascontiguousarray(img_bgr[..., ::-1])
            tensors.append(self._tf(self._image.fromarray(rgb)))
        batch = self._torch.stack(tensors)
        with self._torch.inference_mode():
            y = self._model(batch)
            y = self._torch.flatten(y, 1)
        arr = y.cpu().numpy().astype(np.float32)
        return [row for row in arr]


def _torch_home() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"])
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "torch"


def _torchvision_weights_cached(weights) -> bool:
    url = getattr(weights, "url", "")
    if not url:
        return False
    return (_torch_home() / "hub" / "checkpoints" / url.rsplit("/", 1)[-1]).exists()


def build_extractor(kind: str, *, batch_size: int, allow_download: bool):
    if kind in {"auto", "efficientnet"}:
        try:
            return EfficientNetExtractor(batch_size=batch_size, allow_download=allow_download)
        except Exception as exc:
            if kind == "efficientnet":
                raise SystemExit(f"efficientnet embeddings unavailable: {exc}") from exc
            print(f"[cluster] efficientnet unavailable ({exc}); falling back to visual features")
    return VisualExtractor()


def visual_feature(img_bgr: np.ndarray) -> np.ndarray:
    """Small deterministic visual descriptor: texture + edges + color stats."""
    import cv2

    img = cv2.resize(img_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    small = (small - small.mean()) / (small.std() + 1e-6)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = np.floor(ang * (8.0 / (2.0 * math.pi))).astype(np.int32) % 8
    hog = []
    for y in range(0, 48, 12):
        for x in range(0, 48, 12):
            cell_bins = bins[y:y + 12, x:x + 12].ravel()
            cell_mag = mag[y:y + 12, x:x + 12].ravel()
            hist = np.bincount(cell_bins, weights=cell_mag, minlength=8).astype(np.float32)
            hist /= hist.sum() + 1e-6
            hog.append(hist)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).ravel()
    s_hist = cv2.calcHist([hsv], [1], None, [6], [0, 256]).ravel()
    v_hist = cv2.calcHist([hsv], [2], None, [6], [0, 256]).ravel()
    color = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color /= color.sum() + 1e-6

    return np.concatenate([small.ravel(), *hog, color]).astype(np.float32)


def standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sigma, 1e-6)


def prepare_embeddings(raw: np.ndarray, *, out_dim: int, seed: int) -> np.ndarray:
    """Standardize, optionally random-project, then L2-normalize features."""
    x = standardize(raw.astype(np.float32))
    if out_dim > 0 and x.shape[1] > out_dim:
        rng = np.random.default_rng(seed)
        proj = rng.normal(
            0.0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], out_dim)
        ).astype(np.float32)
        x = standardize(x @ proj)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-6)).astype(np.float32)


def dedupe_nearby(items: list[dict], x: np.ndarray | None, *, window_sec: float,
                  threshold: float) -> tuple[list[dict], np.ndarray | None, int]:
    """Drop near-identical crops close in time on the same camera.

    With embeddings (``x`` given), they are L2-normalized so the dot product is
    cosine similarity, and a crop is dropped only when it is BOTH within the
    per-camera window AND cosine-similar (>= ``threshold``) to a kept neighbour —
    conservative, so same-looking cats minutes apart survive.

    Without embeddings (``x is None``, the time-mode path), there is no similarity
    signal, so dedup is purely temporal: a crop within the per-camera window of a
    kept crop is treated as a duplicate. This thins rapid-fire detections to ~one
    per window while keeping the schema identical.
    """
    if window_sec <= 0 or (x is not None and threshold <= 0):
        return items, x, 0
    window_ms = int(window_sec * 1000)
    keep: list[int] = []
    recent_by_camera: dict[str, list[int]] = {}
    dropped = 0
    order = sorted(range(len(items)), key=lambda i: (items[i]["camera"], items[i]["wall_ms"]))
    for i in order:
        item = items[i]
        recent = recent_by_camera.setdefault(item["camera"], [])
        cutoff = item["wall_ms"] - window_ms
        recent[:] = [j for j in recent if items[j]["wall_ms"] >= cutoff]
        duplicate_of = None
        for j in recent:
            # x is None → time-only: any in-window neighbour is a duplicate.
            if x is None or float(x[i] @ x[j]) >= threshold:
                duplicate_of = j
                break
        if duplicate_of is not None:
            dropped += 1
            continue
        keep.append(i)
        recent.append(i)

    keep.sort()
    return [items[i] for i in keep], (None if x is None else x[keep]), dropped


def thin_cluster_members(
    members: list[int],
    items: list[dict],
    *,
    max_size: int,
    seed: int,
    cluster_id: int,
) -> tuple[list[int], int]:
    """Keep a compact, useful review sample from a large cluster.

    We keep center-near examples for fast bulk labelling, a few outliers so
    mixed clusters remain visible, and time-spread samples so a long static
    sequence does not collapse to only adjacent frames.
    """
    if max_size <= 0 or len(members) <= max_size:
        return members, 0

    by_distance = sorted(members, key=lambda i: float(items[i].get("distance", 0.0)))
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(indices: list[int]) -> None:
        for i in indices:
            if len(selected) >= max_size:
                return
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)

    center_quota = max(1, int(max_size * 0.55))
    outlier_quota = max(1, int(max_size * 0.15))
    add(by_distance[:center_quota])
    add(list(reversed(by_distance[-outlier_quota:])))

    remaining = [
        i for i in sorted(members, key=lambda j: (items[j]["camera"], items[j]["wall_ms"]))
        if i not in selected_set
    ]
    slots = max_size - len(selected)
    if slots > 0 and remaining:
        if len(remaining) <= slots:
            add(remaining)
        else:
            step = len(remaining) / slots
            spread = [remaining[min(len(remaining) - 1, int((n + 0.5) * step))]
                      for n in range(slots)]
            add(spread)

    if len(selected) < max_size:
        rng = np.random.default_rng(seed + cluster_id)
        rest = [i for i in members if i not in selected_set]
        if rest:
            shuffled = [rest[i] for i in rng.permutation(len(rest)).tolist()]
            add(shuffled)

    selected.sort(key=lambda i: float(items[i].get("distance", 0.0)))
    return selected, len(members) - len(selected)


def _is_near_duplicate(
    i: int,
    j: int,
    items: list[dict],
    x: np.ndarray | None,
    *,
    window_ms: int,
    threshold: float,
) -> bool:
    if items[i]["camera"] != items[j]["camera"]:
        return False
    if abs(int(items[i]["wall_ms"]) - int(items[j]["wall_ms"])) > window_ms:
        return False
    if x is None:
        return True
    return float(x[i] @ x[j]) >= threshold


# Crop-record fields that imply a hidden/collapsed model — never written.
REVIEW_ONLY_FIELDS = (
    "review_visible", "is_duplicate", "duplicate_group_id",
    "representative_rank", "sampling_reason", "suspicious_score",
    "suspicious_reasons", "hidden", "collapsed",
)
# Cluster-record fields that count shown/hidden crops — never written.
HIDDEN_CLUSTER_FIELDS = ("visible_count", "hidden_duplicate_count")


def _strip_review_fields(item: dict) -> None:
    """Drop legacy hidden/collapsed sampling fields from a crop record. New
    manifests carry only what the UI shows, so these must never be written."""
    for key in REVIEW_ONLY_FIELDS:
        item.pop(key, None)


def assert_review_manifest_invariants(items: list[dict], clusters: list[dict],
                                      max_per_cluster: int) -> None:
    """Fail loudly (before writing) if the manifest is not review-exact. Encodes
    the acceptance criteria so a non-compliant clusters.json can never be written:
    hard cap respected, compact in-range item_indices, size == len(item_indices),
    and no hidden/collapsed crop or cluster fields."""
    n = len(items)
    for item in items:
        bad = set(item) & set(REVIEW_ONLY_FIELDS)
        if bad:
            raise SystemExit(f"manifest invariant: item {item.get('crop_id')!r} "
                             f"still has hidden/collapsed field(s): {sorted(bad)}")
    seen: set[int] = set()
    for c in clusters:
        idxs = c["item_indices"]
        if max_per_cluster > 0 and len(idxs) > max_per_cluster:
            raise SystemExit(f"manifest invariant: cluster {c['cluster_id']} has "
                             f"{len(idxs)} crops > --max-cluster-size {max_per_cluster}")
        if c.get("size") != len(idxs):
            raise SystemExit(f"manifest invariant: cluster {c['cluster_id']} size "
                             f"{c.get('size')} != len(item_indices) {len(idxs)}")
        bad_c = set(c) & set(HIDDEN_CLUSTER_FIELDS)
        if bad_c:
            raise SystemExit(f"manifest invariant: cluster {c['cluster_id']} has "
                             f"shown/hidden counter(s): {sorted(bad_c)}")
        for i in idxs:
            if not (0 <= i < n):
                raise SystemExit(f"manifest invariant: cluster {c['cluster_id']} "
                                 f"item index {i} out of range [0,{n})")
            if i in seen:
                raise SystemExit(f"manifest invariant: item index {i} referenced by "
                                 "more than one cluster")
            seen.add(i)


def select_review_crops(
    items: list[dict],
    cluster_members: dict[int, list[int]],
    x: np.ndarray | None,
    *,
    duplicate_window_sec: float,
    duplicate_threshold: float,
    max_per_cluster: int,
    seed: int,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    """Pick the crops that actually get written to the manifest.

    Two real filters, both of which DROP crops (nothing is hidden-but-kept):
      1. dedupe — collapse near-duplicate neighbours to one representative;
      2. hard cap — at most ``max_per_cluster`` crops per cluster.

    Returns ``{cluster_id: [item index, ...]}`` (each list already <= the cap)
    plus drop counts. The manifest is exactly what the reviewer sees and labels.
    """
    window_ms = int(max(0.0, duplicate_window_sec) * 1000)
    selected_by_cluster: dict[int, list[int]] = {}
    dropped_dedupe = 0
    dropped_cap = 0

    for cluster_id, members in sorted(cluster_members.items()):
        members = list(members)
        if not members:
            selected_by_cluster[cluster_id] = []
            continue

        # 1) dedupe → keep one representative per near-duplicate group.
        ordered = sorted(members, key=lambda i: (
            items[i]["camera"], int(items[i]["wall_ms"]), items[i]["crop_id"]
        ))
        reps: list[int] = []
        for i in ordered:
            duplicate = False
            if window_ms > 0:
                for rep in reps:
                    if _is_near_duplicate(
                        i, rep, items, x,
                        window_ms=window_ms, threshold=duplicate_threshold,
                    ):
                        duplicate = True
                        break
            if not duplicate:
                reps.append(i)
        dropped_dedupe += len(members) - len(reps)

        # 2) hard cap per cluster.
        if max_per_cluster > 0 and len(reps) > max_per_cluster:
            kept, _ = thin_cluster_members(
                reps, items, max_size=max_per_cluster, seed=seed, cluster_id=cluster_id,
            )
            dropped_cap += len(reps) - len(kept)
            reps = kept

        reps.sort(key=lambda i: (float(items[i].get("distance", 0.0)), int(items[i]["wall_ms"])))
        selected_by_cluster[cluster_id] = reps

    return selected_by_cluster, {
        "dropped_dedupe": dropped_dedupe,
        "dropped_cap": dropped_cap,
    }


def compact_items_to_clusters(
    items: list[dict],
    clusters: list[dict],
) -> tuple[list[dict], list[dict]]:
    used = sorted({int(i) for cluster in clusters for i in cluster["item_indices"]})
    old_to_new = {old: new for new, old in enumerate(used)}
    compacted_items = [items[i] for i in used]
    compacted_clusters = []
    for cluster in clusters:
        c = dict(cluster)
        c["item_indices"] = [old_to_new[int(i)] for i in cluster["item_indices"]]
        c["representatives"] = [
            compacted_items[i]["crop_id"] for i in c["item_indices"][:24]
        ]
        c["size"] = len(c["item_indices"])
        compacted_clusters.append(c)
    return compacted_items, compacted_clusters


def assign_chunked(x: np.ndarray, centers: np.ndarray, *, chunk: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-center labels + squared distances in row chunks.

    Never materializes the full N×K distance matrix — only `chunk`×K at a time —
    so memory stays flat as N grows. float32 throughout; temporaries freed each
    chunk. Results are identical to the dense computation (chunking only changes
    *when* rows are processed, not the argmin)."""
    n = x.shape[0]
    step = max(1, int(chunk))
    labels = np.empty(n, dtype=np.int32)
    dists = np.empty(n, dtype=np.float32)
    c_norm = np.sum(centers * centers, axis=1, dtype=np.float32)   # (K,)
    # float32 SIMD matmul can raise spurious FP-exception warnings; the values
    # are fine (argmin is unaffected).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for s in range(0, n, step):
            xb = x[s:s + step]
            x_norm = np.sum(xb * xb, axis=1, dtype=np.float32)[:, None]  # (b,1)
            d = x_norm + c_norm[None, :] - 2.0 * (xb @ centers.T)        # (b,K)
            lb = np.argmin(d, axis=1).astype(np.int32)
            labels[s:s + step] = lb
            dists[s:s + step] = d[np.arange(lb.shape[0]), lb]
            del d, x_norm, lb
    np.maximum(dists, 0.0, out=dists)   # float round-off can dip below 0
    return labels, dists


def kmeans(x: np.ndarray, k: int, *, seed: int, max_iter: int,
           chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[0]
    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int32)

    for _ in range(max_iter):
        new_labels, _ = assign_chunked(x, centers, chunk=chunk)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        sums = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        np.add.at(sums, labels, x)
        empty = counts == 0
        centers = sums / np.maximum(counts[:, None], 1.0)
        if np.any(empty):
            centers[empty] = x[rng.choice(n, size=int(empty.sum()), replace=False)]
        del sums

    labels, assigned_dist = assign_chunked(x, centers, chunk=chunk)
    return labels, assigned_dist.astype(np.float32)


def build_time_episodes(items: list[dict], gap_ms: int) -> dict[int, list[int]]:
    """Group detections into time episodes PER CAMERA — one episode = one cluster.

    Sort by (camera, wall_ms); start a new episode whenever the gap to the
    previous detection of the SAME camera is >= ``gap_ms`` (or the camera
    changes). No embeddings, no torch — fast and fully deterministic. Returns
    ``{episode_id: [item_index, ...]}`` in chronological order, mirroring the
    ``cluster_members`` shape the embedding path produces so the downstream
    (renumber/thin/compact/write) is shared unchanged.

    The >= boundary matches the cluster-review acceptance test: a gap of exactly
    ``gap_ms`` splits. Same idea as the feeder's presence episodes (group by
    time gaps), just pinned to an inclusive boundary here.
    """
    order = sorted(range(len(items)), key=lambda i: (items[i]["camera"], items[i]["wall_ms"]))
    episodes: dict[int, list[int]] = {}
    episode_id = -1
    prev_camera: str | None = None
    prev_wall: int | None = None
    for i in order:
        camera = items[i]["camera"]
        wall = int(items[i]["wall_ms"])
        if (
            prev_camera is None
            or camera != prev_camera
            or wall - prev_wall >= gap_ms
        ):
            episode_id += 1
            episodes[episode_id] = []
        episodes[episode_id].append(i)
        prev_camera = camera
        prev_wall = wall
    return episodes


def default_cluster_count(n: int) -> int:
    if n <= 0:
        return 0
    return max(8, min(96, round(math.sqrt(n / 4))))



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True, help="events.db")
    ap.add_argument("--recordings", type=Path, required=True, help="data/recordings root")
    ap.add_argument("--out", type=Path, required=True, help="clusters.json path")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="drop low detector-score boxes before clustering")
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--default-rotate-deg", type=int, default=0)
    ap.add_argument("--t-from", type=int, default=None)
    ap.add_argument("--t-to", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clusters", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="rows per chunk for k-means assignment; bounds peak RAM "
                         "(never builds the full N×K distance matrix)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=("embedding", "time"), default="embedding",
                    help="clustering strategy: 'embedding' (default, k-means over "
                         "visual/EfficientNet embeddings) or 'time' (group "
                         "detections into per-camera time episodes — fast, "
                         "deterministic, no torch; one episode = one cluster)")
    ap.add_argument("--episode-gap-sec", type=float,
                    default=float(os.environ.get("EPISODE_GAP_SEC", "30")),
                    help="time mode: start a new per-camera episode when the gap "
                         "to the previous detection is >= this many seconds "
                         "(default 30, like the feeder presence episode)")
    ap.add_argument("--embedding", choices=("auto", "visual", "efficientnet"),
                    default="auto",
                    help="feature extractor for clustering; auto uses cached "
                         "EfficientNet ImageNet weights when available")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument("--embedding-dim", type=int, default=64,
                    help="compact stored embedding dim for clustering/splitting")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow torchvision to download EfficientNet weights")
    ap.add_argument("--no-store-embeddings", action="store_true",
                    help="smaller manifest, but disables split-cluster in review UI")
    ap.add_argument("--dedupe-window-sec", type=float, default=15.0,
                    help="mark near-identical crops within this per-camera window "
                         "for default review collapse; 0 disables")
    ap.add_argument("--dedupe-threshold", type=float, default=0.995,
                    help="cosine similarity threshold for duplicate crops")
    ap.add_argument("--max-cluster-size", type=int, default=96,
                    help="HARD CAP on crops written per cluster: at most this many "
                         "crops per cluster end up in clusters.json (the rest are "
                         "dropped, not hidden). 0 disables the cap.")
    ap.add_argument("--ignore-config", type=Path, default=ROOT / "cameras.yaml",
                    help="camera config with ignore_regions (default: cameras.yaml)")
    ap.add_argument("--no-ignore-config", action="store_true",
                    help="do not load ignore_regions from cameras.yaml")
    ap.add_argument("--ignore-region", action="append", default=[],
                    help="extra ignored region as CAMERA:x0,y0,x1,y1 or global "
                         "x0,y0,x1,y1 in camera-normalized coords")
    ap.add_argument("--ignore-region-min-coverage", type=float, default=0.8,
                    help="drop a box only when at least this fraction of the "
                         "box area is inside an ignore region")
    ap.add_argument("--labels", default="",
                    help="optional comma-separated label names for the review UI")
    args = ap.parse_args()

    from training import CropSource
    from training.regions import (
        box_in_ignore_region,
        load_ignore_regions_from_camera_config,
        merge_regions,
        parse_region_specs,
        regions_to_jsonable,
    )

    ignored_by_region = 0
    config_regions = (
        {}
        if args.no_ignore_config
        else load_ignore_regions_from_camera_config(args.ignore_config)
    )
    cli_regions = parse_region_specs(args.ignore_region)
    ignore_regions = merge_regions(config_regions, cli_regions)

    def keep_box(frame, box) -> bool:
        nonlocal ignored_by_region
        if box_in_ignore_region(
            frame.camera_id,
            frame.frame_w,
            frame.frame_h,
            box,
            ignore_regions,
            min_coverage=args.ignore_region_min_coverage,
        ):
            ignored_by_region += 1
            return False
        return True

    src = CropSource(
        db_path=args.db,
        recordings_root=args.recordings,
        camera_id=args.camera,
        model=args.model,
        t_from=args.t_from,
        t_to=args.t_to,
        min_score=args.min_score,
        pad_frac=args.pad_frac,
        default_rotate_deg=args.default_rotate_deg,
        box_filter=keep_box if ignore_regions else None,
    )
    if ignore_regions:
        total_regions = sum(len(v) for v in ignore_regions.values())
        print(f"[cluster] using {total_regions} ignore region(s)")

    items: list[dict] = []

    if args.mode == "time":
        # ---- time mode: per-camera episodes, no decode / embeddings / torch ----
        if args.clusters is not None:
            print(f"[cluster] --clusters={args.clusters} ignored in time mode "
                  "(episodes define the clusters)")
        # Metadata only — iter_crop_refs never decodes pixels.
        for n, (stub, _ref) in enumerate(src.iter_crop_refs()):
            if args.limit is not None and n >= args.limit:
                break
            sb = stub.src_box
            if sb is None or sb.rowid is None:
                continue
            items.append({
                "crop_id": f"{stub.camera_id}:{sb.rowid}",
                "src_event_key": int(sb.rowid),
                "wall_ms": int(stub.wall_ms),
                "camera": stub.camera_id,
                "model": stub.model,
                "score": float(sb.score),
                "box": {"x": int(sb.x), "y": int(sb.y), "w": int(sb.w), "h": int(sb.h)},
                "rotate_deg": int(stub.rotate_deg),
                "pad_frac": float(args.pad_frac),
            })
        if not items:
            raise SystemExit("no usable crops after detector-score filtering")
        if ignored_by_region:
            print(f"[cluster] dropped {ignored_by_region} crops inside ignore region(s)")

        cluster_members = build_time_episodes(items, int(args.episode_gap_sec * 1000))
        for cid, members in cluster_members.items():
            for i in members:
                items[i]["cluster"] = int(cid)
        embedding_name = "time-episode"
        embedding_dim_param = 0
        stored_embeddings_param = False
        x = None
        print(f"[cluster] time mode: {len(cluster_members)} episode(s) "
              f"(per camera, gap >= {args.episode_gap_sec}s)")
    else:
        # ---- embedding mode (default): k-means over visual/EfficientNet feats ----
        extractor = build_extractor(
            args.embedding,
            batch_size=args.embedding_batch_size,
            allow_download=args.allow_download,
        )
        print(f"[cluster] embedding={extractor.name}")

        feats: list[np.ndarray] = []
        pending_images: list[np.ndarray] = []

        def flush_features() -> None:
            nonlocal pending_images
            if not pending_images:
                return
            feats.extend(extractor.encode_batch(pending_images))
            pending_images = []

        for n, sample in enumerate(src):
            if args.limit is not None and n >= args.limit:
                break
            sb = sample.src_box
            if sb is None or sb.rowid is None:
                continue
            pending_images.append(sample.image)
            items.append({
                "crop_id": f"{sample.camera_id}:{sb.rowid}",
                "src_event_key": int(sb.rowid),
                "wall_ms": int(sample.wall_ms),
                "camera": sample.camera_id,
                "model": sample.model,
                "score": float(sb.score),
                "box": {"x": int(sb.x), "y": int(sb.y), "w": int(sb.w), "h": int(sb.h)},
                "rotate_deg": int(sample.rotate_deg),
                "pad_frac": float(args.pad_frac),
            })
            if len(pending_images) >= max(1, args.embedding_batch_size):
                flush_features()
        flush_features()

        if not items:
            raise SystemExit("no usable crops after detector-score filtering")

        x = prepare_embeddings(
            np.stack(feats).astype(np.float32),
            out_dim=args.embedding_dim,
            seed=args.seed,
        )
        if ignored_by_region:
            print(f"[cluster] dropped {ignored_by_region} crops inside ignore region(s)")

        k = args.clusters or default_cluster_count(len(items))
        labels, distances = kmeans(
            x, k, seed=args.seed, max_iter=args.max_iter, chunk=args.chunk_size,
        )

        cluster_members = {}
        for i, (cluster, dist) in enumerate(zip(labels.tolist(), distances.tolist())):
            items[i]["cluster"] = int(cluster)
            items[i]["distance"] = float(dist)
            if not args.no_store_embeddings:
                items[i]["embedding"] = [round(float(v), 5) for v in x[i]]
            cluster_members.setdefault(int(cluster), []).append(i)
        embedding_name = extractor.name
        embedding_dim_param = int(x.shape[1])
        stored_embeddings_param = not args.no_store_embeddings

    # Re-number clusters by size, largest first, for a stable human review queue.
    old_to_new = {
        old: new
        for new, (old, _members) in enumerate(
            sorted(cluster_members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        )
    }
    remapped: dict[int, list[int]] = {}
    for old, members in cluster_members.items():
        new = old_to_new[old]
        remapped[new] = members
        for i in members:
            items[i]["cluster"] = new

    # Pick the crops that actually get written: dedupe + a hard per-cluster cap.
    # Everything else is DROPPED, so the manifest is exactly what the UI shows.
    source_crops = len(items)
    selected_by_cluster, drop_stats = select_review_crops(
        items,
        remapped,
        x,
        duplicate_window_sec=args.dedupe_window_sec,
        duplicate_threshold=args.dedupe_threshold,
        max_per_cluster=args.max_cluster_size,
        seed=args.seed,
    )
    clusters = [
        {"cluster_id": cluster_id, "item_indices": sorted(sel)}
        for cluster_id, sel in sorted(selected_by_cluster.items())
        if sel
    ]
    # Rebuild a compact items list and reindex every cluster's item_indices into
    # it; recomputes size + representatives from the kept crops only.
    items, clusters = compact_items_to_clusters(items, clusters)
    for item in items:
        _strip_review_fields(item)

    labels_hint = [v.strip() for v in args.labels.split(",") if v.strip()]
    out = {
        "version": 1,
        "kind": "cluster_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "min_score": args.min_score,
            "pad_frac": args.pad_frac,
            "default_rotate_deg": args.default_rotate_deg,
            "mode": args.mode,
            "episode_gap_sec": args.episode_gap_sec if args.mode == "time" else None,
            "clusters": len(clusters),
            "seed": args.seed,
            "embedding": embedding_name,
            "embedding_dim": embedding_dim_param,
            "stored_embeddings": stored_embeddings_param,
            "dedupe_window_sec": args.dedupe_window_sec,
            "dedupe_threshold": args.dedupe_threshold,
            "max_cluster_size": args.max_cluster_size,
            "dropped_dedupe": drop_stats["dropped_dedupe"],
            "dropped_cap": drop_stats["dropped_cap"],
            "ignored_by_region": ignored_by_region,
            "ignore_region_min_coverage": args.ignore_region_min_coverage,
            "ignore_regions": regions_to_jsonable(ignore_regions),
        },
        "labels": labels_hint,
        "items": items,
        "clusters": clusters,
    }

    # Enforce the review-exact invariants before writing — a non-compliant
    # manifest is never written, even if a future change regresses the pipeline.
    assert_review_manifest_invariants(items, clusters, args.max_cluster_size)

    written = len(items)
    dropped = source_crops - written
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"source crops scanned: {source_crops}")
    print(f"clusters written: {len(clusters)}")
    print(f"crops written: {written}")
    print(f"crops dropped by dedupe/max-cluster-size: {dropped}")
    print(f"wrote {args.out} (detector min_score={args.min_score})")


if __name__ == "__main__":
    main()
