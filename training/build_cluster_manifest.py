"""Build a cold-start cluster manifest for bulk cat-label review.

Unlike build_review_manifest.py, this does NOT use an existing cat identity
classifier. It only trusts the detector box score ("there is probably a cat
here"), decodes the crops, builds visual or generic ImageNet embeddings,
clusters them, and writes metadata for the cluster-review UI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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


def dedupe_nearby(items: list[dict], x: np.ndarray, *, window_sec: float,
                  threshold: float) -> tuple[list[dict], np.ndarray, int]:
    """Drop near-identical crops close in time on the same camera.

    Embeddings are L2-normalized, so dot product is cosine similarity. The window
    keeps this conservative: same-looking cats minutes apart are still retained.
    """
    if window_sec <= 0 or threshold <= 0:
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
            if float(x[i] @ x[j]) >= threshold:
                duplicate_of = j
                break
        if duplicate_of is not None:
            dropped += 1
            continue
        keep.append(i)
        recent.append(i)

    keep.sort()
    return [items[i] for i in keep], x[keep], dropped


def kmeans(x: np.ndarray, k: int, *, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.full(n, -1, dtype=np.int32)

    x_norm = np.sum(x * x, axis=1, keepdims=True)
    for _ in range(max_iter):
        c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
        dist = x_norm + c_norm - 2.0 * (x @ centers.T)
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
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

    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    dist = x_norm + c_norm - 2.0 * (x @ centers.T)
    assigned_dist = dist[np.arange(n), labels]
    return labels, assigned_dist.astype(np.float32)


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
    ap.add_argument("--seed", type=int, default=42)
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
    ap.add_argument("--dedupe-window-sec", type=float, default=2.0,
                    help="drop near-identical crops within this per-camera window; "
                         "0 disables")
    ap.add_argument("--dedupe-threshold", type=float, default=0.995,
                    help="cosine similarity threshold for duplicate crops")
    ap.add_argument("--labels", default="",
                    help="optional comma-separated label names for the review UI")
    args = ap.parse_args()

    from training import CropSource

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
    )

    extractor = build_extractor(
        args.embedding,
        batch_size=args.embedding_batch_size,
        allow_download=args.allow_download,
    )
    print(f"[cluster] embedding={extractor.name}")

    items: list[dict] = []
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
    items, x, deduped = dedupe_nearby(
        items, x,
        window_sec=args.dedupe_window_sec,
        threshold=args.dedupe_threshold,
    )
    if not items:
        raise SystemExit("no usable crops left after duplicate filtering")
    if deduped:
        print(f"[cluster] dropped {deduped} near-duplicate crops")

    k = args.clusters or default_cluster_count(len(items))
    labels, distances = kmeans(x, k, seed=args.seed, max_iter=args.max_iter)

    cluster_members: dict[int, list[int]] = {}
    for i, (cluster, dist) in enumerate(zip(labels.tolist(), distances.tolist())):
        items[i]["cluster"] = int(cluster)
        items[i]["distance"] = float(dist)
        if not args.no_store_embeddings:
            items[i]["embedding"] = [round(float(v), 5) for v in x[i]]
        cluster_members.setdefault(int(cluster), []).append(i)

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

    clusters = []
    for cluster_id, members in sorted(remapped.items()):
        members.sort(key=lambda i: items[i]["distance"])
        clusters.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "item_indices": members,
            "representatives": [items[i]["crop_id"] for i in members[:24]],
        })

    labels_hint = [v.strip() for v in args.labels.split(",") if v.strip()]
    out = {
        "version": 1,
        "kind": "cluster_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "min_score": args.min_score,
            "pad_frac": args.pad_frac,
            "default_rotate_deg": args.default_rotate_deg,
            "clusters": len(clusters),
            "seed": args.seed,
            "embedding": extractor.name,
            "embedding_dim": int(x.shape[1]),
            "stored_embeddings": not args.no_store_embeddings,
            "dedupe_window_sec": args.dedupe_window_sec,
            "dedupe_threshold": args.dedupe_threshold,
            "deduped": deduped,
        },
        "labels": labels_hint,
        "items": items,
        "clusters": clusters,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {len(items)} crops in {len(clusters)} clusters to {args.out} "
        f"(detector min_score={args.min_score})"
    )


if __name__ == "__main__":
    main()
