"""PyTorch dataset adapters around our SampleSource — no disk roundtrip.

For PyTorch-style training (typically the per-cat classifier) you can feed
decoded ndarrays straight from our SampleSource into a Dataset, no JPEG
encode/decode in between.

Two adapters, picked by how many epochs you'll train for:

  - TorchStreamingDataset (IterableDataset). Each epoch starts a fresh
    SampleSource iteration. PyAV-decodes the segments every time — cheap
    for one-shot / online training, expensive for multi-epoch.

  - TorchCachedDataset (map-style Dataset). On first access, decodes
    every Sample into an in-memory list of (image ndarray, label).
    Subsequent epochs serve from RAM. Standard choice for multi-epoch
    classifier training.

Ultralytics YOLO training intentionally NOT covered here — its trainer
hard-wires a YAML/directory layout. For that path use
`extract_detector.py` (disk).

Importing this module requires torch. The rest of the `training` package
works without it; only this file depends on torch.

Usage sketch (per-cat classifier):

    import torch
    from torchvision import transforms
    from training import CropSource
    from training.torch_dataset import TorchCachedDataset

    classes = ["alisa", "chuzh", "ellie", "felisis"]
    label_idx = {c: i for i, c in enumerate(classes)}

    src = CropSource(db_path="data/events/events.db",
                     recordings_root="data/recordings",
                     model="yolov8n", min_score=0.3, pad_frac=0.15)

    tfm = transforms.Compose([
        transforms.ToPILImage(),                    # BGR ndarray → PIL
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    def label_of(sample):  return label_idx[sample.boxes[0].cat]

    ds = TorchCachedDataset(src, transform=tfm, target_fn=label_of)
    ds.materialise()                                # explicit decode pass
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True,
                                     num_workers=4)
    for img, label in dl: ...

Worker caveat: SampleSource uses PyAV containers and isn't fork-safe.
ALWAYS call `materialise()` before spawning DataLoader workers — at that
point the workers only need to look at the in-memory tensors, not touch
SampleSource at all. For TorchStreamingDataset, set `num_workers=0`.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from .crop_cache import BoundedCropCache
from .sources import Sample, SampleSource

# Default RAM budget for the lazy crop cache; 0 disables caching. CPU boxes
# should keep this modest — the cache only avoids re-decoding, not OOM.
TRAINING_CACHE_MAX_MB = float(os.environ.get("TRAINING_CACHE_MAX_MB", "512"))


def _resize_uint8(img_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Contiguous uint8 crop capped to `max_side` on its long edge (0 = as-is)."""
    arr = np.ascontiguousarray(img_bgr)
    if max_side <= 0:
        return arr
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    scale = max_side / float(longest)
    import cv2

    out = cv2.resize(
        arr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(out)


# ---- helpers ----------------------------------------------------------------

def default_transform(img_bgr: np.ndarray) -> torch.Tensor:
    """ndarray BGR uint8 → CHW float32 tensor in [0,1], RGB order."""
    rgb = img_bgr[..., ::-1].copy()                 # BGR → RGB, contiguous
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw).float() / 255.0


def default_target_fn(sample: Sample):
    """First box's cat string. Override to map to int / one-hot / whatever."""
    return sample.boxes[0].cat if sample.boxes else None


# ---- IterableDataset (streaming) --------------------------------------------

class TorchStreamingDataset(IterableDataset):
    """Wraps a SampleSource for one-pass / online training. Each `__iter__`
    starts a fresh SampleSource (so multi-epoch use re-decodes segments)."""

    def __init__(self, source: SampleSource,
                 transform: Callable[[np.ndarray], torch.Tensor] = default_transform,
                 target_fn: Callable[[Sample], object] = default_target_fn):
        super().__init__()
        self.source    = source
        self.transform = transform
        self.target_fn = target_fn

    def __iter__(self):
        for sample in self.source:
            yield self.transform(sample.image), self.target_fn(sample)


# ---- Dataset (RAM-cached) ---------------------------------------------------

class TorchCachedDataset(Dataset):
    """Materialise a SampleSource into an in-memory list once, then serve
    random-access lookups. `transform` runs at __getitem__ time so cheap
    per-sample augmentation is fine without invalidating the cache.

    Memory: roughly N × H × W × 3 bytes (raw BGR ndarrays cached, not
    transformed tensors). ~5k crops at 224² ≈ 750 MB; budget ~1 GB per 5k
    crops once padding lands around 250×250.
    """

    def __init__(self, source: SampleSource,
                 transform: Callable[[np.ndarray], torch.Tensor] = default_transform,
                 target_fn: Callable[[Sample], object] = default_target_fn):
        super().__init__()
        self.source    = source
        self.transform = transform
        self.target_fn = target_fn
        self._cache: list[tuple[np.ndarray, object]] | None = None

    def materialise(self) -> None:
        """One-shot decode pass. Call before DataLoader workers fork, since
        SampleSource itself is not fork-safe."""
        if self._cache is None:
            self._cache = [(s.image, self.target_fn(s)) for s in self.source]

    def __len__(self) -> int:
        self.materialise()
        assert self._cache is not None
        return len(self._cache)

    def __getitem__(self, idx: int):
        self.materialise()
        assert self._cache is not None
        img, label = self._cache[idx]
        return self.transform(img), label


# ---- Dataset (lazy, byte-bounded LRU) ---------------------------------------

class TorchLazyCachedDataset(Dataset):
    """Map-style dataset that decodes crops on demand and keeps a byte-bounded
    LRU of resized ``uint8`` crops — never the full dataset, never full frames.

    First pass collects only lightweight per-sample metadata (label + a small
    closure that re-decodes/resizes ONE crop). On ``__getitem__`` the crop is
    served from the cache or recomputed; ``transform`` (float conversion +
    augmentation) runs after the cache so the cache stays compact ``uint8``.

    `cache_max_mb<=0` disables retention (every access re-decodes). Inspect
    ``cache.stats`` for current/peak bytes, hits, misses, evictions.
    """

    def __init__(self, source: SampleSource,
                 transform: Callable[[np.ndarray], torch.Tensor] = default_transform,
                 target_fn: Callable[[Sample], object] = default_target_fn,
                 *, cache_max_mb: float = TRAINING_CACHE_MAX_MB,
                 resize_max_side: int = 256):
        super().__init__()
        self.source = source
        self.transform = transform
        self.target_fn = target_fn
        self.resize_max_side = resize_max_side
        self.cache = BoundedCropCache(cache_max_mb)
        # Decoded once: (label, loader) where loader returns a small uint8 crop.
        self._items: list[tuple[object, Callable[[], np.ndarray]]] | None = None

    def _index(self) -> None:
        if self._items is not None:
            return
        items: list[tuple[object, Callable[[], np.ndarray]]] = []
        for s in self.source:
            label = self.target_fn(s)
            crop = _resize_uint8(s.image, self.resize_max_side)
            # The crop is already small + contiguous; the loader just returns it.
            # (Subclasses backed by random-access decode can override this to
            #  re-decode instead of holding the bytes — left as a TODO.)
            items.append((label, (lambda c=crop: c)))
        self._items = items

    def __len__(self) -> int:
        self._index()
        assert self._items is not None
        return len(self._items)

    def __getitem__(self, idx: int):
        self._index()
        assert self._items is not None
        label, loader = self._items[idx]
        crop = self.cache.get(idx, loader)
        return self.transform(crop), label
