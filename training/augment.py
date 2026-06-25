"""Safe, identity-preserving augmentation for the cat classifier.

This is *identity* classification, not generic object classification: left/right
coat markings, tail shape, ear/head visibility and texture carry the signal that
separates visually similar cats (e.g. alisa vs felisis). So the augmentation
here is deliberately conservative — it must not flip, quarter-turn, hard-crop or
heavily recolour a crop, because any of those can erase the very cues the model
needs and make alisa↔felisis confusion *worse*.

Two layers, on purpose:

  * :class:`AugmentSpec` — a pure-Python description of an augmentation level. It
    imports no torch, so it can be inspected (and unit-tested) anywhere. The
    forbidden ops are explicit ``False`` fields, not just "absent", so the
    guarantees are visible and testable.
  * :func:`build_train_transform` / :func:`build_eval_transform` — turn a spec
    into a torchvision pipeline. These import torch lazily.

Augmentation is on-the-fly (applied per ``__getitem__`` during training only).
Nothing here writes augmented images to disk, and validation/test use the
deterministic eval transform (byte-compatible framing with the runtime
preprocessing: resize short side 256 → center-crop 224 → ImageNet normalise).
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("off", "light", "medium")

INPUT_SIZE = 224
RESIZE_SHORT = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentSpec:
    """A single augmentation level. All geometry is mild; the ``*_flip`` /
    ``quarter_turns`` / ``random_resized_crop`` / ``random_erasing`` flags are
    hard-wired ``False`` so the identity-destroying ops can never sneak in."""

    level: str
    rotation_deg: float        # max |rotation|, degrees (small; never a quarter turn)
    translate_frac: float      # max translation as a fraction of size
    scale_min: float           # zoom range (around 1.0; no hard crop)
    scale_max: float
    brightness: float          # ColorJitter brightness jitter
    contrast: float            # ColorJitter contrast jitter
    gamma: float               # max |gamma delta| around 1.0
    noise_std: float           # additive Gaussian noise std in [0,1] pixel space
    blur_prob: float           # probability of a mild Gaussian blur
    blur_sigma_max: float
    pad_frac: float            # reflect-pad before affine so heads/ears/tails aren't cut

    # --- forbidden, identity-destroying ops: never enabled --------------------
    horizontal_flip: bool = False
    vertical_flip: bool = False
    quarter_turns: bool = False        # 90 / 180 / 270
    random_resized_crop: bool = False  # strong random crop
    random_erasing: bool = False

    @property
    def is_random(self) -> bool:
        """True if applying this transform twice can differ (i.e. augmenting)."""
        return self.level != "off"


_SPECS: dict[str, AugmentSpec] = {
    # No randomness — identical to the eval transform. Used for --augment off and
    # for validation/test.
    "off": AugmentSpec(
        level="off",
        rotation_deg=0.0, translate_frac=0.0, scale_min=1.0, scale_max=1.0,
        brightness=0.0, contrast=0.0, gamma=0.0, noise_std=0.0,
        blur_prob=0.0, blur_sigma_max=0.0, pad_frac=0.0,
    ),
    # Conservative defaults — within the "allowed for light" envelope in the spec.
    "light": AugmentSpec(
        level="light",
        rotation_deg=6.0, translate_frac=0.04, scale_min=0.92, scale_max=1.05,
        brightness=0.12, contrast=0.12, gamma=0.08, noise_std=0.01,
        blur_prob=0.10, blur_sigma_max=1.0, pad_frac=0.08,
    ),
    # A modest step up, still in the same safe family (no flips, small rotations).
    "medium": AugmentSpec(
        level="medium",
        rotation_deg=8.0, translate_frac=0.06, scale_min=0.90, scale_max=1.08,
        brightness=0.18, contrast=0.18, gamma=0.12, noise_std=0.02,
        blur_prob=0.25, blur_sigma_max=1.5, pad_frac=0.10,
    ),
}


def augment_spec(level: str) -> AugmentSpec:
    try:
        return _SPECS[level]
    except KeyError:
        raise ValueError(
            f"unknown augment level {level!r}; choose from {', '.join(LEVELS)}"
        ) from None


# --- torchvision builders (lazy torch import) --------------------------------

class _RandomGamma:
    """Mild gamma jitter around 1.0 on a PIL image (texture/contrast only)."""

    def __init__(self, max_delta: float):
        self.max_delta = float(max_delta)

    def __call__(self, img):
        if self.max_delta <= 0:
            return img
        import random
        from torchvision.transforms.functional import adjust_gamma
        g = 1.0 + random.uniform(-self.max_delta, self.max_delta)
        return adjust_gamma(img, max(0.1, g))


class _AddGaussianNoise:
    """Additive Gaussian noise on a [0,1] CHW tensor, clamped back to [0,1]."""

    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, t):
        if self.std <= 0:
            return t
        import torch
        return torch.clamp(t + torch.randn_like(t) * self.std, 0.0, 1.0)


def build_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD,
                         *, size=INPUT_SIZE, resize=RESIZE_SHORT):
    """Deterministic eval/test transform: resize short side → center crop →
    ToTensor → ImageNet normalise. No randomness."""
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(list(mean), list(std)),
    ])


def build_train_transform(level: str, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                          *, size=INPUT_SIZE, resize=RESIZE_SHORT):
    """Build the train-time transform for ``level``.

    ``off`` returns the deterministic eval transform (no augmentation). Otherwise
    the pipeline keeps the *same* center framing as eval (no random crop) and
    only adds mild, identity-preserving jitter: reflect-pad → small affine
    (rotate/translate/scale) → crop back → mild brightness/contrast/gamma →
    optional mild blur → ToTensor → optional mild noise → normalise.
    """
    spec = augment_spec(level)
    if not spec.is_random:
        return build_eval_transform(mean, std, size=size, resize=resize)

    from torchvision import transforms as T
    pad = int(round(size * spec.pad_frac))
    ops = [
        T.Resize(resize),
        T.CenterCrop(size),                       # same framing as eval (NO random crop)
        T.Pad(pad, padding_mode="reflect"),       # margin so affine never clips ears/tail
        T.RandomAffine(
            degrees=spec.rotation_deg,            # small rotation only; never 90/180
            translate=(spec.translate_frac, spec.translate_frac),
            scale=(spec.scale_min, spec.scale_max),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.CenterCrop(size),                       # back to model input size
        # No hue/saturation jitter: IR crops are ~grayscale and colour shifts add
        # nothing but risk. Brightness/contrast/gamma only.
        T.ColorJitter(brightness=spec.brightness, contrast=spec.contrast),
        _RandomGamma(spec.gamma),
    ]
    if spec.blur_prob > 0:
        ops.append(T.RandomApply(
            [T.GaussianBlur(3, sigma=(0.1, spec.blur_sigma_max))],
            p=spec.blur_prob,
        ))
    ops.append(T.ToTensor())
    if spec.noise_std > 0:
        ops.append(_AddGaussianNoise(spec.noise_std))
    ops.append(T.Normalize(list(mean), list(std)))
    return T.Compose(ops)


def tensor_to_pil(t, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Invert Normalize+ToTensor for previews: CHW normalised tensor → PIL RGB."""
    import torch
    from PIL import Image
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    arr = (t.detach().cpu() * s + m).clamp(0, 1)
    arr = (arr * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def save_augmentation_preview(out_dir, crops_rgb, train_transform,
                              eval_transform, *, n_aug=5,
                              mean=IMAGENET_MEAN, std=IMAGENET_STD) -> list:
    """Save one PNG per sample: [eval(original) | aug1 … aug_n]. Debug only —
    nothing here is written back into the dataset. Returns the written paths."""
    from pathlib import Path
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, rgb in enumerate(crops_rgb):
        pil = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
        tiles = [tensor_to_pil(eval_transform(pil), mean, std)]
        for _ in range(n_aug):
            tiles.append(tensor_to_pil(train_transform(pil), mean, std))
        w = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
        h = max(t.height for t in tiles)
        grid = Image.new("RGB", (w, h), (20, 20, 24))
        x = 0
        for t in tiles:
            grid.paste(t, (x, 0))
            x += t.width + 4
        path = out / f"aug_preview_{i:02d}.png"
        grid.save(path)
        written.append(path)
    return written
