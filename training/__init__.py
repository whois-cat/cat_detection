"""Training-data extraction package.

See training/README.md for the modeller-facing doc. Core entry points:

    from training.sources import FullFrameSource, CropSource, Sample
    from training.db      import Box, FrameRecord
"""
from .db import Box, FrameRecord
from .reviews import load_reviews

__all__ = [
    "Box", "FrameRecord",
    "Sample", "SampleSource", "FullFrameSource", "CropSource",
    "CropRef", "CropUnavailable", "decode_one_crop",
    "load_reviews",
]

_SOURCE_EXPORTS = {
    "Sample",
    "SampleSource",
    "FullFrameSource",
    "CropSource",
    "CropRef",
    "CropUnavailable",
    "decode_one_crop",
}


def __getattr__(name: str):
    if name in _SOURCE_EXPORTS:
        from . import sources

        value = getattr(sources, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
