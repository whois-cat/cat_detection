"""Training-data extraction package.

See training/README.md for the modeller-facing doc. Core entry points:

    from training.sources import FullFrameSource, CropSource, Sample
    from training.db      import Box, FrameRecord
"""
from .db import Box, FrameRecord
from .sources import Sample, SampleSource, FullFrameSource, CropSource

__all__ = [
    "Box", "FrameRecord",
    "Sample", "SampleSource", "FullFrameSource", "CropSource",
]
