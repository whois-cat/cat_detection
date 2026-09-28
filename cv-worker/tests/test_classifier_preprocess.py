import numpy as np
import pytest

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")
Image = pytest.importorskip("PIL.Image")

from cv_worker.models.classifier import preprocess  # noqa: E402


@pytest.mark.parametrize("shape", [(180, 97), (97, 180), (300, 301), (224, 224), (41, 500)])
def test_matches_training_transform(shape):
    """Runtime preprocessing must equal the training transform exactly
    (resize-256 short side, center-crop 224, ImageNet normalization)."""
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    ref = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(Image.fromarray(crop)).numpy()[np.newaxis]
    assert np.abs(preprocess(crop) - ref).max() < 1e-5
