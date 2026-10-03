import io

import numpy as np
import pytest
from PIL import Image

from defect.inference import InvalidImageError, decode_image
from defect.preprocess import preprocess


def _png(w=64, h=64, mode="L"):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(h, w) * 255).astype("uint8"), mode).save(buf, "PNG")
    return buf.getvalue()


def test_preprocess_shape_dtype_grayscale_and_rgb():
    for mode in ("L", "RGB"):
        img = Image.new(mode, (300, 200))
        a = preprocess(img, 128)
        assert a.shape == (3, 128, 128) and a.dtype == np.float32


def test_decode_ok():
    assert decode_image(_png()).size == (64, 64)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\nbroken"])
def test_decode_rejects_garbage(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_decode_rejects_tiny():
    with pytest.raises(InvalidImageError):
        decode_image(_png(8, 8))
