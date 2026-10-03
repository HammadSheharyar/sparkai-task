"""Single source of truth for preprocessing (used by training, evaluation AND the API)."""
import numpy as np
from PIL import Image

from .constants import MEAN, STD


def resize_pil(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def pil_to_array(img: Image.Image, mean=MEAN, std=STD) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - mean) / std
    return np.ascontiguousarray(a.transpose(2, 0, 1), dtype=np.float32)


def preprocess(img: Image.Image, size: int, mean=MEAN, std=STD) -> np.ndarray:
    return pil_to_array(resize_pil(img, size), mean, std)
