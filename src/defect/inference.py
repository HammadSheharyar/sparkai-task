"""Runtime inference (ONNX Runtime only - no torch needed in the serving image)."""
import io
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .preprocess import preprocess

log = logging.getLogger("inference")
Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes, min_side: int = 32) -> Image.Image:
    if not data:
        raise InvalidImageError("empty file")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as e:
        raise InvalidImageError(f"cannot decode image: {type(e).__name__}") from e
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"unsupported format {fmt}; allowed: {sorted(ALLOWED_FORMATS)}")
    if min(img.size) < min_side:
        raise InvalidImageError(f"image too small {img.size}; min side {min_side}px")
    return img


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class Predictor:
    def __init__(self, model_path, meta_path, review_confidence: float = 0.8):
        import onnxruntime as ort

        self.meta = json.loads(Path(meta_path).read_text())
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.size = int(self.meta["input_size"])
        self.mean, self.std = self.meta["mean"], self.meta["std"]
        thr = min(max(float(self.meta["threshold"]), 1e-6), 1 - 1e-6)
        self.threshold = thr
        self._logit_thr = math.log(thr / (1 - thr))
        self.classes = self.meta["class_names"]
        self.version = self.meta.get("model_version", "unknown")
        self.review_confidence = review_confidence
        self.input_name = self.meta.get("input_name", "input")
        self.session.run(None, {self.input_name: np.zeros((1, 3, self.size, self.size), np.float32)})  # warm-up
        log.info("model loaded", extra={"ctx": {"providers": providers, "version": self.version,
                                                "threshold": thr, "size": self.size}})

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.perf_counter()
        img = decode_image(data)
        x = preprocess(img, self.size, self.mean, self.std)[None]
        t1 = time.perf_counter()
        logit = float(self.session.run(None, {self.input_name: x})[0].reshape(-1)[0])
        t2 = time.perf_counter()
        defective = logit >= self._logit_thr  # identical to p >= threshold
        # confidence is re-centred so the operating threshold maps to 0.5 (label and confidence never disagree)
        p_def_adj = _sigmoid(logit - self._logit_thr)
        confidence = p_def_adj if defective else 1.0 - p_def_adj
        return {
            "predicted_class": self.classes[1 if defective else 0],
            "confidence": round(confidence, 4),
            "defect_probability": round(_sigmoid(logit), 4),
            "threshold": round(self.threshold, 4),
            "needs_human_review": bool(confidence < self.review_confidence),
            "latency": {"preprocess_ms": round((t1 - t0) * 1000, 2), "inference_ms": round((t2 - t1) * 1000, 2),
                        "total_ms": round((t2 - t0) * 1000, 2)},
            "model_version": self.version,
        }
