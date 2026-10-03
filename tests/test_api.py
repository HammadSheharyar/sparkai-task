import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defect.api import create_app
from defect.inference import InvalidImageError, decode_image


class FakePredictor:
    meta = {"model_name": "fake", "threshold": 0.4}

    def predict_bytes(self, data):
        decode_image(data)  # real validation path
        return {"predicted_class": "defective", "confidence": 0.93, "defect_probability": 0.91, "threshold": 0.4,
                "needs_human_review": False, "model_version": "test",
                "latency": {"preprocess_ms": 1.0, "inference_ms": 2.0, "total_ms": 3.0}}


def png(w=64, h=64):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(h, w) * 255).astype("uint8"), "L").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(create_app(FakePredictor())) as c:
        yield c


def test_health_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["predicted_class"] == "defective" and 0 <= j["confidence"] <= 1 and j["request_id"]
    assert r.headers["x-request-id"] == j["request_id"]


def test_predict_bad_image_422(client):
    assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 422


def test_predict_missing_file_422(client):
    assert client.post("/predict").status_code == 422


def test_batch_mixed(client):
    files = [("files", ("ok.png", png(), "image/png")), ("files", ("bad.txt", b"x", "text/plain"))]
    res = client.post("/predict/batch", files=files).json()["results"]
    assert res[0]["prediction"] is not None and res[1]["error"]


def test_too_large(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    with TestClient(create_app(FakePredictor())) as c:
        assert c.post("/predict", files={"file": ("a.png", png(256, 256), "image/png")}).status_code == 413


def test_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    with TestClient(create_app(FakePredictor())) as c:
        f = {"file": ("a.png", png(), "image/png")}
        assert c.post("/predict", files=f).status_code == 401
        assert c.post("/predict", files=f, headers={"X-API-Key": "secret"}).status_code == 200


def test_not_ready_without_model():
    with TestClient(create_app(None)) as c:  # default paths don't exist in CI -> 503
        assert c.get("/ready").status_code == 503
        assert c.get("/health").status_code == 200
