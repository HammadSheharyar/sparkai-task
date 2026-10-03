"""FastAPI inference service.
Env: MODEL_PATH, META_PATH, MAX_UPLOAD_MB (10), MAX_BATCH (16), API_KEY (optional), LOG_DIR (optional),
     LOG_LEVEL (INFO), PREDICTION_LOG (path to JSONL audit log, optional), REVIEW_CONFIDENCE (0.8)
"""
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .inference import InvalidImageError, Predictor
from .logging_config import setup_logging

log = logging.getLogger("api")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent   # src/defect/api.py → project root


class Latency(BaseModel):
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class Prediction(BaseModel):
    predicted_class: str
    confidence: float
    defect_probability: float
    threshold: float
    needs_human_review: bool
    latency: Latency
    model_version: str


class PredictResponse(Prediction):
    request_id: str
    filename: Optional[str] = None


class BatchItem(BaseModel):
    filename: Optional[str]
    prediction: Optional[Prediction] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    request_id: str
    results: List[BatchItem]

_SENTINEL = object()
def create_app(predictor=None) -> FastAPI:
    log_dir = os.getenv("LOG_DIR")
    setup_logging(os.getenv("LOG_LEVEL", "INFO"), json_logs=True,
                  log_file=os.path.join(log_dir, "api.log") if log_dir else None)
    max_bytes = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    max_batch = int(os.getenv("MAX_BATCH", "16"))
    api_key = os.getenv("API_KEY")
    pred_log = os.getenv("PREDICTION_LOG")
    stats = {"requests": 0, "errors_4xx": 0, "errors_5xx": 0, "predictions": {}, "inference_ms_sum": 0.0}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            model_path = os.getenv("MODEL_PATH", str(ROOT / "models" / "model.onnx"))
            meta_path  = os.getenv("META_PATH",  str(ROOT / "models" / "model_meta.json"))
            print(f"[startup] cwd          = {os.getcwd()}")
            print(f"[startup] model_path   = {model_path}  exists={os.path.exists(model_path)}")
            print(f"[startup] meta_path    = {meta_path}  exists={os.path.exists(meta_path)}")
            try:
                app.state.predictor = Predictor(
                    os.getenv("MODEL_PATH", "models/model.onnx"), os.getenv("META_PATH", "models/model_meta.json"),
                    float(os.getenv("REVIEW_CONFIDENCE", "0.8")))
            except Exception:
                log.exception("model failed to load; service will report not ready")
        yield

    app = FastAPI(title="Casting Defect Detection API", version="1.0.0", lifespan=lifespan)
    app.state.predictor = predictor

    def auth(x_api_key: Optional[str] = Header(default=None)):
        if api_key and not (x_api_key and secrets.compare_digest(x_api_key, api_key)):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def get_predictor(request: Request):
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded")
        return p

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = rid
        with lock:
            stats["requests"] += 1
            if 400 <= response.status_code < 500:
                stats["errors_4xx"] += 1
            elif response.status_code >= 500:
                stats["errors_5xx"] += 1
        log.info("request", extra={"ctx": {"request_id": rid, "method": request.method, "path": request.url.path,
                                           "status": response.status_code, "ms": ms}})
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log.error("unhandled error", exc_info=exc, extra={"ctx": {"request_id": rid}})
        return JSONResponse({"error": "internal server error", "request_id": rid}, status_code=500)

    def read_upload(f: UploadFile) -> bytes:
        data = f.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"file larger than {max_bytes // (1024 * 1024)} MB")
        return data

    def record(rid, filename, data, out):
        with lock:
            c = out["predicted_class"]
            stats["predictions"][c] = stats["predictions"].get(c, 0) + 1
            stats["inference_ms_sum"] += out["latency"]["total_ms"]
        log.info("prediction", extra={"ctx": {"request_id": rid, "filename": filename, **{
            k: out[k] for k in ("predicted_class", "confidence", "defect_probability", "needs_human_review")},
            "total_ms": out["latency"]["total_ms"]}})
        if pred_log:  # audit trail for monitoring / drift analysis (no raw images stored)
            rec = {"ts": time.time(), "request_id": rid, "filename": filename,
                   "sha256": hashlib.sha256(data).hexdigest(), **out}
            with lock, open(pred_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready"}

    @app.get("/model-info", dependencies=[Depends(auth)])
    def model_info(p=Depends(get_predictor)):
        return p.meta

    @app.get("/stats", dependencies=[Depends(auth)])
    def get_stats():
        with lock:
            return dict(stats)

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(auth)])
    def predict(request: Request, file: UploadFile = File(...), p=Depends(get_predictor)):
        data = read_upload(file)
        try:
            out = p.predict_bytes(data)
        except InvalidImageError as e:
            raise HTTPException(422, str(e))
        rid = request.state.request_id
        record(rid, file.filename, data, out)
        return PredictResponse(**out, request_id=rid, filename=file.filename)

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(auth)])
    def predict_batch(request: Request, files: List[UploadFile] = File(...), p=Depends(get_predictor)):
        if len(files) > max_batch:
            raise HTTPException(413, f"max {max_batch} files per request")
        rid, results = request.state.request_id, []
        for f in files:
            try:
                data = read_upload(f)
                out = p.predict_bytes(data)
                record(rid, f.filename, data, out)
                results.append(BatchItem(filename=f.filename, prediction=Prediction(**out)))
            except InvalidImageError as e:
                results.append(BatchItem(filename=f.filename, error=str(e)))
            except HTTPException as e:
                results.append(BatchItem(filename=f.filename, error=e.detail))
        return BatchResponse(request_id=rid, results=results)

    return app


app = create_app()
