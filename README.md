# Casting Defect Detection

A production-style computer vision pipeline for detecting casting defects from images. The project combines dataset cleaning, leakage-safe splitting, transfer learning, ONNX export, and a FastAPI inference service for deployment.

## Setup instructions

### 1 Clone and create the environment
```bash
git clone <repo-url>
cd casting-defect-detection
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you want to run the API container, also make sure Docker and Docker Compose are available.

### 2) Prepare the dataset
The project expects the Kaggle casting defect dataset in a raw folder structure similar to:

```text
data/
  raw/
    .../cast_def_0_*.jpeg   # defective
    .../cast_ok_0_*.jpeg    # normal
```

The labels are inferred from the parent folder name (`def` vs `ok`), and the pipeline re-splits the data using a leakage-safe strategy instead of trusting the original train/test layout.

### 3) Run the full pipeline
```bash
make prepare
make train
make evaluate
make export
make benchmark
make examples
make test
```

Equivalent direct commands:

```powershell
$env:PYTHONPATH = "src"
python -m defect.data_prep
python -m defect.train
python -m defect.evaluate --split test
python -m defect.evaluate --split val
python -m defect.export_onnx
python -m defect.benchmark
python scripts/predict_examples.py
python -m pytest -q tests
```

### 4 Run the API locally or with Docker
```bash
docker compose up -d --build
# or
make serve
```

Then send a sample request:

```bash
curl -F "file=@examples/images/cast_def_0_1.jpeg" http://localhost:8000/predict
```

## Model approach

The solution uses an EfficientNet-B0 backbone initialized with ImageNet weights and a single-logit classification head for binary prediction (defective vs normal).

Key design choices:
- Transfer learning is used because the dataset is small and a pretrained feature extractor generalizes better than training from scratch.
- The model is trained in two phases: first the classifier head, then full fine-tuning with a lower learning rate.
- Binary cross-entropy with logits is used with `pos_weight` to handle class imbalance.
- Validation average precision is used as the early-stopping target so the model is optimized for defect detection rather than simple accuracy.
- The decision threshold is tuned on the validation set and chosen to maximize recall for defective items while keeping false alarms controlled.

This is a practical choice for industrial inspection where a missed defect is more costly than a false alarm.

## Dataset strategy

### Source and labeling
The project uses the Kaggle casting product image data for quality inspection. The raw images are kept as the ground truth source, and labels are derived from the image folder structure.

### Data quality controls
- Corrupt or unreadable files are removed.
- Exact duplicate images are removed via MD5 hashing.
- Conflicting duplicates with inconsistent labels are dropped.
- Near-duplicate images are grouped using a 256-bit dHash strategy to avoid leakage across train/val/test splits.

### Leakage-safe split
The biggest risk in visual inspection datasets is information leakage through near-duplicate versions of the same part. To reduce this, the project uses `StratifiedGroupKFold` and assigns whole duplicate clusters to a single split. This preserves class balance while ensuring group-disjoint train/validation/test partitions.

### Augmentation and preprocessing
Training uses augmentation such as flips, rotation, translation, scale changes, and mild brightness/contrast adjustments. The shared preprocessing pipeline converts grayscale to 3-channel images, resizes to 256x256, and applies ImageNet normalization. The same preprocessing path is used during training, evaluation, and inference to avoid train/serve skew.

## Evaluation results

The current model was evaluated on the test split using the tuned operating threshold chosen on validation data.

### Test-split results
Model: `efficientnet_b0` @ 256px | operating threshold: `0.9995`

| metric (defective = positive) | tuned threshold | threshold 0.5 |
|---|---:|---:|
| precision | 1.0000 | 1.0000 |
| recall | 0.9850 | 0.9950 |
| F1 | 0.9925 | 0.9975 |
| accuracy | 0.9914 | 0.9971 |
| ROC-AUC | 1.0000 | - |
| Avg. precision | 1.0000 | - |

Confusion matrix (tuned threshold): TN = 439, FP = 0, FN = 9, TP = 593.

Interpretation:
- The model is extremely strong on the validation/test dataset.
- It achieves perfect precision with only a small number of false negatives.
- Because the project prioritizes defect recall, the tuned threshold is deliberately conservative to reduce missed defects.

### Benchmark snapshot
Latency and model-size measurements are stored in `reports/benchmark.json`.

| mode | p50 (ms) | p95 (ms) | img/s |
|---|---:|---:|---:|
| PyTorch CPU batch 1 | 41.62 | 84.69 | 24.0 |
| ONNX Runtime CPU batch 1 | 13.61 | 15.83 | 73.5 |
| PyTorch CPU batch 8 | 315.24 | 383.89 | 25.4 |
| ONNX Runtime CPU batch 8 | 153.60 | 201.21 | 52.1 |

This shows the ONNX export is substantially faster for deployment while preserving model quality.

## Known limitations

- The dataset comes from a single capture setup, so performance on other lines, lighting conditions, or product variants is not yet validated.
- The model performs image-level classification only; it does not localize the defect itself.
- Resizing images to 256x256 may hide very small defects, which can contribute to false negatives in edge cases.
- Confidence is not fully calibrated beyond threshold tuning; it is useful for triage but not as a standalone business metric.
- The dataset is still relatively small, so metrics have wide confidence intervals and should be interpreted alongside raw counts and error cases.
- The current service is optimized for single-image synchronous inference; high-throughput production lines may benefit from batching or queueing.
- The system is designed for a controlled environment and may drift over time without periodic revalidation on new factory samples.

## Summary

This project demonstrates a practical, deployable defect-detection pipeline that balances model quality, industrial constraints, and operational reliability. It is suitable as a baseline for real-world inspection use, with clear paths for drift monitoring, higher-resolution models, and future defect localization work.

## Deployment (Docker)

### Build
docker compose build

### Run
docker compose up -d

The API will be available at http://localhost:8000.

### Verify
curl http://localhost:8000/ready        # {"status":"ready"}
curl http://localhost:8000/health       # {"status":"ok"}

### Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@examples/sample_defective.jpg"

### Stop
docker compose down

### Configuration (env vars)
| Variable | Default | Purpose |
|---|---|---|
| MODEL_PATH | /app/models/model.onnx | ONNX model path |
| META_PATH | /app/models/model_meta.json | threshold, class names |
| MAX_UPLOAD_MB | 10 | upload size cap |
| MAX_BATCH | 16 | max files per batch request |
| API_KEY | unset | if set, require X-API-Key header |
| PREDICTION_LOG | unset | JSONL audit log path |
| LOG_DIR | /app/logs | structured log directory |
| REVIEW_CONFIDENCE | 0.8 | threshold for "needs human review" |

### Image design notes
- Base: python:3.11-slim
- Non-root user (app, uid 1000)
- Model baked into the image at /app/models/ — no external download needed
- HEALTHCHECK hits /ready every 30s
- Logs written to /app/logs (mounted to host via compose)
- CPU-only (onnxruntime CPUExecutionProvider); swap to onnxruntime-gpu for GPU inference
