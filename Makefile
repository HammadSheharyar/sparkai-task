export PYTHONPATH := src
PY ?= python

install:    ; pip install -r requirements.txt
prepare:    ; $(PY) -m defect.data_prep
train:      ; $(PY) -m defect.train
evaluate:   ; $(PY) -m defect.evaluate --split test && $(PY) -m defect.evaluate --split val
export:     ; $(PY) -m defect.export_onnx
benchmark:  ; $(PY) -m defect.benchmark
examples:   ; $(PY) scripts/predict_examples.py
serve:      ; uvicorn defect.api:app --host 0.0.0.0 --port 8000 --no-access-log
test:       ; $(PY) -m pytest -q tests
docker-build: ; docker build -t casting-defect-api:1.0.0 .
docker-run:   ; docker compose up -d --build
all: prepare train evaluate export benchmark examples

# optional model comparison (justifies the architecture choice)
compare:
	@for m in efficientnet_b0 mobilenet_v3_large resnet18; do \
	  $(PY) -m defect.train --model $$m --out-dir models/$$m && \
	  $(PY) -m defect.evaluate --ckpt models/$$m/best.pt --split test --out-dir reports/compare/$$m ; done
