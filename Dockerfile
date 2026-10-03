FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.onnx META_PATH=/app/models/model_meta.json LOG_DIR=/app/logs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN useradd -m -u 1000 app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY models/model.onnx models/model_meta.json ./models/
RUN mkdir -p logs && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')" || exit 1
CMD ["uvicorn", "defect.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--workers", "2"]
