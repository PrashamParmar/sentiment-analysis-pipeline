# syntax=docker/dockerfile:1.7
# CPU inference image. For GPU: FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 + torch cu128 index.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/hf \
    TOKENIZERS_PARALLELISM=false \
    LOG_JSON=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

# Heaviest, least frequently changing layer first -> maximal build cache reuse.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY configs ./configs
COPY src ./src

RUN useradd --create-home --uid 10001 app && mkdir -p /app/artifacts /app/logs /app/.cache && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=2).status==200 else 1)"

# One worker per container; scale horizontally. --timeout-graceful-shutdown drains in-flight requests.
CMD ["uvicorn", "sentiment.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--timeout-graceful-shutdown", "20", "--no-access-log"]
