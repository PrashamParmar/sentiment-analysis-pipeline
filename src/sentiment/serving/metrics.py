"""Prometheus metrics (scraped at /metrics; dashboards in deploy/grafana)."""
from prometheus_client import Counter, Gauge, Histogram, Info

_LAT = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

HTTP_REQUESTS = Counter("sentiment_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("sentiment_http_request_duration_seconds", "End-to-end request latency", ["route"], buckets=_LAT)
INFER_LATENCY = Histogram("sentiment_inference_duration_seconds", "Model inference latency per micro-batch", buckets=_LAT)
BATCH_SIZE = Histogram("sentiment_inference_batch_size", "Texts per micro-batch", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
QUEUE_DEPTH = Gauge("sentiment_batch_queue_depth", "Requests waiting for the micro-batcher")
PREDICTIONS = Counter("sentiment_predictions_total", "Predictions by label", ["label"])
UNCERTAIN = Counter("sentiment_uncertain_predictions_total", "Predictions flagged uncertain")
EMPTY_AFTER_CLEAN = Counter("sentiment_empty_after_cleaning_total", "Inputs with no usable content after cleaning")
SCORE = Histogram("sentiment_score", "P(positive) distribution", buckets=tuple(i / 10 for i in range(1, 11)))
OOV = Histogram("sentiment_oov_rate", "Per-text out-of-vocabulary rate", buckets=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 1.0))
CACHE = Counter("sentiment_cache_total", "Prediction cache lookups", ["result"])
RATE_LIMITED = Counter("sentiment_rate_limited_total", "Requests rejected by the rate limiter")
OVERLOADED = Counter("sentiment_overloaded_total", "Requests shed because the queue was full")
DRIFT_PSI = Gauge("sentiment_drift_psi", "Population Stability Index vs training reference", ["feature"])
DRIFT_ALERT = Gauge("sentiment_drift_alert", "1 if any feature PSI exceeds the alert threshold")
DRIFT_WINDOW = Gauge("sentiment_drift_window_size", "Rows in the drift window")
JOBS = Gauge("sentiment_jobs", "Async batch jobs by state", ["state"])
MODEL_INFO = Info("sentiment_model", "Currently served model")
