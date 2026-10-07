"""Regenerates deploy/grafana/dashboards/sentiment.json (dashboards-as-code). Run: python deploy/grafana/build_dashboard.py"""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
PANELS = [
    ("Requests / s by status", "sum by (status) (rate(sentiment_http_requests_total{route=~\"/v1/.*\"}[1m]))", "reqps", "{{status}}"),
    ("/v1/predict latency p50 / p95 / p99", [
        ("histogram_quantile(0.50, sum by (le) (rate(sentiment_http_request_duration_seconds_bucket{route=\"/v1/predict\"}[5m])))", "p50"),
        ("histogram_quantile(0.95, sum by (le) (rate(sentiment_http_request_duration_seconds_bucket{route=\"/v1/predict\"}[5m])))", "p95"),
        ("histogram_quantile(0.99, sum by (le) (rate(sentiment_http_request_duration_seconds_bucket{route=\"/v1/predict\"}[5m])))", "p99"),
    ], "s", None),
    ("Inference latency p95 per micro-batch", "histogram_quantile(0.95, sum by (le) (rate(sentiment_inference_duration_seconds_bucket[5m])))", "s", "p95"),
    ("Avg micro-batch size", "rate(sentiment_inference_batch_size_sum[5m]) / clamp_min(rate(sentiment_inference_batch_size_count[5m]), 1e-9)", "short", "texts/batch"),
    ("Predictions / s by label", "sum by (label) (rate(sentiment_predictions_total[1m]))", "short", "{{label}}"),
    ("Positive share & uncertain share", [
        ("sum(rate(sentiment_predictions_total{label=\"positive\"}[5m])) / clamp_min(sum(rate(sentiment_predictions_total[5m])), 1e-9)", "positive"),
        ("sum(rate(sentiment_uncertain_predictions_total[5m])) / clamp_min(sum(rate(sentiment_predictions_total[5m])), 1e-9)", "uncertain"),
    ], "percentunit", None),
    ("Drift PSI by feature (alert > 0.2)", "sentiment_drift_psi", "short", "{{feature}}"),
    ("Mean OOV rate", "rate(sentiment_oov_rate_sum[5m]) / clamp_min(rate(sentiment_oov_rate_count[5m]), 1e-9)", "percentunit", "oov"),
    ("Cache hit ratio", "sum(rate(sentiment_cache_total{result=\"hit\"}[5m])) / clamp_min(sum(rate(sentiment_cache_total[5m])), 1e-9)", "percentunit", "hit ratio"),
    ("Queue depth / rate-limited / shed", [
        ("sentiment_batch_queue_depth", "queue depth"),
        ("rate(sentiment_rate_limited_total[1m])", "429/s"),
        ("rate(sentiment_overloaded_total[1m])", "503/s"),
    ], "short", None),
]


def panel(i: int, title: str, expr, unit: str, legend) -> dict:
    exprs = expr if isinstance(expr, list) else [(expr, legend)]
    return {
        "id": i + 1, "type": "timeseries", "title": title, "datasource": DS,
        "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"refId": chr(65 + j), "datasource": DS, "expr": e, "legendFormat": lg} for j, (e, lg) in enumerate(exprs)],
    }


dash = {
    "uid": "sentiment-api", "title": "Sentiment API", "schemaVersion": 39, "version": 1, "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"}, "tags": ["ml", "sentiment"],
    "panels": [panel(i, *p) for i, p in enumerate(PANELS)],
}
out = Path(__file__).parent / "dashboards" / "sentiment.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dash, indent=2))
print(f"wrote {out}")
