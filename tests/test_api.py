import time

import pytest
from fastapi.testclient import TestClient

from sentiment.models.bundle import Predictor
from sentiment.serving.app import create_app
from sentiment.serving.runtime import TokenBucketLimiter, TTLCache


@pytest.fixture()
def client(trained_bundle, cfg):
    cfg.serve.rate_limit_burst = 1000
    cfg.serve.rate_limit_per_minute = 100_000
    app = create_app(cfg, predictor=Predictor.load(trained_bundle))
    with TestClient(app) as c:
        yield c


def test_health_and_model(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200
    info = client.get("/v1/model").json()
    assert info["kind"] == "classical" and 0 < info["threshold"] < 1


def test_predict_batch_preserves_order_and_caches(client):
    texts = ["I love this great day", "terrible awful horrible", "I love this great day", ""]
    r = client.post("/v1/predict", json={"texts": texts})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 4
    assert preds[0]["score"] == preds[2]["score"] and preds[0]["score"] > preds[1]["score"]
    assert preds[3]["empty_after_cleaning"]
    again = client.post("/v1/predict", json={"texts": [texts[1]]}).json()["predictions"][0]
    assert again["cached"] and again["score"] == preds[1]["score"]


def test_docs_work_offline(client):
    html = client.get("/docs").text
    assert "/static/swagger/swagger-ui-bundle.js" in html and "cdn.jsdelivr" not in html
    assert client.get("/static/swagger/swagger-ui-bundle.js").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "Sentiment API"


def test_validation_errors(client, cfg):
    assert client.post("/v1/predict", json={"texts": []}).status_code == 422
    assert client.post("/v1/predict", json={"texts": ["x"] * (cfg.serve.max_batch_items + 1)}).status_code == 422
    assert client.post("/v1/predict", json={"texts": ["x" * (cfg.serve.max_text_chars + 1)]}).status_code == 422
    assert client.post("/v1/predict", json={"wrong": 1}).status_code == 422


def test_async_job(client):
    r = client.post("/v1/jobs", json={"texts": [f"great day number {i}" for i in range(150)]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded" and len(job["results"]) == 150
    assert client.get("/v1/jobs/nope").status_code == 404


def test_metrics_and_drift_endpoints(client):
    client.post("/v1/predict", json={"texts": [f"text {i}" for i in range(10)]})
    body = client.get("/metrics").text
    assert "sentiment_predictions_total" in body and "sentiment_inference_batch_size" in body
    assert client.get("/v1/drift").json()["status"] in ("ok", "drift", "insufficient_data")


def test_rate_limit_returns_429(trained_bundle, cfg):
    cfg.serve.rate_limit_burst = 3
    cfg.serve.rate_limit_per_minute = 1
    app = create_app(cfg, predictor=Predictor.load(trained_bundle))
    with TestClient(app) as c:
        codes = [c.post("/v1/predict", json={"texts": ["hi"]}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
        assert c.get("/health/live").status_code == 200        # health is never rate limited


def test_no_model_is_not_ready(cfg, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    with TestClient(create_app(cfg)) as c:
        assert c.get("/health/ready").status_code == 503
        assert c.post("/v1/predict", json={"texts": ["x"]}).status_code == 503


def test_cache_and_limiter_units():
    cache = TTLCache(maxsize=2, ttl_s=60)
    for k in "abc":
        cache.set(k, k)
    assert cache.get("a") is None and cache.get("c") == "c" and len(cache) == 2
    lim = TokenBucketLimiter(rate_per_minute=60, burst=2)
    assert lim.allow("x")[0] and lim.allow("x")[0]
    ok, retry = lim.allow("x")
    assert not ok and 0 < retry <= 1.0
    assert lim.allow("y")[0]
