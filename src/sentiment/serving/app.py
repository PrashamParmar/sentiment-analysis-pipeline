"""Async FastAPI service.

Endpoints
    POST /v1/predict        1..max_batch_items texts -> predictions (micro-batched, cached)
    POST /v1/jobs           large async batch (up to job_max_items) -> 202 + job id
    GET  /v1/jobs/{id}      job status / results
    GET  /v1/model          served model metadata
    GET  /v1/drift          live drift (PSI) vs training reference
    POST /admin/reload      hot-swap to the current registry champion (needs X-Admin-Token)
    GET  /health/live, /health/ready, /metrics (Prometheus), / (demo UI)

Run:  uvicorn sentiment.serving.app:app --host 0.0.0.0 --port 8000 --workers 1
(Scale with replicas, not workers: each worker would load its own model copy, and Prometheus multi-process
mode would be required.)
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from sentiment.config import PipelineConfig, configure_caches, get_config
from sentiment.logging_utils import get_logger, setup_logging
from sentiment.models.bundle import Prediction, Predictor
from sentiment.monitoring.drift import DriftMonitor
from sentiment.preprocessing.cleaner import text_features
from sentiment.registry import LocalRegistry
from sentiment.serving import metrics as M
from sentiment.serving.runtime import JobStore, MicroBatcher, Overloaded, PredictionLogger, TokenBucketLimiter, TTLCache

log = get_logger("sentiment.api")
STATIC = Path(__file__).parent / "static"


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw texts to classify")


class PredictionOut(BaseModel):
    text_id: str
    label: str
    score: float
    confidence: float
    uncertain: bool
    oov_rate: float
    empty_after_cleaning: bool
    cached: bool = False


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[PredictionOut]


class ServiceState:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        s = cfg.serve
        self.predictor: Predictor | None = None
        self.batcher: MicroBatcher[Prediction] | None = None
        self.cache = TTLCache(s.cache_size, s.cache_ttl_s)
        self.limiter = TokenBucketLimiter(s.rate_limit_per_minute, s.rate_limit_burst)
        self.jobs = JobStore()
        self.drift = DriftMonitor(None, cfg.monitoring.window_size, cfg.monitoring.min_window, cfg.monitoring.psi_alert)
        self.pred_log = PredictionLogger(Path(cfg.paths.logs_dir) / "predictions", s.log_text) if s.log_predictions else None
        self.load_error: str | None = None

    def resolve_model_dir(self) -> Path | None:
        explicit = os.environ.get("MODEL_DIR") or self.cfg.serve.model_dir
        if explicit:
            return Path(explicit)
        reg = LocalRegistry(Path(self.cfg.paths.artifacts_dir) / "registry", self.cfg.registry.model_name)
        return reg.champion_path()

    def load_model(self) -> Predictor:
        path = self.resolve_model_dir()
        if path is None or not path.exists():
            raise FileNotFoundError("no model: set MODEL_DIR or train + promote a champion (`sentiment pipeline`)")
        s = self.cfg.serve
        pred = Predictor.load(path, quantize=s.quantize_transformer, strict_fingerprint=s.strict_fingerprint,
                              uncertainty_margin=s.uncertainty_margin)
        if s.torch_threads and pred.metadata["kind"] == "transformer":
            import torch
            torch.set_num_threads(s.torch_threads)   # overrides the training-time thread count
        self.predictor = pred
        self.cache.clear()   # cached scores belong to the previous model
        self.drift = DriftMonitor(pred.reference_profile, self.cfg.monitoring.window_size,
                                  self.cfg.monitoring.min_window, self.cfg.monitoring.psi_alert)
        M.MODEL_INFO.info({k: str(v) for k, v in pred.info().items() if k != "test_metrics"})
        log.info("serving model %s from %s", pred.version, path)
        return pred

    # Runs on the inference thread: predict + observability side effects, off the event loop.
    def infer(self, texts: list[str]) -> list[Prediction]:
        pred = self.predictor
        assert pred is not None
        out = pred.predict(texts)
        rows = []
        for t, p in zip(texts, out):
            M.PREDICTIONS.labels(p.label).inc()
            M.SCORE.observe(p.score)
            M.OOV.observe(p.oov_rate)
            if p.uncertain:
                M.UNCERTAIN.inc()
            if p.empty_after_cleaning:
                M.EMPTY_AFTER_CLEAN.inc()
            feats = text_features(t)
            feats.update(score=p.score, oov_rate=p.oov_rate, pred=int(p.label == "positive"))
            rows.append(feats)
        self.drift.add(rows)
        if self.pred_log:
            ts = datetime.now(timezone.utc).isoformat()
            self.pred_log.write([
                {"ts": ts, "model": pred.version, "text_sha1": hashlib.sha1(t.encode("utf-8", "replace")).hexdigest(),
                 **({"text": t} if self.pred_log.log_text else {}), **r, "label": p.label, "uncertain": p.uncertain}
                for t, p, r in zip(texts, out, rows)
            ])
        return out


def _client_id(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "anonymous")


def create_app(cfg: PipelineConfig | None = None, predictor: Predictor | None = None) -> FastAPI:
    cfg = cfg or get_config()
    configure_caches(cfg)
    state = ServiceState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        try:
            if predictor is not None:
                state.predictor = predictor
                state.drift = DriftMonitor(predictor.reference_profile, cfg.monitoring.window_size,
                                           cfg.monitoring.min_window, cfg.monitoring.psi_alert)
            else:
                await asyncio.get_running_loop().run_in_executor(None, state.load_model)
        except Exception as exc:  # start anyway: liveness OK, readiness reports the problem
            state.load_error = str(exc)
            log.error("model load failed: %s", exc)
        s = cfg.serve
        state.batcher = MicroBatcher(state.infer, s.microbatch_max_size, s.microbatch_max_wait_ms, s.max_queue)
        await state.batcher.start()
        drift_task = asyncio.create_task(_drift_loop())
        yield
        drift_task.cancel()
        await state.batcher.stop()

    async def _drift_loop() -> None:
        while True:
            await asyncio.sleep(10)
            _refresh_drift()

    def _refresh_drift() -> dict[str, Any]:
        res = state.drift.compute()
        M.DRIFT_WINDOW.set(res["window"])
        for feat, val in res["psi"].items():
            M.DRIFT_PSI.labels(feat).set(val)
        M.DRIFT_ALERT.set(1 if res["alerts"] else 0)
        return res

    # docs_url=None: Swagger UI is served from bundled static files (below), not from a CDN, so /docs also
    # works offline / behind firewalls that block cdn.jsdelivr.net.
    app = FastAPI(title="Sentiment API", version="1.0.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.service = state
    job_tasks: set[asyncio.Task[None]] = set()

    @app.middleware("http")
    async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        route = request.url.path
        if route.startswith("/v1/jobs/"):
            route = "/v1/jobs/{id}"                    # bounded label cardinality
        if route.startswith("/v1/"):
            ok, retry = state.limiter.allow(_client_id(request))
            if not ok:
                M.RATE_LIMITED.inc()
                M.HTTP_REQUESTS.labels(request.method, route, "429").inc()
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429,
                                    headers={"Retry-After": str(max(1, int(retry + 0.999)))})
        started = time.perf_counter()
        response = await call_next(request)
        M.HTTP_LATENCY.labels(route).observe(time.perf_counter() - started)
        M.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        return response

    def _require_model() -> Predictor:
        if state.predictor is None:
            raise HTTPException(503, f"model not loaded: {state.load_error}")
        return state.predictor

    def _validate(texts: list[str], max_items: int) -> None:
        if len(texts) > max_items:
            raise HTTPException(422, f"too many texts: {len(texts)} > {max_items}")
        too_long = [i for i, t in enumerate(texts) if len(t) > cfg.serve.max_text_chars]
        if too_long:
            raise HTTPException(422, f"texts {too_long[:10]} exceed {cfg.serve.max_text_chars} chars")

    async def _predict(texts: list[str]) -> list[PredictionOut]:
        pred = _require_model()
        assert state.batcher is not None
        keys = [TTLCache.key(pred.version, t) for t in texts]
        results: list[PredictionOut | None] = [None] * len(texts)
        todo: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            hit = state.cache.get(k)
            if hit is not None:
                M.CACHE.labels("hit").inc()
                results[i] = hit.model_copy(update={"cached": True})
            else:
                M.CACHE.labels("miss").inc()
                todo.setdefault(k, []).append(i)        # in-request duplicates computed once
        if todo:
            uniq = [texts[idxs[0]] for idxs in todo.values()]
            try:
                preds = await state.batcher.submit(uniq)
            except Overloaded:
                raise HTTPException(503, "server overloaded, retry later", headers={"Retry-After": "1"}) from None
            for (k, idxs), p in zip(todo.items(), preds):
                out = PredictionOut(text_id=k[:16], **asdict(p))
                state.cache.set(k, out)
                for i in idxs:
                    results[i] = out
        return [r for r in results if r is not None]

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        _validate(req.texts, cfg.serve.max_batch_items)
        preds = await _predict(req.texts)
        pred = _require_model()
        return PredictResponse(model_version=pred.version, threshold=pred.threshold, predictions=preds)

    @app.post("/v1/jobs", status_code=202)
    async def create_job(req: PredictRequest) -> dict[str, Any]:
        _require_model()
        _validate(req.texts, cfg.serve.job_max_items)
        job = state.jobs.create(len(req.texts))

        async def run() -> None:
            state.jobs.update(job, status="running")
            out: list[dict[str, Any]] = []
            try:
                step = cfg.serve.microbatch_max_size
                for i in range(0, len(req.texts), step):   # chunked -> interactive traffic interleaves
                    out.extend(p.model_dump() for p in await _predict(req.texts[i:i + step]))
                    state.jobs.update(job, done=len(out))
                state.jobs.update(job, status="succeeded", results=out, finished_at=datetime.now(timezone.utc).isoformat())
            except Exception as exc:
                log.exception("job %s failed", job.id)
                state.jobs.update(job, status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())

        job_task = asyncio.create_task(run())
        job_tasks.add(job_task)                    # strong reference, or the task can be GC'd mid-run
        job_task.add_done_callback(job_tasks.discard)
        return {"job_id": job.id, "status": job.status, "total": job.total, "status_url": f"/v1/jobs/{job.id}"}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return asdict(job)

    @app.get("/v1/model")
    async def model_info() -> dict[str, Any]:
        return _require_model().info()

    @app.get("/v1/drift")
    async def drift() -> dict[str, Any]:
        return _refresh_drift()

    @app.post("/admin/reload")
    async def reload(request: Request) -> dict[str, Any]:
        token = os.environ.get("ADMIN_TOKEN")
        if not token or request.headers.get("x-admin-token") != token:
            raise HTTPException(403, "forbidden")
        try:
            # Load + verify on a worker thread; the swap is a single reference assignment (atomic).
            pred = await asyncio.get_running_loop().run_in_executor(None, state.load_model)
        except Exception as exc:
            raise HTTPException(500, f"reload failed, previous model kept: {exc}") from exc
        state.load_error = None
        return pred.info()

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        if state.predictor is None:
            return JSONResponse({"status": "not_ready", "error": state.load_error}, status_code=503)
        return JSONResponse({"status": "ready", "model": state.predictor.version})

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/docs", include_in_schema=False)
    async def docs() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json", title="Sentiment API - docs",
            swagger_js_url="/static/swagger/swagger-ui-bundle.js", swagger_css_url="/static/swagger/swagger-ui.css",
            swagger_favicon_url="data:,",
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def _make_default_app() -> FastAPI:
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"), json_format=os.environ.get("LOG_JSON", "0") == "1")
    return create_app()


app = _make_default_app()
