"""Serving building blocks: micro-batcher, TTL-LRU cache, token-bucket rate limiter, async job store,
prediction logger. Each is in-process. For multi-replica deployments the cache and limiter move to Redis behind
the same methods. The batcher stays per-process by design."""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, TypeVar

from sentiment.serving import metrics as M

T = TypeVar("T")


class Overloaded(RuntimeError):
    pass


class MicroBatcher(Generic[T]):
    """Coalesces concurrent requests into one model call (max ``max_batch`` texts or ``max_wait_ms``).

    Inference runs on ONE dedicated worker thread: torch/BLAS already parallelise inside a call, so more
    threads would oversubscribe the CPU, and the event loop stays free for I/O.
    """

    def __init__(self, fn: Callable[[list[str]], list[T]], max_batch: int = 64, max_wait_ms: int = 10,
                 max_queue: int = 2000) -> None:
        self.fn = fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.max_queue = max_queue
        self.queue: asyncio.Queue[tuple[list[str], asyncio.Future[list[T]]]] = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def submit(self, texts: list[str]) -> list[T]:
        if self.queue.qsize() >= self.max_queue:
            M.OVERLOADED.inc()
            raise Overloaded("inference queue full")
        fut: asyncio.Future[list[T]] = asyncio.get_running_loop().create_future()
        await self.queue.put((texts, fut))
        M.QUEUE_DEPTH.set(self.queue.qsize())
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            n = len(items[0][0])
            deadline = loop.time() + self.max_wait
            while n < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                items.append(item)
                n += len(item[0])
            M.QUEUE_DEPTH.set(self.queue.qsize())
            live = [(t, f) for t, f in items if not f.cancelled()]   # client may have disconnected
            if not live:
                continue
            flat = [t for texts, _ in live for t in texts]
            M.BATCH_SIZE.observe(len(flat))
            started = time.perf_counter()
            try:
                preds = await loop.run_in_executor(self.executor, self.fn, flat)
            except Exception as exc:
                for _, f in live:
                    if not f.done():
                        f.set_exception(exc)
                continue
            finally:
                M.INFER_LATENCY.observe(time.perf_counter() - started)
            i = 0
            for texts, f in live:
                if not f.done():
                    f.set_result(preds[i:i + len(texts)])
                i += len(texts)


class TTLCache:
    """LRU cache with per-entry TTL; keys are hashed so memory is bounded regardless of text length."""

    def __init__(self, maxsize: int = 50_000, ttl_s: float = 3600) -> None:
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    @staticmethod
    def key(namespace: str, text: str) -> str:
        return hashlib.blake2b(f"{namespace}\x1f{text}".encode("utf-8", "replace"), digest_size=16).hexdigest()

    def get(self, key: str) -> Any | None:
        hit = self._d.get(key)
        if hit is None or hit[0] < time.monotonic():
            if hit is not None:
                del self._d[key]
            return None
        self._d.move_to_end(key)
        return hit[1]

    def set(self, key: str, value: Any) -> None:
        self._d[key] = (time.monotonic() + self.ttl, value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class TokenBucketLimiter:
    """Per-client token bucket: ``rate_per_minute`` sustained, ``burst`` instantaneous."""

    def __init__(self, rate_per_minute: int, burst: int, max_clients: int = 100_000) -> None:
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self.max_clients = max_clients
        self._b: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, client: str, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        tokens, last = self._b.get(client, (self.burst, now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens >= cost:
            self._b[client] = (tokens - cost, now)
            self._b.move_to_end(client)
            if len(self._b) > self.max_clients:
                self._b.popitem(last=False)
            return True, 0.0
        self._b[client] = (tokens, now)
        return False, (cost - tokens) / self.rate if self.rate else 60.0


@dataclass
class Job:
    id: str
    total: int
    status: str = "queued"          # queued | running | succeeded | failed
    done: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    error: str | None = None
    results: list[dict[str, Any]] | None = None


class JobStore:
    def __init__(self, max_jobs: int = 200) -> None:
        self.max_jobs = max_jobs
        self.jobs: OrderedDict[str, Job] = OrderedDict()

    def create(self, total: int) -> Job:
        job = Job(id=uuid.uuid4().hex, total=total)
        self.jobs[job.id] = job
        while len(self.jobs) > self.max_jobs:   # evict oldest finished first
            victim = next((k for k, j in self.jobs.items() if j.status in ("succeeded", "failed")), None)
            self.jobs.pop(victim or next(iter(self.jobs)))
        self._gauge()
        return job

    def _gauge(self) -> None:
        for state in ("queued", "running", "succeeded", "failed"):
            M.JOBS.labels(state).set(sum(j.status == state for j in self.jobs.values()))

    def update(self, job: Job, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(job, k, v)
        self._gauge()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


class PredictionLogger:
    """Append-only daily JSONL of features + predictions (raw text only if explicitly enabled)."""

    def __init__(self, directory: Path, log_text: bool = False) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_text = log_text
        self._lock = threading.Lock()

    def write(self, rows: list[dict[str, Any]]) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock, open(self.dir / f"predictions-{day}.jsonl", "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
