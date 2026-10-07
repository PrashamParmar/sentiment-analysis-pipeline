"""Streaming ingestion: async source -> validate -> dedup -> micro-batch -> partitioned parquet.

Design:
* A bounded ``asyncio.Queue`` gives backpressure. A fast producer blocks instead of growing memory without limit.
* Records are validated one at a time with ``Record``. Invalid ones go to a dead-letter JSONL file (no silent drops).
* Dedup runs over a bounded LRU window of ids, so replays and at-least-once delivery are idempotent.
* Flushes happen on size OR time. Each write goes to a temp file and is then renamed, so a reader never sees a
  half-written parquet file. The final partial batch is flushed on shutdown.

In production the source would be Kafka / Kinesis / Pub/Sub. Any ``AsyncIterator[dict]`` plugs in.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from sentiment.data.schema import Record, records_to_frame
from sentiment.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class StreamStats:
    received: int = 0
    written: int = 0
    duplicates: int = 0
    dead_letter: int = 0
    files: list[str] = field(default_factory=list)


class _LRUSet:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._d: OrderedDict[str, None] = OrderedDict()

    def add(self, key: str) -> bool:
        """Return True if newly added, False if already present."""
        if key in self._d:
            self._d.move_to_end(key)
            return False
        self._d[key] = None
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
        return True


class StreamIngestor:
    def __init__(
        self,
        out_dir: Path,
        flush_size: int = 500,
        flush_interval_s: float = 2.0,
        queue_maxsize: int = 10_000,
        dedup_window: int = 200_000,
        on_batch: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.flush_size = flush_size
        self.flush_interval_s = flush_interval_s
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=queue_maxsize)
        self.seen = _LRUSet(dedup_window)
        self.stats = StreamStats()
        self.on_batch = on_batch
        self._buffer: list[dict[str, Any]] = []
        self.dead_letter_path = self.out_dir / "dead_letter.jsonl"

    async def _produce(self, source: AsyncIterator[dict[str, Any]]) -> None:
        async for raw in source:
            await self.queue.put(raw)   # blocks when full -> backpressure
        await self.queue.put(None)      # sentinel

    def _dead_letter(self, raw: Any, error: str) -> None:
        self.stats.dead_letter += 1
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dead_letter_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"error": error, "record": raw, "at": datetime.now(timezone.utc).isoformat()},
                                default=str) + "\n")

    def _flush(self) -> None:
        if not self._buffer:
            return
        df = records_to_frame(self._buffer)
        self._buffer = []
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        part_dir = self.out_dir / f"date={day}"
        part_dir.mkdir(parents=True, exist_ok=True)
        final = part_dir / f"part-{uuid.uuid4().hex[:12]}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)          # atomic on the same filesystem
        self.stats.written += len(df)
        self.stats.files.append(str(final))
        if self.on_batch:
            self.on_batch(df)
        log.debug("stream flush: %d rows -> %s", len(df), final.name)

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        last_flush = loop.time()
        while True:
            timeout = max(0.0, self.flush_interval_s - (loop.time() - last_flush))
            try:
                raw = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self._flush()
                last_flush = loop.time()
                continue
            if raw is None:
                break
            self.stats.received += 1
            try:
                rec = Record.model_validate(raw)
            except ValidationError as exc:
                self._dead_letter(raw, exc.errors(include_url=False).__repr__())
                continue
            if not self.seen.add(rec.id):
                self.stats.duplicates += 1
                continue
            self._buffer.append(rec.model_dump())
            if len(self._buffer) >= self.flush_size:
                self._flush()
                last_flush = loop.time()
        self._flush()  # graceful shutdown: never lose the tail

    async def run(self, source: AsyncIterator[dict[str, Any]]) -> StreamStats:
        producer = asyncio.create_task(self._produce(source))
        try:
            await self._consume()
        finally:
            producer.cancel()
            self._flush()
        log.info("stream done: %s", {k: v for k, v in vars(self.stats).items() if k != "files"})
        return self.stats


# --------------------------------------------------------------------------------------------- sources
async def replay_csv_source(
    path: Path, source_name: str = "replay", rate_per_s: float | None = None, limit: int | None = None,
    encoding: str = "latin-1",
) -> AsyncIterator[dict[str, Any]]:
    """Replays a Sentiment140-format CSV as if it were a live feed (useful for load/drift testing)."""
    from sentiment.config import get_config
    from sentiment.data.ingest import parse_timestamps

    tz = get_config().ingest.timezone_offsets
    cols = ["label", "id", "timestamp", "query", "author", "text"]
    n = 0
    for chunk in pd.read_csv(path, names=cols, header=None, dtype=str, encoding=encoding, chunksize=5000,
                             keep_default_na=False, encoding_errors="replace"):
        ts = parse_timestamps(chunk["timestamp"], tz)
        for row, t in zip(chunk.itertuples(index=False), ts):
            yield {
                "id": f"{source_name}:{row.id}",
                "text": row.text,
                "label": None,
                "label_source": "none",
                "timestamp": t.to_pydatetime() if pd.notna(t) else None,
                "author": row.author,
                "source": source_name,
            }
            n += 1
            if limit and n >= limit:
                return
            if rate_per_s:
                await asyncio.sleep(1 / rate_per_s)


async def http_poll_source(
    url: str, interval_s: float = 5.0, cursor_param: str = "since_id", max_backoff_s: float = 60.0,
    max_polls: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Polls a JSON API returning ``{"items": [...], "next_cursor": ...}`` with exponential backoff."""
    import httpx

    cursor: str | None = None
    backoff = interval_s
    polls = 0
    async with httpx.AsyncClient(timeout=30) as client:
        while max_polls is None or polls < max_polls:
            polls += 1
            try:
                resp = await client.get(url, params={cursor_param: cursor} if cursor else None)
                resp.raise_for_status()
                payload = resp.json()
                backoff = interval_s
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("poll failed (%s); backing off %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff_s)
                continue
            for item in payload.get("items", []):
                yield item
            cursor = payload.get("next_cursor", cursor)
            await asyncio.sleep(interval_s)
