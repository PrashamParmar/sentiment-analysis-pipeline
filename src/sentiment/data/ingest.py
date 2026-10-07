"""Batch ingestion: CSV / JSONL files and SQL databases -> canonical parquet + manifest.

Silent-failure guards implemented here:
* Mixed timezone abbreviations ("PDT" vs "UTC" in the same dataset) are normalised to UTC explicitly.
  ``pd.to_datetime`` would otherwise drop or mis-parse them, silently corrupting temporal splits.
* Labels are mapped with an explicit table. Unknown labels are quarantined, never coerced.
* Raw-file SHA-256 and row accounting are written to a manifest, so data versions are traceable
  (pairs with DVC, see dvc.yaml).
* Duplicate ids with conflicting labels are dropped and reported, rather than keep-first.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentiment.config import PipelineConfig, SourceConfig
from sentiment.data.schema import CANONICAL_COLUMNS, stable_id, validate_frame
from sentiment.logging_utils import get_logger

log = get_logger(__name__)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def parse_timestamps(values: pd.Series, tz_offsets: dict[str, str]) -> pd.Series:
    """Parse heterogeneous timestamps to tz-aware UTC.

    Handles Twitter style ``Mon Apr 06 22:19:45 PDT 2009`` (abbreviation -> numeric offset) and ISO-8601.
    Unparseable values become NaT and are rejected downstream (visible in the manifest).
    """
    s = values.astype("string")
    if tz_offsets:
        pattern = r"\b(" + "|".join(map(re.escape, tz_offsets)) + r")\b"
        s = s.str.replace(pattern, lambda m: tz_offsets[m.group(1)], regex=True)
    out = pd.to_datetime(s, format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce")
    todo = out.isna() & s.notna()
    if todo.any():
        iso = pd.to_datetime(s[todo], format="ISO8601", utc=True, errors="coerce")
        out = out.where(~todo, iso)
    return out


def _read_chunks(src: SourceConfig, root: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    if src.kind == "csv":
        path = root / src.path if src.path else None
        if path is None or not path.exists():
            raise FileNotFoundError(f"[{src.name}] csv not found: {path}")
        yield from pd.read_csv(
            path,
            encoding=src.encoding,
            header=0 if src.has_header else None,
            names=None if src.has_header else src.columns,
            dtype=str,
            keep_default_na=False,   # the literal tweet "NA" or "null" is text, not a missing value
            na_values=[""],
            chunksize=chunk_size,
            on_bad_lines="warn",
            encoding_errors="replace",
        )
    elif src.kind == "jsonl":
        path = root / src.path if src.path else None
        if path is None or not path.exists():
            raise FileNotFoundError(f"[{src.name}] jsonl not found: {path}")
        yield from pd.read_json(path, lines=True, dtype=False, chunksize=chunk_size, encoding=src.encoding)
    elif src.kind == "sql":
        from sqlalchemy import create_engine, text

        if not (src.uri and src.query):
            raise ValueError(f"[{src.name}] sql source needs uri + query")
        engine = create_engine(src.uri)
        with engine.connect() as conn:
            yield from pd.read_sql_query(text(src.query), conn, chunksize=chunk_size)
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(src.kind)


def to_canonical(chunk: pd.DataFrame, src: SourceConfig, tz_offsets: dict[str, str]) -> pd.DataFrame:
    """Map a raw chunk to canonical columns (no validation yet)."""
    df = chunk.rename(columns=src.column_map)
    out = pd.DataFrame(index=df.index)
    out["text"] = df["text"].astype("string") if "text" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    if "id" in df:
        out["id"] = df["id"].astype("string")
    else:
        out["id"] = out["text"].fillna("").map(lambda t: stable_id(src.name, t)).astype("string")
    # Namespace ids by source so that the same numeric id in two systems never collides.
    out["id"] = src.name + ":" + out["id"]
    if "label" in df and src.label_map:
        out["label"] = df["label"].astype("string").str.strip().map(src.label_map).astype("Float64")
    elif "label" in df:
        out["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Float64")
    else:
        out["label"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    out["label_source"] = src.label_source
    out["timestamp"] = (
        parse_timestamps(df["timestamp"], tz_offsets) if "timestamp" in df
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    )
    out["author"] = df["author"].astype("string") if "author" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    out["source"] = src.name
    return out[CANONICAL_COLUMNS]


def _sample_fraction(src: SourceConfig, root: Path, chunk_size: int) -> float:
    if not src.sample_size:
        return 1.0
    total = sum(len(c) for c in _read_chunks(src, root, chunk_size))
    return min(1.0, src.sample_size / max(total, 1))


def ingest_source(src: SourceConfig, cfg: PipelineConfig, root: Path) -> dict:
    """Ingest one source to ``interim/<name>.parquet`` (+ rejects + manifest). Returns the manifest."""
    out_dir = Path(cfg.paths.interim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frac = _sample_fraction(src, root, cfg.ingest.chunk_size)
    valid_parts, reject_parts, rows_read = [], [], 0

    for i, chunk in enumerate(_read_chunks(src, root, cfg.ingest.chunk_size)):
        rows_read += len(chunk)
        canon = to_canonical(chunk, src, cfg.ingest.timezone_offsets)
        valid, rejected = validate_frame(canon, require_timestamp="timestamp" in chunk.rename(columns=src.column_map))
        if frac < 1.0 and len(valid):
            # Stratified by label inside each chunk -> preserves the original class prior.
            valid = valid.groupby("label", group_keys=False, dropna=False).sample(
                frac=frac, random_state=cfg.seed + i
            )
        valid_parts.append(valid)
        reject_parts.append(rejected)

    df = pd.concat(valid_parts, ignore_index=True)
    rejects = pd.concat(reject_parts, ignore_index=True)

    # Same id, different labels -> label noise we cannot resolve automatically.
    conflict_ids = df.groupby("id")["label"].nunique(dropna=True)
    conflict_ids = conflict_ids[conflict_ids > 1].index
    if len(conflict_ids):
        conflicts = df[df["id"].isin(conflict_ids)].assign(reject_reason="conflicting_label_same_id")
        rejects = pd.concat([rejects, conflicts], ignore_index=True)
        df = df[~df["id"].isin(conflict_ids)]
    n_before = len(df)
    df = df.drop_duplicates("id", keep="first").sort_values("timestamp", kind="stable").reset_index(drop=True)

    df.to_parquet(out_dir / f"{src.name}.parquet", index=False)
    if len(rejects):
        rejects.astype({"timestamp": "string"}).to_parquet(out_dir / f"{src.name}.rejects.parquet", index=False)

    raw_path = root / src.path if src.path else None
    manifest = {
        "source": src.name,
        "kind": src.kind,
        "raw_path": str(raw_path) if raw_path else src.uri,
        "raw_sha256": file_sha256(raw_path) if raw_path and raw_path.exists() else None,
        "rows_read": rows_read,
        "sample_fraction": round(frac, 6),
        "rows_valid": len(df),
        "rows_rejected": len(rejects),
        "duplicate_ids_dropped": n_before - len(df),
        "reject_reasons": rejects["reject_reason"].value_counts().to_dict() if len(rejects) else {},
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())] if len(df) else None,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"{src.name}.manifest.json").write_text(encoding="utf-8", data=json.dumps(manifest, indent=2, default=str))
    log.info("ingested %s: %d valid / %d rejected (%s)", src.name, len(df), len(rejects), manifest["reject_reasons"])
    return manifest


def ingest_all(cfg: PipelineConfig, root: Path | None = None) -> list[dict]:
    from sentiment.config import PROJECT_ROOT

    root = root or PROJECT_ROOT
    if not cfg.ingest.sources:
        raise ValueError("no ingest sources configured")
    return [ingest_source(src, cfg, root) for src in cfg.ingest.sources]
