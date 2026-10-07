"""Canonical record schema shared by batch + stream ingestion.

Every source is mapped into exactly these columns. Rows that violate the contract are *quarantined* with a
reason (never silently coerced) so data loss is visible in the ingestion manifest.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

CANONICAL_COLUMNS: list[str] = ["id", "text", "label", "label_source", "timestamp", "author", "source"]
LabelSource = Literal["gold", "human", "weak", "none"]


class Record(BaseModel):
    """Row-level contract used by the streaming path (batch path uses the vectorised `validate_frame`)."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: int | None = None
    label_source: LabelSource = "none"
    timestamp: datetime
    author: str | None = None
    source: str = "stream"

    @field_validator("text")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("blank text")
        return v

    @field_validator("label")
    @classmethod
    def _binary(cls, v: int | None) -> int | None:
        if v is not None and v not in (0, 1):
            raise ValueError(f"label must be 0/1, got {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        # Naive timestamps are ambiguous -> mixing them with UTC ones silently corrupts temporal splits.
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v


def stable_id(source: str, text: str) -> str:
    """Deterministic id for sources without one, so re-ingesting the same data is idempotent."""
    return hashlib.sha1(f"{source}\x1f{text}".encode("utf-8", "replace")).hexdigest()[:20]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Series(dtype="string"),
            "text": pd.Series(dtype="string"),
            "label": pd.Series(dtype="Int8"),
            "label_source": pd.Series(dtype="string"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "author": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
        }
    )


def validate_frame(df: pd.DataFrame, require_timestamp: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (valid, rejected); rejected rows carry a ``reject_reason`` column."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing canonical columns: {missing}")

    df = df[CANONICAL_COLUMNS].copy()
    df["id"] = df["id"].astype("string")
    df["text"] = df["text"].astype("string")
    df["author"] = df["author"].astype("string")
    df["source"] = df["source"].astype("string")
    df["label_source"] = df["label_source"].astype("string")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Float64")

    reason = pd.Series(pd.NA, index=df.index, dtype="string")

    def flag(mask: pd.Series, why: str) -> None:
        nonlocal reason
        mask = mask.fillna(True).astype(bool) & reason.isna()
        reason = reason.mask(mask, why)

    flag(df["id"].isna() | (df["id"].str.strip() == ""), "missing_id")
    flag(df["text"].isna() | (df["text"].str.strip() == ""), "empty_text")
    labelled = df["label_source"].isin(["gold", "human", "weak"])
    flag(labelled & df["label"].isna(), "unmapped_label")
    flag(df["label"].notna() & ~df["label"].isin([0, 1]), "bad_label")
    if require_timestamp:
        flag(df["timestamp"].isna(), "bad_timestamp")

    rejected = df[reason.notna()].assign(reject_reason=reason[reason.notna()])
    valid = df[reason.isna()].copy()
    valid["label"] = valid["label"].astype("Int8")
    return valid.reset_index(drop=True), rejected.reset_index(drop=True)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return empty_frame()
    df = pd.DataFrame.from_records(records)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df[CANONICAL_COLUMNS]
