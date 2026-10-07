"""Data quality and leakage guards. These catch errors that don't crash anything but inflate metrics.

* ``deduplicate``: near-duplicate texts (retweets, copy-paste spam, the same text with different URLs) that land
  on both sides of a split let the model memorise test answers. Keys are built from normalised text, not raw.
* Conflicting labels for the same normalised text are label noise and are dropped, not resolved with keep-first.
* ``check_split_integrity``: asserts there is no id, text or group overlap and that temporal order holds.
* ``find_shortcut_tokens``: flags single tokens that predict the label almost perfectly. These are usually
  artefacts of how the labels were made (e.g. the emoticons used for distant supervision).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentiment.logging_utils import get_logger

log = get_logger(__name__)

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^\s*rt\b[:\s]*", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class LeakageError(AssertionError):
    """Raised when splits share information they must not share."""


def dedup_key(text: str) -> str:
    t = _RT.sub(" ", str(text))
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t.lower()).strip()
    return hashlib.sha1(t.encode()).hexdigest()[:16] if t else ""


@dataclass
class DedupReport:
    rows_in: int
    no_content: int
    conflicting_label_rows: int
    duplicates: int
    rows_out: int

    def as_dict(self) -> dict[str, int]:
        return vars(self).copy()


def deduplicate(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> tuple[pd.DataFrame, DedupReport]:
    """Drop content-free rows, label-conflicting duplicate groups and remaining duplicates (keeps earliest)."""
    n0 = len(df)
    df = df.assign(_key=df[text_col].map(dedup_key))
    no_content = int((df["_key"] == "").sum())
    df = df[df["_key"] != ""]

    n_labels = df.groupby("_key")[label_col].transform("nunique")
    conflict = n_labels > 1
    n_conflict = int(conflict.sum())
    df = df[~conflict]

    if "timestamp" in df:
        df = df.sort_values("timestamp", kind="stable")
    n_before = len(df)
    df = df.drop_duplicates("_key", keep="first")
    rep = DedupReport(n0, no_content, n_conflict, n_before - len(df), len(df))
    log.info("dedup: %s", rep.as_dict())
    return df.drop(columns="_key").reset_index(drop=True), rep


def check_split_integrity(
    splits: dict[str, pd.DataFrame],
    group_col: str | None = None,
    time_col: str | None = None,
    order: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, int]:
    """Raise ``LeakageError`` if any pair of splits shares ids, normalised text or (optionally) groups."""
    names = [n for n in order if n in splits]
    keys = {n: set(splits[n]["text"].map(dedup_key)) for n in names}
    problems: list[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ov := len(set(splits[a]["id"]) & set(splits[b]["id"])):
                problems.append(f"{ov} shared ids between {a}/{b}")
            if ov := len(keys[a] & keys[b]):
                problems.append(f"{ov} near-duplicate texts between {a}/{b}")
            if group_col and (ov := len(set(splits[a][group_col].dropna()) & set(splits[b][group_col].dropna()))):
                problems.append(f"{ov} shared {group_col} values between {a}/{b}")
    if time_col:
        for a, b in zip(names, names[1:]):
            if len(splits[a]) and len(splits[b]) and splits[a][time_col].max() > splits[b][time_col].min():
                problems.append(f"temporal order violated: max({a}) > min({b})")
    if problems:
        raise LeakageError("; ".join(problems))
    return {n: len(splits[n]) for n in names}


def find_shortcut_tokens(
    texts: pd.Series, labels: pd.Series, min_support_frac: float = 0.002, min_purity: float = 0.985, top: int = 20
) -> list[dict]:
    """Tokens that occur often AND are near-perfectly associated with one class (likely label artefacts)."""
    from sklearn.feature_extraction.text import CountVectorizer

    y = np.asarray(labels, dtype=int)
    min_df = max(5, int(min_support_frac * len(y)))
    vec = CountVectorizer(binary=True, min_df=min_df, token_pattern=r"(?u)[^\s]+", lowercase=True)
    try:
        X = vec.fit_transform(texts.astype(str))
    except ValueError:  # empty vocabulary
        return []
    support = np.asarray(X.sum(axis=0)).ravel()
    pos = np.asarray(X[y == 1].sum(axis=0)).ravel()
    purity = np.maximum(pos, support - pos) / np.maximum(support, 1)
    vocab = np.array(vec.get_feature_names_out())
    idx = np.where(purity >= min_purity)[0]
    idx = idx[np.argsort(-support[idx])][:top]
    return [
        {"token": vocab[i], "support": int(support[i]), "purity": round(float(purity[i]), 4),
         "class": int(pos[i] * 2 >= support[i])}
        for i in idx
    ]
