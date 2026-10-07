"""Labeling: weak supervision (labeling functions + label model) and active learning.

Weak supervision
----------------
Labeling functions (LFs) vote 1 (positive), 0 (negative) or ABSTAIN (-1). ``LabelModel`` combines the votes
with a weighted majority vote. Each LF's weight is the log-odds of its accuracy, measured on a small gold dev
set when one exists (otherwise weights are uniform). The output is a probability, so low-confidence weak labels
can be filtered out.

Silent-failure guard: once an LF has labelled a text from a surface signal (emoticons, emoji), that signal must
be removed from the training text (``strip_lf_signals``). Otherwise the model learns to detect the LF instead of
sentiment: it scores ~100% on weakly-labelled validation data and fails on real text. Sentiment140 did exactly
this, which is why its training file has the emoticons removed.

Active learning
---------------
``select_for_annotation`` picks the unlabelled texts where the current model is least certain. It removes
duplicates and spreads the picks across text-length bins, so annotators don't spend their budget on
near-identical tweets.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.data.quality import dedup_key
from sentiment.logging_utils import get_logger

log = get_logger(__name__)
ABSTAIN, NEG, POS = -1, 0, 1

# (?<![a-z0-9]) stops "http://" from matching ":/" -- otherwise every URL would be labelled negative.
_POS_EMOTICON = re.compile(r"(?<![a-zA-Z0-9])(?:[:=;]-?[)\]dDpP](?![a-zA-Z0-9])|\(\s?:|<3|\^_?\^)")
_NEG_EMOTICON = re.compile(r"(?<![a-zA-Z0-9])(?:[:=]'?-?(?:[(\[\\]|/(?!/))|D:(?![a-zA-Z0-9/])|</3)")
_POS_EMOJI = set("😀😃😄😁😆😊🙂😍🥰😘❤💕💖👍🎉✨🙌👏😎🤩")
_NEG_EMOJI = set("😢😭😞😔😟😠😡🤬👎💔😒😩😫🙁☹😤")
_NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "cant", "can't", "nothing"}
_POS_WORDS = {"love", "great", "awesome", "amazing", "happy", "excellent", "fantastic", "best", "wonderful", "glad",
              "excited", "perfect", "beautiful", "thanks", "thank", "enjoy", "fun", "yay", "congrats"}
_NEG_WORDS = {"hate", "sad", "awful", "terrible", "worst", "bad", "sucks", "sick", "angry", "upset", "miss",
              "tired", "hurt", "annoying", "horrible", "ugh", "boring", "cry", "sorry", "disappointed"}


@dataclass(frozen=True)
class LabelingFunction:
    name: str
    fn: Callable[[str], int]

    def __call__(self, text: str) -> int:
        return self.fn(text)


def _lexicon_lf(words: set[str], label: int) -> Callable[[str], int]:
    def lf(text: str) -> int:
        toks = re.findall(r"[a-z']+", text.lower())
        for i, t in enumerate(toks):
            if t in words:
                # Negation in the preceding 3 tokens flips meaning -> abstain rather than guess.
                if any(w in _NEGATORS for w in toks[max(0, i - 3):i]):
                    return ABSTAIN
                return label
        return ABSTAIN
    return lf


_vader = None


def _vader_lf(text: str) -> int:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    c = _vader.polarity_scores(text)["compound"]
    return POS if c >= 0.5 else NEG if c <= -0.5 else ABSTAIN


DEFAULT_LFS: list[LabelingFunction] = [
    LabelingFunction("pos_emoticon", lambda t: POS if _POS_EMOTICON.search(t) else ABSTAIN),
    LabelingFunction("neg_emoticon", lambda t: NEG if _NEG_EMOTICON.search(t) else ABSTAIN),
    LabelingFunction("pos_emoji", lambda t: POS if any(c in _POS_EMOJI for c in t) else ABSTAIN),
    LabelingFunction("neg_emoji", lambda t: NEG if any(c in _NEG_EMOJI for c in t) else ABSTAIN),
    LabelingFunction("pos_lexicon", _lexicon_lf(_POS_WORDS, POS)),
    LabelingFunction("neg_lexicon", _lexicon_lf(_NEG_WORDS, NEG)),
    LabelingFunction("vader", _vader_lf),
]


def apply_lfs(texts: Sequence[str], lfs: Sequence[LabelingFunction] = DEFAULT_LFS) -> np.ndarray:
    """Label matrix L of shape (n_texts, n_lfs) with values in {-1, 0, 1}."""
    return np.array([[lf(str(t)) for lf in lfs] for t in texts], dtype=np.int8).reshape(len(texts), len(lfs))


def lf_summary(L: np.ndarray, lfs: Sequence[LabelingFunction] = DEFAULT_LFS, y: np.ndarray | None = None) -> pd.DataFrame:
    """Coverage / overlap / conflict (and empirical accuracy if gold ``y`` is given) per LF."""
    voted = L != ABSTAIN
    rows = []
    for j, lf in enumerate(lfs):
        others = np.delete(voted, j, axis=1).any(axis=1)
        other_votes = np.delete(L, j, axis=1)
        conflict = voted[:, j] & np.any((other_votes != ABSTAIN) & (other_votes != L[:, [j]]), axis=1)
        row = {"lf": lf.name, "coverage": voted[:, j].mean(), "overlap": (voted[:, j] & others).mean(),
               "conflict": conflict.mean()}
        if y is not None and voted[:, j].any():
            row["accuracy"] = float((L[voted[:, j], j] == y[voted[:, j]]).mean())
            row["n_gold_votes"] = int(voted[:, j].sum())
        rows.append(row)
    return pd.DataFrame(rows)


class LabelModel:
    """Accuracy-weighted majority vote producing P(positive)."""

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None

    def fit(self, L: np.ndarray, y_gold: np.ndarray | None = None) -> "LabelModel":
        m = L.shape[1]
        if y_gold is None:
            self.weights = np.ones(m)
            return self
        acc = np.empty(m)
        for j in range(m):
            v = L[:, j] != ABSTAIN
            correct = (L[v, j] == y_gold[v]).sum()
            acc[j] = (correct + 1) / (v.sum() + 2)        # Laplace smoothing for rarely-firing LFs
        acc = np.clip(acc, 0.501, 0.99)                   # an LF worse than chance gets ~0 weight
        self.weights = np.log(acc / (1 - acc))
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        assert self.weights is not None, "call fit() first"
        signed = np.where(L == POS, 1.0, np.where(L == NEG, -1.0, 0.0))
        score = signed @ self.weights
        p = 1.0 / (1.0 + np.exp(-score))
        p[(L == ABSTAIN).all(axis=1)] = np.nan          # nobody voted -> unlabeled, not 0.5
        return p


def weak_label(
    df: pd.DataFrame, gold: pd.DataFrame | None = None, min_confidence: float = 0.7,
    lfs: Sequence[LabelingFunction] = DEFAULT_LFS, strip_signals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weakly label ``df['text']``. Returns (labelled rows, LF summary)."""
    L = apply_lfs(df["text"].tolist(), lfs)
    model = LabelModel()
    summary_y = None
    if gold is not None and len(gold):
        Lg = apply_lfs(gold["text"].tolist(), lfs)
        summary_y = gold["label"].to_numpy(dtype=int)
        model.fit(Lg, summary_y)
        summary = lf_summary(Lg, lfs, summary_y)
    else:
        model.fit(L)
        summary = lf_summary(L, lfs)
    p = model.predict_proba(L)
    conf = np.where(np.isnan(p), 0.0, np.maximum(p, 1 - p))
    keep = conf >= min_confidence
    out = df.loc[keep].copy()
    out["label"] = (p[keep] >= 0.5).astype("int8")
    out["label_confidence"] = conf[keep]
    out["label_source"] = "weak"
    if strip_signals:
        out["text"] = out["text"].map(strip_lf_signals)
    log.info("weak labels: %d/%d rows kept (min_conf=%.2f)", keep.sum(), len(df), min_confidence)
    return out.reset_index(drop=True), summary


def strip_lf_signals(text: str) -> str:
    """Remove the surface cues LFs used, so the classifier cannot learn the labeling rule itself."""
    text = _POS_EMOTICON.sub(" ", text)
    text = _NEG_EMOTICON.sub(" ", text)
    text = "".join(c for c in text if c not in _POS_EMOJI and c not in _NEG_EMOJI)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------------------------ active learning
def uncertainty(p: np.ndarray, strategy: str = "entropy") -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    if strategy == "entropy":
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    if strategy == "margin":
        return 1 - np.abs(2 * p - 1)
    if strategy == "least_confidence":
        return 1 - np.maximum(p, 1 - p)
    raise ValueError(strategy)


def select_for_annotation(
    pool: pd.DataFrame, proba: np.ndarray, k: int, strategy: str = "entropy", n_bins: int = 4
) -> pd.DataFrame:
    """Top-``k`` most uncertain, de-duplicated, round-robin across text-length bins for diversity."""
    df = pool.assign(model_score=proba, uncertainty=uncertainty(proba, strategy), _key=pool["text"].map(dedup_key))
    df = df[df["_key"] != ""].sort_values("uncertainty", ascending=False).drop_duplicates("_key")
    cand = df.head(max(k * 3, k)).copy()
    lengths = cand["text"].str.len()
    cand["_bin"] = pd.qcut(lengths.rank(method="first"), q=min(n_bins, max(len(cand), 1)), labels=False)
    picks: list[pd.Series] = []
    groups = [g for _, g in cand.groupby("_bin")]       # each group is sorted by uncertainty (desc)
    i = 0
    while len(picks) < k and any(len(g) > i for g in groups):
        # One item per bin per round; within a round the most uncertain go first, so a bin holding only
        # confident items never displaces a genuinely uncertain text from another bin.
        round_items = sorted((g.iloc[i] for g in groups if len(g) > i), key=lambda r: -r["uncertainty"])
        picks.extend(round_items[: k - len(picks)])
        i += 1
    out = pd.DataFrame(picks).drop(columns=["_key", "_bin"])
    out["label"] = pd.NA
    return out.reset_index(drop=True)


def export_annotation_batch(batch: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["id", "text", "model_score", "uncertainty", "label"] if c in batch]
    batch[cols].to_csv(path, index=False, encoding="utf-8")
    return path


def merge_annotations(path: Path) -> pd.DataFrame:
    """Load an annotated CSV; accepts 0/1 or negative/positive; drops blanks; rejects anything else loudly."""
    df = pd.read_csv(path, dtype={"id": str, "label": str}, keep_default_na=False)
    raw = df["label"].str.strip().str.lower()
    mapped = raw.map({"0": 0, "1": 1, "negative": 0, "positive": 1, "neg": 0, "pos": 1})
    bad = (raw != "") & mapped.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have invalid labels, e.g. {raw[bad].unique()[:5].tolist()}")
    df = df[mapped.notna()].assign(label=mapped[mapped.notna()].astype("int8"), label_source="human")
    return df
