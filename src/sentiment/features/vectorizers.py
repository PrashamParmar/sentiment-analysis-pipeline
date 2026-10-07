"""Pluggable vectorizers behind one sklearn-compatible interface (``fit`` / ``transform``).

Swapping the embedding mechanism means changing the registry key (``build_vectorizer("word2vec")``). The
classifier, the search space and the serving code stay the same.

| key          | representation                     | trainable on our data | when to use                     |
|--------------|------------------------------------|-----------------------|---------------------------------|
| tfidf        | word 1-2g + char_wb 2-5g TF-IDF    | yes (vocab/idf)       | strong, fast baseline           |
| count        | word 1-2g counts                   | yes                   | NB / interpretable baselines    |
| word2vec     | TF-IDF-weighted mean of vectors    | yes (or GloVe file)   | dense, small model              |
| hf_embed     | mean-pooled transformer embeddings | no (frozen)           | semantic, small labelled sets   |

Silent-failure guards:
* All state (vocabulary, IDF, Word2Vec weights) is fitted in ``fit`` on the TRAIN split only. It lives inside
  the sklearn Pipeline, so cross-validation refits it per fold. Fitting TF-IDF on train+test before splitting
  is the most common silent leak in text classification.
* OOV: sparse vectorizers silently ignore unseen tokens, so a text made only of new slang becomes an all-zero
  vector and gets the intercept-only prediction. ``oov_rate`` measures this, and serving exports it as a
  Prometheus metric and a drift feature.
* Transformer truncation: ``truncation_rate_`` records how many inputs were cut at ``max_length``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

TOKEN_PATTERN = r"(?u)[\w@']+|[!?]"   # keeps "not_neg", "@user", single-char tokens and "!"/"?"


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(TOKEN_PATTERN, text)


def tfidf_vectorizer(max_features_word: int = 200_000, max_features_char: int = 300_000, min_df: int = 2,
                     word_ngram_max: int = 2, use_char: bool = True, sublinear_tf: bool = True) -> FeatureUnion | TfidfVectorizer:
    word = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, ngram_range=(1, word_ngram_max), min_df=min_df, max_df=0.9,
        sublinear_tf=sublinear_tf, max_features=max_features_word, lowercase=False, dtype=np.float32,
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=sublinear_tf,
        max_features=max_features_char, lowercase=False, dtype=np.float32,
    )
    # char n-grams are robust to misspellings and unseen slang -> mitigates OOV drops of the word model.
    return FeatureUnion([("word", word), ("char", char)])


def count_vectorizer(max_features_word: int = 200_000, min_df: int = 2, **_: Any) -> CountVectorizer:
    return CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), min_df=min_df,
                           max_features=max_features_word, lowercase=False, dtype=np.float32)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Average (optionally IDF-weighted) of word vectors. Trains Word2Vec on the fit data or loads GloVe."""

    def __init__(self, vector_size: int = 200, window: int = 5, min_count: int = 3, epochs: int = 10,
                 idf_weighting: bool = True, pretrained_path: str | None = None, seed: int = 42, workers: int = 4) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.idf_weighting = idf_weighting
        self.pretrained_path = pretrained_path
        self.seed = seed
        self.workers = workers

    def fit(self, X: Iterable[str], y: Any = None) -> "Word2VecVectorizer":
        docs = [_tokenize(t) for t in X]
        if self.pretrained_path:
            from gensim.models import KeyedVectors
            self.kv_ = KeyedVectors.load_word2vec_format(self.pretrained_path, binary=False, no_header=True)
        else:
            from gensim.models import Word2Vec
            model = Word2Vec(docs, vector_size=self.vector_size, window=self.window, min_count=self.min_count,
                             epochs=self.epochs, seed=self.seed, workers=self.workers, sg=1)
            self.kv_ = model.wv
        self.dim_ = self.kv_.vector_size
        if self.idf_weighting:
            df: dict[str, int] = {}
            for d in docs:
                for w in set(d):
                    df[w] = df.get(w, 0) + 1
            n = len(docs)
            self.idf_ = {w: float(np.log((1 + n) / (1 + c)) + 1) for w, c in df.items()}
        else:
            self.idf_ = {}
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        out = []
        for text in X:
            toks = [w for w in _tokenize(text) if w in self.kv_.key_to_index]
            if not toks:
                out.append(np.zeros(self.dim_, dtype=np.float32))   # all-OOV -> explicit zero (monitored)
                continue
            weights = np.array([self.idf_.get(w, 1.0) for w in toks], dtype=np.float32)
            vecs = np.stack([self.kv_[w] for w in toks])
            out.append((vecs * weights[:, None]).sum(0) / weights.sum())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim_), np.float32)

    def vocabulary(self) -> set[str]:
        return set(self.kv_.key_to_index)


class HFEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Frozen transformer encoder + attention-masked mean pooling + L2 normalisation (CPU friendly)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 64,
                 batch_size: int = 128, device: str = "cpu") -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def _load(self) -> None:
        if getattr(self, "_model", None) is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def __getstate__(self) -> dict[str, Any]:     # never pickle the torch model; reload from hub cache
        return {k: v for k, v in self.__dict__.items() if k not in ("_model", "_tok")}

    def fit(self, X: Any, y: Any = None) -> "HFEmbeddingVectorizer":
        self._load()
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        import torch
        self._load()
        texts = list(X)
        chunks, truncated = [], 0
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                full = self._tok(batch, add_special_tokens=True, truncation=False)["input_ids"]
                truncated += sum(len(ids) > self.max_length for ids in full)
                enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        self.truncation_rate_ = truncated / max(len(texts), 1)
        return np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, 1), np.float32)


VECTORIZERS: dict[str, Callable[..., Any]] = {
    "tfidf": tfidf_vectorizer,
    "count": count_vectorizer,
    "word2vec": Word2VecVectorizer,
    "hf_embed": HFEmbeddingVectorizer,
}


def build_vectorizer(name: str, **params: Any) -> Any:
    if name not in VECTORIZERS:
        raise KeyError(f"unknown vectorizer '{name}'. options: {sorted(VECTORIZERS)}")
    return VECTORIZERS[name](**params)


def vectorizer_vocabulary(vec: Any) -> set[str] | None:
    """Word-level vocabulary of a fitted vectorizer (None when not applicable, e.g. hf_embed)."""
    if isinstance(vec, FeatureUnion):
        for name, sub in vec.transformer_list:
            if name == "word":
                return vectorizer_vocabulary(sub)
        return None
    if hasattr(vec, "vocabulary_"):
        return {t for t in vec.vocabulary_ if " " not in t}   # unigrams only
    if isinstance(vec, Word2VecVectorizer):
        return vec.vocabulary()
    return None


def oov_rate(cleaned_texts: Iterable[str], vocabulary: set[str] | None) -> np.ndarray:
    """Per-text fraction of tokens unknown to the model (0 when vocabulary is not applicable)."""
    texts = list(cleaned_texts)
    if not vocabulary:
        return np.zeros(len(texts), dtype=np.float32)
    rates = []
    for t in texts:
        toks = _tokenize(t)
        rates.append(sum(w not in vocabulary for w in toks) / len(toks) if toks else 1.0)
    return np.asarray(rates, dtype=np.float32)


def is_sparse(X: Any) -> bool:
    return sparse.issparse(X)
