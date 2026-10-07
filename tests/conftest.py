from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment.config import PipelineConfig, PathsConfig

POS = ["love", "great", "awesome", "happy", "amazing", "fantastic", "wonderful", "best", "enjoy", "excited"]
NEG = ["hate", "awful", "terrible", "sad", "worst", "boring", "angry", "upset", "horrible", "annoying"]
FILL = ["the", "movie", "day", "food", "game", "weather", "phone", "show", "team", "music", "trip", "class"]


def make_corpus(n: int = 1200, seed: int = 0, n_authors: int = 150) -> pd.DataFrame:
    """Synthetic tweets: learnable signal + noise + negation + realistic metadata (authors, timestamps)."""
    rng = np.random.default_rng(seed)
    start = datetime(2009, 4, 6, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        y = int(rng.random() < 0.5)
        words = list(rng.choice(FILL, 4))
        sent = rng.choice(POS if y else NEG)
        if rng.random() < 0.15:                     # negated opposite word ("not bad" -> positive)
            sent = "not " + rng.choice(NEG if y else POS)
        words.insert(int(rng.integers(0, 4)), sent)
        if rng.random() < 0.1:                      # label noise
            y = 1 - y
        rows.append({
            "id": f"syn:{i}", "text": " ".join(words) + f" #{i}", "label": y, "label_source": "weak",
            # Authors are active in bursts (8 consecutive tweets), like real users.
            "timestamp": start + timedelta(minutes=int(i * 60)), "author": f"user{i // 8}",
            "source": "synthetic",
        })
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int8")
    return df


@pytest.fixture()
def corpus() -> pd.DataFrame:
    return make_corpus()


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    c = PipelineConfig()
    c.paths = PathsConfig().resolve(tmp_path)
    c.tracking.enabled = False
    c.train.cv_folds = 2
    c.train.search_iter = 2
    c.train.n_jobs = 1
    c.train.max_features_char = 5000
    c.train.max_features_word = 5000
    c.evaluate.bootstrap_samples = 20
    c.evaluate.max_fpr = 0.1
    c.monitoring.min_window = 5
    return c


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One small TF-IDF+LR bundle shared across tests (training is the slow part)."""
    from sentiment.models.bundle import build_model, save_bundle
    from sentiment.splitting import split_dataset
    from sentiment.training import EDGE_CASES

    tmp = tmp_path_factory.mktemp("bundle")
    c = PipelineConfig()
    c.paths = PathsConfig().resolve(tmp)
    c.tracking.enabled = False
    c.train.cv_folds, c.train.search_iter, c.train.n_jobs = 2, 2, 1
    c.train.max_features_char = c.train.max_features_word = 5000
    s = split_dataset(make_corpus(1500), c.split, c.seed)
    model = build_model("tfidf_logreg", c)
    model.fit(s.train, s.val)
    out = tmp / "bundle"
    save_bundle(model, out, threshold=0.6, threshold_info={}, metrics={"test": {"fpr": 0.05, "recall": 0.8}},
                reference_profile={"n": 0, "numeric": {}, "binary": {"has_url": 0.1}},
                golden_texts=EDGE_CASES + s.val["text"].head(10).tolist(), data_versions={}, config_fingerprint="x")
    return out
