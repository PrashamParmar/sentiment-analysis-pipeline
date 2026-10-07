"""Classical baselines: (TF-IDF | counts | Word2Vec | frozen transformer embeddings) + linear classifier.

Training is a randomized hyperparameter search with leakage-safe cross-validation:
* The vectorizer sits inside the Pipeline, so each CV fold fits its own vocabulary and IDF.
* Folds follow the data's structure. ``TimeSeriesSplit`` is used for temporal data (each fold validates on
  later data than it trained on). Otherwise ``StratifiedGroupKFold`` keeps an author's tweets out of both
  sides of a fold.
* The search metric is recall at FPR <= max_fpr, which is the business objective. Accuracy would favour
  models that produce more false positives.

Cleaning is stateless (no fitted parameters), so texts are cleaned once before the search. This is
leakage-free, avoids cleaning again in every fold, and the final Pipeline still contains the cleaner so
serving uses identical code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, TimeSeriesSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from sentiment.config import PipelineConfig
from sentiment.data.quality import find_shortcut_tokens
from sentiment.features.vectorizers import build_vectorizer, vectorizer_vocabulary
from sentiment.logging_utils import get_logger
from sentiment.models.base import SentimentModel
from sentiment.preprocessing.cleaner import CleanerTransformer, TextCleaner

log = get_logger(__name__)

# name -> (vectorizer key, classifier key, cleaner profile)
CLASSICAL_MODELS: dict[str, tuple[str, str, str]] = {
    "tfidf_logreg": ("tfidf", "logreg", "classic"),
    "tfidf_svm": ("tfidf", "svm", "classic"),
    "count_nb": ("count", "nb", "classic"),
    "w2v_logreg": ("word2vec", "logreg", "classic"),
    "embed_logreg": ("hf_embed", "logreg", "transformer"),
}


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


class ClassicalModel(SentimentModel):
    kind = "classical"

    def __init__(self, name: str, cfg: PipelineConfig) -> None:
        if name not in CLASSICAL_MODELS:
            raise KeyError(f"unknown classical model {name}; options {sorted(CLASSICAL_MODELS)}")
        self.name = name
        self.cfg = cfg
        self.vec_name, self.clf_name, profile = CLASSICAL_MODELS[name]
        self.cleaner = TextCleaner(cfg.preprocess, profile)
        self.pipeline: Pipeline | None = None
        self.best_params_: dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------ build
    def _vectorizer(self) -> Any:
        t = self.cfg.train
        if self.vec_name == "tfidf":
            return build_vectorizer("tfidf", max_features_word=t.max_features_word, max_features_char=t.max_features_char)
        if self.vec_name == "count":
            return build_vectorizer("count", max_features_word=t.max_features_word)
        if self.vec_name == "word2vec":
            w = t.word2vec
            return build_vectorizer("word2vec", vector_size=w.vector_size, window=w.window, min_count=w.min_count,
                                    epochs=w.epochs, pretrained_path=w.pretrained_path, seed=self.cfg.seed,
                                    workers=t.n_jobs)
        e = t.embedding
        return build_vectorizer("hf_embed", model_name=e.model_name, max_length=e.max_length, batch_size=e.batch_size)

    def _classifier(self) -> Any:
        cw = self.cfg.train.class_weight
        if self.clf_name == "logreg":
            return LogisticRegression(C=1.0, class_weight=cw, max_iter=3000, solver="liblinear")
        if self.clf_name == "svm":
            return LinearSVC(C=0.5, class_weight=cw)
        return ComplementNB(alpha=0.5)

    def _search_space(self) -> dict[str, Any]:
        space: dict[str, Any] = {}
        if self.clf_name in ("logreg", "svm"):
            space["clf__C"] = loguniform(0.05, 10) if self.clf_name == "logreg" else loguniform(0.01, 2)
        else:
            space["clf__alpha"] = loguniform(0.05, 2)
        if self.vec_name == "tfidf":
            space["vec__word__ngram_range"] = [(1, 1), (1, 2), (1, 3)]
            space["vec__word__min_df"] = [1, 2, 3]
            space["vec__char"] = ["drop", build_vectorizer("tfidf").transformer_list[1][1]]
        elif self.vec_name == "count":
            space["vec__ngram_range"] = [(1, 1), (1, 2)]
        return space

    def _cv(self, train: pd.DataFrame) -> tuple[Any, np.ndarray | None]:
        k = self.cfg.train.cv_folds
        if self.cfg.split.strategy == "temporal":
            return TimeSeriesSplit(n_splits=k), None     # train is already time-sorted
        groups = train[self.cfg.split.group_col].fillna(train["id"]).to_numpy()
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.cfg.seed), groups

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
        t = self.cfg.train
        if self.cfg.split.strategy == "temporal" and "timestamp" in train:
            train = train.sort_values("timestamp", kind="stable")
        if self.vec_name == "hf_embed" and t.embedding.train_subsample and len(train) > t.embedding.train_subsample:
            train = train.sample(t.embedding.train_subsample, random_state=self.cfg.seed).sort_values("timestamp")
        y = train["label"].to_numpy(dtype=int)
        X = self.cleaner.clean_many(train["text"].tolist())

        shortcuts = find_shortcut_tokens(pd.Series(X), pd.Series(y))
        if shortcuts:
            log.warning("possible label-artefact tokens (too predictive): %s", shortcuts[:5])

        vec = self._vectorizer()
        precomputed = self.vec_name == "hf_embed"
        if precomputed:
            # Frozen encoder = stateless -> embed once; search only over the classifier.
            log.info("embedding %d texts with %s", len(X), t.embedding.model_name)
            X_fit: Any = vec.fit(X).transform(X)
            search_pipe = Pipeline([("vec", FunctionTransformer()), ("clf", self._classifier())])
        else:
            X_fit = X
            search_pipe = Pipeline([("vec", vec), ("clf", self._classifier())])

        cv, groups = self._cv(train)
        scorer = make_scorer(recall_at_fpr, response_method=("decision_function", "predict_proba"),
                             max_fpr=self.cfg.evaluate.max_fpr)
        expensive = self.vec_name == "word2vec"
        search = RandomizedSearchCV(
            search_pipe, self._search_space(), n_iter=max(2, t.search_iter // (4 if expensive else 1)),
            scoring={"recall_at_fpr": scorer, "roc_auc": "roc_auc"}, refit="recall_at_fpr", cv=cv,
            n_jobs=1 if expensive else t.n_jobs, random_state=self.cfg.seed, error_score="raise",
            return_train_score=True,
        )
        log.info("hyperparameter search: %s, %d candidates x %d folds", self.name, search.n_iter, t.cv_folds)
        search.fit(X_fit, y, groups=groups)
        self.best_params_ = {k: ("on" if hasattr(v, "get_params") else str(v)) for k, v in search.best_params_.items()}
        best = search.best_estimator_
        fitted_vec = vec if precomputed else best.named_steps["vec"]
        clf = best.named_steps["clf"]

        if self.clf_name == "svm":
            # LinearSVC has no probabilities -> sigmoid calibration (internal CV on train only).
            Xv = fitted_vec.transform(X) if not precomputed else X_fit
            clf = CalibratedClassifierCV(LinearSVC(C=clf.C, class_weight=t.class_weight), method="sigmoid", cv=3)
            clf.fit(Xv, y)

        if list(clf.classes_) != [0, 1]:
            raise AssertionError(f"unexpected class order {clf.classes_} -> probabilities would be flipped")

        self.pipeline = Pipeline([
            ("clean", CleanerTransformer(self.cfg.preprocess.model_dump(), self.cleaner.profile).fit(None)),
            ("vec", fitted_vec),
            ("clf", clf),
        ])
        res = search.cv_results_
        i = search.best_index_
        info = {
            "best_params": self.best_params_,
            "cv_recall_at_fpr": float(res["mean_test_recall_at_fpr"][i]),
            "cv_recall_at_fpr_std": float(res["std_test_recall_at_fpr"][i]),
            "cv_roc_auc": float(res["mean_test_roc_auc"][i]),
            "train_recall_at_fpr": float(res["mean_train_recall_at_fpr"][i]),
            "shortcut_tokens": shortcuts[:10],
            "n_train": len(train),
        }
        # Big train/CV gap = overfitting; tiny gap with near-perfect score = suspect leakage.
        info["generalization_gap"] = info["train_recall_at_fpr"] - info["cv_recall_at_fpr"]
        log.info("best %s: %s | cv recall@fpr=%.4f", self.name, self.best_params_, info["cv_recall_at_fpr"])
        return info

    # ------------------------------------------------------------------------------------------ inference
    def predict_proba_cleaned(self, cleaned: list[str]) -> np.ndarray:
        assert self.pipeline is not None, "model not fitted"
        if not cleaned:
            return np.zeros(0, dtype=float)
        return self.pipeline[1:].predict_proba(list(cleaned))[:, 1]   # skip the cleaning step

    def vocabulary(self) -> set[str] | None:
        return vectorizer_vocabulary(self.pipeline.named_steps["vec"]) if self.pipeline else None

    def params(self) -> dict[str, Any]:
        return {"model": self.name, "vectorizer": self.vec_name, "classifier": self.clf_name, **self.best_params_}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "pipeline": self.pipeline, "best_params": self.best_params_,
                     "cfg": self.cfg.model_dump(mode="json")}, path / "model.joblib", compress=3)

    @classmethod
    def load(cls, path: Path, **_: Any) -> "ClassicalModel":
        blob = joblib.load(path / "model.joblib")
        cfg = PipelineConfig(**blob["cfg"])
        obj = cls(blob["name"], cfg)
        obj.pipeline = blob["pipeline"]
        obj.best_params_ = blob["best_params"]
        return obj
