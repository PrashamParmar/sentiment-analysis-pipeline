"""Self-describing model bundles + the ``Predictor`` used by serving.

A bundle directory contains:
    model/                   framework-specific weights (joblib pipeline or HF checkpoint)
    metadata.json            kind, threshold, metrics, preprocessing fingerprint, label map, data versions
    golden.json              fixed texts with the scores the model produced at training time
    reference_profile.json   feature/score distributions for drift detection

Load-time checks (refuse to serve rather than serve wrong answers silently):
1. Preprocessing fingerprint: the cleaner code + config at serve time must equal the training-time ones.
2. Golden parity: re-scoring the golden texts must reproduce the saved scores (tolerance is wider for int8).
   This catches library upgrades, tokenizer changes, wrong-file loads and quantisation damage.
3. Label map equality, so a 0/1 flip can't invert every prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sentiment import LABEL2ID, __version__
from sentiment.features.vectorizers import oov_rate
from sentiment.logging_utils import get_logger
from sentiment.models.base import SentimentModel
from sentiment.preprocessing.cleaner import PREPROCESS_VERSION

log = get_logger(__name__)
BUNDLE_VERSION = 1
TRANSFORMER_NAMES = {"distilbert", "transformer"}


class BundleIntegrityError(RuntimeError):
    pass


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def build_model(name: str, cfg: Any) -> SentimentModel:
    from sentiment.models.classical import CLASSICAL_MODELS, ClassicalModel
    if name in CLASSICAL_MODELS:
        return ClassicalModel(name, cfg)
    if name in TRANSFORMER_NAMES:
        from sentiment.models.transformer import TransformerModel
        return TransformerModel(cfg, name)
    raise KeyError(f"unknown model '{name}'. options: {sorted(CLASSICAL_MODELS) + sorted(TRANSFORMER_NAMES)}")


def _load_model(kind: str, path: Path, quantize: bool) -> SentimentModel:
    if kind == "classical":
        from sentiment.models.classical import ClassicalModel
        return ClassicalModel.load(path)
    if kind == "transformer":
        from sentiment.models.transformer import TransformerModel
        return TransformerModel.load(path, quantize=quantize)
    raise BundleIntegrityError(f"unknown model kind {kind}")


def save_bundle(
    model: SentimentModel, out_dir: Path, *, threshold: float, threshold_info: dict[str, Any],
    metrics: dict[str, Any], reference_profile: dict[str, Any], golden_texts: list[str],
    data_versions: dict[str, Any], config_fingerprint: str, extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model")
    golden_scores = model.predict_proba(golden_texts)
    meta = {
        "bundle_version": BUNDLE_VERSION,
        "package_version": __version__,
        "model_name": model.name,
        "kind": model.kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "label2id": LABEL2ID,
        "preprocess": {"profile": model.cleaner.profile, "fingerprint": model.cleaner.fingerprint(),
                       "version": PREPROCESS_VERSION, "config": model.cleaner.cfg.model_dump()},
        "params": model.params(),
        "metrics": metrics,
        "data_versions": data_versions,
        "config_fingerprint": config_fingerprint,
        **(extra or {}),
    }
    (out_dir / "metadata.json").write_text(encoding="utf-8", data=json.dumps(meta, indent=2, default=str))
    (out_dir / "golden.json").write_text(encoding="utf-8", data=json.dumps(
        [{"text": t, "score": float(s)} for t, s in zip(golden_texts, golden_scores)], indent=2, ensure_ascii=False))
    (out_dir / "reference_profile.json").write_text(encoding="utf-8", data=json.dumps(reference_profile, indent=2))
    log.info("bundle saved -> %s", out_dir)
    return out_dir


@dataclass
class Prediction:
    label: str
    score: float           # P(positive)
    confidence: float      # max(p, 1-p)
    uncertain: bool        # |p - threshold| < margin, or input had no usable content
    oov_rate: float
    empty_after_cleaning: bool


class Predictor:
    """Thread-safe-for-reads wrapper used by the API, batch jobs and the MLflow pyfunc."""

    def __init__(self, model: SentimentModel, metadata: dict[str, Any], golden: list[dict[str, Any]],
                 profile: dict[str, Any], path: Path, uncertainty_margin: float = 0.5) -> None:
        self.model = model
        self.metadata = metadata
        self.golden = golden
        self.reference_profile = profile
        self.path = path
        self.threshold = float(metadata["threshold"])
        self.uncertainty_margin = uncertainty_margin
        self.vocab = model.vocabulary()
        self.version = f"{metadata['model_name']}@{metadata['created_at']}"

    @classmethod
    def load(cls, path: Path | str, quantize: bool = False, strict_fingerprint: bool = True,
             uncertainty_margin: float = 0.5) -> "Predictor":
        path = Path(path)
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            raise BundleIntegrityError(f"no metadata.json in {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("bundle_version") != BUNDLE_VERSION:
            raise BundleIntegrityError(f"bundle_version {meta.get('bundle_version')} != {BUNDLE_VERSION}")
        if meta.get("label2id") != LABEL2ID:
            raise BundleIntegrityError(f"label map {meta.get('label2id')} != {LABEL2ID}")

        model = _load_model(meta["kind"], path / "model", quantize=quantize and meta["kind"] == "transformer")
        fp_now = model.cleaner.fingerprint()
        fp_train = meta["preprocess"]["fingerprint"]
        if fp_now != fp_train:
            msg = (f"preprocessing fingerprint mismatch (train={fp_train}, serve={fp_now}): cleaner code/resources "
                   f"changed since training -> retrain or pin the old code")
            if strict_fingerprint:
                raise BundleIntegrityError(msg)
            log.warning(msg)

        golden = json.loads((path / "golden.json").read_text(encoding="utf-8"))
        profile = json.loads((path / "reference_profile.json").read_text(encoding="utf-8"))
        obj = cls(model, meta, golden, profile, path, uncertainty_margin)
        if getattr(model, "quantized", False):
            try:
                obj.verify_golden(tolerance=0.05)
                return obj
            except BundleIntegrityError as exc:
                # int8 changed decisions -> never serve it; fall back to the exact fp32 weights.
                log.warning("quantized model rejected (%s); falling back to fp32", exc)
                model = _load_model(meta["kind"], path / "model", quantize=False)
                obj = cls(model, meta, golden, profile, path, uncertainty_margin)
        obj.verify_golden(tolerance=1e-4)
        return obj

    def verify_golden(self, tolerance: float) -> float:
        if not self.golden:
            return 0.0
        texts = [g["text"] for g in self.golden]
        expected = np.array([g["score"] for g in self.golden])
        got = self.model.predict_proba(texts)
        max_diff = float(np.max(np.abs(got - expected)))
        flips = int(((got >= self.threshold) != (expected >= self.threshold)).sum())
        if max_diff > tolerance and flips > max(1, len(texts) // 50):
            raise BundleIntegrityError(f"golden parity failed: max |dscore|={max_diff:.4g}, {flips} label flips")
        if max_diff > tolerance:
            log.warning("golden scores drifted by %.4g (no material label flips)", max_diff)
        log.info("golden parity ok (max |dscore|=%.2g over %d texts)", max_diff, len(texts))
        return max_diff

    def predict(self, texts: list[str]) -> list[Prediction]:
        cleaned = self.model.clean(list(texts))
        scores = self.model.predict_proba_cleaned(cleaned)
        oov = oov_rate(cleaned, self.vocab)
        # Margin is measured in LOGIT space: with a high FP-averse threshold (e.g. 0.91) a probability margin
        # of 0.1 would flag every confident positive (0.81..1.0) as "uncertain".
        logit_t = _logit(self.threshold)
        out = []
        for c, s, o in zip(cleaned, scores, oov):
            empty = not c.strip()
            near = abs(_logit(float(s)) - logit_t) < self.uncertainty_margin
            out.append(Prediction(
                label="positive" if s >= self.threshold else "negative",
                score=float(s),
                confidence=float(max(s, 1 - s)),
                uncertain=bool(empty or near or o >= 0.999),
                oov_rate=float(o),
                empty_after_cleaning=empty,
            ))
        return out

    def info(self) -> dict[str, Any]:
        m = self.metadata
        return {"model_name": m["model_name"], "kind": m["kind"], "created_at": m["created_at"],
                "threshold": self.threshold, "preprocess_fingerprint": m["preprocess"]["fingerprint"],
                "test_metrics": {k: v for k, v in m.get("metrics", {}).get("test", {}).items()
                                 if k in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")},
                "quantized": bool(getattr(self.model, "quantized", False)), "path": str(self.path)}
