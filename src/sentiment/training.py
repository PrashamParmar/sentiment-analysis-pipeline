"""End-to-end training run: fit -> threshold on val -> evaluate (val/test/gold + slices + CIs) -> bundle ->
MLflow log/register -> local registry -> champion/challenger promotion.

Discipline: the TEST split is scored exactly once, with a threshold frozen on VALIDATION. The external gold set
(human-labelled, different time period) is reported separately. It is the most honest estimate of real-world
performance, because the training labels are weak (emoticon-derived).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment.config import PipelineConfig
from sentiment.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    choose_threshold,
    save_plots,
    slice_metrics,
)
from sentiment.features.vectorizers import oov_rate
from sentiment.logging_utils import get_logger
from sentiment.models.bundle import build_model, save_bundle
from sentiment.monitoring.drift import build_reference_profile
from sentiment.preprocessing.cleaner import TextCleaner, text_features
from sentiment.registry import LocalRegistry
from sentiment.splitting import Splits
from sentiment.tracking import NullTracker

log = get_logger(__name__)

EDGE_CASES = [
    "", "   ", "NA", "<3", "&lt;3 love it", "not good at all", "not bad, actually pretty great",
    "I can't stand this", "😍😍😍", "😡", "http://example.com", "@someone", "LOVE IT!!!!!!",
    "this is soooooo boring", "yeah right, best day ever /s", "donâ€™t like it", "<b>great</b> product",
    "meh", "a" * 3000,
]


def features_frame(df: pd.DataFrame, cleaner: TextCleaner | None = None) -> pd.DataFrame:
    feats = pd.DataFrame([text_features(t, cleaner) for t in df["text"].tolist()])
    if "timestamp" in df:
        feats["timestamp"] = df["timestamp"].to_numpy()
    return feats


def data_fingerprint(splits: Splits) -> str:
    h = hashlib.sha256()
    for name, part in splits.as_dict().items():
        h.update(name.encode())
        h.update("\n".join(sorted(part["id"].astype(str))).encode())
    return h.hexdigest()[:16]


def evaluate_split(name: str, df: pd.DataFrame, p: np.ndarray, threshold: float, cfg: PipelineConfig,
                   cleaner: TextCleaner, out_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["label"].to_numpy(dtype=int)
    e = cfg.evaluate
    metrics = binary_metrics(y, p, threshold)
    metrics["ci95"] = bootstrap_ci(y, p, threshold, n_boot=e.bootstrap_samples, seed=cfg.seed)
    feats = features_frame(df, cleaner)
    slices = slice_metrics(feats, y, p, threshold, e.slice_min_support, e.max_fpr, e.slice_fpr_alert_factor)
    slices.to_csv(out_dir / f"{name}_slices.csv", index=False)
    metrics["slice_alerts"] = slices.loc[slices["fpr_alert"], "slice"].tolist() if len(slices) else []
    save_plots(y, p, threshold, out_dir, prefix=f"{name}_")
    if metrics["slice_alerts"]:
        log.warning("[%s] slices breaching FPR budget: %s", name, metrics["slice_alerts"])
    return metrics, feats


def _golden_texts(val: pd.DataFrame, seed: int, n: int = 40) -> list[str]:
    sample = val.groupby("label", group_keys=False).sample(n=min(n // 2, val["label"].value_counts().min()),
                                                           random_state=seed)
    return EDGE_CASES + sample["text"].astype(str).tolist()


def train_and_register(
    name: str, cfg: PipelineConfig, splits: Splits, gold: pd.DataFrame | None, tracker: NullTracker,
    promote: bool = True, data_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{name}-{stamp}"
    report_dir = Path(cfg.paths.reports_dir) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    data_fp = data_fingerprint(splits)
    model = build_model(name, cfg)

    with tracker.run(run_name, tags={"model": name, "data_fingerprint": data_fp, "split": cfg.split.strategy}):
        tracker.log_params({"model": name, "config": cfg.model_dump(mode="json"), "data_fingerprint": data_fp,
                            "rows": {k: len(v) for k, v in splits.as_dict().items()}})
        log.info("training %s on %d rows", name, len(splits.train))
        info = model.fit(splits.train, splits.val)
        for rec in info.get("history", []):
            tracker.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=int(rec["epoch"]))

        e = cfg.evaluate
        p_val = model.predict_proba(splits.val["text"].tolist())
        choice = choose_threshold(splits.val["label"].to_numpy(int), p_val, e.threshold_strategy, e.max_fpr,
                                  e.cost_fp, e.cost_fn)
        log.info("threshold %.4f (%s, feasible=%s): val FPR=%.4f recall=%.4f", choice.threshold, choice.strategy,
                 choice.feasible, choice.val_fpr, choice.val_recall)
        thr = choice.threshold

        metrics: dict[str, Any] = {"threshold": thr, "threshold_feasible": choice.feasible}
        metrics["val"], val_feats = evaluate_split("val", splits.val, p_val, thr, cfg, model.cleaner, report_dir)
        p_test = model.predict_proba(splits.test["text"].tolist())
        metrics["test"], _ = evaluate_split("test", splits.test, p_test, thr, cfg, model.cleaner, report_dir)
        if gold is not None and len(gold):
            p_gold = model.predict_proba(gold["text"].tolist())
            metrics["gold"], _ = evaluate_split("gold", gold, p_gold, thr, cfg, model.cleaner, report_dir)
        metrics["train_info"] = {k: v for k, v in info.items() if k not in ("history",)}

        # Reference distribution for online drift detection (validation = most recent pre-test data).
        val_feats["oov_rate"] = oov_rate(model.clean(splits.val["text"].tolist()), model.vocabulary())
        val_feats["score"] = p_val
        val_feats["pred"] = (p_val >= thr).astype(int)
        profile = build_reference_profile(val_feats)

        staging = Path(cfg.paths.artifacts_dir) / "staging" / run_name
        save_bundle(model, staging, threshold=thr, threshold_info=choice.as_dict(), metrics=metrics,
                    reference_profile=profile, golden_texts=_golden_texts(splits.val, cfg.seed),
                    data_versions={**(data_versions or {}), "split_fingerprint": data_fp},
                    config_fingerprint=cfg.fingerprint(), extra={"run_name": run_name, "mlflow_run_id": tracker.run_id})

        summary = {k: metrics[k] for k in ("val", "test", "gold") if k in metrics}
        (report_dir / "metrics.json").write_text(encoding="utf-8", data=json.dumps(metrics, indent=2, default=str))
        tracker.log_metrics({f"{split}.{k}": v for split, m in summary.items() for k, v in m.items()
                             if k not in ("ci95", "slice_alerts")})
        tracker.log_metrics({"threshold": thr})
        tracker.log_artifacts(report_dir, "reports")

        mlflow_version = None
        try:
            mlflow_version = tracker.register_bundle(staging, cfg.registry.model_name)
        except Exception as exc:
            log.warning("MLflow model registration failed (%s); local registry still updated", exc)

        registry = LocalRegistry(Path(cfg.paths.artifacts_dir) / "registry", cfg.registry.model_name)
        version = registry.register(staging, metrics=metrics, data_fingerprint=data_fp, run_id=tracker.run_id,
                                    mlflow_version=mlflow_version, model_name=name)
        promoted, reasons = (registry.promote(version, cfg.registry.promotion) if promote else (False, ["promote=False"]))
        if promoted and mlflow_version:
            try:
                tracker.set_alias(cfg.registry.model_name, mlflow_version, "champion")
            except Exception as exc:
                log.warning("could not set MLflow alias: %s", exc)

    result = {
        "run_name": run_name, "model": name, "version": version, "promoted": promoted, "promotion_notes": reasons,
        "threshold": thr, "report_dir": str(report_dir),
        **{f"{s}_{k}": round(float(m[k]), 4) for s, m in summary.items() for k in ("precision", "recall", "f1", "fpr", "roc_auc")},
    }
    (report_dir / "summary.json").write_text(encoding="utf-8", data=json.dumps(result, indent=2, default=str))
    return result
