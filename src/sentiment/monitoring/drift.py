"""Data and prediction drift detection.

* ``build_reference_profile``: at training time, quantile-binned distributions of model-agnostic text features,
  the OOV rate and the model score on the validation split are stored inside the model bundle.
* ``DriftMonitor`` (online, in the API): keeps a sliding window of live traffic and computes the Population
  Stability Index (PSI) per feature against that reference. Results go to Prometheus gauges, and PSI > 0.2
  triggers an alert. It needs no labels, so it catches silent degradation before ground truth arrives
  (new slang -> OOV rate up -> score distribution shifts).
* ``evidently_report`` (offline): a full Evidently HTML/JSON report comparing a reference frame with logged
  production traffic.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ("n_words", "n_chars", "caps_ratio", "non_ascii_ratio", "oov_rate", "score")
BINARY_FEATURES = ("has_url", "has_mention", "has_emoji", "has_emoticon", "has_negation", "has_hashtag")
_EPS = 1e-4


def _edges(values: np.ndarray, n_bins: int = 10) -> list[float]:
    """Quantile bin edges that stay informative for zero-inflated / low-cardinality features.

    Silent-failure guard: if a feature is ~always 0 in the reference (e.g. non_ascii_ratio), plain quantiles
    collapse into ONE bin and PSI is identically 0 no matter how much production traffic changes. A split
    just above the minimum is always added, so a "point mass at min vs. the rest" bin always exists.
    """
    qs = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    lo = float(qs[0])
    inner = set(qs[1:-1].tolist())
    inner.add(lo + max(abs(lo) * 1e-9, 1e-9))
    return [-np.inf, *sorted(inner), np.inf]


def _hist(values: np.ndarray, edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=np.asarray(edges, float))
    return (counts / max(counts.sum(), 1)).tolist()


def psi(expected: list[float] | np.ndarray, actual: list[float] | np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), _EPS, None)
    a = np.clip(np.asarray(actual, float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(features: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    prof: dict[str, Any] = {"n": int(len(features)), "numeric": {}, "binary": {}}
    for col in NUMERIC_FEATURES:
        if col in features:
            v = features[col].to_numpy(float)
            edges = _edges(v, n_bins)
            prof["numeric"][col] = {"edges": [None if not np.isfinite(x) else float(x) for x in edges],
                                    "probs": _hist(v, edges), "mean": float(v.mean())}
    for col in BINARY_FEATURES:
        if col in features:
            prof["binary"][col] = float(features[col].astype(float).mean())
    if "pred" in features:
        prof["binary"]["pred_positive"] = float(features["pred"].astype(float).mean())
    return prof


def _edges_from_profile(raw: list[float | None]) -> list[float]:
    out = list(raw)
    out[0], out[-1] = -np.inf, np.inf
    return [float(x) for x in out]


def compare_to_profile(profile: dict[str, Any], current: pd.DataFrame) -> dict[str, float]:
    """PSI per numeric feature; for binary features, PSI of the 2-bin (p, 1-p) distribution."""
    out: dict[str, float] = {}
    for col, ref in profile.get("numeric", {}).items():
        if col in current and len(current):
            edges = _edges_from_profile(ref["edges"])
            out[col] = psi(ref["probs"], _hist(current[col].to_numpy(float), edges))
    for col, p_ref in profile.get("binary", {}).items():
        src = "pred" if col == "pred_positive" else col
        if src in current and len(current):
            p_cur = float(current[src].astype(float).mean())
            out[col] = psi([1 - p_ref, p_ref], [1 - p_cur, p_cur])
    return out


class DriftMonitor:
    """Thread-safe sliding-window PSI monitor used by the serving layer."""

    def __init__(self, profile: dict[str, Any] | None, window_size: int = 2000, min_window: int = 200,
                 psi_alert: float = 0.2) -> None:
        self.profile = profile or {}
        self.window: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.min_window = min_window
        self.psi_alert = psi_alert
        self._lock = threading.Lock()
        self.last: dict[str, float] = {}

    def add(self, rows: list[dict[str, Any]]) -> None:
        with self._lock:
            self.window.extend(rows)

    def compute(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self.window)
        if not self.profile or len(rows) < self.min_window:
            return {"status": "insufficient_data", "window": len(rows), "psi": {}, "alerts": []}
        self.last = compare_to_profile(self.profile, pd.DataFrame(rows))
        alerts = sorted(k for k, v in self.last.items() if v > self.psi_alert)
        return {"status": "drift" if alerts else "ok", "window": len(rows), "psi": self.last, "alerts": alerts}


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Offline drift report with Evidently (supports the >=0.7 API and the legacy 0.4-0.6 API)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in (*NUMERIC_FEATURES, *BINARY_FEATURES, "pred") if c in reference and c in current]
    ref = reference[cols].astype(float)
    cur = current[cols].astype(float)
    html_path = out_dir / "drift_report.html"
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
        snapshot.save_html(str(html_path))
        summary = json.loads(snapshot.json())
    except ImportError:
        from evidently.metric_preset import DataDriftPreset  # type: ignore[no-redef]
        from evidently.report import Report  # type: ignore[no-redef]

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(html_path))
        summary = report.as_dict()
    (out_dir / "drift_report.json").write_text(encoding="utf-8", data=json.dumps(summary, indent=2, default=str))
    return {"html": str(html_path), "columns": cols}
