"""Evaluation for an asymmetric-cost objective: false positives (negative text called positive) are expensive.

How it works:
1. The model outputs P(positive). The decision threshold is a separate business parameter, chosen on the
   VALIDATION split so that FPR <= ``max_fpr`` while recall is maximised (or so that expected cost is minimal).
2. The test split is scored once with that frozen threshold. Tuning the threshold on test is a silent leak that
   makes the reported FPR look met when it isn't.
3. Point metrics come with bootstrap confidence intervals. On ~350 gold tweets, a 2-point F1 gain is noise.
4. Slice metrics (short/long, negation, emoji, sarcasm markers, month) expose failures that the averages hide,
   and an alert is raised when a slice's FPR goes above ``alert_factor * max_fpr``.
5. Calibration (ECE, Brier) is reported because the FPR threshold transfers to production only if scores mean
   the same thing there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdChoice:
    threshold: float
    strategy: str
    feasible: bool
    val_fpr: float
    val_recall: float
    val_precision: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def choose_threshold(
    y: np.ndarray, p: np.ndarray, strategy: str = "max_fpr", max_fpr: float = 0.05,
    cost_fp: float = 5.0, cost_fn: float = 1.0,
) -> ThresholdChoice:
    y, p = np.asarray(y, int), np.asarray(p, float)
    if strategy == "max_fpr":
        fpr, tpr, thr = roc_curve(y, p)
        finite = np.isfinite(thr)
        fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
        ok = fpr <= max_fpr
        feasible = bool(ok.any())
        if feasible:
            cand = np.where(ok)[0]
            idx = cand[np.argmax(tpr[cand])]
        else:  # cannot meet target -> strictest threshold, flagged as infeasible
            idx = int(np.argmin(fpr))
        t = float(thr[idx])
    elif strategy == "cost":
        grid = np.unique(np.quantile(p, np.linspace(0, 1, 501)))
        costs = [cost_fp * ((p >= t) & (y == 0)).sum() + cost_fn * ((p < t) & (y == 1)).sum() for t in grid]
        t, feasible = float(grid[int(np.argmin(costs))]), True
    else:
        raise ValueError(strategy)
    tn, fp, fn, tp = _counts(y, (p >= t).astype(int))
    return ThresholdChoice(
        threshold=t, strategy=strategy, feasible=feasible,
        val_fpr=fp / max(fp + tn, 1), val_recall=tp / max(tp + fn, 1), val_precision=tp / max(tp + fp, 1),
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = _counts(y, pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": tn / max(tn + fp, 1),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)) if both else float("nan"),
        "ece": expected_calibration_error(y, p),
        "mcc": float(matthews_corrcoef(y, pred)) if both else float("nan"),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
        "actual_positive_rate": float(y.mean()) if len(y) else float("nan"),
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 500, alpha: float = 0.05, seed: int = 0,
    keys: tuple[str, ...] = ("precision", "recall", "f1", "fpr", "roc_auc"),
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, int), np.asarray(p, float)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.nanquantile(v, alpha / 2)), float(np.nanquantile(v, 1 - alpha / 2))] for k, v in samples.items()}


def build_slices(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks over a frame that has ``text_features`` columns (+ optional timestamp)."""
    s: dict[str, pd.Series] = {
        "len:short(<=5w)": frame["n_words"] <= 5,
        "len:medium(6-15w)": frame["n_words"].between(6, 15),
        "len:long(>15w)": frame["n_words"] > 15,
    }
    for flag in ("has_negation", "has_emoji", "has_emoticon", "has_url", "has_mention", "has_hashtag",
                 "has_sarcasm_marker", "has_elongation"):
        if flag in frame:
            s[f"{flag}=1"] = frame[flag].astype(bool)
            s[f"{flag}=0"] = ~frame[flag].astype(bool)
    if "caps_ratio" in frame:
        s["shouting(caps>50%)"] = frame["caps_ratio"] > 0.5
    if "timestamp" in frame and frame["timestamp"].notna().any():
        month = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime("%Y-%m")
        for m in sorted(month.dropna().unique()):
            s[f"month={m}"] = month == m
    return s


def slice_metrics(
    frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float, min_support: int = 50,
    max_fpr: float = 0.05, alert_factor: float = 1.5,
) -> pd.DataFrame:
    y, p = np.asarray(y, int), np.asarray(p, float)
    rows = []
    for name, mask in build_slices(frame).items():
        m = np.asarray(mask, bool)
        n = int(m.sum())
        if n == 0:
            continue
        met = binary_metrics(y[m], p[m], threshold)
        negatives = int((y[m] == 0).sum())
        rows.append({
            "slice": name, "n": n, "negatives": negatives, "positive_rate": met["actual_positive_rate"],
            "precision": met["precision"], "recall": met["recall"], "f1": met["f1"], "fpr": met["fpr"],
            "fp": met["fp"], "roc_auc": met["roc_auc"],
            "low_support": n < min_support,
            "fpr_alert": bool(negatives >= min_support and met["fpr"] > alert_factor * max_fpr),
        })
    return pd.DataFrame(rows)


def save_plots(y: np.ndarray, p: np.ndarray, threshold: float, out_dir: Path, prefix: str = "") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    y, p = np.asarray(y, int), np.asarray(p, float)
    paths: list[Path] = []
    if len(np.unique(y)) < 2:
        return paths

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    ax = axes[0, 0]
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n({v / cm[i].sum():.1%})", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred neg", "pred pos"])
    ax.set_yticks([0, 1], ["true neg", "true pos"])
    ax.set_title(f"Confusion matrix @ t={threshold:.3f}")

    fpr, tpr, _ = roc_curve(y, p)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.axvline(((p >= threshold) & (y == 0)).sum() / max((y == 0).sum(), 1), color="red", ls=":", label="operating FPR")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (recall)")
    ax.set_title("ROC")
    ax.legend()

    prec, rec, _ = precision_recall_curve(y, p)
    ax = axes[1, 0]
    ax.plot(rec, prec, label=f"AP={average_precision_score(y, p):.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall")
    ax.legend()

    ax = axes[1, 1]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(xs, ys, "o-", label=f"ECE={expected_calibration_error(y, p):.3f}")
    ax.set_xlabel("mean predicted P(pos)")
    ax.set_ylabel("observed positive rate")
    ax.set_title("Calibration")
    ax.legend()

    fig.tight_layout()
    path = out_dir / f"{prefix}evaluation.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths
