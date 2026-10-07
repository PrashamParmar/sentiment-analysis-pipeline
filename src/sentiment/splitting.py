"""Train / validation / test splitting with leakage guards.

Strategies
* ``temporal`` (default): train on the past, evaluate on the future. A random split leaks future vocabulary,
  topics and events into training and overstates production accuracy. The cut is made on *timestamps*,
  not row positions, so a timestamp shared by rows on both sides of a boundary can't leak.
* ``stratified``: keeps the class prior identical across splits (use when data has no meaningful time axis).
* ``group``: stratified + no author in two splits (the model must not recognise a user's writing style).

Every strategy runs ``deduplicate`` first and ``check_split_integrity`` last, and fails loudly if a split is
almost single-class, because a temporal cut on label-sorted data silently produces a one-class test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from sentiment.config import SplitConfig
from sentiment.data.quality import check_split_integrity, deduplicate
from sentiment.logging_utils import get_logger

log = get_logger(__name__)
SPLITS = ("train", "val", "test")


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    report: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _temporal(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = df[cfg.time_col]
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} rows without {cfg.time_col}; temporal split impossible")
    q_train = 1 - cfg.val_frac - cfg.test_frac
    t1, t2 = ts.quantile(q_train), ts.quantile(1 - cfg.test_frac)
    train = df[ts < t1]
    val = df[(ts >= t1) & (ts < t2)]
    test = df[ts >= t2]
    return train, val, test


def _stratified(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rest, test = train_test_split(df, test_size=cfg.test_frac, stratify=df["label"], random_state=seed)
    rel_val = cfg.val_frac / (1 - cfg.test_frac)
    train, val = train_test_split(rest, test_size=rel_val, stratify=rest["label"], random_state=seed)
    return train, val, test


def _group(df: pd.DataFrame, cfg: SplitConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[cfg.group_col].fillna(df["id"])            # ungrouped rows become their own group
    n_test = max(2, round(1 / cfg.test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgkf.split(df, df["label"], groups))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]
    n_val = max(2, round((1 - cfg.test_frac) / cfg.val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed + 1)
    tr_idx, va_idx = next(sgkf.split(rest, rest["label"], groups.iloc[rest_idx]))
    return rest.iloc[tr_idx], rest.iloc[va_idx], test


def split_dataset(df: pd.DataFrame, cfg: SplitConfig, seed: int = 42) -> Splits:
    labelled = df[df["label"].notna()].copy()
    start, end = cfg.time_window
    n_window_dropped = 0
    if (start or end) and cfg.time_col in labelled:
        ts = labelled[cfg.time_col]
        keep = pd.Series(True, index=labelled.index)
        if start:
            keep &= ts >= pd.Timestamp(start, tz="UTC")
        if end:
            keep &= ts < pd.Timestamp(end, tz="UTC")
        n_window_dropped = int((~keep).sum())
        labelled = labelled[keep]
        log.info("time_window %s: dropped %d rows outside window", cfg.time_window, n_window_dropped)
    labelled, dedup_rep = deduplicate(labelled)

    if cfg.strategy == "temporal":
        train, val, test = _temporal(labelled, cfg)
    elif cfg.strategy == "stratified":
        train, val, test = _stratified(labelled, cfg, seed)
    else:
        train, val, test = _group(labelled, cfg, seed)

    dropped = {"val": 0, "test": 0}
    # group: disjoint by construction. temporal: optionally drop eval rows of authors already seen in training.
    # stratified: authors deliberately allowed to overlap (use `group` if that is a leak for your use case).
    enforce_groups = cfg.strategy == "group" or (cfg.strategy == "temporal" and cfg.drop_eval_groups_seen_in_train)
    if enforce_groups and cfg.strategy == "temporal" and cfg.group_col in labelled:
        seen = set(train[cfg.group_col].dropna())
        keep = ~val[cfg.group_col].isin(seen)
        dropped["val"] = int((~keep).sum())
        val = val[keep]
        seen |= set(val[cfg.group_col].dropna())
        keep = ~test[cfg.group_col].isin(seen)
        dropped["test"] = int((~keep).sum())
        test = test[keep]

    splits = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    check_split_integrity(
        splits,
        group_col=cfg.group_col if enforce_groups and cfg.group_col in labelled else None,
        time_col=cfg.time_col if cfg.strategy == "temporal" else None,
    )

    report: dict = {"strategy": cfg.strategy, "time_window": list(cfg.time_window),
                    "rows_outside_time_window": n_window_dropped, "dedup": dedup_rep.as_dict(),
                    "eval_rows_dropped_for_group_overlap": dropped}
    for name, part in splits.items():
        if len(part) == 0:
            raise ValueError(f"split '{name}' is empty")
        pos = float(part["label"].mean())
        if min(pos, 1 - pos) < cfg.min_class_frac:
            raise ValueError(
                f"split '{name}' is nearly single-class (positive rate {pos:.3f}). "
                "If data is sorted by label within time, use strategy=stratified or fix the source."
            )
        report[name] = {
            "rows": len(part),
            "positive_rate": round(pos, 4),
            "time_min": str(part[cfg.time_col].min()) if cfg.time_col in part else None,
            "time_max": str(part[cfg.time_col].max()) if cfg.time_col in part else None,
            "authors": int(part[cfg.group_col].nunique()) if cfg.group_col in part else None,
        }
    # Prior shift between splits is not an error (it is real drift), but it must be visible.
    rates = [report[s]["positive_rate"] for s in SPLITS]
    report["max_prior_shift"] = round(float(np.ptp(rates)), 4)
    if report["max_prior_shift"] > 0.15:
        log.warning("label prior shifts by %.2f across splits -> calibrate thresholds on val, monitor in prod",
                    report["max_prior_shift"])
    log.info("splits: %s", {s: report[s]["rows"] for s in SPLITS})
    return Splits(splits["train"], splits["val"], splits["test"], report)


def save_splits(splits: Splits, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.as_dict().items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "split_report.json").write_text(encoding="utf-8", data=json.dumps(splits.report, indent=2, default=str))


def load_splits(in_dir: Path) -> Splits:
    missing = [s for s in SPLITS if not (in_dir / f"{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"missing splits {missing} in {in_dir}; run `sentiment split` first")
    parts = {s: pd.read_parquet(in_dir / f"{s}.parquet") for s in SPLITS}
    report_path = in_dir / "split_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return Splits(parts["train"], parts["val"], parts["test"], report)
