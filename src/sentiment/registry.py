"""File-based model registry with champion/challenger promotion (mirrored to MLflow aliases when enabled).

Layout:  artifacts/registry/<model_name>/registry.json + v1/, v2/, ... (immutable bundles)

Promotion policy (all must pass, otherwise the champion is kept):
* the candidate's test FPR is at or below ``max_test_fpr`` (the business constraint), and its threshold was
  feasible on validation
* when a champion exists and was evaluated on the SAME split (same data fingerprint): candidate recall must be
  at least champion recall + ``min_recall_gain``. Scores on different test sets are not comparable, and
  comparing them silently is a common way a worse model gets promoted.
Writes go to a temp file and are then renamed, so a crashed promotion can't corrupt the index.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentiment.config import PromotionConfig
from sentiment.logging_utils import get_logger

log = get_logger(__name__)


class LocalRegistry:
    def __init__(self, root: Path, model_name: str) -> None:
        self.dir = Path(root) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "registry.json"
        self.model_name = model_name

    def _read(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"model_name": self.model_name, "versions": [], "champion": None, "history": []}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(encoding="utf-8", data=json.dumps(data, indent=2, default=str))
        os.replace(tmp, self.index_path)

    def register(self, bundle_dir: Path, *, metrics: dict[str, Any], data_fingerprint: str,
                 run_id: str | None = None, mlflow_version: str | None = None, model_name: str = "") -> int:
        data = self._read()
        version = max((v["version"] for v in data["versions"]), default=0) + 1
        dest = self.dir / f"v{version}"
        shutil.move(str(bundle_dir), str(dest))
        data["versions"].append({
            "version": version, "path": dest.name, "model": model_name, "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics, "data_fingerprint": data_fingerprint, "mlflow_run_id": run_id,
            "mlflow_version": mlflow_version,
        })
        self._write(data)
        log.info("registered %s v%d (%s)", self.model_name, version, model_name)
        return version

    def versions(self) -> list[dict[str, Any]]:
        return self._read()["versions"]

    def get(self, version: int) -> dict[str, Any]:
        for v in self.versions():
            if v["version"] == version:
                return v
        raise KeyError(f"version {version} not found")

    def path(self, version: int) -> Path:
        return self.dir / self.get(version)["path"]

    def champion(self) -> dict[str, Any] | None:
        data = self._read()
        return self.get(data["champion"]) if data["champion"] else None

    def champion_path(self) -> Path | None:
        c = self.champion()
        return self.dir / c["path"] if c else None

    def evaluate_promotion(self, version: int, policy: PromotionConfig) -> tuple[bool, list[str]]:
        cand = self.get(version)
        test = cand["metrics"].get("test", {})
        reasons: list[str] = []
        if test.get("fpr", 1.0) > policy.max_test_fpr:
            reasons.append(f"test FPR {test.get('fpr'):.4f} > {policy.max_test_fpr}")
        if not cand["metrics"].get("threshold_feasible", True):
            reasons.append("FPR target infeasible on validation")
        champ = self.champion()
        if champ and champ["version"] != version:
            if champ.get("data_fingerprint") != cand.get("data_fingerprint"):
                log.warning("champion v%s was evaluated on different data; recall comparison skipped", champ["version"])
            else:
                c_rec = champ["metrics"].get("test", {}).get("recall", 0.0)
                gain = test.get("recall", 0.0) - c_rec
                if gain < policy.min_recall_gain:
                    reasons.append(f"recall {test.get('recall'):.4f} vs champion {c_rec:.4f} (gain {gain:+.4f})")
        return (not reasons), reasons

    def promote(self, version: int, policy: PromotionConfig, force: bool = False) -> tuple[bool, list[str]]:
        ok, reasons = self.evaluate_promotion(version, policy)
        if not ok and not force:
            log.warning("v%d NOT promoted: %s", version, "; ".join(reasons))
            return False, reasons
        data = self._read()
        data["history"].append({"from": data["champion"], "to": version, "at": datetime.now(timezone.utc).isoformat(),
                                "forced": bool(force and not ok), "reasons_overridden": reasons if force else []})
        data["champion"] = version
        self._write(data)
        log.info("v%d promoted to champion", version)
        return True, reasons
