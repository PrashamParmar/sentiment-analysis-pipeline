"""Experiment tracking (MLflow) behind a tiny interface; degrades to a no-op tracker.

* The tracking URI and artifact root are made ABSOLUTE. A relative ``sqlite:///mlflow.db`` resolves against
  the current working directory, so running the CLI from another folder silently creates a second, empty
  experiment store.
* Bundles are logged as an MLflow ``pyfunc`` model and registered in the MLflow Model Registry, so
  ``mlflow models serve -m models:/sentiment-classifier@champion`` also works.
* To use Weights & Biases instead, implement the same 6 methods. Nothing else depends on MLflow.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sentiment.config import PROJECT_ROOT, PipelineConfig
from sentiment.logging_utils import get_logger

log = get_logger(__name__)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _requirements() -> list[str]:
    """Inline requirement lines (a `-r <path>` reference breaks on paths with spaces/apostrophes)."""
    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        return ["scikit-learn", "pandas", "numpy"]
    lines = [ln.split("#")[0].strip() for ln in req.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith(("-", "torch"))]


class NullTracker:
    run_id: str | None = None

    @contextlib.contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator["NullTracker"]:
        yield self

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...
    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None: ...
    def log_dict(self, data: dict[str, Any], name: str) -> None: ...
    def register_bundle(self, bundle_dir: Path, model_name: str) -> str | None: return None
    def set_alias(self, model_name: str, version: str, alias: str) -> None: ...


class MLflowTracker(NullTracker):
    def __init__(self, cfg: PipelineConfig) -> None:
        import mlflow

        self.mlflow = mlflow
        uri = cfg.tracking.tracking_uri
        if uri.startswith("sqlite:///") and not Path(uri.removeprefix("sqlite:///")).is_absolute():
            uri = "sqlite:///" + (PROJECT_ROOT / uri.removeprefix("sqlite:///")).as_posix()
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        exp = mlflow.get_experiment_by_name(cfg.tracking.experiment)
        if exp is None:
            mlflow.create_experiment(cfg.tracking.experiment, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
        mlflow.set_experiment(cfg.tracking.experiment)
        self.run_id = None

    @contextlib.contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator["MLflowTracker"]:
        with self.mlflow.start_run(run_name=name, tags=tags) as r:
            self.run_id = r.info.run_id
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: str(v)[:500] for k, v in _flatten(params).items()}
        items = list(flat.items())
        for i in range(0, len(items), 100):                      # MLflow batch limit
            self.mlflow.log_params(dict(items[i:i + 100]))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in _flatten(metrics).items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        self.mlflow.log_metrics(clean, step=step)

    def log_artifacts(self, path: Path, artifact_path: str | None = None) -> None:
        if path.is_dir():
            self.mlflow.log_artifacts(str(path), artifact_path)
        elif path.exists():
            self.mlflow.log_artifact(str(path), artifact_path)

    def log_dict(self, data: dict[str, Any], name: str) -> None:
        self.mlflow.log_dict(json.loads(json.dumps(data, default=str)), name)

    def register_bundle(self, bundle_dir: Path, model_name: str) -> str | None:
        from sentiment.serving.pyfunc import SentimentPyfunc

        info = self.mlflow.pyfunc.log_model(
            name="model",
            python_model=SentimentPyfunc(),
            artifacts={"bundle": str(bundle_dir)},
            code_paths=[str(PROJECT_ROOT / "src" / "sentiment")],
            pip_requirements=_requirements(),
            registered_model_name=model_name,
        )
        version = getattr(info, "registered_model_version", None)
        log.info("registered %s version %s in MLflow", model_name, version)
        return str(version) if version is not None else None

    def set_alias(self, model_name: str, version: str, alias: str) -> None:
        from mlflow import MlflowClient
        MlflowClient().set_registered_model_alias(model_name, alias, version)


def get_tracker(cfg: PipelineConfig) -> NullTracker:
    if not cfg.tracking.enabled:
        return NullTracker()
    try:
        return MLflowTracker(cfg)
    except Exception as exc:  # tracking must never take training down
        log.warning("MLflow unavailable (%s); continuing without tracking", exc)
        return NullTracker()
