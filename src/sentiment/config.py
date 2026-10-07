"""Typed configuration loaded from YAML (single source of truth for every stage).

Unknown keys are rejected (``extra="forbid"``) so a typo such as ``max_fp: 0.05`` fails loudly instead of
silently falling back to a default -- a classic silent misconfiguration.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class PathsConfig(_Strict):
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    cache_dir: Path = Path(".cache")
    logs_dir: Path = Path("logs")

    def resolve(self, root: Path) -> "PathsConfig":
        """Make every relative path absolute w.r.t. the project root (CWD-independent)."""
        data = {k: (v if Path(v).is_absolute() else root / v) for k, v in self.model_dump().items()}
        return PathsConfig(**data)


class SourceConfig(_Strict):
    name: str
    kind: Literal["csv", "jsonl", "sql"]
    path: str | None = None
    uri: str | None = None            # SQLAlchemy URI for kind=sql
    query: str | None = None
    encoding: str = "utf-8"
    has_header: bool = True
    columns: list[str] | None = None  # names to assign when has_header=False
    column_map: dict[str, str] = Field(default_factory=dict)  # source column -> canonical column
    label_map: dict[str, int] = Field(default_factory=dict)
    label_source: Literal["gold", "human", "weak", "none"] = "gold"
    sample_size: int | None = None


class IngestConfig(_Strict):
    sources: list[SourceConfig] = Field(default_factory=list)
    chunk_size: int = 200_000
    timezone_offsets: dict[str, str] = Field(default_factory=lambda: {"UTC": "+0000"})


class StreamConfig(_Strict):
    queue_maxsize: int = 10_000
    flush_size: int = 500
    flush_interval_s: float = 2.0
    dedup_window: int = 200_000


class PreprocessConfig(_Strict):
    language: str = "en"
    max_chars: int = 5000
    url_token: str = " httpurl "
    user_token: str = " @user "
    demojize: bool = True
    fix_mojibake: bool = True
    strip_html: bool = True
    max_char_repeat: int = 2
    expand_slang: bool = True
    expand_contractions: bool = True
    lowercase: bool = True
    mark_negation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = False


class SplitConfig(_Strict):
    strategy: Literal["temporal", "stratified", "group"] = "temporal"
    val_frac: float = Field(0.15, gt=0, lt=0.5)
    test_frac: float = Field(0.15, gt=0, lt=0.5)
    time_col: str = "timestamp"
    group_col: str = "author"
    drop_eval_groups_seen_in_train: bool = True
    min_class_frac: float = 0.10
    # [start, end) ISO timestamps; restricts data to the period in which ALL classes were collected.
    time_window: tuple[str | None, str | None] = (None, None)


class TransformerConfig(_Strict):
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    train_subsample: int | None = 8000
    eval_subsample: int | None = 2000
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 1
    torch_threads: int = 8


class EmbeddingConfig(_Strict):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 64
    batch_size: int = 128
    train_subsample: int | None = 30_000


class Word2VecConfig(_Strict):
    vector_size: int = 200
    window: int = 5
    min_count: int = 3
    epochs: int = 10
    pretrained_path: str | None = None  # e.g. a GloVe .txt file (no header) -> used instead of training


class TrainConfig(_Strict):
    models: list[str] = Field(default_factory=lambda: ["tfidf_logreg"])
    cv_folds: int = 3
    search_iter: int = 8
    n_jobs: int = 4
    class_weight: Literal["balanced"] | None = "balanced"
    max_features_word: int = 200_000
    max_features_char: int = 300_000
    transformer: TransformerConfig = TransformerConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    word2vec: Word2VecConfig = Word2VecConfig()


class EvaluateConfig(_Strict):
    positive_label: int = 1
    threshold_strategy: Literal["max_fpr", "cost"] = "max_fpr"
    max_fpr: float = Field(0.05, gt=0, lt=1)
    cost_fp: float = 5.0
    cost_fn: float = 1.0
    bootstrap_samples: int = 500
    slice_min_support: int = 50
    slice_fpr_alert_factor: float = 1.5


class PromotionConfig(_Strict):
    max_test_fpr: float = 0.06
    min_recall_gain: float = -0.005


class RegistryConfig(_Strict):
    model_name: str = "sentiment-classifier"
    promotion: PromotionConfig = PromotionConfig()


class TrackingConfig(_Strict):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "sentiment"


class ServeConfig(_Strict):
    model_dir: str | None = None
    max_batch_items: int = 256
    max_text_chars: int = 5000
    microbatch_max_size: int = 64
    microbatch_max_wait_ms: int = 10
    max_queue: int = 2000
    cache_size: int = 50_000
    cache_ttl_s: int = 3600
    rate_limit_per_minute: int = 600
    rate_limit_burst: int = 100
    job_max_items: int = 20_000
    uncertainty_margin: float = 0.5   # logit units around the decision threshold
    quantize_transformer: bool = True
    torch_threads: int | None = None   # intra-op threads for inference; None = torch default
    log_predictions: bool = True
    log_text: bool = False
    strict_fingerprint: bool = True


class MonitoringConfig(_Strict):
    window_size: int = 2000
    min_window: int = 200
    psi_alert: float = 0.2


class PipelineConfig(_Strict):
    project: str = "sentiment"
    seed: int = 42
    paths: PathsConfig = PathsConfig()
    ingest: IngestConfig = IngestConfig()
    stream: StreamConfig = StreamConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    split: SplitConfig = SplitConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvaluateConfig = EvaluateConfig()
    registry: RegistryConfig = RegistryConfig()
    tracking: TrackingConfig = TrackingConfig()
    serve: ServeConfig = ServeConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config from ``path`` > $SENTIMENT_CONFIG > configs/pipeline.yaml. Paths resolve to project root."""
    path = Path(path or os.environ.get("SENTIMENT_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = PipelineConfig(**(raw or {}))
    root = Path(os.environ.get("SENTIMENT_ROOT", PROJECT_ROOT))
    cfg.paths = cfg.paths.resolve(root)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> PipelineConfig:
    return load_config()


def configure_caches(cfg: PipelineConfig) -> None:
    """Point HuggingFace/torch caches at the project cache dir BEFORE transformers is imported.

    On this machine C: has ~2 GB free; the default ~/.cache/huggingface would silently fill it and later
    downloads would fail half-way with cryptic errors.
    """
    cache = Path(cfg.paths.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids fork deadlock warnings in dataloaders
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows w/o developer mode
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
