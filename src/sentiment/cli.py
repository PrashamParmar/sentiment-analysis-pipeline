"""Command line entry point: `python -m sentiment <command>` (or `sentiment <command>` after pip install -e .).

    ingest        batch-ingest configured sources -> data/interim (+ manifests)
    stream        replay a CSV as a live stream through the streaming ingestor
    weak-label    apply labeling functions + label model to unlabelled data
    split         dedup + leakage-checked train/val/test -> data/processed
    train         train one or more models, evaluate, register, (auto-)promote
    pipeline      ingest -> split -> train (config models) in one go
    registry      list registered versions;  promote  -> manual promotion
    active-learn  pick the most informative unlabelled texts for annotation
    drift-report  Evidently report: training reference vs logged production traffic
    serve         run the API;  loadtest -> fire concurrent traffic at a running API
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from sentiment.config import PROJECT_ROOT, PipelineConfig, configure_caches, load_config
from sentiment.logging_utils import get_logger, setup_logging

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
log = get_logger("sentiment.cli")
_state: dict[str, PipelineConfig] = {}


def cfg() -> PipelineConfig:
    return _state["cfg"]


@app.callback()
def main(config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config"),
         log_level: str = typer.Option("INFO", "--log-level"), json_logs: bool = typer.Option(False, "--json-logs")) -> None:
    setup_logging(log_level, json_logs)
    _state["cfg"] = load_config(config)
    configure_caches(_state["cfg"])


def _gold_sources(c: PipelineConfig) -> set[str]:
    return {s.name for s in c.ingest.sources if s.label_source == "gold"}


def _load_interim(c: PipelineConfig, gold: bool):
    import pandas as pd
    names = [s.name for s in c.ingest.sources if (s.name in _gold_sources(c)) == gold]
    parts = [pd.read_parquet(Path(c.paths.interim_dir) / f"{n}.parquet") for n in names
             if (Path(c.paths.interim_dir) / f"{n}.parquet").exists()]
    return pd.concat(parts, ignore_index=True) if parts else None


def _data_versions(c: PipelineConfig) -> dict:
    out = {}
    for s in c.ingest.sources:
        mf = Path(c.paths.interim_dir) / f"{s.name}.manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            out[s.name] = {"raw_sha256": m.get("raw_sha256"), "rows_valid": m.get("rows_valid")}
    return out


@app.command()
def ingest() -> None:
    """Batch-ingest every configured source."""
    from sentiment.data.ingest import ingest_all
    for m in ingest_all(cfg()):
        typer.echo(json.dumps({k: m[k] for k in ("source", "rows_read", "rows_valid", "rows_rejected", "reject_reasons",
                                                   "label_distribution", "time_range")}, default=str))


@app.command()
def stream(path: Path = typer.Option(..., help="Sentiment140-format CSV to replay"),
           limit: int = typer.Option(5000), rate: float = typer.Option(0.0, help="records/s (0 = as fast as possible)")) -> None:
    """Replay a CSV through the async streaming ingestor (validation, dedup, micro-batched parquet)."""
    from sentiment.data.stream import StreamIngestor, replay_csv_source
    c = cfg()
    ing = StreamIngestor(Path(c.paths.interim_dir) / "stream", c.stream.flush_size, c.stream.flush_interval_s,
                         c.stream.queue_maxsize, c.stream.dedup_window)
    stats = asyncio.run(ing.run(replay_csv_source(path, "stream", rate or None, limit)))
    typer.echo(json.dumps({k: v for k, v in vars(stats).items() if k != "files"} | {"files": len(stats.files)}))


@app.command("weak-label")
def weak_label_cmd(input: Path = typer.Option(..., help="parquet/csv file, or a directory of parquet partitions"),
                   output: Path = typer.Option(Path("data/interim/weak_labelled.parquet")),
                   min_confidence: float = typer.Option(0.7)) -> None:
    """Weak supervision: labeling functions + accuracy-weighted label model (weights fit on gold if present)."""
    import pandas as pd
    from sentiment.data.labeling import weak_label
    if input.is_dir():
        df = pd.concat([pd.read_parquet(f) for f in sorted(input.rglob("*.parquet"))], ignore_index=True)
    else:
        df = pd.read_parquet(input) if input.suffix == ".parquet" else pd.read_csv(input)
    gold = _load_interim(cfg(), gold=True)
    out, summary = weak_label(df, gold=gold, min_confidence=min_confidence)
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    typer.echo(summary.round(4).to_string(index=False))
    typer.echo(f"{len(out)} weakly-labelled rows -> {output}")


@app.command()
def split() -> None:
    """Deduplicate, remove gold-set contamination, split, and verify integrity."""
    from sentiment.data.quality import dedup_key
    from sentiment.splitting import save_splits, split_dataset
    c = cfg()
    df = _load_interim(c, gold=False)
    if df is None:
        raise typer.BadParameter("no interim data: run `ingest` first")
    gold = _load_interim(c, gold=True)
    if gold is not None:
        # Gold texts that also appear in training data would inflate the external evaluation silently.
        gold_keys = set(gold["text"].map(dedup_key))
        contaminated = df["text"].map(dedup_key).isin(gold_keys)
        if contaminated.any():
            log.warning("removing %d training rows that duplicate gold texts", int(contaminated.sum()))
            df = df[~contaminated]
    s = split_dataset(df, c.split, c.seed)
    save_splits(s, Path(c.paths.processed_dir))
    typer.echo(json.dumps(s.report, indent=2, default=str))


@app.command()
def train(models: list[str] = typer.Option(None, "--model", "-m", help="repeatable; default = config train.models"),
          no_promote: bool = typer.Option(False, "--no-promote")) -> None:
    """Train, evaluate (val/test/gold/slices), log to MLflow, register and promote if policy passes."""
    from sentiment.splitting import load_splits
    from sentiment.tracking import get_tracker
    from sentiment.training import train_and_register
    c = cfg()
    splits = load_splits(Path(c.paths.processed_dir))
    gold = _load_interim(c, gold=True)
    if gold is not None:
        gold = gold[gold["label"].notna()].reset_index(drop=True)
    tracker = get_tracker(c)
    results = []
    for name in models or c.train.models:
        res = train_and_register(name, c, splits, gold, tracker, promote=not no_promote, data_versions=_data_versions(c))
        results.append(res)
        typer.echo(json.dumps(res, indent=2, default=str))
    if len(results) > 1:
        import pandas as pd
        cols = [k for k in results[0] if k.startswith(("test_", "gold_"))]
        typer.echo(pd.DataFrame(results).set_index("model")[cols + ["promoted"]].to_string())


@app.command()
def pipeline(skip_ingest: bool = typer.Option(False, help="reuse data/interim")) -> None:
    """ingest -> split -> train -> register/promote (models from config)."""
    if not skip_ingest:
        ingest()
    split()
    train(models=None, no_promote=False)


@app.command()
def registry() -> None:
    """List registered model versions with headline metrics."""
    from sentiment.registry import LocalRegistry
    c = cfg()
    reg = LocalRegistry(Path(c.paths.artifacts_dir) / "registry", c.registry.model_name)
    champ = reg.champion()
    for v in reg.versions():
        t, g = v["metrics"].get("test", {}), v["metrics"].get("gold", {})
        mark = "*" if champ and champ["version"] == v["version"] else " "
        typer.echo(f"{mark} v{v['version']:<3} {v.get('model', ''):<14} test P={t.get('precision', 0):.3f} "
                   f"R={t.get('recall', 0):.3f} FPR={t.get('fpr', 0):.3f} AUC={t.get('roc_auc', 0):.3f} | gold "
                   f"P={g.get('precision', 0):.3f} R={g.get('recall', 0):.3f} FPR={g.get('fpr', 0):.3f}  {v['registered_at'][:19]}")


@app.command()
def promote(version: int, force: bool = typer.Option(False, help="override failed policy checks (audited)")) -> None:
    """Promote a registered version to champion (subject to policy unless --force)."""
    from sentiment.registry import LocalRegistry
    c = cfg()
    reg = LocalRegistry(Path(c.paths.artifacts_dir) / "registry", c.registry.model_name)
    ok, reasons = reg.promote(version, c.registry.promotion, force=force)
    typer.echo(json.dumps({"promoted": ok, "reasons": reasons}))
    if ok:
        mv = reg.get(version).get("mlflow_version")
        if mv:
            try:
                from sentiment.tracking import get_tracker
                get_tracker(c).set_alias(c.registry.model_name, mv, "champion")
            except Exception as exc:
                log.warning("MLflow alias not updated: %s", exc)
    raise typer.Exit(0 if ok else 1)


@app.command("active-learn")
def active_learn(k: int = typer.Option(200), strategy: str = typer.Option("entropy"),
                 pool: Optional[Path] = typer.Option(None, help="parquet of unlabelled texts (default: stream data)"),
                 output: Path = typer.Option(Path("data/annotation/batch.csv"))) -> None:
    """Export the K most informative unlabelled texts for human annotation."""
    import pandas as pd
    from sentiment.data.labeling import export_annotation_batch, select_for_annotation
    from sentiment.models.bundle import Predictor
    from sentiment.registry import LocalRegistry
    c = cfg()
    path = LocalRegistry(Path(c.paths.artifacts_dir) / "registry", c.registry.model_name).champion_path()
    if path is None:
        raise typer.BadParameter("no champion model; train first")
    pred = Predictor.load(path)
    if pool is None:
        files = sorted((Path(c.paths.interim_dir) / "stream").rglob("*.parquet"))
        if not files:
            raise typer.BadParameter("no pool given and no stream data; run `stream` or pass --pool")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(pool)
    proba = pred.model.predict_proba(df["text"].astype(str).tolist())
    batch = select_for_annotation(df, proba, k, strategy)
    output = output if output.is_absolute() else PROJECT_ROOT / output
    export_annotation_batch(batch, output)
    typer.echo(f"{len(batch)} texts -> {output} (fill the `label` column with 0/1 or negative/positive)")


@app.command("drift-report")
def drift_report(logs: Optional[Path] = typer.Option(None, help="prediction log dir (default logs/predictions)")) -> None:
    """Offline Evidently drift report: validation-set reference vs logged production traffic."""
    import pandas as pd
    from sentiment.features.vectorizers import oov_rate
    from sentiment.models.bundle import Predictor
    from sentiment.monitoring.drift import compare_to_profile, evidently_report
    from sentiment.registry import LocalRegistry
    from sentiment.splitting import load_splits
    from sentiment.training import features_frame
    c = cfg()
    path = LocalRegistry(Path(c.paths.artifacts_dir) / "registry", c.registry.model_name).champion_path()
    if path is None:
        raise typer.BadParameter("no champion model")
    pred = Predictor.load(path)
    val = load_splits(Path(c.paths.processed_dir)).val
    ref = features_frame(val).drop(columns=["timestamp"], errors="ignore")
    ref["score"] = pred.model.predict_proba(val["text"].tolist())
    ref["oov_rate"] = oov_rate(pred.model.clean(val["text"].tolist()), pred.vocab)
    ref["pred"] = (ref["score"] >= pred.threshold).astype(int)
    log_dir = logs or Path(c.paths.logs_dir) / "predictions"
    files = sorted(Path(log_dir).glob("*.jsonl"))
    if not files:
        raise typer.BadParameter(f"no prediction logs in {log_dir}; send traffic to the API first")
    cur = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    out_dir = Path(c.paths.reports_dir) / "drift"
    res = evidently_report(ref, cur, out_dir)
    psi = compare_to_profile(pred.reference_profile, cur)
    typer.echo(json.dumps({"report": res["html"], "rows_current": len(cur), "psi": {k: round(v, 4) for k, v in psi.items()}}, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API with uvicorn (single worker; scale with replicas)."""
    import uvicorn
    uvicorn.run("sentiment.serving.app:app", host=host, port=port, workers=1, log_level="info")


@app.command()
def loadtest(url: str = "http://127.0.0.1:8000", n: int = 2000, concurrency: int = 32,
             source: str = typer.Option("test", help="test | gold: which texts to send"),
             spread_keys: bool = typer.Option(False, help="one API key per request (bypass per-client rate limit)")) -> None:
    """Fire concurrent single-text requests at a running API and report latency percentiles."""
    import time
    import httpx
    import numpy as np
    from sentiment.splitting import load_splits
    c = cfg()
    if source == "gold":
        texts = _load_interim(c, gold=True)["text"].tolist()
    else:
        texts = load_splits(Path(c.paths.processed_dir)).test["text"].tolist()
    texts = (texts * (n // max(len(texts), 1) + 1))[:n]
    lat: list[float] = []
    codes: dict[int, int] = {}

    async def run() -> None:
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            async def one(i: int, t: str) -> None:
                async with sem:
                    t0 = time.perf_counter()
                    key = f"loadtest-{i}" if spread_keys else "loadtest"
                    r = await client.post("/v1/predict", json={"texts": [t]}, headers={"x-api-key": key})
                    lat.append(time.perf_counter() - t0)
                    codes[r.status_code] = codes.get(r.status_code, 0) + 1
            t0 = time.perf_counter()
            await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
            typer.echo(f"{n} requests in {time.perf_counter() - t0:.1f}s -> {n / (time.perf_counter() - t0):.0f} req/s")

    asyncio.run(run())
    a = np.array(lat) * 1000
    typer.echo(f"status codes: {codes} | latency ms p50={np.percentile(a, 50):.1f} p95={np.percentile(a, 95):.1f} "
               f"p99={np.percentile(a, 99):.1f}")


if __name__ == "__main__":
    app()
