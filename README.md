# Sentiment Analysis Pipeline (production-grade, CPU-first)

An end-to-end pipeline covering the steps below. Every stage has explicit guards against **silent failures**, meaning bugs that don't crash anything but inflate metrics or degrade production quietly.

```
ingest (batch CSV/JSONL/SQL + async stream) -> label (weak supervision, active learning)
  -> clean (2 profiles) -> vectorize (TF-IDF | counts | Word2Vec/GloVe | transformer embeddings)
  -> split (temporal / stratified / group, leakage-checked)
  -> train (CV search, early stopping, class weights) -> evaluate (FPR-constrained threshold, slices, CIs)
  -> bundle (+ fingerprint, golden parity, drift reference) -> MLflow + local registry -> champion/challenger
  -> FastAPI (async micro-batching, cache, rate limit, jobs) -> Prometheus/Grafana + PSI drift + Evidently
```

Tailored to the machine it was built on: Windows 11, i7-1255U (12 threads), 16 GB RAM, **no CUDA GPU**, and **C: with about 2 GB free**. All caches (pip, HuggingFace, temp) are forced onto the project drive. The Docker Engine was not running and its storage is on C:, so the API and tests were verified natively. The container files are included and ready to use.

---

## Quick start (Windows / PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # venv + CPU torch + deps (add -Gpu for CUDA)
.\.venv\Scripts\Activate.ps1

# data: Sentiment140 (1.6M tweets, emoticon-derived labels + 359 human-labelled gold tweets)
Invoke-WebRequest http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip -OutFile data\raw\s140.zip
Expand-Archive data\raw\s140.zip data\raw\sentiment140

python -m sentiment pipeline                      # ingest -> split -> train (config models) -> register -> promote
python -m sentiment train -m tfidf_svm -m distilbert   # more models; promotion policy decides the champion
python -m sentiment registry                      # list versions (* = champion)
python -m sentiment serve                         # http://127.0.0.1:8000  (UI at /, docs at /docs)
python -m pytest                                  # 49 tests, incl. offline transformer + API tests
mlflow ui --backend-store-uri sqlite:///mlflow.db # experiment tracking UI on :5000
```

Docker (once Docker Desktop is running; move its disk image off C: first: *Settings -> Resources -> Disk image location*):

```powershell
docker compose up -d --build       # API :8000, Prometheus :9090, Grafana :3000 (dashboard "Sentiment API")
Invoke-RestMethod -Method Post localhost:8000/admin/reload -Headers @{"x-admin-token"="change-me"}  # hot-swap champion
```

---

## Results on this machine (Sentiment140, temporal split, threshold tuned for FPR ≤ 5% on validation)

The test set is the **future** period (2009-06-07 to 06-16). The gold set is **human-labelled** tweets from a different collection, so it is the most honest estimate.

| model | test precision | test recall | **test FPR** | test ROC-AUC | gold precision | gold recall | gold FPR | gold ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **distilbert** (champion v5, 8k-tweet CPU fine-tune, 2 epochs) | 0.926 | **0.546** | **0.050** | **0.891** | 0.940 | **0.687** | **0.045** | **0.917** |
| tfidf_logreg (v1, previous champion) | 0.920 | 0.461 | 0.046 | 0.866 | 0.897 | 0.621 | 0.073 | 0.898 |
| tfidf_svm (calibrated) | 0.910 | 0.443 | 0.050 | 0.860 | 0.928 | 0.566 | 0.045 | 0.898 |
| count_nb | 0.893 | 0.358 | 0.049 | 0.836 | 0.941 | 0.527 | 0.034 | 0.883 |
| w2v_logreg | 0.898 | 0.384 | 0.049 | 0.831 | 0.935 | 0.478 | 0.034 | 0.873 |

DistilBERT was promoted automatically by the champion/challenger policy. At the same 5% FPR it finds 18% more positives on test. It was trained on only 8k of the 68k training tweets (a CPU budget of about 1 hour); on a GPU, raise `train.transformer.train_subsample` for more gains.

Reading the table: under a strict 5% FPR budget the threshold is high (0.80 for TF-IDF, 0.91 for DistilBERT), so recall is roughly 46-55%. That is the intended trade-off ("I care more about minimising false positives"). To move along the curve, change `evaluate.max_fpr` or use `threshold_strategy: cost` with `cost_fp`/`cost_fn`. Gold has only 177 negatives, so the bootstrap 95% CI on the champion's gold FPR is **3.6% to 11.4%** (see `reports/<run>/metrics.json` -> `ci95`). The differences between the four baselines on gold are therefore not statistically meaningful. Test (11.6k rows) is the deciding set.

API throughput (measured locally, single uvicorn worker, zero errors in every run):

| served model | throughput | latency | notes |
|---|---|---|---|
| tfidf_logreg | **131 req/s** | p50 436 ms / p99 742 ms @ 64 clients | measured while DistilBERT trained on the same CPU |
| distilbert fp32, 8 threads | 6 req/s | p50 5.2 s @ 32 clients | the hybrid-CPU slowdown, see below |
| distilbert fp32, **2 threads** | **14 req/s** | p50 2.5 s / p99 3.6 s @ 32 clients | `serve.torch_threads: 2` |

This is a real latency-vs-recall trade-off. For high-QPS CPU serving, pin the TF-IDF version (`MODEL_DIR=artifacts/registry/sentiment-classifier/v1`). For DistilBERT, scale out with replicas or use a GPU.

Findings from running it on this hardware:
- **int8 quantisation was rejected automatically.** Dynamic quantisation changed one golden score by 0.53 and flipped 3 labels, so `Predictor.load` refused it and fell back to fp32, which passed parity exactly (Δ = 0).
- **Thread count on hybrid CPUs.** The i7-1255U has 2 P-cores + 8 E-cores. DistilBERT inference took 1.41 s per batch at 8 threads but 0.61 s at 2.
- **Don't train next to other heavy jobs.** Running tests and load tests concurrently made training 7× slower (23 s/step vs 3 s).

---

## Stage by stage

### 1. Data collection & labeling (`src/sentiment/data/`)
| Essential | Production readiness |
|---|---|
| `ingest.py`: chunked CSV/JSONL/SQL (SQLAlchemy) to a canonical schema, saved as parquet | Manifest per source: raw SHA-256, row accounting, reject reasons (pairs with `dvc.yaml`) |
| `stream.py`: async ingestor (any `AsyncIterator[dict]`: Kafka/Kinesis/HTTP poll/CSV replay) | Bounded queue for backpressure, LRU id dedup (idempotent replays), size/time micro-batches, atomic parquet writes, dead-letter JSONL |
| `labeling.py`: labeling functions (emoticons, emoji, lexicon w/ negation, VADER) + accuracy-weighted label model | LF coverage/overlap/conflict/accuracy table; `active-learn` exports uncertain, de-duplicated, length-diverse batches; `merge_annotations` validates human labels |

**Silent errors handled.** Every one of these was hit in this project.
- **Mixed timezones** (`PDT` vs `UTC` in the same dataset) are parsed explicitly to UTC. Otherwise temporal splits are silently wrong.
- **Same id with conflicting labels** (15 tweet IDs, 30 rows, in Sentiment140) are quarantined instead of resolved with keep-first.
- **Literal "NA"/"null" tweets** are kept as text (`keep_default_na=False`). Pandas would otherwise make them NaN and drop them.
- **Unknown labels** (neutral = 2) are rejected with a reason and never coerced.
- **`:/` emoticon regex matching `http://`** would have labelled every URL negative. Fixed and tested.
- **LF label leakage**: emoticons used to label a text are stripped from it (`strip_lf_signals`), so the model can't learn the labeling rule.

### 2. Text preprocessing (`src/sentiment/preprocessing/`)
There are two profiles in the same `TextCleaner`. **classic** (for sparse/static embeddings) lowercases, marks negation scope (`not good` → `not good_neg`), removes stopwords and can optionally lemmatise with spaCy. **transformer** applies light normalisation only, because subword models need near-raw text. Both profiles fix mojibake (ftfy), unescape HTML entities, strip real tags, apply NFKC, use URL/@user placeholders, map emoticons and emoji to words, cap elongation, and expand contractions and slang.

Multilingual: lexical resources live in `resources.py`, and `register_language(LanguageResources(...))` adds a language. Transformer profiles defer to the model tokenizer, so `xlm-roberta-base` or `cardiffnlp/twitter-xlm-roberta-base-sentiment` drop in via `train.transformer.model_name`. The drift monitor's `non_ascii_ratio` and OOV PSI flag unexpected languages in production.

**Silent errors handled**
- **sklearn's stopword list contains `not`, `no`, `never`, `very`**, and removing them flips sentiment. Those words are protected.
- **Naive `<[^>]+>` HTML stripping deletes `<3`**. Only real tags are stripped (tested).
- **`&lt;3`, `&amp;`** in Sentiment140 become top features unless unescaped.
- **Emoji become `[UNK]`** in uncased BERT vocabularies. They are demojized into words.
- **Train/serve skew**: the cleaner is serialised inside the model and its **fingerprint** (code version + config + resources) is stored in the bundle. Serving refuses a mismatching model.
- **Sarcasm** can't be fixed by preprocessing. It is detected as a slice (`has_sarcasm_marker`) and evaluated separately. Predictions near the threshold are returned with `uncertain: true`.

### 3. Feature extraction (`src/sentiment/features/vectorizers.py`)
`build_vectorizer(name)` gives one sklearn interface over `tfidf` (word 1-3g + char_wb 2-5g), `count`, `word2vec` (trained on train only, or a GloVe file via `train.word2vec.pretrained_path`; TF-IDF-weighted mean) and `hf_embed` (frozen transformer, masked mean pooling). Swapping is a config or CLI change: `-m embed_logreg`, `-m w2v_logreg`.

**Silent errors handled**
- **Vectorizer fitted on all data before splitting**: vectorizers live inside the Pipeline, so CV refits them on every fold.
- **OOV silent drops** (an all-unknown text gets the intercept prediction): the per-text OOV rate is returned by the API, exported to Prometheus and used as a drift feature. Char n-grams soften the effect.
- **Transformer truncation and `[UNK]` rate** are measured at training time and warned about.

### 4. Data splitting (`src/sentiment/splitting.py`, `data/quality.py`)
Strategies are `temporal` (default), `stratified` and `group` (author-disjoint). The pipeline always deduplicates first and checks integrity last.

**Silent errors handled**, with real numbers from this dataset:
- Near-duplicates (retweets, URL variants): **1,669** removed before the split. Same normalised text with different labels: **700** rows dropped.
- **Gold-set contamination**: 1 training tweet duplicated a gold tweet and was removed.
- **Author leakage**: **6,425** val/test rows from authors seen in training were dropped, so the model can't recognise a user's style.
- **Temporal cut on timestamps, not row positions** (no shared timestamp across a boundary), with order asserted.
- **One-class test set**: the Sentiment140 file contains *only negatives* after 2009-06-17. The `min_class_frac` guard stopped the run, and `split.time_window` now restricts the data to the period where both classes were collected.

### 5. Model selection & training (`src/sentiment/models/`, `training.py`)
- Baselines: TF-IDF + LogReg / calibrated LinearSVC, ComplementNB, Word2Vec + LR, embeddings + LR. Each runs a `RandomizedSearchCV` with **TimeSeriesSplit** for temporal data (StratifiedGroupKFold otherwise). The scorer is **recall at FPR ≤ max_fpr**, the business metric.
- Transformer: DistilBERT fine-tuned with a plain PyTorch loop (stable across transformers 4.x/5.x). It uses class-weighted CE, AdamW + warmup/decay, gradient clipping, per-epoch validation on the business metric, **early stopping that restores the best weights**, and length-sorted batched inference that is un-sorted afterwards (tested). The default subsample is CPU-sized; raise `train.transformer.train_subsample` on a GPU (used automatically if available).
- Imbalance: `class_weight: balanced` for the classical models and a weighted loss for the transformer. Prior shift between splits is reported.
- Tracking: MLflow (SQLite at an absolute path) records params, per-epoch metrics, reports and plots. Models are registered as `pyfunc`, with the alias `champion`. A W&B tracker only has to implement the same 6 methods in `tracking.py`.

**Silent errors handled**
- Class order is asserted, so probabilities can't be silently inverted.
- The shortcut-token detector flags features that are too predictive. It found `#followfriday`, which is 98.7% positive.
- The train-vs-CV generalisation gap is logged.

### 6. Model evaluation (`src/sentiment/evaluation/metrics.py`)
Reported: precision, recall, F1, specificity, **FPR**, FNR, ROC-AUC, PR-AUC, MCC, Brier, ECE, confusion matrix, **bootstrap 95% CIs**, and ROC/PR/calibration plots. These are computed for val, test and gold, and the slice CSVs (length buckets, negation, emoji, emoticon, URL, mention, hashtag, sarcasm marker, caps, month) are written to `reports/<run>/`.

FP-first policy: the threshold is chosen **on validation** to maximise recall subject to FPR ≤ `max_fpr` (or minimum expected cost). The test set is scored **once** with the frozen threshold. Slices whose FPR exceeds 1.5× the budget are flagged. The current champion breaches it on `has_url=1` (link-sharing tweets), which is a targeted next step.

### 7. Deployment & monitoring (`src/sentiment/serving/`, `deploy/`, `Dockerfile`, `docker-compose.yml`)
- **Async FastAPI.** `POST /v1/predict` accepts 1-256 texts. An **async micro-batcher** merges concurrent requests into one model call, which runs on a dedicated inference thread so the event loop never blocks.
- `POST /v1/jobs` / `GET /v1/jobs/{id}` handle up to 20k texts asynchronously, processed in chunks so interactive traffic keeps flowing.
- **TTL-LRU cache** keyed by (model version, text). It is cleared on model swap, and duplicates within a request are computed once.
- **Token-bucket rate limiting** per API key/IP returns `429` with `Retry-After`. Load shedding returns `503` when the queue is full.
- **Bundles verify themselves at load**: label map, preprocessing fingerprint, and golden-set score parity (catches wrong weights, library upgrades and int8 quantisation damage). `/admin/reload` hot-swaps to the new champion and keeps the old model if verification fails.
- **Prometheus metrics**: request/inference latency, batch size, queue depth, predictions by label, uncertain rate, score and OOV histograms, cache hits, 429/503 counts, drift PSI per feature. **Alert rules** cover p95 latency, error rate, load shedding, drift, OOV and positive-rate shift. A **Grafana dashboard** is provisioned automatically and generated as code by `deploy/grafana/build_dashboard.py`.
- **Drift**: online PSI over a sliding window against the reference profile stored in the bundle, without needing labels. Verified: Spanish and URL-heavy traffic raised alerts with OOV PSI 0.64 and has_url PSI 0.86. `python -m sentiment drift-report` writes an **Evidently** HTML report from the prediction logs. Raw text is not logged by default (`serve.log_text: false`); only hashes and features are.
- **Retraining loop**: `stream` → `active-learn` → annotate → `merge_annotations` → `train`. The registry then promotes the challenger only if it meets the FPR budget and doesn't lose recall on the *same* data split.
- Not included: gRPC (REST covers the stated needs; a gRPC front-end would reuse `Predictor` unchanged). Multi-replica deployments should move the cache and rate limiter to Redis behind the same interfaces in `serving/runtime.py`.

---

## Silent-failure catalogue (each one has code + a test)

| # | Failure | Where it would hide | Guard |
|---|---|---|---|
| 1 | Timezone-abbreviation misparse | temporal split order | `parse_timestamps` + test |
| 2 | Conflicting duplicate labels | label noise inflates/deflates | quarantine in ingest + dedup |
| 3 | Near-duplicates across splits | test memorisation | normalised dedup key + `check_split_integrity` |
| 4 | Author overlap | style memorisation | author drop / `group` strategy |
| 5 | Future data in training | over-optimistic metrics | temporal split + TimeSeriesSplit CV |
| 6 | One-class eval split | meaningless AUC/accuracy | `min_class_frac` + `time_window` |
| 7 | Gold-set contamination | inflated external eval | gold-key removal in `split` |
| 8 | Vectorizer fit on test | leakage | Pipeline-internal fitting |
| 9 | Stopword list removes negations | flipped sentiment | protected word set + test |
| 10 | `<3`, `&lt;3`, mojibake, emoji→[UNK] | lost signal | cleaner + tests |
| 11 | LF signal left in text | model learns the labeler | `strip_lf_signals` + shortcut detector |
| 12 | Threshold tuned on test | fake FPR compliance | val-only threshold, test scored once |
| 13 | Train/serve preprocessing skew | quality decay after deploy | fingerprint check (strict) |
| 14 | Wrong weights / lib upgrade / int8 damage | silently different scores | golden parity on load |
| 15 | Label-id order flip | inverted predictions | label map + class-order asserts |
| 16 | Unsorted batch outputs | scores swapped between texts | scatter-back + equality test |
| 17 | OOV-only inputs | confident garbage | OOV rate in API + metric + drift |
| 18 | Truncation | lost content | truncation rate logged |
| 19 | Comparing champion on different test sets | worse model promoted | data fingerprint check in promotion |
| 20 | Zero-inflated drift feature → single PSI bin | drift invisible | min-point-mass bin edge + test |
| 21 | Relative MLflow/sqlite paths | runs land in a second store | absolute URIs |
| 22 | Windows cp1252 default encoding | crash or mojibake on emoji | explicit UTF-8 everywhere |
| 23 | Cache served after model swap | stale predictions | cache cleared on reload, keyed by version |
| 24 | Active learning picks confident items | wasted annotation budget | uncertainty-ranked round-robin + test |
| 25 | Probability-space "uncertain" band near a high threshold | every confident positive flagged | logit-space margin + test |
| 26 | Raw file sorted by label | `head()`/`nrows` samples are one-class | chunk-wise stratified sampling in ingest |

---

## Layout

```
configs/pipeline.yaml        single source of truth (strict: unknown keys fail)
src/sentiment/
  data/        ingest.py stream.py labeling.py quality.py schema.py
  preprocessing/ cleaner.py resources.py
  features/    vectorizers.py
  models/      classical.py transformer.py bundle.py base.py
  evaluation/  metrics.py
  monitoring/  drift.py
  serving/     app.py runtime.py metrics.py pyfunc.py static/index.html
  splitting.py training.py registry.py tracking.py config.py cli.py
tests/                       49 tests (offline; tiny random DistilBERT for the transformer path)
deploy/                      prometheus (+alerts), grafana (provisioned datasource + dashboard)
Dockerfile, docker-compose.yml, dvc.yaml, requirements*.txt, scripts/setup.ps1
```
