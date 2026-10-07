"""Fine-tuned transformer classifier (default: distilbert-base-uncased), CPU-first and GPU-ready.

Uses a plain PyTorch loop instead of ``transformers.Trainer``, so the code is stable across transformers
major versions and every production concern is explicit:
* class-weighted cross-entropy (imbalanced data)
* AdamW + linear warmup/decay + gradient clipping
* validation every epoch, monitoring recall at FPR <= max_fpr (the business metric, not accuracy)
* early stopping with restoration of the BEST weights. Keeping the last epoch's weights is a common silent
  regression.
* length-sorted batched inference that is un-sorted afterwards. Forgetting to un-sort silently pairs texts
  with other texts' scores.
* truncation and [UNK] rates are measured and reported (silent information loss)
* optional dynamic int8 quantisation for CPU serving. The bundle's golden-set parity check verifies it
  didn't change predictions beyond tolerance.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment import ID2LABEL, LABEL2ID
from sentiment.config import PipelineConfig
from sentiment.logging_utils import get_logger
from sentiment.models.base import SentimentModel
from sentiment.models.classical import recall_at_fpr
from sentiment.preprocessing.cleaner import TextCleaner

log = get_logger(__name__)


def _stratified_subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if not n or len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)


class TransformerModel(SentimentModel):
    kind = "transformer"

    def __init__(self, cfg: PipelineConfig, name: str = "distilbert") -> None:
        self.cfg = cfg
        self.name = name
        self.tcfg = cfg.train.transformer
        self.cleaner = TextCleaner(cfg.preprocess, "transformer")
        self.model: Any = None
        self.tokenizer: Any = None
        self.history: list[dict[str, float]] = []
        self.quantized = False

    # ------------------------------------------------------------------------------------------ helpers
    def _encode(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, truncation=True, max_length=self.tcfg.max_length)["input_ids"]

    def _collate(self, batch: list[tuple[list[int], int]]) -> dict[str, Any]:
        import torch
        pad = self.tokenizer.pad_token_id
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}

    def tokenization_stats(self, cleaned: list[str]) -> dict[str, float]:
        full = self.tokenizer(cleaned, truncation=False)["input_ids"]
        unk = self.tokenizer.unk_token_id
        n_tok = sum(len(x) for x in full)
        return {
            "truncation_rate": float(np.mean([len(x) > self.tcfg.max_length for x in full])) if full else 0.0,
            "unk_rate": (sum(t == unk for x in full for t in x) / n_tok) if n_tok else 0.0,
            "p99_length": float(np.percentile([len(x) for x in full], 99)) if full else 0.0,
        }

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        t = self.tcfg
        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(t.torch_threads)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        train = _stratified_subsample(train, t.train_subsample, self.cfg.seed)
        val = _stratified_subsample(val, t.eval_subsample, self.cfg.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(t.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            t.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        ).to(device)

        x_train = self.cleaner.clean_many(train["text"].tolist())
        x_val = self.cleaner.clean_many(val["text"].tolist())
        y_train = train["label"].to_numpy(dtype=int)
        y_val = val["label"].to_numpy(dtype=int)
        tok_stats = self.tokenization_stats(x_train)
        if tok_stats["truncation_rate"] > 0.02:
            log.warning("%.1f%% of training texts truncated at max_length=%d", 100 * tok_stats["truncation_rate"], t.max_length)
        if tok_stats["unk_rate"] > 0.01:
            log.warning("[UNK] rate %.2f%%: tokenizer vocabulary does not cover the input", 100 * tok_stats["unk_rate"])

        ds = list(zip(self._encode(x_train), y_train.tolist()))
        loader = DataLoader(ds, batch_size=t.batch_size, shuffle=True, collate_fn=self._collate, num_workers=0,
                            generator=torch.Generator().manual_seed(self.cfg.seed))

        counts = np.bincount(y_train, minlength=2).astype(float)
        weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights if self.cfg.train.class_weight else None)

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": t.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optim = torch.optim.AdamW(groups, lr=t.learning_rate)
        total_steps = len(loader) * t.epochs
        sched = get_linear_schedule_with_warmup(optim, math.ceil(t.warmup_ratio * total_steps), total_steps)

        best_score, best_state, bad_epochs = -1.0, None, 0
        started = time.time()
        for epoch in range(1, t.epochs + 1):
            self.model.train()
            running, n_batches = 0.0, 0
            for step, batch in enumerate(loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                logits = self.model(**batch).logits
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                running += loss.item()
                n_batches += 1
                if step % 50 == 0:
                    log.info("epoch %d step %d/%d loss %.4f (%.0fs)", epoch, step, len(loader), running / n_batches,
                             time.time() - started)

            p_val = self._predict_cleaned(x_val)
            score = recall_at_fpr(y_val, p_val, self.cfg.evaluate.max_fpr)
            from sklearn.metrics import roc_auc_score
            rec = {"epoch": epoch, "train_loss": running / max(n_batches, 1), "val_recall_at_fpr": score,
                   "val_roc_auc": float(roc_auc_score(y_val, p_val)), "elapsed_s": time.time() - started}
            self.history.append(rec)
            log.info("epoch %d: %s", epoch, {k: round(v, 4) for k, v in rec.items()})
            if score > best_score + 1e-4:
                best_score, bad_epochs = score, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items()})
            else:
                bad_epochs += 1
                if bad_epochs > t.early_stopping_patience:
                    log.info("early stopping after epoch %d (best val recall@fpr %.4f)", epoch, best_score)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        return {"history": self.history, "best_val_recall_at_fpr": best_score, "n_train": len(train),
                "n_val_used": len(val), "tokenization": tok_stats, "device": device}

    # ------------------------------------------------------------------------------------------ inference
    def _predict_cleaned(self, cleaned: list[str], batch_size: int = 64) -> np.ndarray:
        import torch
        if not cleaned:
            return np.zeros(0, dtype=float)
        device = next(self.model.parameters()).device if not self.quantized else "cpu"
        self.model.eval()
        ids = self._encode(cleaned)
        order = np.argsort([len(x) for x in ids], kind="stable")   # length-sorted -> minimal padding
        out = np.empty(len(ids), dtype=float)
        with torch.inference_mode():
            for i in range(0, len(order), batch_size):
                idx = order[i:i + batch_size]
                batch = self._collate([(ids[j], 0) for j in idx])
                batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                probs = torch.softmax(self.model(**batch).logits.float(), dim=-1)[:, LABEL2ID["positive"]]
                out[idx] = probs.cpu().numpy()                    # scatter back to ORIGINAL positions
        return out

    def predict_proba_cleaned(self, cleaned: list[str]) -> np.ndarray:
        assert self.model is not None, "model not fitted"
        return self._predict_cleaned(list(cleaned))

    def params(self) -> dict[str, Any]:
        return {"model": self.name, **{k: v for k, v in self.tcfg.model_dump().items()}}

    # ------------------------------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        if self.quantized:
            raise RuntimeError("refusing to save a quantized model; save the fp32 model and quantize at load time")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "hf")
        self.tokenizer.save_pretrained(path / "hf")
        (path / "transformer_meta.json").write_text(encoding="utf-8", data=json.dumps({
            "name": self.name, "cfg": self.cfg.model_dump(mode="json"), "history": self.history,
        }, indent=2))

    @classmethod
    def load(cls, path: Path, quantize: bool = False, **_: Any) -> "TransformerModel":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((path / "transformer_meta.json").read_text(encoding="utf-8"))
        obj = cls(PipelineConfig(**meta["cfg"]), meta["name"])
        obj.history = meta.get("history", [])
        obj.tokenizer = AutoTokenizer.from_pretrained(path / "hf")
        obj.model = AutoModelForSequenceClassification.from_pretrained(path / "hf").eval()
        if obj.model.config.label2id.get("positive") != LABEL2ID["positive"]:
            raise AssertionError(f"label mapping mismatch: {obj.model.config.label2id} vs {LABEL2ID}")
        torch.set_num_threads(obj.tcfg.torch_threads)
        if quantize:
            try:
                from torch.ao.quantization import quantize_dynamic
                obj.model = quantize_dynamic(obj.model, {torch.nn.Linear}, dtype=torch.qint8)
                obj.quantized = True
            except Exception as exc:  # quantization backends vary by platform/torch version
                log.warning("dynamic quantization unavailable (%s); serving fp32", exc)
        return obj
