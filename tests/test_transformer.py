"""Transformer path tested OFFLINE with a tiny randomly-initialised DistilBERT + local vocab (no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sentiment.models.transformer import TransformerModel  # noqa: E402


@pytest.fixture()
def tiny_model_dir(tmp_path, corpus):
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

    words = sorted({w for t in corpus["text"] for w in t.lower().split()} | {"not", "great", "terrible", "love"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"), do_lower_case=True)
    config = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                              max_position_embeddings=128, num_labels=2,
                              id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification(config)
    out = tmp_path / "tiny"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def test_finetune_loop_and_unsorted_batch_inference(tiny_model_dir, corpus, cfg):
    from sentiment.splitting import split_dataset

    cfg.train.transformer.model_name = str(tiny_model_dir)
    cfg.train.transformer.epochs = 2
    cfg.train.transformer.batch_size = 16
    cfg.train.transformer.learning_rate = 1e-3
    cfg.train.transformer.train_subsample = 600
    cfg.train.transformer.eval_subsample = 150
    cfg.train.transformer.torch_threads = 2
    s = split_dataset(corpus, cfg.split, cfg.seed)
    m = TransformerModel(cfg)
    info = m.fit(s.train, s.val)
    assert info["history"] and 0 <= info["tokenization"]["truncation_rate"] <= 1

    # Batched length-sorted inference must equal one-by-one inference (catches un-sort bugs).
    texts = ["short", "a much much longer text about the great movie today", "mid length text", "x"] * 20
    batched = m.predict_proba(texts)
    single = np.array([m.predict_proba([t])[0] for t in texts[:4]])
    np.testing.assert_allclose(batched[:4], single, atol=1e-5)

    out = cfg.paths.artifacts_dir / "tm"
    m.save(out)
    re = TransformerModel.load(out)
    np.testing.assert_allclose(re.predict_proba(texts[:4]), single, atol=1e-5)
    q = TransformerModel.load(out, quantize=True)
    assert np.abs(q.predict_proba(texts[:4]) - single).max() < 0.1
