import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sentiment.config import SourceConfig
from sentiment.data.ingest import ingest_source, parse_timestamps
from sentiment.data.labeling import (
    NEG,
    POS,
    LabelModel,
    apply_lfs,
    merge_annotations,
    select_for_annotation,
    strip_lf_signals,
    weak_label,
)
from sentiment.data.quality import LeakageError, check_split_integrity, dedup_key, deduplicate
from sentiment.data.stream import StreamIngestor
from sentiment.splitting import split_dataset

TZ = {"UTC": "+0000", "PDT": "-0700"}


def test_mixed_timezones_normalised_to_utc():
    s = pd.Series(["Mon Apr 06 22:19:45 PDT 2009", "Mon May 11 03:17:40 UTC 2009", "2009-06-01T10:00:00Z", "garbage"])
    out = parse_timestamps(s, TZ)
    assert out.iloc[0] == pd.Timestamp("2009-04-07 05:19:45", tz="UTC")     # PDT = UTC-7
    assert out.iloc[1] == pd.Timestamp("2009-05-11 03:17:40", tz="UTC")
    assert out.iloc[2] == pd.Timestamp("2009-06-01 10:00:00", tz="UTC")
    assert pd.isna(out.iloc[3])


def test_csv_ingest_quarantines_instead_of_coercing(cfg, tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(
        '"0","1","Mon Apr 06 22:19:45 PDT 2009","q","u1","sad day"\n'
        '"4","2","Mon Apr 06 22:20:45 PDT 2009","q","u2","happy day"\n'
        '"2","3","Mon Apr 06 22:21:45 PDT 2009","q","u3","neutral day"\n'
        '"4","4","not a date","q","u4","bad timestamp"\n'
        '"4","5","Mon Apr 06 22:22:45 PDT 2009","q","u5",""\n'
        '"4","6","Mon Apr 06 22:23:45 PDT 2009","q","u6","NA"\n'.encode("latin-1")
    )
    cfg.ingest.timezone_offsets = TZ
    src = SourceConfig(name="t", kind="csv", path=str(raw), encoding="latin-1", has_header=False,
                       columns=["label", "id", "timestamp", "query", "author", "text"], label_map={"0": 0, "4": 1},
                       label_source="weak")
    m = ingest_source(src, cfg, tmp_path)
    assert m["rows_valid"] == 3                                  # "NA" tweet is text, not missing
    assert m["reject_reasons"] == {"unmapped_label": 1, "bad_timestamp": 1, "empty_text": 1}
    df = pd.read_parquet(cfg.paths.interim_dir / "t.parquet")
    assert set(df["id"]) == {"t:1", "t:2", "t:6"}
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_dedup_drops_conflicts_and_near_duplicates():
    df = pd.DataFrame({
        "id": list("abcdef"),
        "text": ["I love it http://a.co", "RT @x: I love it http://b.co", "so bad", "SO BAD!!", "@user", "fine"],
        "label": [1, 1, 0, 1, 1, 1],
    })
    out, rep = deduplicate(df)
    assert rep.no_content == 1 and rep.conflicting_label_rows == 2 and rep.duplicates == 1
    assert sorted(out["id"]) == ["a", "f"]


def test_split_integrity_detects_leaks():
    a = pd.DataFrame({"id": ["1", "2"], "text": ["hello world", "x"], "author": ["u1", "u2"]})
    b = pd.DataFrame({"id": ["3"], "text": ["Hello, world!"], "author": ["u3"]})
    with pytest.raises(LeakageError, match="near-duplicate"):
        check_split_integrity({"train": a, "test": b})
    c = pd.DataFrame({"id": ["4"], "text": ["new"], "author": ["u1"]})
    with pytest.raises(LeakageError, match="author"):
        check_split_integrity({"train": a, "test": c}, group_col="author")


def test_temporal_split_is_ordered_and_author_disjoint(corpus, cfg):
    s = split_dataset(corpus, cfg.split, cfg.seed)
    assert s.train["timestamp"].max() < s.val["timestamp"].min() <= s.val["timestamp"].max() < s.test["timestamp"].min()
    assert not set(s.train["author"]) & set(s.test["author"])
    assert s.report["eval_rows_dropped_for_group_overlap"]["test"] > 0


@pytest.mark.parametrize("strategy", ["stratified", "group"])
def test_other_strategies(corpus, cfg, strategy):
    cfg.split.strategy = strategy
    s = split_dataset(corpus, cfg.split, cfg.seed)
    assert abs(s.train["label"].mean() - s.test["label"].mean()) < 0.1
    if strategy == "group":
        assert not set(s.train["author"]) & set(s.test["author"])


def test_single_class_split_fails_loudly(corpus, cfg):
    df = corpus.sort_values("label").reset_index(drop=True)
    df["timestamp"] = pd.date_range("2009-01-01", periods=len(df), freq="h", tz="UTC")   # label-sorted time
    cfg.split.drop_eval_groups_seen_in_train = False
    with pytest.raises(ValueError, match="single-class"):
        split_dataset(df, cfg.split, cfg.seed)


def test_lfs_do_not_fire_on_urls():
    L = apply_lfs(["see http://example.com/page", "so sad :(", "yay :)", "not happy at all"])
    names_neg_emoticon = 1
    assert L[0, names_neg_emoticon] == -1          # "://" is not a ":/" emoticon
    assert L[1, names_neg_emoticon] == NEG
    assert L[2, 0] == POS
    assert L[3, 4] == -1                            # negated lexicon hit -> abstain


def test_label_model_and_signal_stripping():
    L = np.array([[1, -1, 1], [0, 0, -1], [-1, -1, -1]])
    y = np.array([1, 0, 1])
    p = LabelModel().fit(L, y).predict_proba(L)
    assert p[0] > 0.5 > p[1] and np.isnan(p[2])
    assert strip_lf_signals("great day :) 😍") == "great day"


def test_weak_label_pipeline():
    df = pd.DataFrame({"text": ["I love this :)", "worst day ever :(", "the sky", "so happy 😍"]})
    out, summary = weak_label(df)
    assert set(out["label"]) == {0, 1} and "the sky" not in out["text"].tolist()
    assert ":)" not in " ".join(out["text"])
    assert {"lf", "coverage", "conflict"} <= set(summary.columns)


def test_active_learning_selects_uncertain_unique():
    pool = pd.DataFrame({"id": [str(i) for i in range(8)],
                         "text": ["a b", "a b!", "c d e f", "g", "h i j", "k l", "m", "n o p q r"]})
    proba = np.array([0.5, 0.5, 0.9, 0.45, 0.99, 0.55, 0.01, 0.6])
    sel = select_for_annotation(pool, proba, k=3)
    assert len(sel) == 3 and sel["text"].map(dedup_key).is_unique
    assert 0.99 not in sel["model_score"].tolist()


def test_merge_annotations_rejects_garbage(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,text,label\n1,a,positive\n2,b,\n3,c,0\n")
    out = merge_annotations(p)
    assert out["label"].tolist() == [1, 0] and (out["label_source"] == "human").all()
    p.write_text("id,text,label\n1,a,maybe\n")
    with pytest.raises(ValueError):
        merge_annotations(p)


def test_stream_ingestor_dedups_and_dead_letters(tmp_path):
    now = datetime.now(timezone.utc)
    recs = [
        {"id": "1", "text": "hello", "timestamp": now},
        {"id": "1", "text": "hello again", "timestamp": now},       # duplicate id
        {"id": "2", "text": "   ", "timestamp": now},               # blank
        {"id": "3", "text": "naive ts", "timestamp": datetime(2020, 1, 1)},
        {"id": "4", "text": "ok", "timestamp": now, "label": 1},
    ]

    async def source():
        for r in recs:
            yield r

    ing = StreamIngestor(tmp_path, flush_size=2, flush_interval_s=0.05)
    stats = asyncio.run(ing.run(source()))
    assert (stats.received, stats.written, stats.duplicates, stats.dead_letter) == (5, 2, 1, 2)
    files = list(tmp_path.rglob("*.parquet"))
    assert sum(len(pd.read_parquet(f)) for f in files) == 2
    assert not list(tmp_path.rglob("*.tmp"))
