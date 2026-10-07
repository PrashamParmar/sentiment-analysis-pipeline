import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sentiment.evaluation.metrics import binary_metrics, choose_threshold, slice_metrics
from sentiment.models.bundle import BundleIntegrityError, Predictor, build_model
from sentiment.monitoring.drift import DriftMonitor, build_reference_profile, compare_to_profile, psi
from sentiment.registry import LocalRegistry
from sentiment.splitting import split_dataset
from sentiment.training import features_frame, train_and_register
from sentiment.tracking import NullTracker


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.clip(y * 0.35 + rng.normal(0.35, 0.2, 5000), 0, 1)
    ch = choose_threshold(y, p, "max_fpr", max_fpr=0.05)
    m = binary_metrics(y, p, ch.threshold)
    assert ch.feasible and m["fpr"] <= 0.05 + 1e-9
    looser = choose_threshold(y, p, "max_fpr", max_fpr=0.2)
    assert looser.threshold < ch.threshold and looser.val_recall > ch.val_recall
    cost = choose_threshold(y, p, "cost", cost_fp=10, cost_fn=1)
    assert binary_metrics(y, p, cost.threshold)["fpr"] < binary_metrics(y, p, 0.5)["fpr"]


def test_metrics_single_class_does_not_crash():
    m = binary_metrics(np.zeros(10, int), np.linspace(0, 1, 10), 0.5)
    assert np.isnan(m["roc_auc"]) and m["fp"] == 5


def test_slices_flag_fpr_breach():
    frame = pd.DataFrame({"n_words": [3] * 100 + [20] * 100, "has_negation": [True] * 100 + [False] * 100})
    y = np.zeros(200, int)
    p = np.r_[np.full(100, 0.9), np.full(100, 0.1)]       # short texts all false positives
    sl = slice_metrics(frame, y, p, 0.5, min_support=50, max_fpr=0.05).set_index("slice")
    assert sl.loc["len:short(<=5w)", "fpr_alert"] and not sl.loc["len:long(>15w)", "fpr_alert"]


def test_psi_and_drift_monitor():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"n_words": rng.poisson(12, 3000), "score": rng.random(3000), "has_url": rng.random(3000) < 0.1})
    prof = build_reference_profile(ref)
    same = pd.DataFrame({"n_words": rng.poisson(12, 1000), "score": rng.random(1000), "has_url": rng.random(1000) < 0.1})
    shifted = pd.DataFrame({"n_words": rng.poisson(40, 1000), "score": rng.random(1000) ** 4, "has_url": rng.random(1000) < 0.6})
    assert max(compare_to_profile(prof, same).values()) < 0.1
    drift = compare_to_profile(prof, shifted)
    assert drift["n_words"] > 0.2 and drift["has_url"] > 0.2
    mon = DriftMonitor(prof, window_size=500, min_window=100)
    assert mon.compute()["status"] == "insufficient_data"
    mon.add(shifted.to_dict("records"))
    assert mon.compute()["status"] == "drift"
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0


def test_psi_sees_drift_in_zero_inflated_feature():
    ref = pd.DataFrame({"non_ascii_ratio": np.r_[np.zeros(2990), np.full(10, 0.1)]})
    prof = build_reference_profile(ref)
    cur = pd.DataFrame({"non_ascii_ratio": np.r_[np.zeros(500), np.full(500, 0.05)]})   # half the traffic accented
    assert compare_to_profile(prof, cur)["non_ascii_ratio"] > 0.2
    assert compare_to_profile(prof, ref)["non_ascii_ratio"] < 0.01


@pytest.mark.parametrize("name", ["tfidf_logreg", "tfidf_svm", "count_nb", "w2v_logreg"])
def test_classical_models_learn_and_respect_negation(corpus, cfg, name):
    s = split_dataset(corpus, cfg.split, cfg.seed)
    model = build_model(name, cfg)
    info = model.fit(s.train, s.val)
    p = model.predict_proba(s.test["text"].tolist())
    assert binary_metrics(s.test["label"].to_numpy(int), p, 0.5)["roc_auc"] > 0.75
    assert 0 <= p.min() and p.max() <= 1
    assert "cv_recall_at_fpr" in info
    pp = model.predict_proba(["the movie was terrible", "the movie was great"])
    assert pp[0] < pp[1]


def test_bundle_roundtrip_and_parity(trained_bundle):
    pred = Predictor.load(trained_bundle)
    out = pred.predict(["what a great day", "", "awful awful awful"])
    assert out[0].label == "positive" or out[0].score > out[2].score
    assert out[1].empty_after_cleaning and out[1].uncertain
    assert pred.predict(["zzzqqq xxyyzz"])[0].oov_rate == 1.0


def test_uncertainty_is_scale_free_near_high_threshold(trained_bundle, monkeypatch):
    pred = Predictor.load(trained_bundle)
    pred.threshold = 0.91                                    # FP-averse operating point
    scores = iter([0.98, 0.90, 0.30])
    monkeypatch.setattr(pred.model, "predict_proba_cleaned", lambda cleaned: np.array([next(scores) for _ in cleaned]))
    confident_pos, borderline, confident_neg = pred.predict(["great movie", "the game", "awful day"])  # in-vocab
    assert not confident_pos.uncertain and confident_pos.label == "positive"
    assert borderline.uncertain and borderline.label == "negative"
    assert not confident_neg.uncertain


def test_bundle_refuses_preprocessing_skew(trained_bundle, monkeypatch):
    import sentiment.preprocessing.cleaner as cl
    monkeypatch.setattr(cl, "PREPROCESS_VERSION", "999")
    with pytest.raises(BundleIntegrityError, match="fingerprint"):
        Predictor.load(trained_bundle)
    Predictor.load(trained_bundle, strict_fingerprint=False)     # explicit opt-out works


def test_bundle_refuses_golden_mismatch(trained_bundle, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(trained_bundle, bad)
    g = json.loads((bad / "golden.json").read_text(encoding="utf-8"))
    for row in g:
        row["score"] = 1 - row["score"]                            # simulates loading the wrong weights
    (bad / "golden.json").write_text(json.dumps(g), encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="golden"):
        Predictor.load(bad)


def test_train_register_promote_end_to_end(corpus, cfg):
    s = split_dataset(corpus, cfg.split, cfg.seed)
    cfg.registry.promotion.max_test_fpr = 1.0
    gold = s.test.head(60)
    res = train_and_register("tfidf_logreg", cfg, s, gold, NullTracker())
    assert res["promoted"] and res["version"] == 1
    reg = LocalRegistry(cfg.paths.artifacts_dir / "registry", cfg.registry.model_name)
    assert reg.champion_path().exists()
    assert (cfg.paths.reports_dir / res["run_name"] / "test_slices.csv").exists()
    # A challenger that is much worse in recall must not replace the champion.
    reg_data = json.loads(reg.index_path.read_text())
    reg_data["versions"][0]["metrics"]["test"]["recall"] = 0.999
    reg.index_path.write_text(json.dumps(reg_data))
    res2 = train_and_register("count_nb", cfg, s, None, NullTracker())
    assert not res2["promoted"] and reg.champion()["version"] == 1
    assert features_frame(s.test).shape[0] == len(s.test)
