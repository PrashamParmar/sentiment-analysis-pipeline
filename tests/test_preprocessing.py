import pickle

import pytest

from sentiment.config import PreprocessConfig
from sentiment.preprocessing.cleaner import TextCleaner, text_features


@pytest.fixture(scope="module")
def classic() -> TextCleaner:
    return TextCleaner(PreprocessConfig(), "classic")


@pytest.fixture(scope="module")
def light() -> TextCleaner:
    return TextCleaner(PreprocessConfig(), "transformer")


def test_negation_survives_stopword_removal(classic):
    out = classic.clean("This is not good at all")
    assert "not" in out.split()
    assert "good_neg" in out.split()          # negation scope marked
    assert "is" not in out.split()             # real stopword removed


def test_negation_scope_ends_at_punctuation(classic):
    toks = classic.clean("not good. great movie").split()
    assert "good_neg" in toks and "great" in toks and "great_neg" not in toks


def test_contractions_and_slang(classic, light):
    assert "not" in classic.clean("I don't like it").split()
    assert "can not" in light.clean("I can't believe u did this")
    assert "you" in light.clean("I can't believe u did this").split()


def test_heart_emoticon_not_eaten_by_html_stripper(light):
    assert "heart" in light.clean("<3 you so much")
    assert "heart" in light.clean("&lt;3 love")             # HTML entity, as stored in Sentiment140
    assert "great" in light.clean("<b>great</b> product") and "<b>" not in light.clean("<b>great</b>")
    assert "a < b" in light.clean("a < b and c > d")


def test_emoticon_attached_to_word(light):
    assert "smile" in light.clean("great:)")
    assert "laugh" not in light.clean("maxd out")


def test_mojibake_fixed(light):
    assert light.clean("donâ€™t like it") == "do not like it"


def test_emoji_demojized(light):
    out = light.clean("love it 😍")
    assert "smiling face with heart-eyes" in out


def test_urls_mentions_and_elongation(light):
    out = light.clean("@bob soooooo goooood http://t.co/xyz")
    assert "@user" in out and "httpurl" in out
    assert "soo" in out and "sooo" not in out


def test_degenerate_inputs(classic):
    assert classic.clean(None) == ""
    assert classic.clean(float("nan")) == ""
    assert classic.clean("   ") == ""
    assert len(classic.clean("a " * 100_000)) < 20_000          # truncated before regex work


def test_fingerprint_stable_and_sensitive():
    a = TextCleaner(PreprocessConfig(), "classic")
    b = TextCleaner(PreprocessConfig(), "classic")
    c = TextCleaner(PreprocessConfig(remove_stopwords=False), "classic")
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert a.fingerprint() != TextCleaner(PreprocessConfig(), "transformer").fingerprint()


def test_pickle_roundtrip(classic):
    clone = pickle.loads(pickle.dumps(classic))
    s = "I don't love it :( http://x.co"
    assert clone.clean(s) == classic.clean(s)
    assert clone.fingerprint() == classic.fingerprint()


def test_text_features(classic):
    f = text_features("Yeah right, I LOVE Mondays /s 😒 http://x.co", classic)
    assert f["has_url"] and f["has_emoji"] and f["has_sarcasm_marker"]
    assert text_features(None)["n_chars"] == 0
