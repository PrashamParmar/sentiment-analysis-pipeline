"""Text cleaning shared byte-for-byte by training and serving.

Two profiles:
* ``classic``: aggressive normalisation for sparse / static-embedding models. Lowercases, marks negation
  scope, removes stopwords (but never negations) and optionally lemmatises.
* ``transformer``: light normalisation only. Subword tokenizers do better on near-raw text, and lowercasing,
  stemming or stopword removal there destroys signal the pretrained model relies on.

Silent-failure guards:
* Train/serve skew: the cleaner has a ``fingerprint()`` (config + code version + resources). It is stored in
  every model bundle, and serving refuses to load a model whose fingerprint differs.
* ``<3``, ``a < b``: the HTML stripper only removes real tags, not every ``<...>`` span.
* ``&lt;3`` and ``&amp;``: HTML entities are unescaped first. Otherwise "lt" and "amp" become top features.
* Mojibake (``donâ€™t``): repaired with ftfy. Otherwise "don't" and "donâ€™t" become different tokens.
* Emoji: demojized into words, because uncased BERT vocabularies map most emoji to [UNK].
* Negations are protected from stopword removal (see resources.py).
* Very long inputs are truncated *before* any regex runs, to avoid ReDoS-style latency spikes.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any, Literal

import emoji
import ftfy
from sklearn.base import BaseEstimator, TransformerMixin

from sentiment.config import PreprocessConfig
from sentiment.preprocessing.resources import LanguageResources, get_resources

PREPROCESS_VERSION = "1.0.0"   # bump on ANY behavioural change -> old models are refused at serving time
Profile = Literal["classic", "transformer"]

_HTML_TAG = re.compile(r"<!--.*?-->|</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>", re.S)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_MENTION = re.compile(r"(?<!\w)@\w{1,30}")
_HASHTAG = re.compile(r"#(\w+)")
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​‌‎‏﻿]")
_WS = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([!?.])\1{2,}")
_CLASSIC_TOKEN = re.compile(r"[a-z0-9_@']+|[.,!?;:]")
_CLAUSE_END = {".", ",", "!", "?", ";", ":"}
_NEGATION_WINDOW = 4


def _alternation(keys: Iterable[str]) -> str:
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


class TextCleaner:
    def __init__(self, cfg: PreprocessConfig | dict[str, Any] | None = None, profile: Profile = "classic") -> None:
        self.cfg = cfg if isinstance(cfg, PreprocessConfig) else PreprocessConfig(**(cfg or {}))
        if profile not in ("classic", "transformer"):
            raise ValueError(f"unknown profile {profile}")
        self.profile = profile
        self.res: LanguageResources = get_resources(self.cfg.language)
        self._compile()

    # -- pickling: regexes are rebuilt, so bundles stay small and portable
    def __getstate__(self) -> dict[str, Any]:
        return {"cfg": self.cfg.model_dump(), "profile": self.profile}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["cfg"], state["profile"])  # type: ignore[misc]

    def _compile(self) -> None:
        r = self.res
        n = self.cfg.max_char_repeat
        self._elong = re.compile(r"(\w)\1{%d,}" % n)
        # Boundaries only where the emoticon edge is alphanumeric: "great:)" must match, ":dog" / "maxd" must not.
        parts = []
        for k in sorted(r.emoticons, key=len, reverse=True):
            pre = r"(?<![a-z0-9])" if k[0].isalnum() else ""
            post = r"(?![a-z])" if k[-1].isdigit() else (r"(?![a-z0-9])" if k[-1].isalnum() else "")
            parts.append(pre + re.escape(k) + post)
        self._emoticon = re.compile("(" + "|".join(parts) + ")", re.I)
        self._emoticon_map = {k.lower(): v for k, v in r.emoticons.items()}
        contr = {k: v for k, v in r.contractions.items() if not k.startswith(("n'", "'"))}
        self._contr_word = re.compile(r"\b(" + _alternation(contr) + r")\b", re.I) if contr else None
        self._contr_map = {k.lower(): v for k, v in contr.items()}
        self._contr_suffix = [(re.compile(re.escape(k) + r"\b", re.I), v)
                              for k, v in r.contractions.items() if k.startswith(("n'", "'"))]
        self._slang = re.compile(r"(?<![\w'])(" + _alternation(r.slang) + r")(?![\w'])", re.I) if r.slang else None
        self._slang_map = {k.lower(): v for k, v in r.slang.items()}
        self._sarcasm = re.compile("|".join(re.escape(m) for m in r.sarcasm_markers), re.I)

    # ------------------------------------------------------------------------------------------ public
    def fingerprint(self) -> str:
        payload = json.dumps(
            {"v": PREPROCESS_VERSION, "profile": self.profile, "cfg": self.cfg.model_dump(),
             "res": [sorted(self.res.stopwords), sorted(self.res.slang.items()), sorted(self.res.emoticons.items()),
                     sorted(self.res.contractions.items())]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def clean(self, text: Any) -> str:
        text = self._normalise(text)
        return self._classic(text) if self.profile == "classic" else text

    def clean_many(self, texts: Iterable[Any]) -> list[str]:
        return [self.clean(t) for t in texts]

    # ------------------------------------------------------------------------------------------ steps
    def _normalise(self, text: Any) -> str:
        if text is None or (isinstance(text, float) and text != text):  # None / NaN
            return ""
        text = str(text)[: self.cfg.max_chars]
        if self.cfg.fix_mojibake:
            text = ftfy.fix_text(text, unescape_html=False)
        text = html.unescape(text)
        if self.cfg.strip_html:
            text = _HTML_TAG.sub(" ", text)
        text = unicodedata.normalize("NFKC", text)
        text = _CTRL.sub("", text)
        text = _URL.sub(self.cfg.url_token, text)
        text = _MENTION.sub(self.cfg.user_token, text)
        text = _HASHTAG.sub(r"\1", text)
        text = self._emoticon.sub(lambda m: self._emoticon_map[m.group(1).lower()], text)
        if self.cfg.demojize:
            text = emoji.replace_emoji(text, replace=lambda e, d: " " + d["en"].strip(":").replace("_", " ") + " ")
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        text = self._elong.sub(lambda m: m.group(1) * self.cfg.max_char_repeat, text)
        text = _REPEAT_PUNCT.sub(r"\1\1", text)
        if self.cfg.expand_contractions:
            if self._contr_word:
                text = self._contr_word.sub(lambda m: self._contr_map[m.group(1).lower()], text)
            for pat, rep in self._contr_suffix:
                text = pat.sub(rep, text)
        if self.cfg.expand_slang and self._slang:
            text = self._slang.sub(lambda m: self._slang_map[m.group(1).lower()], text)
        return _WS.sub(" ", text).strip()

    def _classic(self, text: str) -> str:
        if self.cfg.lowercase:
            text = text.lower()
        tokens = _CLASSIC_TOKEN.findall(text)
        out: list[str] = []
        negating = 0
        for tok in tokens:
            if tok in _CLAUSE_END:
                negating = 0
                if tok in ("!", "?"):
                    out.append(tok)         # "!" and "?" carry sentiment intensity; keep them as tokens
                continue
            if tok in self.res.negators:
                negating = _NEGATION_WINDOW if self.cfg.mark_negation else 0
                out.append(tok)
                continue
            if self.cfg.remove_stopwords and tok in self.res.stopwords:
                continue
            if negating:
                tok = f"{tok}_neg"
                negating -= 1
            out.append(tok)
        if self.cfg.lemmatize:
            out = _lemmatize(out)
        return " ".join(out)

    def has_sarcasm_marker(self, text: str) -> bool:
        return bool(self._sarcasm.search(text))


_nlp = None


def _lemmatize(tokens: list[str]) -> list[str]:
    """Optional spaCy lemmatisation (keeps `_neg` suffixes)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("lemmatize=true needs `pip install spacy && python -m spacy download en_core_web_sm`") from exc
    bases = [t.removesuffix("_neg") for t in tokens]
    lemmas = [tok.lemma_ for tok in _nlp(" ".join(bases))]
    if len(lemmas) != len(tokens):   # spaCy re-tokenised (e.g. "can't") -> fall back rather than misalign
        return tokens
    return [lem + ("_neg" if t.endswith("_neg") else "") for lem, t in zip(lemmas, tokens)]


# ---------------------------------------------------------------------------------- slice / drift features
_EMOTICON_ANY = re.compile(r"(?<!\w)(?:[:;=]'?-?[()\[\]dDpP/|]|<3|</3|\(:)(?!\w)")
_NEG_ANY = re.compile(r"\b(?:not|no|never|nothing|none|nobody|cannot|can't|won't|n't|dont|cant|isnt|didnt)\b|n't\b", re.I)


def text_features(text: Any, cleaner: TextCleaner | None = None) -> dict[str, Any]:
    """Cheap, model-agnostic features of the *raw* text, used for slice evaluation and drift monitoring."""
    t = "" if text is None else str(text)
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "has_url": bool(_URL.search(t)),
        "has_mention": bool(_MENTION.search(t)),
        "has_hashtag": "#" in t,
        "has_emoji": emoji.emoji_count(t) > 0,
        "has_emoticon": bool(_EMOTICON_ANY.search(t)),
        "has_negation": bool(_NEG_ANY.search(t)),
        "has_sarcasm_marker": bool(cleaner.has_sarcasm_marker(t)) if cleaner else False,
        "has_elongation": bool(re.search(r"(\w)\1{2,}", t)),
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "non_ascii_ratio": (sum(ord(c) > 127 for c in t) / len(t)) if t else 0.0,
    }


class CleanerTransformer(BaseEstimator, TransformerMixin):
    """Stateless sklearn wrapper so cleaning lives *inside* the serialized Pipeline (train == serve)."""

    def __init__(self, cfg: dict[str, Any] | None = None, profile: Profile = "classic") -> None:
        self.cfg = cfg
        self.profile = profile

    def fit(self, X: Any, y: Any = None) -> "CleanerTransformer":
        self.cleaner_ = TextCleaner(self.cfg, self.profile)
        return self

    def transform(self, X: Iterable[Any]) -> list[str]:
        cleaner = getattr(self, "cleaner_", None) or TextCleaner(self.cfg, self.profile)
        return cleaner.clean_many(X)
