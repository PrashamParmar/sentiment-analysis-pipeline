"""Per-language lexical resources. Adding a language means registering one ``LanguageResources``; nothing else
in the pipeline is English-specific (transformer profiles defer to the model's own tokenizer)."""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(frozen=True)
class LanguageResources:
    code: str
    negators: frozenset[str]
    protected: frozenset[str]          # never removed as stopwords (sentiment-bearing function words)
    stopwords: frozenset[str]
    contractions: dict[str, str] = field(default_factory=dict)
    slang: dict[str, str] = field(default_factory=dict)
    emoticons: dict[str, str] = field(default_factory=dict)
    sarcasm_markers: tuple[str, ...] = ()


_EN_NEGATORS = frozenset({
    "not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "without",
    "hardly", "barely", "scarcely",
})
# Silent bug guard: sklearn's ENGLISH_STOP_WORDS contains "not", "no", "never", "nothing", "very"...
# removing them turns "not good" into "good" and quietly flips sentiment.
_EN_PROTECTED = _EN_NEGATORS | frozenset({
    "very", "too", "but", "however", "although", "though", "yet", "only", "more", "most", "less", "least",
    "much", "few", "really", "so", "against", "off", "down", "up", "well", "again", "all", "enough",
})

_EN_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "shan't": "shall not", "ain't": "is not", "y'all": "you all",
    "let's": "let us", "n't": " not", "'re": " are", "'m": " am", "'ll": " will", "'ve": " have", "'d": " would",
}

_EN_SLANG = {
    "u": "you", "ur": "your", "ya": "you", "yall": "you all", "luv": "love", "gr8": "great", "thx": "thanks",
    "thanx": "thanks", "ty": "thank you", "pls": "please", "plz": "please", "b4": "before", "2day": "today",
    "2nite": "tonight", "tmrw": "tomorrow", "tmr": "tomorrow", "idk": "i do not know", "imo": "in my opinion",
    "imho": "in my opinion", "omg": "oh my god", "smh": "shaking my head", "bc": "because", "cuz": "because",
    "coz": "because", "ppl": "people", "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "dont": "do not", "doesnt": "does not", "didnt": "did not", "cant": "can not", "wont": "will not",
    "isnt": "is not", "wasnt": "was not", "arent": "are not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "shouldnt": "should not", "couldnt": "could not", "wouldnt": "would not",
    "im": "i am", "ive": "i have", "youre": "you are", "thats": "that is", "whats": "what is", "nvm": "never mind",
    "btw": "by the way", "tbh": "to be honest", "rly": "really", "srsly": "seriously", "w/": "with",
    "w/o": "without", "bday": "birthday", "fav": "favorite", "fave": "favorite", "awsome": "awesome",
}

# Emoticons -> words (works for both sparse models and wordpiece vocabularies that lack the symbols).
_EN_EMOTICONS = {
    ":-)": " smile ", ":)": " smile ", "(:": " smile ", "=)": " smile ", ":]": " smile ", ":-d": " laugh ",
    ":d": " laugh ", "xd": " laugh ", ";)": " wink ", ";-)": " wink ", ":p": " playful ", ":-p": " playful ",
    "<3": " heart ", ":-(": " frown ", ":(": " frown ", ":'(": " cry ", ":/": " skeptical ",
    ":-/": " skeptical ", ":|": " neutral face ", "</3": " broken heart ", "-_-": " annoyed ",
}

_EN_SARCASM = (
    "/s", "#sarcasm", "yeah right", "oh great", "just great", "thanks a lot", "just what i needed",
    "love how", "love it when", "sure jan", "big surprise", "what a surprise", "oh joy",
)

EN = LanguageResources(
    code="en",
    negators=_EN_NEGATORS,
    protected=_EN_PROTECTED,
    stopwords=frozenset(ENGLISH_STOP_WORDS) - _EN_PROTECTED,
    contractions=_EN_CONTRACTIONS,
    slang=_EN_SLANG,
    emoticons=_EN_EMOTICONS,
    sarcasm_markers=_EN_SARCASM,
)

LANGUAGES: dict[str, LanguageResources] = {"en": EN}


def register_language(resources: LanguageResources) -> None:
    LANGUAGES[resources.code] = resources


def get_resources(code: str) -> LanguageResources:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise KeyError(f"language '{code}' not registered; available: {sorted(LANGUAGES)}") from exc
