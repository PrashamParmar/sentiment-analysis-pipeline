"""Production sentiment analysis pipeline."""

__version__ = "1.0.0"

LABELS: tuple[str, str] = ("negative", "positive")
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = dict(enumerate(LABELS))
