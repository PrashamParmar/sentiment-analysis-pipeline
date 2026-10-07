"""Model interface: every model maps raw texts to P(positive), and carries its own cleaner (train == serve)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment.preprocessing.cleaner import TextCleaner


class SentimentModel(ABC):
    kind: str = "base"
    name: str = "base"
    cleaner: TextCleaner

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
        """Train on ``train`` (``val`` only for early stopping / model selection). Returns training info."""

    def clean(self, texts: list[str]) -> list[str]:
        return self.cleaner.clean_many(texts)

    @abstractmethod
    def predict_proba_cleaned(self, cleaned: list[str]) -> np.ndarray:
        """P(positive) for texts ALREADY cleaned by ``self.clean`` (lets serving clean once)."""

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(positive) for each RAW text."""
        return self.predict_proba_cleaned(self.clean(list(texts)))

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs: Any) -> "SentimentModel": ...

    def vocabulary(self) -> set[str] | None:
        """Word vocabulary for OOV monitoring (None when the model has no closed vocabulary)."""
        return None

    def params(self) -> dict[str, Any]:
        return {}
