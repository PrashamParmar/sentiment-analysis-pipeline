"""MLflow pyfunc adapter so any bundle can be served by `mlflow models serve` or loaded with mlflow.pyfunc."""
from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd


class SentimentPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from sentiment.models.bundle import Predictor
        self.predictor = Predictor.load(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist() if "text" in model_input else model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        return pd.DataFrame([vars(p) for p in self.predictor.predict(texts)])
