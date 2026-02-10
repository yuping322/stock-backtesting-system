"""prediction_loader: load and validate multi-model predictions
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass
class Prediction:
    model: str
    date: str
    code: str
    score: float


class PredictionLoader:
    """Load predictions from multiple models and provide simple aggregation utilities."""

    def __init__(self):
        self._predictions: List[Prediction] = []

    def load_from_df(self, df: pd.DataFrame):
        # expect columns: model, date, code, score
        for _, r in df.iterrows():
            self._predictions.append(Prediction(str(r.get("model")), str(r.get("date")), str(r.get("code")), float(r.get("score"))))

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in self._predictions])

    def aggregate_mean(self) -> pd.DataFrame:
        df = self.as_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["code", "mean_score"])
        return df.groupby("code").score.mean().reset_index().rename(columns={"score": "mean_score"})
