"""Multi-model training and fusion utilities for the factor workflow.

This module orchestrates training a suite of qlib models, evaluates their
rolling Information Coefficient (IC), and blends their predictions using
stability-aware dynamic weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model
from qlib.utils import init_instance_by_config


def _coerce_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return np.nan
        return float(arr.reshape(-1)[0])
    return value


@dataclass
class ModelRunResult:
    name: str
    model: Model
    prediction: pd.Series
    daily_ic: pd.Series
    ic_mean: float
    ic_std: float
    ic_ema: float
    weight: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_ema": self.ic_ema,
            "weight": self.weight,
        }


@dataclass
class SuiteResult:
    name: str
    fused_prediction: pd.Series
    model_results: List[ModelRunResult]
    weights: Dict[str, float]
    label_series: pd.Series

    def metrics_table(self) -> pd.DataFrame:
        records = []
        for res in self.model_results:
            records.append({"model": res.name, **res.to_dict()})
        return pd.DataFrame(records).set_index("model")


def _prepare_label_series(dataset: DatasetH) -> pd.Series:
    label_df = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_I)
    if isinstance(label_df, pd.DataFrame):
        if isinstance(label_df.columns, pd.MultiIndex):
            try:
                label_series = label_df.xs("label", level=0, axis=1)
            except (KeyError, TypeError):
                label_series = label_df.iloc[:, 0]
        else:
            label_series = label_df.get("label", label_df.iloc[:, 0])
        if isinstance(label_series, pd.DataFrame):
            label_series = label_series.iloc[:, 0]
    elif isinstance(label_df, pd.Series):
        label_series = label_df
    else:
        label_series = pd.Series(label_df)
    label_series = label_series.map(_coerce_scalar)
    return pd.to_numeric(label_series, errors="coerce")


def _calc_daily_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    aligned = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    def _corr(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        if group["pred"].nunique(dropna=True) <= 1 or group["label"].nunique(dropna=True) <= 1:
            return 0.0
        return group["pred"].corr(group["label"], method="spearman")

    daily_ic = aligned.groupby(level=0).apply(_corr).dropna()
    return daily_ic


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level=0)

    def _zscore(values: pd.Series) -> pd.Series:
        std = values.std(ddof=0)
        if std <= 1e-8 or np.isnan(std):
            return values * 0.0
        return (values - values.mean()) / (std + 1e-8)

    return grouped.transform(_zscore)


def _compute_weight(
    daily_ic: pd.Series,
    ema_span: int,
    min_periods: int,
    ic_floor: float,
    vol_penalty: float = 5.0,
) -> float:
    if daily_ic.empty:
        return 0.0
    ema = daily_ic.ewm(span=ema_span, min_periods=min_periods).mean().iloc[-1]
    if np.isnan(ema):
        return 0.0
    adj = ema - ic_floor
    if adj <= 0:
        return 0.0
    ic_std = daily_ic.std(ddof=0)
    if np.isnan(ic_std):
        ic_std = 0.0
    stability = 1.0 / (1.0 + vol_penalty * ic_std)
    weight = adj * max(stability, 0.0)
    return max(weight, 0.0)


def _blend_predictions(preds: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    if not preds:
        return pd.Series(dtype=float)
    aligned_index = None
    for ser in preds.values():
        aligned_index = ser.index if aligned_index is None else aligned_index.intersection(ser.index)
    if aligned_index is None:
        return pd.Series(dtype=float)
    aligned_index = aligned_index.sort_values()

    zscores = {}
    for name, ser in preds.items():
        zscores[name] = _cross_sectional_zscore(ser.loc[aligned_index])

    total_weight = sum(weights.values())
    if total_weight <= 0:
        equal_weight = 1.0 / len(preds)
        weights = {name: equal_weight for name in preds}
        total_weight = 1.0

    fused = sum(weights[name] * zscores[name] for name in preds) / total_weight
    return fused


def train_model_suite(
    suite_name: str,
    dataset: DatasetH,
    model_specs: Iterable[Dict],
    fusion_defaults: Dict[str, float],
) -> SuiteResult:
    label_series = _prepare_label_series(dataset)
    preds: Dict[str, pd.Series] = {}
    results: List[ModelRunResult] = []
    weights: Dict[str, float] = {}

    for spec in model_specs:
        name = spec["name"]
        config = spec["config"]
        weighting_overrides = spec.get("weighting", {})
        fusion_params = {**fusion_defaults, **weighting_overrides}

        model = init_instance_by_config(config)
        model.fit(dataset)
        prediction = model.predict(dataset, "test")
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.iloc[:, 0]
        prediction = prediction.map(_coerce_scalar)
        prediction = pd.to_numeric(prediction, errors="coerce")
        preds[name] = prediction

        daily_ic = _calc_daily_ic(prediction, label_series)
        ic_mean = float(daily_ic.mean()) if not daily_ic.empty else 0.0
        ic_std = float(daily_ic.std(ddof=0)) if not daily_ic.empty else 0.0
        ic_ema = float(
            daily_ic.ewm(span=fusion_params["ema_span"], min_periods=fusion_params["min_periods"]).mean().iloc[-1]
        ) if len(daily_ic) >= fusion_params["min_periods"] else 0.0

        weight = _compute_weight(
            daily_ic,
            ema_span=fusion_params["ema_span"],
            min_periods=fusion_params["min_periods"],
            ic_floor=fusion_params.get("ic_floor", 0.0),
            vol_penalty=fusion_params.get("vol_penalty", 5.0),
        )
        weights[name] = weight

        results.append(
            ModelRunResult(
                name=name,
                model=model,
                prediction=prediction,
                daily_ic=daily_ic,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_ema=ic_ema,
                weight=weight,
            )
        )

    fused_prediction = _blend_predictions(preds, weights)
    return SuiteResult(
        name=suite_name,
        fused_prediction=fused_prediction,
        model_results=results,
        weights=weights,
        label_series=label_series,
    )


__all__ = ["train_model_suite", "SuiteResult", "ModelRunResult"]
