"""Local sklearn-based models to avoid heavy optional dependencies.
"""
from __future__ import annotations

from typing import Optional, Text, Union

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.base import Model
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression


class _RidgeSolver:
    """Minimal ridge regression solver with L2 regularization."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "_RidgeSolver":
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
            sw = np.sqrt(w)
            X_mat = X_mat * sw
            y_vec = y_vec * sw.ravel()

        if self.fit_intercept:
            X_mean = X_mat.mean(axis=0)
            y_mean = y_vec.mean()
            X_centered = X_mat - X_mean
            y_centered = y_vec - y_mean
        else:
            X_mean = np.zeros(X_mat.shape[1], dtype=np.float64)
            y_mean = 0.0
            X_centered = X_mat
            y_centered = y_vec

        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        if s.size == 0:
            self.coef_ = np.zeros(X_mat.shape[1], dtype=np.float64)
            self.intercept_ = float(y_mean)
            return self

        d = s / (s**2 + self.alpha)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            proj = U.T @ y_centered
            coef = (Vt.T * d) @ proj
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        self.coef_ = coef

        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Ridge solver must be fitted before predicting")
        X_mat = np.asarray(X, dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            preds = X_mat @ self.coef_ + self.intercept_
        return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_dataset(
    dataset: DatasetH,
    include_valid: bool = True,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    if include_valid:
        try:
            df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            df_train = pd.concat([df_train, df_valid])
        except KeyError:
            pass
    # Handle case where labels are all NaN (no price data available)
    label_values = df_train["label"].values
    if pd.isna(label_values).all():
        print("Warning: All labels are NaN, creating dummy labels for training")
        # Create dummy labels (zeros) for training when no real labels exist
        df_train = df_train.assign(label=0.0)
    else:
        df_train = df_train.dropna()

    if df_train.empty:
        raise ValueError("Empty training data after processing")

    features = np.asarray(df_train["feature"].values, dtype=np.float64)
    labels = np.asarray(np.squeeze(df_train["label"].values), dtype=np.float64)

    # Replace stray infinities with finite fallbacks and clip extreme magnitudes.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    col_max = np.nanmax(np.abs(features), axis=0)
    col_max[col_max == 0] = 1.0
    features = features / col_max

    return features, labels, df_train


class SklearnLinear(Model):
    """Lightweight linear model wrapper for qlib DatasetH objects."""

    def __init__(
        self,
        estimator: str = "ridge",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        include_valid: bool = True,
    ) -> None:
        estimators = {"ols", "ridge", "lasso"}
        if estimator not in estimators:
            raise ValueError(f"Unsupported estimator '{estimator}'. Choose from {sorted(estimators)}")
        if estimator == "ols" and alpha != 0:
            raise ValueError("alpha should be 0 for OLS")
        self.estimator = estimator
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.include_valid = include_valid
        self._model: Optional[Union[LinearRegression, Lasso, _RidgeSolver]] = None

    def _build_model(self):
        if self.estimator == "ols":
            return LinearRegression(fit_intercept=self.fit_intercept)
        if self.estimator == "ridge":
            return _RidgeSolver(alpha=self.alpha, fit_intercept=self.fit_intercept)
        return Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        features, labels, df_train = _prepare_dataset(dataset, include_valid=self.include_valid)

        sample_weight = None
        if reweighter is not None:
            weights = reweighter.reweight(df_train)
            sample_weight = weights.values

        self._model = self._build_model()
        self._model.fit(features, labels, sample_weight=sample_weight)
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self._model is None:
            raise ValueError("Model must be fitted before calling predict")
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._model.predict(df_test.values)
        return pd.Series(preds, index=df_test.index)


class SklearnHistGB(Model):
    """Histogram-based Gradient Boosting wrapper suitable for large factor sets."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        max_iter: int = 400,
        include_valid: bool = True,
        random_state: int | None = 7,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.include_valid = include_valid
        self.random_state = random_state
        self._model: Optional[HistGradientBoostingRegressor] = None

    def fit(self, dataset: DatasetH, reweighter: Reweighter = None):
        features, labels, _ = _prepare_dataset(dataset, include_valid=self.include_valid)

        self._model = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._model.fit(features, labels)
        return self

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self._model is None:
            raise ValueError("Model must be fitted before calling predict")
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._model.predict(df_test.values)
        return pd.Series(preds, index=df_test.index)


__all__ = ["SklearnLinear", "SklearnHistGB"]
