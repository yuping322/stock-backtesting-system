"""Combine long & short model predictions with neutralization and clipping.
Usage example after running workflow_main:
    from combine_signal import combine_predictions
    final = combine_predictions(pred_long, pred_short, industry_series, mkt_cap_series)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float, lstsq_rcond: float = 1e-6) -> np.ndarray:
    """Solve a stabilized linear system via ridge regression.

    Args:
        X: Design matrix.
        y: Target vector.
        alpha: Ridge penalty added to the diagonal of X^T X.
        lstsq_rcond: Fallback conditioning threshold when using lstsq.
    """

    if X.size == 0:
        return np.zeros(0, dtype=float)

    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        LOGGER.debug("SVD failed; falling back to lstsq", exc_info=True)
        beta, *_ = np.linalg.lstsq(X, y, rcond=lstsq_rcond)
        return beta

    if s.size == 0:
        return np.zeros(X.shape[1], dtype=float)

    damp = s / (s**2 + float(alpha))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = U.T @ y
        beta = (Vt.T * damp) @ proj
    return np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)


def _neutralize(
    pred: pd.Series,
    industry: pd.Series,
    mkt_cap: pd.Series,
    extra_styles: dict | None = None,
    ridge_alpha: float = 1e-3,
):
    """Neutralize prediction against industry dummies & log market cap.

    Robustness additions:
    - Coerce mkt_cap to numeric (errors->NaN) then fill with median to avoid object dtype
    - Drop rows with any NaN in design matrix or target
    - Remove constant / near-constant columns to avoid singular matrices
    - If after filtering columns <1, return original pred (no neutralization)
    """
    pred = pd.to_numeric(pred, errors="coerce")
    mkt_cap_numeric = pd.to_numeric(mkt_cap, errors="coerce")
    # fill nan with median (if still nan -> fill 1.0)
    if mkt_cap_numeric.isna().all():
        mkt_cap_numeric = pd.Series(1.0, index=mkt_cap.index)
    else:
        mkt_cap_numeric = mkt_cap_numeric.fillna(mkt_cap_numeric.median())

    df = pd.DataFrame({
        "pred": pred,
        "industry": industry.astype(str),
        "log_mkt": np.log(mkt_cap_numeric.clip(lower=1) + 1e-9),
    })

    X = pd.get_dummies(df["industry"], dummy_na=False)
    X["log_mkt"] = df["log_mkt"].values
    if extra_styles:
        for name, series in extra_styles.items():
            aligned = series.reindex(pred.index)
            X[name] = pd.to_numeric(aligned, errors="coerce").fillna(0).values

    # Align & drop rows with NaN in any regressor or target
    X["_y"] = df["pred"].values
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if X.empty:
        LOGGER.warning("Neutralization skipped because all rows were dropped (NaNs or infs)")
        return pred  # fallback
    y = X.pop("_y").values.astype(float)
    kept_index = X.index
    # Remove constant columns
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]
    if X.shape[1] == 0:
        LOGGER.info("Neutralization design matrix reduced to zero columns; returning demeaned residuals")
        resid = y - y.mean() if len(y) else y
        result = pd.Series(np.nan, index=df.index)
        result.loc[kept_index] = resid
        return result.fillna(pred)
    X_mat = X.values.astype(float)
    # Standardize columns to improve conditioning (avoid scaling issues with sparse dummies)
    col_std = X_mat.std(axis=0, ddof=0)
    scale = np.where(col_std <= 1e-8, 1.0, col_std)
    X_mat_scaled = X_mat / scale
    try:
        beta = _solve_ridge(X_mat_scaled, y, alpha=ridge_alpha)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            y_hat = X_mat_scaled @ beta
        y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:  # pragma: no cover - defensive branch
        LOGGER.exception("Neutralization failed; returning original predictions")
        return pred

    resid = y - y_hat
    result = pd.Series(np.nan, index=df.index)
    result.loc[kept_index] = resid
    # Fill any NaN (dropped rows) with original pred so length matches
    result = result.fillna(pred)
    return result


def combine_predictions(
    pred_long: pd.Series,
    pred_short: pd.Series,
    industry: pd.Series,
    mkt_cap: pd.Series,
    w_long: float = 0.7,
    w_short: float = 0.3,
    clip_q: tuple[float, float] = (0.01, 0.99),
    ridge_alpha: float = 1e-3,
):
    # Align indexes
    common_index = pred_long.index.intersection(pred_short.index)
    pred_long = pred_long.loc[common_index]
    pred_short = pred_short.loc[common_index]
    industry = industry.loc[common_index]
    mkt_cap = mkt_cap.loc[common_index]

    z_long = (pred_long - pred_long.mean()) / (pred_long.std() + 1e-8)
    z_short = (pred_short - pred_short.mean()) / (pred_short.std() + 1e-8)

    raw = w_long * z_long + w_short * z_short
    neutral = _neutralize(raw, industry, mkt_cap, ridge_alpha=ridge_alpha)

    lower = neutral.quantile(clip_q[0])
    upper = neutral.quantile(clip_q[1])
    clipped = neutral.clip(lower=lower, upper=upper)
    final = (clipped - clipped.mean()) / (clipped.std() + 1e-8)
    return final

__all__ = ["combine_predictions"]
