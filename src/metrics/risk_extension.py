"""Extended risk and distribution metrics."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

SMALL_VALUE = 1e-12
TRADING_DAYS = 252


def _ensure_series(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.Series(series, dtype=float)


def downside_deviation(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    downside = returns.copy()
    downside[downside > 0] = 0.0
    squared = np.square(downside.values)
    return float(np.sqrt(np.mean(squared)) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    mean_return = float(returns.mean())
    annual_return = mean_return * TRADING_DAYS
    dd = downside_deviation(returns)
    if dd <= SMALL_VALUE:
        return 0.0
    return float(annual_return / dd)


def tail_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    upper = returns.quantile(0.95)
    lower = returns.quantile(0.05)
    if lower >= 0:
        lower = returns.min()
    if abs(lower) <= SMALL_VALUE:
        return 0.0
    return float(upper / abs(lower))


def ulcer_index(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    clean_nav = nav.astype(float)
    base = clean_nav.iloc[0]
    if base <= 0:
        clean_nav = clean_nav / (abs(base) + SMALL_VALUE)
    else:
        clean_nav = clean_nav / base
    running_max = clean_nav.cummax()
    drawdown = clean_nav / running_max - 1.0
    squared = np.square(np.minimum(drawdown, 0))
    return float(np.sqrt(np.mean(squared)))


def distribution_stats(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
        }
    return {
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }


def extended_risk_metrics(strategy_nav: Optional[pd.Series]) -> Dict[str, float]:
    nav = _ensure_series(strategy_nav)
    returns = nav.pct_change().dropna()

    metrics = {
        "sortino_ratio": sortino_ratio(returns),
        "downside_deviation": downside_deviation(returns),
        "tail_ratio": tail_ratio(returns),
        "ulcer_index": ulcer_index(nav),
    }

    metrics.update(distribution_stats(returns))
    metrics["return_count"] = int(len(returns))
    return metrics
