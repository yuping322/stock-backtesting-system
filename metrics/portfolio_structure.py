"""Portfolio structure and industry concentration metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data import get_industry_category


SMALL_VALUE = 1e-12


def _to_timestamp(value) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def weights_from_holdings(snapshot: Optional[dict]) -> pd.Series:
    """Return normalized weight series from a holdings snapshot."""

    if not snapshot or not snapshot.get("holdings"):
        return pd.Series(dtype=float)

    holdings = snapshot["holdings"]
    weights = {}
    total_value = snapshot.get("total_value")

    for item in holdings:
        code = str(item.get("code", "")).zfill(6)
        if not code:
            continue
        weight = item.get("weight")
        if weight is None:
            value = item.get("value")
            if value is None and total_value:
                continue
            if value is None:
                continue
            if total_value in (None, 0):
                continue
            weight = float(value) / float(total_value)
        weights[code] = float(weight)

    if not weights:
        return pd.Series(dtype=float)

    series = pd.Series(weights, dtype=float)
    total = series.sum()
    if total > 0:
        series = series / total
    return series


def industry_breakdown(snapshot: Optional[dict]) -> Dict[str, float]:
    """Aggregate weights by industry using latest holdings snapshot."""

    weights = weights_from_holdings(snapshot)
    if weights.empty:
        return {}

    industries = get_industry_category(list(weights.index))
    if isinstance(industries, str):
        industries = {weights.index[0]: industries}

    aggregates: Dict[str, float] = defaultdict(float)
    for code, weight in weights.items():
        industry = industries.get(code)
        if not industry:
            industry = "UNKNOWN"
        aggregates[industry] += float(weight)

    sorted_items = sorted(aggregates.items(), key=lambda kv: kv[1], reverse=True)
    return dict(sorted_items)


def diversification_basic(weights: pd.Series) -> Dict[str, float]:
    """Calculate diversification metrics based on position weights."""

    if weights.empty:
        return {
            "effective_positions": 0.0,
            "weight_entropy": 0.0,
            "normalized_entropy": 0.0,
            "gini_coefficient": 0.0,
            "max_single_weight": 0.0,
        }

    normalized = weights / (weights.sum() or 1.0)
    squared_sum = float((normalized ** 2).sum())
    effective = 1.0 / squared_sum if squared_sum > SMALL_VALUE else float(len(normalized))

    entropy = float(-(normalized * np.log(normalized + SMALL_VALUE)).sum())
    normalized_entropy = float(entropy / np.log(len(normalized))) if len(normalized) > 1 else 0.0

    sorted_weights = np.sort(normalized.values)
    n = len(sorted_weights)
    cumulative = np.cumsum(sorted_weights)
    gini = float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n) if cumulative[-1] > SMALL_VALUE else 0.0

    return {
        "effective_positions": effective,
        "weight_entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini_coefficient": max(0.0, min(1.0, gini)),
        "max_single_weight": float(sorted_weights[-1]),
    }


def industry_concentration(industry_weights: Dict[str, float]) -> Dict[str, float]:
    if not industry_weights:
        return {
            "industry_hhi": 0.0,
            "top_industry_weight": 0.0,
            "industry_count": 0,
        }

    weights = np.array(list(industry_weights.values()), dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    hhi = float(np.sum(np.square(weights)))
    top_weight = float(weights.max()) if len(weights) else 0.0

    return {
        "industry_hhi": hhi,
        "top_industry_weight": top_weight,
        "industry_count": int(np.sum(weights > SMALL_VALUE)),
    }


def industry_rotation(prev: Dict[str, float], curr: Dict[str, float]) -> Optional[float]:
    if not prev or not curr:
        return None

    all_keys = sorted(set(prev.keys()) | set(curr.keys()))
    if not all_keys:
        return None

    prev_vector = np.array([prev.get(k, 0.0) for k in all_keys], dtype=float)
    curr_vector = np.array([curr.get(k, 0.0) for k in all_keys], dtype=float)

    if prev_vector.sum() > 0:
        prev_vector = prev_vector / prev_vector.sum()
    if curr_vector.sum() > 0:
        curr_vector = curr_vector / curr_vector.sum()

    numerator = float(np.dot(prev_vector, curr_vector))
    denom = float(np.linalg.norm(prev_vector) * np.linalg.norm(curr_vector))
    if denom <= SMALL_VALUE:
        return None

    cosine_similarity = max(-1.0, min(1.0, numerator / denom))
    return 1.0 - cosine_similarity


def structure_metrics(daily_holdings: Iterable[dict]) -> Dict[str, object]:
    records = list(daily_holdings or [])
    latest = records[-1] if records else None
    previous = records[-2] if len(records) >= 2 else None

    weights = weights_from_holdings(latest)
    basics = diversification_basic(weights)
    industry_weights = industry_breakdown(latest)
    concentration = industry_concentration(industry_weights)
    rotation_value = industry_rotation(
        industry_breakdown(previous) if previous else {},
        industry_weights,
    )

    return {
        **basics,
        **concentration,
        "industry_weights": industry_weights,
        "industry_rotation": rotation_value,
        "snapshot_date": _to_timestamp(latest.get("date")) if latest else None,
    }
