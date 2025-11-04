"""Backtest evaluation helpers for the simplified factor workflow.

This module trains the ridge + HistGB suites, prepares a neutralized signal,
and offers utilities for backtesting and exporting the scores.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:  # prefer package-relative imports
    from .workflow_main import train_long_short  # reuse training
    from .combine_signal import combine_predictions
    from .paths import (
        BACKTEST_OUTPUT_DIR,
        CLEAN_PRICES_FILE,
        META_FILE,
        RAW_PRICES_FILE,
    )
except ImportError:  # fallback for ``python factor_workflow/backtest_evaluation.py``
    import sys

    _PKG_ROOT = Path(__file__).resolve().parent
    _REPO_ROOT = _PKG_ROOT.parent
    _REPO_PATH = str(_REPO_ROOT)
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)

    from factor_workflow.workflow_main import train_long_short
    from factor_workflow.combine_signal import combine_predictions
    from factor_workflow.paths import (
        BACKTEST_OUTPUT_DIR,
        CLEAN_PRICES_FILE,
        META_FILE,
        RAW_PRICES_FILE,
    )

# --- Config ---
PROVIDER_URI = "~/.qlib/qlib_data"
REGION = "cn"
TOP_K = 50  # number of instruments selected daily
MIN_SIGNAL = None  # optional: minimum threshold
COST_RATE = 0.001  # commission + slippage per trade (round trip approximated)
BENCHMARK = "LOCAL_MEAN"  # use the average level of local instruments as benchmark
PRICES_CLEAN_PATH = CLEAN_PRICES_FILE
PRICES_RAW_PATH = RAW_PRICES_FILE
META_PICKLE_PATH = META_FILE

LOGGER = logging.getLogger(__name__)


def _fallback_meta(index: pd.MultiIndex) -> Tuple[pd.Series, pd.Series]:
    dummy_industry = pd.Series("IND000", index=index, dtype="object")
    dummy_mkt_cap = pd.Series(1e9, index=index, dtype="float64")
    return dummy_industry, dummy_mkt_cap


def _prepare_meta_series(index: pd.MultiIndex) -> Tuple[pd.Series, pd.Series]:
    if not META_PICKLE_PATH.exists():
        LOGGER.warning("Meta file %s not found; using fallback neutralization factors", META_PICKLE_PATH)
        return _fallback_meta(index)

    try:
        with META_PICKLE_PATH.open("rb") as f:
            meta = pickle.load(f)
    except Exception as exc:  # pragma: no cover - defensive I/O branch
        LOGGER.warning("Failed to load meta pickle %s (%s); using fallback", META_PICKLE_PATH, exc)
        return _fallback_meta(index)

    industry_raw: pd.Series | None = meta.get("industry")
    mkt_cap_raw: pd.Series | None = meta.get("mkt_cap")
    if industry_raw is None or mkt_cap_raw is None:
        LOGGER.warning("meta_series.pkl missing industry or mkt_cap; using fallback neutralization factors")
        return _fallback_meta(index)

    industry_series = industry_raw.reindex(index)
    tmp = industry_series.fillna("").astype(str)
    mask_unknown = tmp.str.strip().str.lower().isin(["", "unknown", "nan"])
    index_names = list(tmp.index.names)
    inst_level = index_names.index("instrument") if "instrument" in index_names else len(index_names) - 1
    inst_codes = tmp.index.get_level_values(inst_level)
    fallback_codes = inst_codes.str.split(".").str[0].str[:3]
    fallback_series = pd.Series("IND" + fallback_codes, index=tmp.index).fillna("IND000")
    tmp.loc[mask_unknown] = fallback_series.loc[mask_unknown]
    industry_series = tmp

    if industry_series.isna().all():
        LOGGER.warning("Industry series entirely NaN after alignment; reverting to fallback")
        industry_series, _ = _fallback_meta(index)

    mkt_cap_series = pd.to_numeric(mkt_cap_raw.reindex(index), errors="coerce")
    cross_section_median = mkt_cap_series.groupby(level=0).transform("median")
    mkt_cap_series = mkt_cap_series.fillna(cross_section_median)
    mkt_cap_series = mkt_cap_series.fillna(mkt_cap_series.median())
    mkt_cap_series = mkt_cap_series.fillna(1e9)

    return industry_series, mkt_cap_series

# --- Helper utilities ---

def get_close_series(instruments: list[str], start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    data: dict[str, pd.Series] = {}
    raw_cache: pd.DataFrame | None = None

    for inst in instruments:
        try:
            df = D.features([inst], ["$close"], start_time=start, end_time=end)
        except ValueError:
            df = pd.DataFrame()
        if not df.empty:
            ser = df.xs(inst, level=1, axis=1)["$close"]
            ser.name = inst
            data[inst] = ser
            continue

        if raw_cache is None:
            for candidate in (PRICES_CLEAN_PATH, PRICES_RAW_PATH):
                if candidate.exists():
                    raw = pd.read_csv(candidate, parse_dates=["date"], dtype={"stock": str})
                    if "stock" not in raw.columns and "instrument" in raw.columns:
                        raw = raw.rename(columns={"instrument": "stock"})
                    raw_cache = raw.pivot(index="date", columns="stock", values="close")
                    break
        if raw_cache is not None and inst in raw_cache.columns:
            ser = raw_cache[inst].loc[(raw_cache.index >= start_dt) & (raw_cache.index <= end_dt)]
            ser.name = inst
            data[inst] = ser

    if not data:
        return pd.DataFrame()

    close_df = pd.DataFrame(data)
    close_df.index.name = "datetime"
    close_df = close_df.sort_index().loc[start_dt:end_dt]
    return close_df.reindex(columns=instruments)


def construct_positions(signal: pd.Series) -> pd.DataFrame:
    """Convert signal to daily weights using proportional allocation."""
    df = signal.reset_index()
    df.columns = ["datetime", "instrument", "score"]
    daily_groups = []
    for dt, sub in df.groupby("datetime"):
        sub = sub.sort_values("score", ascending=False)
        if MIN_SIGNAL is not None:
            sub = sub[sub["score"] >= MIN_SIGNAL]
        chosen = sub.head(TOP_K)
        if chosen.empty:
            continue
        weights = chosen["score"].clip(lower=0)
        if weights.sum() == 0:
            continue
        weights = weights / weights.sum()
        out = pd.DataFrame({"datetime": dt, "instrument": chosen["instrument"], "weight": weights})
        daily_groups.append(out)
    if not daily_groups:
        return pd.DataFrame(columns=["datetime", "instrument", "weight"])
    return pd.concat(daily_groups, axis=0)


def calc_metrics(nav: pd.Series) -> Dict[str, float]:
    rets = nav.pct_change().dropna()
    ann_factor = 252
    max_dd = ((nav / nav.cummax()) - 1).min() if not nav.empty else 0.0
    if rets.empty:
        return {
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": float(max_dd) if pd.notna(max_dd) else 0.0,
        }
    ann_ret = (1 + rets.mean()) ** ann_factor - 1
    ann_vol = rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / (ann_vol + 1e-8)
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(max_dd),
    }


def apply_transaction_cost(positions: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Estimate turnover and deduct simple cost."""
    weight_wide = positions.pivot(index="datetime", columns="instrument", values="weight").fillna(0)
    weight_wide = weight_wide.reindex(price_df.index).fillna(0)

    diff = weight_wide.diff().abs().sum(axis=1) / 2
    turnover = diff.fillna(0)

    rets = price_df.pct_change(fill_method=None).fillna(0)
    port_ret_gross = (weight_wide.shift(1) * rets).sum(axis=1)

    cost = turnover * COST_RATE
    port_ret_net = port_ret_gross - cost

    nav = (1 + port_ret_net).cumprod()
    return weight_wide, nav


def get_benchmark_nav(code: str, start: str, end: str, price_df: pd.DataFrame | None = None) -> pd.Series:
    from qlib.data import D

    if code is None:
        return pd.Series(dtype=float)
    if code.upper() == "LOCAL_MEAN":
        if price_df is None or price_df.empty:
            return pd.Series(dtype=float)
        mean_level = price_df.mean(axis=1).dropna()
        if mean_level.empty:
            return pd.Series(dtype=float)
        mean_level = mean_level.loc[(mean_level.index >= pd.Timestamp(start)) & (mean_level.index <= pd.Timestamp(end))]
        if mean_level.empty:
            return pd.Series(dtype=float)
        nav = mean_level / mean_level.iloc[0]
        nav.name = "LOCAL_MEAN"
        nav.index.name = "datetime"
        return nav
    try:
        df = D.features([code], ["$close"], start_time=start, end_time=end)
    except ValueError:
        df = pd.DataFrame()
    if df.empty:
        idx = pd.date_range(start=start, end=end, freq="D")
        if idx.empty:
            return pd.Series(dtype=float)
        ser = pd.Series(1.0, index=idx, name=code)
        ser.index.name = "datetime"
        return ser
    ser = df.xs(code, level=1, axis=1)["$close"].dropna()
    return ser / ser.iloc[0]


# --- End-to-end run ---


def generate_final_signal(
    provider_uri: str = PROVIDER_URI,
    region: str = REGION,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Train model suites and return the neutralized score series.

    Returns:
        final_signal: pd.Series indexed by (datetime, instrument) with "score" name.
        pred_long: Aligned long-only fused prediction.
        pred_short: Aligned short-only fused prediction.
    """

    pred_long, pred_short, *_ = train_long_short(provider_uri, region)

    if isinstance(pred_long, pd.DataFrame):
        pred_long = pred_long.iloc[:, 0]
    if isinstance(pred_short, pd.DataFrame):
        pred_short = pred_short.iloc[:, 0]

    idx = pred_long.index.intersection(pred_short.index)
    if idx.empty:
        LOGGER.error("Predictions from long/short suites have no overlapping index")
        raise RuntimeError("Prediction indices are empty; check dataset configuration")

    pred_long_aligned = pred_long.loc[idx]
    pred_short_aligned = pred_short.loc[idx]

    industry_series, mkt_cap_series = _prepare_meta_series(idx)

    final_signal = combine_predictions(pred_long_aligned, pred_short_aligned, industry_series, mkt_cap_series)
    final_signal = final_signal.sort_index()
    final_signal.name = "score"

    if final_signal.empty:
        LOGGER.error("Final signal is empty after combination")
        raise RuntimeError("Final signal is empty; inspect training outputs")

    return final_signal, pred_long_aligned, pred_short_aligned


def run_backtest():
    final_signal, _, _ = generate_final_signal(PROVIDER_URI, REGION)

    positions = construct_positions(final_signal)
    if positions.empty:
        print("No positions constructed.")
        return

    start = positions["datetime"].min()
    end = positions["datetime"].max()
    instruments = sorted(positions["instrument"].unique())

    price_df = get_close_series(instruments, start, end)
    if price_df.empty:
        print("Price data empty; check instruments or date range.")
        return

    weight_wide, nav = apply_transaction_cost(positions, price_df)
    metrics = calc_metrics(nav)
    bench_nav = get_benchmark_nav(BENCHMARK, start, end, price_df)

    excess = None
    if not bench_nav.empty:
        bench_ret = bench_nav.pct_change().reindex(nav.index).fillna(0)
        strat_ret = nav.pct_change().fillna(0)
        excess = (strat_ret - bench_ret).mean() * 252

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if excess is not None:
        print(f"  annual_excess_return_vs_bench: {excess:.4f}")

    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nav.to_csv(BACKTEST_OUTPUT_DIR / "nav.csv")
    weight_wide.to_csv(BACKTEST_OUTPUT_DIR / "weights.csv")
    positions.to_csv(BACKTEST_OUTPUT_DIR / "positions_raw.csv", index=False)
    print("Backtest results saved to", BACKTEST_OUTPUT_DIR)


__all__ = ["generate_final_signal", "run_backtest"]


if __name__ == "__main__":
    run_backtest()
