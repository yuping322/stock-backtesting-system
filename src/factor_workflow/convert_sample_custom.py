"""Convert the real ``formatted_data_all.csv`` export into qlib static assets.

Outputs:
- ``features_panel.pkl``: MultiIndex DataFrame with the core factor columns.
- ``label_panel.pkl``: forward-return label aligned to the feature index.
- ``meta_series.pkl``: industry & market-cap series for neutralisation utilities.
- ``factor_ic_daily.pkl``: genuine daily Spearman IC scores for the factors.

Run:
    python convert_sample_custom.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .config_factors import core_factor_list
    from .paths import (
        CLEAN_PRICES_FILE,
        DATA_ROOT,
        FEATURES_FILE,
        IC_FILE,
        LABEL_FILE,
        META_FILE,
    )
except ImportError:  # allow running as plain script
    import sys

    _PKG_ROOT = Path(__file__).resolve().parent
    _REPO_ROOT = _PKG_ROOT.parent
    _REPO_PATH = str(_REPO_ROOT)
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)

    from factor_workflow.config_factors import core_factor_list
    from factor_workflow.paths import (
        CLEAN_PRICES_FILE,
        DATA_ROOT,
        FEATURES_FILE,
        IC_FILE,
        LABEL_FILE,
        META_FILE,
    )


# Custom paths for this specific conversion
CUSTOM_DATA_ROOT = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/data/model_tasks")
CUSTOM_INPUT_FILE = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/data/merge_tasks/output_file")

DEFAULT_RAW_FILE = CUSTOM_INPUT_FILE
DEFAULT_IC_PICKLE = CUSTOM_DATA_ROOT / "factor_ic_daily.pkl"
DEFAULT_FEATURE_PKL = CUSTOM_DATA_ROOT / "features_panel.pkl"
DEFAULT_LABEL_PKL = CUSTOM_DATA_ROOT / "label_panel.pkl"
DEFAULT_META_PKL = CUSTOM_DATA_ROOT / "meta_series.pkl"
DEFAULT_CLEAN_RAW = CUSTOM_DATA_ROOT / "prices_cleaned.csv"

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount", "mkt_cap"]
INFO_COLS = ["industry"]
# FACTOR_COLUMNS = list(dict.fromkeys(core_factor_list))  # preserve order, drop dups
FACTOR_COLUMNS = None  # Will be auto-detected from input file
LABEL_NAME = "LABEL1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw CSV into qlib static dataset")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_RAW_FILE, help="Raw CSV with columns date, stock, prices")
    parser.add_argument("--feature-out", type=Path, default=DEFAULT_FEATURE_PKL, help="Output pickle path for feature panel")
    parser.add_argument("--label-out", type=Path, default=DEFAULT_LABEL_PKL, help="Output pickle path for label panel")
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_PKL, help="Output pickle path for meta series")
    parser.add_argument("--ic-out", type=Path, default=DEFAULT_IC_PICKLE, help="Output pickle path for factor IC data")
    parser.add_argument("--clean-csv-out", type=Path, default=DEFAULT_CLEAN_RAW, help="Optional path to write cleaned raw CSV")
    parser.add_argument("--no-clean-csv", action="store_true", help="Skip writing the cleaned raw CSV output")
    parser.add_argument("--limit-stocks", type=int, default=None, help="Optional number of stocks to keep (alphabetical order)")
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive end date filter (YYYY-MM-DD)")
    return parser.parse_args()


def normalize_instrument(raw: str) -> str:
    code = str(raw).strip().upper()
    if not code:
        return code
    if "." in code:
        return code
    if code.startswith("6") or code.startswith("9"):
        suffix = ".XSHG"
    else:
        suffix = ".XSHE"
    return f"{code}{suffix}"


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if not non_null.empty:
        return non_null.iloc[0]
    return np.nan


def get_factor_columns_from_csv(path: Path) -> list[str]:
    """Auto-detect factor columns from CSV file."""
    header = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()

    # Exclude known non-factor columns
    exclude_cols = {"date", "stock_code", "open", "high", "low", "close", "volume", "amount", "mkt_cap", "market_cap", "industry"}
    factor_cols = [col for col in available if col not in exclude_cols]

    print(f"Detected {len(factor_cols)} factor columns from input file")
    return factor_cols


def load_raw(path: Path, factor_cols: Sequence[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()

    # Auto-detect factor columns if not provided
    if factor_cols is None:
        factor_cols = get_factor_columns_from_csv(path)

    missing_factors = [c for c in factor_cols if c not in available]
    if missing_factors:
        raise ValueError(f"Missing factor columns in raw csv: {missing_factors}")

    base_candidates: Iterable[str] = ["date", "stock_code", "market_cap", *PRICE_COLS, *INFO_COLS]
    selected = [col for col in base_candidates if col in available]
    selected += [col for col in factor_cols if col not in selected]

    dtype_map = {"stock_code": str}
    if "industry" in selected:
        dtype_map["industry"] = str
    df = pd.read_csv(path, usecols=selected, parse_dates=["date"], dtype=dtype_map)
    df["stock"] = df["stock_code"].str.strip().str.upper()
    return df


def merge_price_and_factors(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["instrument"] = df["stock"].apply(normalize_instrument)
    df["raw_code"] = df["instrument"].str.split(".").str[0]

    group_cols = ["date", "instrument"]
    value_cols = [c for c in df.columns if c not in group_cols]
    combined = (
        df.groupby(group_cols)[value_cols]
        .agg(first_non_null)
        .reset_index()
        .sort_values(["instrument", "date"])
    )

    # Ensure essential numeric columns exist
    if "market_cap" in combined.columns:
        combined["market_cap"] = pd.to_numeric(combined["market_cap"], errors="coerce")
        if "mkt_cap" in combined.columns:
            combined["mkt_cap"] = pd.to_numeric(combined["mkt_cap"], errors="coerce").fillna(combined["market_cap"])
        else:
            combined["mkt_cap"] = combined["market_cap"]
    else:
        combined["mkt_cap"] = pd.to_numeric(combined.get("mkt_cap"), errors="coerce")

    combined = combined.drop(columns=[c for c in ["market_cap"] if c in combined])

    # Drop rows without prices (if price data exists)
    price_cols = ["open", "high", "low", "close"]
    available_price_cols = [col for col in price_cols if col in combined.columns]

    if available_price_cols:
        combined[available_price_cols] = combined[available_price_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        combined = combined.dropna(subset=["close"])
    else:
        print("Warning: No price columns found in data")

    for col in ["volume", "amount", "mkt_cap"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
            combined[col] = (
                combined.groupby("instrument")[col]
                .transform(lambda s: s.ffill().bfill())
            )

    combined["industry"] = combined.get("industry", "Unknown")
    combined["industry"] = combined["industry"].fillna("Unknown").astype(str)
    unknown_mask = combined["industry"].str.strip().eq("") | combined["industry"].eq("Unknown")
    prefixes = combined.loc[unknown_mask, "instrument"].str.split(".").str[0].str[:3]
    combined.loc[unknown_mask, "industry"] = "IND" + prefixes.fillna("000")

    for col in factor_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["stock"] = combined["instrument"]  # for compatibility with downstream pivots
    combined = combined.reset_index(drop=True)
    return combined


def apply_filters(df: pd.DataFrame, start: str | None, end: str | None, limit_stocks: int | None) -> pd.DataFrame:
    df = df.copy()
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if limit_stocks is not None:
        keep = sorted(df["instrument"].unique())[:limit_stocks]
        df = df[df["instrument"].isin(keep)]
    return df.sort_values(["instrument", "date"]).reset_index(drop=True)


def compute_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Check if we have price data to compute forward returns
    if "close" in df.columns:
        df[LABEL_NAME] = (
            df.groupby("instrument")["close"].shift(-1) / df["close"] - 1.0
        )
        df[LABEL_NAME] = df[LABEL_NAME].replace([np.inf, -np.inf], np.nan)
    else:
        # No price data available, create NaN labels
        print("Warning: No price data found, creating NaN labels")
        df[LABEL_NAME] = np.nan
    return df


def build_panel(df: pd.DataFrame, factor_cols: Sequence[str]):
    idx = pd.MultiIndex.from_frame(df[["date", "instrument"]])
    idx = idx.set_names(["datetime", "instrument"])

    feature_df = pd.DataFrame(df[factor_cols].values, index=idx, columns=factor_cols)
    feature_df.columns = pd.MultiIndex.from_product([["feature"], factor_cols])

    label_df = pd.DataFrame(
        df[LABEL_NAME].values,
        index=idx,
        columns=pd.MultiIndex.from_tuples([( "label", LABEL_NAME)]),
    )

    industry_series = pd.Series(df["industry"].values, index=idx, name="industry")
    mkt_cap_series = pd.Series(df["mkt_cap"].values, index=idx, name="mkt_cap")
    return feature_df, label_df, industry_series, mkt_cap_series


def compute_factor_ic(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    records = {}
    for date, sub in df.groupby("date"):
        valid = sub[[LABEL_NAME] + list(factor_cols)].dropna()
        if len(valid) < 2:
            continue
        ic = valid[factor_cols].corrwith(valid[LABEL_NAME], method="spearman")
        if ic.isna().all():
            continue
        records[date] = ic
    if not records:
        return pd.DataFrame(columns=factor_cols)
    ic_df = pd.DataFrame(records).T.sort_index()
    ic_df.index.name = "date"
    return ic_df


def save_outputs(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    industry_series: pd.Series,
    mkt_cap_series: pd.Series,
    ic_df: pd.DataFrame,
    feature_path: Path,
    label_path: Path,
    meta_path: Path,
    ic_path: Path,
) -> None:
    feature_df.to_pickle(feature_path)
    label_df.to_pickle(label_path)
    with meta_path.open("wb") as f:
        pickle.dump({"industry": industry_series, "mkt_cap": mkt_cap_series}, f)
    with ic_path.open("wb") as f:
        pickle.dump(ic_df, f)


def main():
    args = parse_args()
    raw_df = load_raw(args.input_csv, FACTOR_COLUMNS)
    merged_df = merge_price_and_factors(raw_df, FACTOR_COLUMNS if FACTOR_COLUMNS else get_factor_columns_from_csv(args.input_csv))
    filtered_df = apply_filters(merged_df, start=args.start_date, end=args.end_date, limit_stocks=args.limit_stocks)
    labelled_df = compute_forward_return(filtered_df)

    # Get final factor columns after processing
    final_factor_cols = FACTOR_COLUMNS if FACTOR_COLUMNS else get_factor_columns_from_csv(args.input_csv)
    missing_after_merge = [c for c in final_factor_cols if c not in labelled_df.columns]
    if missing_after_merge:
        raise ValueError(f"Missing factors after merge: {missing_after_merge}")

    feature_df, label_df, industry_series, mkt_cap_series = build_panel(labelled_df, final_factor_cols)
    ic_df = compute_factor_ic(labelled_df, final_factor_cols)

    feature_path = args.feature_out.resolve()
    label_path = args.label_out.resolve()
    meta_path = args.meta_out.resolve()
    ic_path = args.ic_out.resolve()

    save_outputs(feature_df, label_df, industry_series, mkt_cap_series, ic_df, feature_path, label_path, meta_path, ic_path)
    if not args.no_clean_csv:
        clean_path = (args.clean_csv_out or DEFAULT_CLEAN_RAW).resolve()
        labelled_df.to_csv(clean_path, index=False)
        print("Saved cleaned raw ->", clean_path)
    print("Saved feature panel ->", feature_path)
    print("Saved label panel ->", label_path)
    print("Saved meta series ->", meta_path)
    print("Saved factor IC ->", ic_path)


if __name__ == "__main__":
    main()