"""Convert the real ``formatted_data_all.csv`` export into qlib static assets.

Outputs:
- ``features_panel.pkl``: MultiIndex DataFrame with the core factor columns.
- ``label_panel.pkl``: forward-return label aligned to the feature index.
- ``meta_series.pkl``: industry & market-cap series for neutralisation utilities.
- ``factor_ic_daily.pkl``: genuine daily Spearman IC scores for the factors.

Run:
    python convert_sample.py
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from datetime import datetime
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


DEFAULT_IC_PICKLE = IC_FILE
DEFAULT_FEATURE_PKL = FEATURES_FILE
DEFAULT_LABEL_PKL = LABEL_FILE
DEFAULT_META_PKL = META_FILE
DEFAULT_CLEAN_RAW = CLEAN_PRICES_FILE

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount", "mkt_cap"]
INFO_COLS = ["industry", "concepts"]
ID_COLS = ["stock", "stock_code", "code", "order_book_id", "instrument", "symbol", "ticker"]
DATE_COLS = ["date", "datetime", "trade_date"]
RESERVED_COLS = set(["raw_code", "market_cap", "outstanding_share", "float_shares"]) | set(PRICE_COLS) | set(INFO_COLS) | set(ID_COLS) | set(DATE_COLS)
FACTOR_COLUMNS = list(dict.fromkeys(core_factor_list))  # default universe; will be trimmed to available columns
LABEL_NAME = "LABEL1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert formatted factor CSV into qlib静态数据集")
    parser.add_argument("--factor-csv", type=Path, required=True, help="包含价格、元数据和因子列的格式化CSV文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可选：自定义输出根目录（默认写入 data/model_tasks/<timestamp>/）",
    )
    return parser.parse_args()


def normalize_instrument(raw: str) -> str:
    code = str(raw).strip().upper()
    if not code:
        return code
    if "." in code:
        return code
    if code.isdigit():
        code = code.zfill(6)
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


def _resolve_date_column(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in DATE_COLS:
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            if candidate != "date":
                df = df.rename(columns={candidate: "date"})
            return df
    raise ValueError("Expected a date column (one of: date, datetime, trade_date)")


def _resolve_stock_column(df: pd.DataFrame) -> pd.DataFrame:
    for candidate in ID_COLS:
        if candidate in df.columns:
            if candidate != "stock":
                df = df.rename(columns={candidate: "stock"})
            df["stock"] = df["stock"].astype(str).str.strip().str.upper()
            return df
    raise ValueError("Expected a stock identifier column (one of: stock, stock_code, code, order_book_id, instrument, symbol, ticker)")


def _infer_factor_columns(df: pd.DataFrame, preferred_order: Sequence[str]) -> list[str]:
    candidate_cols = [c for c in df.columns if c not in RESERVED_COLS]
    ordered = [c for c in preferred_order if c in candidate_cols]
    extras = [c for c in candidate_cols if c not in preferred_order]
    return ordered + extras


def _fetch_market_cap_from_oss(
    instruments: pd.DataFrame,
) -> pd.DataFrame:
    try:
        from data import load_oss_complex_stocks  # type: ignore
    except ImportError:
        try:
            from data.data import load_oss_complex_stocks  # type: ignore
        except ImportError:
            return pd.DataFrame()

    if instruments.empty:
        return pd.DataFrame()

    raw_codes = instruments.get("raw_code")
    if raw_codes is None or raw_codes.isna().all():
        return pd.DataFrame()

    codes = sorted({code for code in raw_codes.dropna().astype(str) if code})
    if not codes:
        return pd.DataFrame()

    date_min = instruments["date"].min()
    date_max = instruments["date"].max()
    if pd.isna(date_min) or pd.isna(date_max):
        return pd.DataFrame()

    requested_fields = [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "outstanding_share",
    ]

    try:
        oss_data = load_oss_complex_stocks(
            codes=codes,
            start=pd.Timestamp(date_min).strftime("%Y-%m-%d"),
            end=pd.Timestamp(date_max).strftime("%Y-%m-%d"),
            fields=requested_fields,
        )
    except Exception as exc:  # pragma: no cover - remote call failures
        print(f"Warning: failed to fetch OSS market data: {exc}")
        return pd.DataFrame()

    if isinstance(oss_data, pd.DataFrame):
        oss_data = {requested_fields[0]: oss_data}

    if not isinstance(oss_data, dict):
        return pd.DataFrame()

    merged: pd.DataFrame | None = None
    for field, df_field in oss_data.items():
        if df_field is None or df_field.empty:
            continue
        series = df_field.stack(dropna=False).rename(f"{field}_oss").reset_index()
        series["date"] = pd.to_datetime(series["date"])
        # 标准化资产列名称，兼容不同字段返回的列名（如 level_1、instrument 等）
        non_value_cols = {"date", f"{field}_oss"}
        asset_cols = [col for col in series.columns if col not in non_value_cols]
        if asset_cols:
            series = series.rename(columns={asset_cols[0]: "asset"})
        if "asset" not in series.columns:
            # 如果无法识别资产列，则跳过该字段以避免 merge 出错
            continue
        if merged is None:
            merged = series
        else:
            merged = merged.merge(series, on=["date", "asset"], how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = merged.rename(columns={"level_1": "asset"})
    merged["raw_code"] = merged["asset"].astype(str).str.extract(r"(\d{6})", expand=False)
    merged["raw_code"] = merged["raw_code"].fillna(merged["asset"].astype(str)).str.zfill(6)

    close_series = merged.get("close_oss")
    share_series = merged.get("outstanding_share_oss")
    if close_series is not None:
        merged["close_oss"] = pd.to_numeric(close_series, errors="coerce")
    if share_series is not None:
        merged["outstanding_share_oss"] = pd.to_numeric(share_series, errors="coerce")
    if "close_oss" in merged.columns and "outstanding_share_oss" in merged.columns:
        merged["mkt_cap_oss"] = merged["close_oss"] * merged["outstanding_share_oss"]

    useful_cols = {"date", "raw_code"}
    useful_cols.update(col for col in merged.columns if col.endswith("_oss"))
    result = merged[list(useful_cols)]
    if "asset" in result.columns:
        result = result.drop(columns=["asset"])

    return result


def load_raw(source: Path | pd.DataFrame, factor_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        df = _resolve_date_column(df)
        df = _resolve_stock_column(df)
        available_factors = _infer_factor_columns(df, factor_cols)
        return df, available_factors

    path = source
    header = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()

    base_candidates: Iterable[str] = [*DATE_COLS, *ID_COLS, "market_cap", "mkt_cap", "outstanding_share", *PRICE_COLS, *INFO_COLS]
    selected = [col for col in base_candidates if col in available]
    factor_candidates = [col for col in available if col not in RESERVED_COLS]
    for col in factor_candidates:
        if col not in selected:
            selected.append(col)

    dtype_map = {}
    if "industry" in selected:
        dtype_map["industry"] = str
    for col in ID_COLS:
        if col in selected:
            dtype_map[col] = str

    parse_date_cols = [col for col in DATE_COLS if col in selected]
    df = pd.read_csv(path, usecols=selected, parse_dates=parse_date_cols or None, dtype=dtype_map)

    df = _resolve_date_column(df)
    df = _resolve_stock_column(df)

    if "industry" in df.columns:
        df["industry"] = df["industry"].fillna("Unknown")

    available_factors = _infer_factor_columns(df, factor_cols)
    return df, available_factors
def merge_price_and_factors(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["instrument"] = df["stock"].apply(normalize_instrument)
    df["raw_code"] = df["instrument"].str.split(".").str[0].str.zfill(6)

    group_cols = ["date", "instrument"]
    value_cols = [c for c in df.columns if c not in group_cols]
    combined = (
        df.groupby(group_cols)[value_cols]
        .agg(first_non_null)
        .reset_index()
        .sort_values(["instrument", "date"])
    )

    oss_cap = _fetch_market_cap_from_oss(combined)
    if oss_cap.empty:
        raise ValueError("Failed to fetch OSS pricing data for provided instruments")

    combined = combined.merge(oss_cap, on=["date", "raw_code"], how="left")

    # Expect all关键行情字段来自 OSS。
    price_fields = ["close", "open", "high", "low"]
    quantity_fields = ["volume", "amount", "outstanding_share"]

    for field in price_fields + quantity_fields:
        oss_col = f"{field}_oss"
        if oss_col in combined.columns:
            values = pd.to_numeric(combined.pop(oss_col), errors="coerce")
        else:
            values = pd.Series(np.nan, index=combined.index, dtype=float)
        combined[field] = values

    if combined["close"].isna().all():
        raise ValueError("OSS data missing close prices for all records")

    for field in ["open", "high", "low"]:
        if combined[field].isna().all():
            combined[field] = combined["close"]
        else:
            combined[field] = combined[field].fillna(combined["close"])

    for field in ["volume", "amount", "outstanding_share"]:
        if field in combined.columns:
            combined[field] = combined[field].astype(float)
            combined[field] = (
                combined.groupby("instrument")[field]
                .transform(lambda s: s.ffill().bfill())
            )

    if "mkt_cap_oss" in combined.columns:
        mkt_cap_series = pd.to_numeric(combined.pop("mkt_cap_oss"), errors="coerce")
    else:
        mkt_cap_series = pd.Series(np.nan, index=combined.index, dtype=float)
    combined["mkt_cap"] = mkt_cap_series

    if combined["mkt_cap"].isna().any():
        combined["mkt_cap"] = combined["mkt_cap"].fillna(combined["close"] * combined["outstanding_share"])

    if combined["mkt_cap"].isna().all():
        raise ValueError("OSS data missing market cap information for all records")

    combined = combined.dropna(subset=["close"])

    combined["industry"] = combined.get("industry", "Unknown")
    combined["industry"] = combined["industry"].fillna("Unknown").astype(str)
    unknown_mask = combined["industry"].str.strip().eq("") | combined["industry"].eq("Unknown")
    prefixes = combined.loc[unknown_mask, "instrument"].str.split(".").str[0].str[:3]
    combined.loc[unknown_mask, "industry"] = "IND" + prefixes.fillna("000")

    # Try to get industry and concept data from data module if available
    try:
        import data
        code_list = combined['instrument'].str.split('.').str[0].unique().tolist()
        if code_list:
            ind_map = data.get_industry_category(code_list)
            cpt_map = data.get_concept_categories(code_list)
            
            def _get_industry(code: str) -> str:
                return ind_map.get(code, "Unknown") if isinstance(ind_map, dict) else "Unknown"
            
            def _get_concepts(code: str) -> str:
                vals = cpt_map.get(code, []) if isinstance(cpt_map, dict) else []
                return ','.join([str(v) for v in vals if v]) if vals else ''
            
            # Update industry if we got better data
            if ind_map:
                clean_codes = combined['instrument'].str.split('.').str[0]
                combined['industry'] = clean_codes.map(_get_industry).fillna(combined['industry'])
            
            # Add concepts column
            if 'concepts' not in combined.columns:
                combined['concepts'] = clean_codes.map(_get_concepts)
            else:
                combined['concepts'] = combined['concepts'].fillna(clean_codes.map(_get_concepts))
    except ImportError:
        print("Warning: Could not import data module for industry/concept data")
        if 'concepts' not in combined.columns:
            combined['concepts'] = ''
    except Exception as e:
        print(f"Warning: Could not load industry/concept data: {e}")
        if 'concepts' not in combined.columns:
            combined['concepts'] = ''

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
    # Compute forward returns from price data
    df[LABEL_NAME] = (
        df.groupby("instrument")["close"].shift(-1) / df["close"] - 1.0
    )
    df[LABEL_NAME] = df[LABEL_NAME].replace([np.inf, -np.inf], np.nan)
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
    raw_source: Path | pd.DataFrame = args.factor_csv

    base_output_dir: Path = args.output_dir or DATA_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = (base_output_dir / timestamp).resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, available_factors = load_raw(raw_source, FACTOR_COLUMNS)
    if not available_factors:
        raise ValueError("No factor columns found in provided CSV")

    merged_df = merge_price_and_factors(raw_df, available_factors)
    filtered_df = apply_filters(merged_df, start=None, end=None, limit_stocks=None)
    labelled_df = compute_forward_return(filtered_df)

    feature_df, label_df, industry_series, mkt_cap_series = build_panel(labelled_df, available_factors)
    ic_df = compute_factor_ic(labelled_df, available_factors)

    feature_path = run_output_dir / FEATURES_FILE.name
    label_path = run_output_dir / LABEL_FILE.name
    meta_path = run_output_dir / META_FILE.name
    ic_path = run_output_dir / IC_FILE.name

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    ic_path.parent.mkdir(parents=True, exist_ok=True)

    save_outputs(feature_df, label_df, industry_series, mkt_cap_series, ic_df, feature_path, label_path, meta_path, ic_path)

    clean_path = run_output_dir / CLEAN_PRICES_FILE.name
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    labelled_df.to_csv(clean_path, index=False)
    latest_marker = base_output_dir / "latest_run.txt"
    try:
        latest_marker.write_text(str(run_output_dir))
    except Exception as exc:
        print(f"Warning: Failed to update latest_run.txt marker: {exc}")

    latest_link = base_output_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
        latest_link.symlink_to(run_output_dir, target_is_directory=True)
    except OSError as exc:
        print(f"Warning: Could not update latest symlink: {exc}")

    print(f"Output directory: {run_output_dir}")
    print("Saved cleaned raw ->", clean_path)
    print("Saved feature panel ->", feature_path)
    print("Saved label panel ->", label_path)
    print("Saved meta series ->", meta_path)
    print("Saved factor IC ->", ic_path)


if __name__ == "__main__":
    main()
