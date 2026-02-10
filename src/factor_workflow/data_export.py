"""Reusable data export helpers for building formatted factor datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    import data  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("data module is required for building formatted datasets") from exc


def _normalize_stock_code(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.upper()
    series = (
        series.str.replace('.XSHG', '', regex=False)
        .str.replace('.XSHE', '', regex=False)
        .str.replace('.XBJ', '', regex=False)
    )
    extracted = series.str.extract(r'(\d{6})')[0]
    return extracted.where(extracted.notna(), series)


def build_formatted_frame(
    codes: Sequence[str] | None,
    start_date: str,
    end_date: str,
    *,
    factors: Iterable[str] | None = None,
    industry_default: str = "Unknown",
    source_csv: Path | None = None,
) -> pd.DataFrame:
    """Build a unified DataFrame combining prices, industries, concepts, and factors."""
    base_cols = [
        "date",
        "stock",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "outstanding_share",
        "mkt_cap",
        "industry",
        "concepts",
    ]

    factor_cols: List[str] = []

    if source_csv is None:
        raise ValueError("source_csv must be provided; dynamic factor fetching has been removed")

    source_path = Path(source_csv)
    if not source_path.exists():
        raise FileNotFoundError(f"Formatted source CSV not found: {source_path}")
    merged = pd.read_csv(source_path)
    if "date" not in merged.columns:
        raise ValueError("Formatted source CSV must contain a 'date' column")
    merged["date"] = pd.to_datetime(merged["date"])
    stock_col = None
    for candidate in ["stock", "stock_code", "instrument", "code"]:
        if candidate in merged.columns:
            stock_col = candidate
            break
    if stock_col is None:
        raise ValueError("Formatted source CSV must contain a stock identifier column")
    if stock_col != "stock":
        merged = merged.rename(columns={stock_col: "stock"})
    merged["stock"] = _normalize_stock_code(merged["stock"])

    try:
        code_list = merged["stock"].dropna().astype(str).unique().tolist()
        ind_map = data.get_industry_category(code_list) if code_list else {}
        cpt_map = data.get_concept_categories(code_list) if code_list else {}
    except Exception:
        ind_map, cpt_map = {}, {}

    def _industry_lookup(code: str) -> str:
        if isinstance(ind_map, dict):
            return ind_map.get(code) or industry_default
        return industry_default

    def _concept_lookup(code: str) -> str:
        if isinstance(cpt_map, dict):
            vals = cpt_map.get(code) or []
            if isinstance(vals, (list, tuple)):
                return ",".join(str(v) for v in vals if v)
        return ""

    merged["stock"] = _normalize_stock_code(merged["stock"])

    industry_series = merged.get("industry")
    if industry_series is None:
        merged["industry"] = merged["stock"].map(_industry_lookup)
    else:
        merged["industry"] = industry_series.fillna("Unknown").astype(str)
        missing_mask = merged["industry"].str.strip().eq("Unknown")
        merged.loc[missing_mask, "industry"] = merged.loc[missing_mask, "stock"].map(_industry_lookup)

    concept_series = merged.get("concepts")
    if concept_series is None:
        merged["concepts"] = merged["stock"].map(_concept_lookup)
    else:
        merged["concepts"] = concept_series.fillna("").astype(str)
        missing_mask = merged["concepts"].str.strip().eq("")
        merged.loc[missing_mask, "concepts"] = merged.loc[missing_mask, "stock"].map(_concept_lookup)

    if "close" in merged.columns and "outstanding_share" in merged.columns and (
        "mkt_cap" not in merged.columns or merged["mkt_cap"].isna().all()
    ):
        with np.errstate(all="ignore"):
            merged["mkt_cap"] = merged["close"].astype(float) * merged["outstanding_share"].astype(float)
    elif "mkt_cap" not in merged.columns:
        merged["mkt_cap"] = np.nan

    from collections import OrderedDict

    existing_factor_cols = [col for col in merged.columns if col not in base_cols]
    factor_cols = list(OrderedDict.fromkeys(factor_cols + existing_factor_cols))

    for col in base_cols + factor_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    out_df = merged[base_cols + factor_cols].sort_values(["date", "stock"]).reset_index(drop=True)
    out_df["date"] = pd.to_datetime(out_df["date"])
    return out_df


__all__ = ["build_formatted_frame"]
