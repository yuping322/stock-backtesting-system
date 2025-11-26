#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom script to export formatted CSV with price data and TALIB factors
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import data

def load_local_factors(factor_dir: str, factor_names: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load factor data from local CSV files"""
    all_factors = []

    for factor_name in factor_names:
        factor_file = Path(factor_dir) / f"{factor_name}_{start_date}_{end_date}.csv"
        if factor_file.exists():
            df = pd.read_csv(factor_file)
            df['date'] = pd.to_datetime(df['date'])
            df['code'] = df['code'].astype(str).str.zfill(6)
            # Rename factor_value to the factor name
            df = df.rename(columns={'factor_value': factor_name})
            df = df[['date', 'code', factor_name]]
            all_factors.append(df)
        else:
            print(f"Warning: Factor file not found: {factor_file}")

    if not all_factors:
        return pd.DataFrame()

    # Merge all factors
    merged = all_factors[0]
    for df in all_factors[1:]:
        merged = merged.merge(df, on=['date', 'code'], how='outer')

    # Set multi-index
    merged = merged.set_index(['date', 'code'])
    return merged

def export_formatted_csv_with_local_factors(
    codes: list,
    start_date: str,
    end_date: str,
    factor_dir: str,
    factor_names: list,
    output_file: str
):
    """Export formatted CSV with price data and local factors"""
    print(f"Loading price data for {len(codes)} stocks from {start_date} to {end_date}")

    # Load price data
    price_dict = data.load_oss_complex_stocks(
        codes=codes,
        start=start_date,
        end=end_date,
        fields="all"
    )

    if not price_dict:
        print("No price data found")
        return None

    # Merge price data into long format
    merged = None
    for fname, fdf in price_dict.items():
        long_df = fdf.reset_index().melt(
            id_vars='date', var_name='stock', value_name=fname
        )
        if merged is None:
            merged = long_df
        else:
            merged = merged.merge(long_df, on=['date', 'stock'], how='outer')

    if merged is None or merged.empty:
        print("No merged price data")
        return None

    merged = merged.sort_values(['stock', 'date']).reset_index(drop=True)

    # Add market cap calculation
    if 'close' in merged.columns and 'outstanding_share' in merged.columns:
        merged['mkt_cap'] = merged['close'] * merged['outstanding_share']
    else:
        merged['mkt_cap'] = pd.NA

    # Add industry and concept data
    try:
        code_list = merged['stock'].dropna().astype(str).unique().tolist()
        ind_map = data.get_industry_category(code_list) if code_list else {}
        cpt_map = data.get_concept_categories(code_list) if code_list else {}
    except Exception as e:
        print(f"Warning: Could not load industry/concept data: {e}")
        ind_map, cpt_map = {}, {}

    def _code_industry(c: str) -> str:
        if isinstance(ind_map, dict):
            return ind_map.get(c, "Unknown")
        return "Unknown"

    def _code_concepts(c: str) -> str:
        vals = []
        if isinstance(cpt_map, dict):
            vals = cpt_map.get(c, [])
        return ','.join([str(v) for v in vals if v]) if vals else ''

    merged['industry'] = merged['stock'].astype(str).map(_code_industry)
    merged['concepts'] = merged['stock'].astype(str).map(_code_concepts)

    # Load local factors
    print(f"Loading {len(factor_names)} factors from {factor_dir}")
    factors_df = load_local_factors(factor_dir, factor_names, start_date, end_date)

    if not factors_df.empty:
        # Merge factors
        factors_df = factors_df.reset_index()
        factors_df = factors_df.rename(columns={'code': 'stock'})

        # Normalize stock codes
        def _normalize_code(s: pd.Series) -> pd.Series:
            s = s.astype(str).str.upper()
            s = s.str.replace('.XSHG','', regex=False).str.replace('.XSHE','', regex=False).str.replace('.XBJ','', regex=False)
            return s.str.zfill(6)

        merged['stock'] = _normalize_code(merged['stock'])
        factors_df['stock'] = _normalize_code(factors_df['stock'])

        merged = merged.merge(factors_df, on=['date', 'stock'], how='outer')

    # Define final columns
    base_cols = ['date', 'stock', 'open', 'high', 'low', 'close', 'volume', 'amount', 'mkt_cap', 'industry', 'concepts']
    factor_cols = [f for f in factor_names if f in merged.columns]
    final_cols = base_cols + factor_cols

    # Fill missing columns with NA
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # Sort and save
    result = merged[final_cols].sort_values(['date', 'stock']).reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"✓ Exported formatted data to: {output_file}")
    print(f"  Records: {len(result)}")
    print(f"  Stocks: {result['stock'].nunique()}")
    print(f"  Date range: {result['date'].min()} ~ {result['date'].max()}")
    print(f"  Factors included: {factor_cols}")

    return output_file

if __name__ == '__main__':
    # Get the stocks that were used in factor generation
    # Read from one of the factor files to get the stock list
    factor_file = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data/TALIB_HT_DCPERIOD_2024-09-01_2024-11-30.csv")
    if factor_file.exists():
        df = pd.read_csv(factor_file)
        codes = df['code'].unique().tolist()
        print(f"Found {len(codes)} stocks from factor data")
    else:
        print("No factor file found, using default small pool")
        codes = ['000001', '000002', '600000']  # fallback

    # Export with local factors
    output_file = "/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data/formatted_data_with_talib.csv"

    export_formatted_csv_with_local_factors(
        codes=codes,
        start_date="2024-09-01",
        end_date="2024-11-30",
        factor_dir="/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data",
        factor_names=['TALIB_HT_DCPERIOD', 'TALIB_MACD_12_26_9', 'TALIB_MACDEXT_12_26_9_0_0_0', 'TALIB_MACDFIX_9'],
        output_file=output_file
    )