#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge TALIB factors with existing formatted data
"""

import pandas as pd
from pathlib import Path

def merge_talib_factors_with_formatted_data():
    """Merge TALIB factors with existing formatted_data.csv"""

    base_dir = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data")

    # Read existing formatted data
    formatted_file = base_dir / "formatted_data.csv"
    if not formatted_file.exists():
        print("Error: formatted_data.csv not found")
        return

    print("Reading existing formatted data...")
    df = pd.read_csv(formatted_file)
    df['date'] = pd.to_datetime(df['date'])
    df['stock'] = df['stock'].astype(str).str.zfill(6)

    print(f"Loaded {len(df)} records with {df['stock'].nunique()} stocks")

    # List of TALIB factors to merge
    talib_factors = [
        'TALIB_HT_DCPERIOD',
        'TALIB_MACD_12_26_9',
        'TALIB_MACDEXT_12_26_9_0_0_0',
        'TALIB_MACDFIX_9'
    ]

    # Merge each TALIB factor
    for factor_name in talib_factors:
        factor_file = base_dir / f"{factor_name}_2024-09-01_2024-11-30.csv"
        if factor_file.exists():
            print(f"Merging {factor_name}...")
            factor_df = pd.read_csv(factor_file)
            factor_df['date'] = pd.to_datetime(factor_df['date'])
            factor_df['code'] = factor_df['code'].astype(str).str.zfill(6)

            # Merge on date and stock/code
            df = df.merge(
                factor_df[['date', 'code', 'factor_value']],
                left_on=['date', 'stock'],
                right_on=['date', 'code'],
                how='left'
            )

            # Rename factor_value to factor name
            df = df.rename(columns={'factor_value': factor_name})

            # Drop the code column
            if 'code' in df.columns:
                df = df.drop('code', axis=1)

            print(f"  Added {factor_name} with {df[factor_name].notna().sum()} non-null values")
        else:
            print(f"Warning: {factor_file} not found")

    # Sort by date and stock
    df = df.sort_values(['date', 'stock']).reset_index(drop=True)

    # Save the merged data
    output_file = base_dir / "formatted_data_with_talib.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✓ Merged data saved to: {output_file}")
    print(f"  Total records: {len(df)}")
    print(f"  Stocks: {df['stock'].nunique()}")
    print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  Columns: {list(df.columns)}")

    # Show sample
    print("\nSample of merged data:")
    sample = df.head(3)
    print(sample.to_string(index=False))

if __name__ == '__main__':
    merge_talib_factors_with_formatted_data()