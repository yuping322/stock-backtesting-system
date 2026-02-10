#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate TALIB factors for stocks from formatted_data.csv
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.factor_old.factor_calculator import create_factor_calculator

def generate_talib_for_formatted_stocks():
    """Generate TALIB factors for stocks in formatted_data.csv"""

    # Read stocks from formatted_data_fixed.csv (which has all 281 stocks)
    formatted_file = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data/formatted_data_fixed.csv")
    df = pd.read_csv(formatted_file, dtype={'stock': str})
    stocks = sorted(df['stock'].unique())

    print(f"Found {len(stocks)} stocks in formatted_data_fixed.csv")

    # Date range for 2025 data (extended to ensure TALIB calculation correctness)
    start_date = "2025-06-29"  # Extended start date for TALIB calculation
    end_date = "2025-11-26"    # Current date
    output_dir = Path("/Users/fengzhi/Downloads/git/stock-backtesting-system/exported_data")

    # TALIB factors to generate (user requested)
    talib_factors = [
        'TALIB_MACD_12_26_9',
        'TALIB_MACDEXT_12_26_9_0_0_0',
        'TALIB_MACDFIX_9',
        'TALIB_HT_DCPERIOD'
    ]

    for factor_name in talib_factors:
        print(f"\nGenerating {factor_name}...")

        # Create calculator
        calc = create_factor_calculator(factor_name=factor_name)

        all_data = []

        for i, stock in enumerate(stocks):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(stocks)} stocks")

            try:
                factor_series = calc.calculate(stock, start_date, end_date)
                if not factor_series.empty:
                    factor_series = factor_series.dropna()
                    if not factor_series.empty:
                        stock_df = pd.DataFrame({
                            'date': factor_series.index,
                            'code': stock,
                            'factor_value': factor_series.values
                        })
                        all_data.append(stock_df)
            except Exception as e:
                print(f"  Error processing {stock}: {e}")
                continue

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            result_df = result_df.sort_values(['date', 'code']).reset_index(drop=True)

            # Save
            filename = f"{factor_name}_{start_date}_{end_date}.csv"
            filepath = output_dir / filename
            result_df.to_csv(filepath, index=False, float_format='%.6f')

            print(f"  ✓ Saved {len(result_df)} records to {filepath}")
        else:
            print(f"  ❌ No data generated for {factor_name}")

if __name__ == '__main__':
    generate_talib_for_formatted_stocks()