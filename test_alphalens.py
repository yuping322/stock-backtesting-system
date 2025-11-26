#!/usr/bin/env python3
"""
手动测试 Alphalens 的 get_clean_factor_and_forward_returns 函数
"""

import sys
import os
sys.path.append('/Users/fengzhi/Downloads/git/stock-backtesting-system')

from factor.factor_calculator import create_factor_calculator
import data
import pandas as pd
import numpy as np
import alphalens as al

def test_alphalens_clean():
    """测试 Alphalens 清理函数"""
    print("测试 Alphalens 清理函数...")

    # 获取因子数据
    calculator = create_factor_calculator(factor_name='TALIB_AD')
    test_stocks = ['000001', '002004', '002006', '002007', '002008']
    start_date = '2024-01-01'
    end_date = '2024-01-10'

    factor_data = {}
    for stock in test_stocks:
        factor_series = calculator.calculate(stock, start_date, end_date)
        if factor_series is not None:
            for date, val in factor_series.items():
                factor_data[(pd.Timestamp(date), stock)] = val

    factor_series = pd.Series(factor_data)
    factor_series.index.names = ['date', 'asset']

    print(f"因子数据形状: {factor_series.shape}")
    print(f"因子数据包含 NaN: {factor_series.isna().any()}")
    print(f"因子数据类型: {factor_series.dtype}")

    # 获取价格数据
    price_data = data.load_oss_stocks(codes=test_stocks, start=start_date, end=end_date)
    print(f"价格数据形状: {price_data.shape}")
    print(f"价格数据包含 NaN: {price_data.isna().any().any()}")

    # 获取行业数据
    industry_data = data.get_industry_category(test_stocks)
    dates = pd.date_range(start_date, end_date, freq='D')
    group_dict = {}
    for code in test_stocks:
        grp = industry_data.get(code, 'Other')
        for d in dates:
            group_dict[(pd.Timestamp(d), code)] = grp
    industry_series = pd.Series(group_dict, name='group')

    print(f"行业数据形状: {industry_series.shape}")
    print(f"行业数据包含 NaN: {industry_series.isna().any()}")

    # 尝试 Alphalens 清理
    print("\n=== 尝试 Alphalens 清理 ===")
    try:
        clean = al.utils.get_clean_factor_and_forward_returns(
            factor_series,
            prices=price_data,
            groupby=industry_series,
            quantiles=10,
            periods=[5, 10, 15],
            max_loss=3
        )
        print("✓ Alphalens 清理成功")
        print(f"清理结果类型: {type(clean)}")
        if hasattr(clean, 'shape'):
            print(f"清理结果形状: {clean.shape}")
    except Exception as e:
        print(f"❌ Alphalens 清理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_alphalens_clean()