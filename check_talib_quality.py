#!/usr/bin/env python3
"""
检查 TALIB_AD 因子数据的质量
"""

import sys
import os
sys.path.append('/Users/fengzhi/Downloads/git/stock-backtesting-system')

from factor.factor_calculator import create_factor_calculator
import pandas as pd
import numpy as np

def check_talib_ad_quality():
    """检查 TALIB_AD 因子数据的质量"""
    print("检查 TALIB_AD 因子数据质量...")

    # 创建计算器
    calculator = create_factor_calculator(factor_name='TALIB_AD')
    print(f"✓ 创建计算器: {type(calculator)}")

    # 测试股票
    test_stock = '000001'
    start_date = '2024-01-01'
    end_date = '2024-01-10'

    print(f"计算股票 {test_stock} 从 {start_date} 到 {end_date}")

    # 计算因子
    factor_series = calculator.calculate(test_stock, start_date, end_date)

    if factor_series is None or len(factor_series) == 0:
        print("❌ 因子计算失败或无数据")
        return

    print(f"✓ 计算成功，共 {len(factor_series)} 个数据点")

    # 检查数据质量
    values = list(factor_series.values)
    print(f"前5个值: {values[:5]}")

    # 检查 NaN
    nan_count = sum(1 for v in values if pd.isna(v))
    print(f"NaN 值数量: {nan_count}")

    # 检查无穷大
    inf_count = sum(1 for v in values if np.isinf(v))
    print(f"无穷大值数量: {inf_count}")

    # 检查数据类型
    print(f"数据类型: {type(values[0])}")

    # 检查数值范围
    finite_values = [v for v in values if pd.notna(v) and not np.isinf(v)]
    if finite_values:
        print(f"最小值: {min(finite_values)}")
        print(f"最大值: {max(finite_values)}")
        print(f"平均值: {np.mean(finite_values)}")
        print(f"标准差: {np.std(finite_values)}")

    # 创建 MultiIndex Series
    factor_data = {}
    for date, val in factor_series.items():
        factor_data[(pd.Timestamp(date), test_stock)] = val

    factor_series_full = pd.Series(factor_data)
    factor_series_full.index.names = ['date', 'asset']

    print(f"创建的因子 Series 长度: {len(factor_series_full)}")
    print(f"因子 Series 包含 NaN: {factor_series_full.isna().any()}")
    print(f"因子 Series 包含无穷大: {np.isinf(factor_series_full).any()}")

    return factor_series_full

if __name__ == '__main__':
    check_talib_ad_quality()