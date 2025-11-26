#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 TALIB 因子计算器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor.factor_calculator import create_factor_calculator

def test_talib_factor():
    """测试单个 TALIB 因子"""
    factor_name = 'TALIB_AD'  # 选择一个简单的 TALIB 因子测试

    print(f"测试因子: {factor_name}")

    try:
        # 创建计算器
        calculator = create_factor_calculator(factor_name=factor_name)
        print(f"✓ 成功创建计算器: {type(calculator).__name__}")

        # 测试计算
        stock_code = '000001'  # 平安银行
        start_date = '2024-01-01'
        end_date = '2024-01-10'

        print(f"计算股票 {stock_code} 从 {start_date} 到 {end_date}")

        factor_series = calculator.calculate(stock_code, start_date, end_date)

        if factor_series is not None and len(factor_series) > 0:
            print(f"✓ 计算成功，共 {len(factor_series)} 个数据点")
            print("前5个数据点:")
            for i, (date, value) in enumerate(factor_series.head().items()):
                print(f"  {date}: {value}")
        else:
            print("✗ 计算失败或无数据")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_talib_factor()