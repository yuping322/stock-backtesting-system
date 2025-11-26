#!/usr/bin/env python3
"""
检查价格数据和行业数据的质量
"""

import sys
import os
sys.path.append('/Users/fengzhi/Downloads/git/stock-backtesting-system')

import data
import pandas as pd
import numpy as np

def check_data_quality():
    """检查价格数据和行业数据的质量"""
    print("检查数据质量...")

    # 测试股票
    test_stocks = ['000001', '002004', '002006', '002007', '002008']
    start_date = '2024-01-01'
    end_date = '2024-01-10'

    print(f"测试股票: {test_stocks}")
    print(f"日期范围: {start_date} 到 {end_date}")

    # 检查价格数据
    print("\n=== 检查价格数据 ===")
    try:
        price_data = data.load_oss_stocks(codes=test_stocks, start=start_date, end=end_date)
        print(f"价格数据形状: {price_data.shape}")
        print(f"价格数据列: {list(price_data.columns)}")
        print(f"价格数据索引: {type(price_data.index)}")
        print(f"价格数据包含 NaN: {price_data.isna().any().any()}")
        print(f"价格数据前5行:\n{price_data.head()}")
    except Exception as e:
        print(f"❌ 价格数据加载失败: {e}")

    # 检查行业数据
    print("\n=== 检查行业数据 ===")
    try:
        industry_data = data.get_industry_category(test_stocks)
        print(f"行业数据: {industry_data}")
        print(f"行业数据类型: {type(industry_data)}")
    except Exception as e:
        print(f"❌ 行业数据加载失败: {e}")

    # 检查概念数据
    print("\n=== 检查概念数据 ===")
    try:
        concept_data = data.get_concept_categories(test_stocks)
        print(f"概念数据: {concept_data}")
        print(f"概念数据类型: {type(concept_data)}")
    except Exception as e:
        print(f"❌ 概念数据加载失败: {e}")

if __name__ == '__main__':
    check_data_quality()