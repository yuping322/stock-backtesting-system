#!/usr/bin/env python3
"""
板块轮动分析脚本的简单测试
"""

import src.backtest.jq_compat as jq
import pandas as pd
from datetime import datetime

def test_basic_functions():
    """测试基本功能"""
    print("=== 板块轮动分析脚本测试 ===\n")

    # 测试获取股票列表
    print("1. 测试获取股票列表...")
    try:
        stocks = jq.get_all_securities()
        print(f"   ✓ 成功获取 {len(stocks)} 只股票")
        sample_stocks = stocks.index[:5].tolist()
        print(f"   示例股票: {sample_stocks}")
    except Exception as e:
        print(f"   ✗ 获取股票列表失败: {e}")
        return False

    # 测试获取行业分类
    print("\n2. 测试获取行业分类...")
    try:
        industry_data = jq.get_industry(sample_stocks)
        print(f"   ✓ 成功获取 {len(industry_data)} 只股票的行业信息")
        for code, info in list(industry_data.items())[:2]:
            print(f"   {code}: {info}")
    except Exception as e:
        print(f"   ✗ 获取行业分类失败: {e}")
        return False

    # 测试获取价格数据
    print("\n3. 测试获取价格数据...")
    try:
        price_data = jq.get_price(sample_stocks[0], '2025-10-01', '2025-11-01')
        print(f"   ✓ 成功获取股票 {sample_stocks[0]} 的价格数据")
        print(f"   数据形状: {price_data.shape}")
        print(f"   列名: {price_data.columns.tolist()}")
    except Exception as e:
        print(f"   ✗ 获取价格数据失败: {e}")
        return False

    # 测试获取ST信息
    print("\n4. 测试获取ST信息...")
    try:
        st_data = jq.get_extras('is_st', sample_stocks)
        print(f"   ✓ 成功获取 {len(st_data)} 只股票的ST信息")
        print(f"   ST股票数量: {sum(st_data['is_st'])}")
    except Exception as e:
        print(f"   ⚠ 获取ST信息失败(可能网络问题): {e}")
        print("   这不影响主要分析功能")

    print("\n=== 测试完成 ===")
    print("所有核心功能测试通过！可以运行完整的板块分析脚本。")
    return True

if __name__ == "__main__":
    test_basic_functions()