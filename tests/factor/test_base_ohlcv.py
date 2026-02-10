"""
测试 _base.py 中的 load_ohlcv_data() 函数
验证是否能正确加载真实的 OHLCV 数据
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.factor.generator._base import load_ohlcv_data


def test_load_ohlcv_basic():
    """测试基本的 OHLCV 加载"""
    print("\n测试 1: 基本 OHLCV 加载")
    print("-" * 60)
    
    stock_codes = ['000001', '000002']
    start_date = '2024-09-01'
    end_date = '2024-09-10'
    
    df = load_ohlcv_data(stock_codes, start_date, end_date)
    
    if df.empty:
        print("❌ 加载失败：返回空 DataFrame")
        return False
    
    print(f"✅ 加载成功")
    print(f"  数据形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  股票数: {df['stock_code'].nunique()}")
    
    # 验证数据结构
    required_cols = ['date', 'stock_code', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要列: {missing_cols}")
        return False
    
    print(f"✅ 数据结构正确")
    return True


def test_load_ohlcv_data_types():
    """测试 OHLCV 数据类型"""
    print("\n测试 2: OHLCV 数据类型")
    print("-" * 60)
    
    stock_codes = ['000001']
    start_date = '2024-09-01'
    end_date = '2024-09-10'
    
    df = load_ohlcv_data(stock_codes, start_date, end_date)
    
    if df.empty:
        print("❌ 加载失败")
        return False
    
    # 检查数据类型
    print(f"date 类型: {df['date'].dtype}")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        print("❌ date 列不是 datetime 类型")
        return False
    
    print(f"stock_code 类型: {df['stock_code'].dtype}")
    if not pd.api.types.is_string_dtype(df['stock_code']) and df['stock_code'].dtype != 'object':
        print("❌ stock_code 列不是字符串类型")
        return False
    
    # 检查数值列
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            print(f"{col} 类型: {df[col].dtype}")
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"⚠️  {col} 列不是数值类型")
    
    print(f"✅ 数据类型检查完成")
    return True


def test_load_ohlcv_content():
    """测试 OHLCV 数据内容"""
    print("\n测试 3: OHLCV 数据内容")
    print("-" * 60)
    
    stock_codes = ['000001', '000002']
    start_date = '2024-09-01'
    end_date = '2024-09-10'
    
    df = load_ohlcv_data(stock_codes, start_date, end_date)
    
    if df.empty:
        print("❌ 加载失败")
        return False
    
    print(f"样本数据:")
    print(df.head(10))
    
    # 验证没有全 NaN 的行
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    available_numeric = [col for col in numeric_cols if col in df.columns]
    
    if available_numeric:
        null_rows = df[available_numeric].isna().all(axis=1).sum()
        print(f"\n全为 NaN 的行数: {null_rows}")
        
        non_null_rows = (~df[available_numeric].isna().all(axis=1)).sum()
        print(f"有数据的行数: {non_null_rows}")
        
        if non_null_rows > 0:
            print(f"✅ 数据有有效值")
            return True
        else:
            print(f"❌ 所有数据都是 NaN")
            return False
    else:
        print("❌ 没有找到任何数值列")
        return False


def test_load_ohlcv_single_stock():
    """测试单只股票的 OHLCV 加载"""
    print("\n测试 4: 单只股票 OHLCV 加载")
    print("-" * 60)
    
    stock_codes = ['000001']
    start_date = '2024-09-01'
    end_date = '2024-09-15'
    
    df = load_ohlcv_data(stock_codes, start_date, end_date)
    
    if df.empty:
        print("❌ 加载失败")
        return False
    
    unique_stocks = df['stock_code'].unique()
    print(f"股票数: {len(unique_stocks)}")
    print(f"股票代码: {unique_stocks}")
    
    if len(unique_stocks) == 1 and unique_stocks[0] == '000001':
        print(f"✅ 正确加载了单只股票")
        return True
    else:
        print(f"❌ 股票代码不符合预期")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("测试 _base.py 中的 load_ohlcv_data() 函数")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("基本 OHLCV 加载", test_load_ohlcv_basic()))
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("基本 OHLCV 加载", False))
    
    try:
        results.append(("OHLCV 数据类型", test_load_ohlcv_data_types()))
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("OHLCV 数据类型", False))
    
    try:
        results.append(("OHLCV 数据内容", test_load_ohlcv_content()))
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("OHLCV 数据内容", False))
    
    try:
        results.append(("单只股票加载", test_load_ohlcv_single_stock()))
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("单只股票加载", False))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
