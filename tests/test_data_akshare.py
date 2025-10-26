"""
data_akshare.py 测试脚本
测试AKShare版本的接口是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("❌ akshare未安装，请运行: pip install akshare")

import pandas as pd
from datetime import datetime, date

if AKSHARE_AVAILABLE:
    try:
        from data_akshare import (
            load_oss_stocks,
            load_modelscope_stocks,
            get_index_stocks,
            get_index_daily,
            get_balance,
            get_income,
            get_cashflow,
            get_valuation,
            load_bt_stocks,
            get_trading_dates,
            code2name,
        )
        print("✅ 成功导入data_akshare模块")
    except Exception as e:
        print(f"❌ 导入data_akshare模块失败: {e}")
        sys.exit(1)


def test_load_oss_stocks():
    """测试加载股票日线数据"""
    print("\n测试: load_oss_stocks")
    try:
        codes = ["000001", "600000"]
        start = "2024-01-01"
        end = "2024-01-10"
        
        df = load_oss_stocks(codes=codes, start=start, end=end)
        print(f"  ✅ 成功加载数据，形状: {df.shape}")
        if not df.empty:
            print(f"  📊 数据预览:\n{df.head()}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_index_stocks():
    """测试获取指数成分股"""
    print("\n测试: get_index_stocks")
    try:
        stocks = get_index_stocks("000300")
        print(f"  ✅ 成功获取沪深300成分股，数量: {len(stocks)}")
        if stocks:
            print(f"  📊 前5只: {stocks[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_index_daily():
    """测试获取指数日线"""
    print("\n测试: get_index_daily")
    try:
        start = "2024-01-01"
        end = "2024-01-10"
        
        nav = get_index_daily("000300", start, end)
        print(f"  ✅ 成功获取指数净值，长度: {len(nav)}")
        if not nav.empty:
            print(f"  📊 净值预览:\n{nav.head()}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_balance():
    """测试获取资产负债表"""
    print("\n测试: get_balance")
    try:
        df = get_balance("000001")
        print(f"  ✅ 成功获取资产负债表，形状: {df.shape}")
        if not df.empty:
            print(f"  📊 列名: {df.columns.tolist()[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_income():
    """测试获取利润表"""
    print("\n测试: get_income")
    try:
        df = get_income("000001")
        print(f"  ✅ 成功获取利润表，形状: {df.shape}")
        if not df.empty:
            print(f"  📊 列名: {df.columns.tolist()[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_cashflow():
    """测试获取现金流量表"""
    print("\n测试: get_cashflow")
    try:
        df = get_cashflow("000001")
        print(f"  ✅ 成功获取现金流量表，形状: {df.shape}")
        if not df.empty:
            print(f"  📊 列名: {df.columns.tolist()[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_valuation():
    """测试获取估值数据"""
    print("\n测试: get_valuation")
    try:
        df = get_valuation("000001")
        print(f"  ✅ 成功获取估值数据，形状: {df.shape}")
        if not df.empty:
            print(f"  📊 列名: {df.columns.tolist()[:5]}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_get_trading_dates():
    """测试获取交易日历"""
    print("\n测试: get_trading_dates")
    try:
        start = "2024-01-01"
        end = "2024-01-10"
        
        dates = get_trading_dates(start, end)
        print(f"  ✅ 成功获取交易日，数量: {len(dates)}")
        if dates:
            print(f"  📊 交易日: {dates[:3]}")
        
        dates_str = get_trading_dates(start, end, as_str=True)
        print(f"  ✅ 成功获取字符串格式交易日")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_code2name():
    """测试代码映射"""
    print("\n测试: code2name")
    try:
        assert isinstance(code2name, dict)
        print(f"  ✅ 代码映射字典，数量: {len(code2name)}")
        if code2name:
            sample = list(code2name.items())[:3]
            print(f"  📊 样本: {sample}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_load_bt_stocks():
    """测试Backtrader数据加载"""
    print("\n测试: load_bt_stocks")
    try:
        codes = ["000001"]
        start = "2024-01-01"
        end = "2024-01-10"
        
        feeds = load_bt_stocks(codes=codes, start=start, end=end)
        print(f"  ✅ 成功加载Backtrader数据，数量: {len(feeds)}")
        if feeds:
            code = list(feeds.keys())[0]
            print(f"  📊 示例: {code}")
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("data_akshare.py 接口测试")
    print("="*60)
    
    if not AKSHARE_AVAILABLE:
        print("\n❌ akshare未安装，无法运行测试")
        print("请运行: pip install akshare")
        return
    
    tests = [
        test_load_oss_stocks,
        test_get_index_stocks,
        test_get_index_daily,
        test_get_balance,
        test_get_income,
        test_get_cashflow,
        test_get_valuation,
        test_get_trading_dates,
        test_code2name,
        test_load_bt_stocks,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_func.__name__} 执行异常: {e}")
            results.append(False)
    
    # 统计结果
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    print(f"📊 通过率: {passed/total*100:.1f}%")


if __name__ == "__main__":
    main()

