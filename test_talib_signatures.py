#!/usr/bin/env python3
"""
测试所有 TALIB 指标的参数和计算
"""

import sys
import os
sys.path.append('/Users/fengzhi/Downloads/git/stock-backtesting-system')

import pandas as pd
import numpy as np
import talib
import inspect
from factor.factor_calculator import create_factor_calculator

def get_talib_function_signature(func_name):
    """获取 TALIB 函数的签名"""
    try:
        func = getattr(talib, func_name.upper(), None)
        if func is None:
            return None

        sig = inspect.signature(func)
        return {
            'name': func_name,
            'signature': str(sig),
            'parameters': list(sig.parameters.keys()),
            'defaults': {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_talib_function(func_name):
    """分析 TALIB 函数的参数需求"""
    sig_info = get_talib_function_signature(func_name)
    if not sig_info or 'error' in sig_info:
        return None

    params = sig_info['parameters']
    defaults = sig_info.get('defaults', {})

    # 分类参数
    price_params = []
    time_params = []
    other_params = []

    for param in params:
        if param in ['open', 'high', 'low', 'close', 'volume']:
            price_params.append(param)
        elif 'timeperiod' in param.lower() or param in ['fastperiod', 'slowperiod', 'signalperiod']:
            time_params.append(param)
        else:
            other_params.append(param)

    return {
        'function': func_name,
        'price_params': price_params,
        'time_params': time_params,
        'other_params': other_params,
        'all_params': params,
        'defaults': defaults,
        'signature': sig_info['signature']
    }

def test_talib_factor_calculation(factor_name, test_stock='000001', start_date='2023-01-01', end_date='2024-12-31'):
    """测试单个 TALIB 因子的计算"""
    print(f"\n=== 测试 {factor_name} ===")

    try:
        # 分析函数签名
        analysis = analyze_talib_function(factor_name.replace('TALIB_', ''))
        if analysis:
            print(f"函数签名: {analysis['signature']}")
            print(f"价格参数: {analysis['price_params']}")
            print(f"时间参数: {analysis['time_params']}")
            print(f"其他参数: {analysis['other_params']}")
            print(f"默认值: {analysis['defaults']}")
        else:
            print("❌ 无法分析函数签名")

        # 创建计算器
        calc = create_factor_calculator(factor_name)
        print(f"✓ 创建计算器成功: {type(calc).__name__}")

        # 计算因子
        result = calc.calculate(test_stock, start_date, end_date)
        print(f"✓ 计算完成，共 {len(result)} 个数据点")
        print(f"✓ 非 NaN 值: {result.notna().sum()}")

        if result.notna().sum() > 0:
            print(f"✓ 示例值: {result.dropna().head(3).to_dict()}")
            return True
        else:
            print("⚠️  所有值都是 NaN")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_talib_factors():
    """从 talib_factors.txt 加载因子列表"""
    factors = []
    try:
        with open('talib_factors.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    factors.append(line)
    except FileNotFoundError:
        print("❌ talib_factors.txt 文件不存在")
        return []

    return factors

def main():
    """主函数"""
    print("TALIB 指标参数测试")
    print("=" * 50)

    # 加载因子列表
    factors = load_talib_factors()
    if not factors:
        print("没有找到因子列表")
        return

    print(f"共找到 {len(factors)} 个 TALIB 因子")

    # 测试所有因子
    test_factors = factors  # 测试所有因子
    print(f"测试所有 {len(test_factors)} 个因子")

    results = {}
    for factor in test_factors:
        success = test_talib_factor_calculation(factor)
        results[factor] = success

    # 统计结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"成功: {successful}/{total}")

    if successful < total:
        print("\n失败的因子:")
        for factor, success in results.items():
            if not success:
                print(f"  ❌ {factor}")

    print("\n成功的因子:")
    for factor, success in results.items():
        if success:
            print(f"  ✅ {factor}")

if __name__ == '__main__':
    main()