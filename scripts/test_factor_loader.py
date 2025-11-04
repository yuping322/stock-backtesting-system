#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接测试因子文件加载功能
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from factor.factor_calculator import create_factor_calculator
import pandas as pd

def test_factor_loading():
    """测试因子加载"""
    print("=" * 80)
    print("测试因子文件加载")
    print("=" * 80)
    print()
    
    # 测试因子
    test_factors = ['KMID', 'KLEN']
    factor_dir = 'factors'
    
    # 测试日期范围（从因子文件中提取）
    factor_file = 'factors/Alpha158_20250805_20251103.csv'
    df_meta = pd.read_csv(factor_file, nrows=1)
    df_all = pd.read_csv(factor_file)
    start_date = df_all['date'].min()
    end_date = df_all['date'].max()
    
    print(f"因子文件: {factor_file}")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"测试因子: {', '.join(test_factors)}")
    print()
    
    # 获取股票代码（从因子文件中提取）
    codes = sorted(df_all['code'].astype(str).str.zfill(6).unique())[:10]  # 只测试前10只股票
    print(f"测试股票: {', '.join(codes[:5])}... (共{len(codes)}只)")
    print()
    
    # 测试每个因子
    results = {}
    for factor_name in test_factors:
        print(f"[测试因子] {factor_name}")
        print("-" * 80)
        
        try:
            # 创建因子计算器
            calc = create_factor_calculator(factor_name=factor_name, factor_dir=factor_dir)
            print(f"✓ 创建因子计算器: {type(calc).__name__}")
            
            # 测试加载数据
            factor_data = {}
            success_count = 0
            
            for code in codes:
                try:
                    result = calc.calculate(code, start_date, end_date)
                    if result is not None and len(result) > 0:
                        for date, val in result.items():
                            factor_data[(pd.Timestamp(date), code)] = val
                        success_count += 1
                except Exception as e:
                    print(f"  ⚠️  股票 {code}: {e}")
            
            if factor_data:
                factor_series = pd.Series(factor_data)
                factor_series.index.names = ['date', 'asset']
                results[factor_name] = factor_series
                print(f"✓ 成功加载因子 {factor_name}")
                print(f"  成功股票数: {success_count}/{len(codes)}")
                print(f"  总数据点: {len(factor_series)}")
                print(f"  日期范围: {factor_series.index.get_level_values('date').min()} 到 {factor_series.index.get_level_values('date').max()}")
            else:
                print(f"✗ 因子 {factor_name} 无数据")
                
        except Exception as e:
            print(f"✗ 因子 {factor_name} 加载失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 总结
    print("=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"成功加载因子数: {len(results)}/{len(test_factors)}")
    for factor_name, factor_series in results.items():
        print(f"  {factor_name}: {len(factor_series)} 个数据点")
    print()
    
    if len(results) == len(test_factors):
        print("✅ 所有因子加载成功！")
        return 0
    else:
        print("❌ 部分因子加载失败")
        return 1

if __name__ == '__main__':
    sys.exit(test_factor_loading())

