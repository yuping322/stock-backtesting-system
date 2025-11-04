#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试从因子文件加载因子，并自动使用文件中的日期范围
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor.factor import FactorTester, CFG, parse_args
from factor.factor_calculator import create_factor_calculator

def test_factor_file_date_range():
    """测试从因子文件加载因子，自动使用文件日期范围"""
    factor_file = 'factors/Alpha158_20250806_20251104.csv'
    
    # 检查文件是否存在
    if not os.path.exists(factor_file):
        print(f'❌ 因子文件不存在: {factor_file}')
        return
    
    # 读取因子文件中的日期范围
    import pandas as pd
    df = pd.read_csv(factor_file, usecols=['date'])
    dates = pd.to_datetime(df['date'])
    file_start = dates.min().strftime('%Y-%m-%d')
    file_end = dates.max().strftime('%Y-%m-%d')
    
    print(f'📊 因子文件日期范围: {file_start} 到 {file_end}')
    
    # 选择前5个因子进行测试
    df_full = pd.read_csv(factor_file, nrows=1)
    factor_names = [col for col in df_full.columns if col not in ['date', 'code']]
    test_factors = factor_names[:5]
    
    print(f'📊 测试因子: {test_factors}')
    
    # 创建因子计算器
    factor_dir = 'factors'
    custom_factors = {}
    for fname in test_factors:
        calc = create_factor_calculator(factor_name=fname, factor_dir=factor_dir)
        if calc:
            custom_factors[fname] = calc
            print(f'  ✓ {fname}: 已创建因子计算器')
    
    if not custom_factors:
        print('❌ 没有成功创建任何因子计算器')
        return
    
    # 创建配置（使用文件日期范围）
    class Args:
        def __init__(self):
            self.start = file_start
            self.end = file_end
            self.stock_pool = 'small'
            self.factors = test_factors
            self.quantiles = 5
            self.periods = [5, 10]
            self.fillna = 0
            self.winsorize = 0
            self.neutralize = 0
            self.standardize = 0
            self.roll_win = 20
            self.monitor_csv = 'monitor.csv'
            self.last_only = False
            self.factor_dir = factor_dir
            self.max_stocks = 20  # 限制股票数量加快速度
    
    args = Args()
    cfg = CFG(args)
    
    # 创建因子测试器
    print(f'\n创建因子测试器，验证 {len(custom_factors)} 个因子...')
    tester = FactorTester(cfg, custom_factors=custom_factors)
    
    # 运行测试（会自动使用文件日期范围）
    print(f'\n开始运行因子测试...')
    print(f'配置日期范围: {cfg.START} 到 {cfg.END}')
    results = tester.run(plot=False)
    
    print(f'\n✅ 测试完成，共测试 {len(results)} 个因子周期组合')
    return results

if __name__ == '__main__':
    test_factor_file_date_range()

