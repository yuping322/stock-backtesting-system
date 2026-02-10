"""
调试脚本：对比本地和JQ结果的差异
用于诊断因子值、评分、排序等环节的差异
"""
import pandas as pd
import numpy as np
import datetime
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

import data as data_module
from src.factor_old.factor import FactorTester, CFG

# 选择一个特定日期进行详细调试
DEBUG_DATE = '2025-11-02'
DEBUG_MODEL_IDX = 0  # 调试第一个模型

# 因子列表
factor_list = [
    (['sales_to_price_ratio', 'share_turnover_monthly', 'natural_log_of_market_cap'], 
     [0.0002666821, -0.0020518674, -0.0023101097], 0.0507803593),
    (['size', 'roe_ttm', 'current_asset_turnover_rate'], 
     [-0.0008979094, -0.0000039691, 0.0002272270], -0.0003337466),
]

def main():
    print("="*80)
    print(f"调试日期: {DEBUG_DATE}")
    print(f"调试模型: {DEBUG_MODEL_IDX}")
    print("="*80)
    
    # 获取因子
    all_factors = set()
    for factors, _, _ in factor_list:
        all_factors.update(factors)
    all_factors = list(all_factors)
    
    class Args:
        start = DEBUG_DATE
        end = DEBUG_DATE
        stock_pool = 'stock'
        factors = all_factors
        quantiles = 5
        periods = [5]
        fillna = 0
        winsorize = 0
        neutralize = 0
        standardize = 0
        roll_win = 20
        monitor_csv = 'monitor.csv'
        last_only = False
        factor_dir = None
        max_stocks = None

    cfg = CFG(Args())
    
    # 获取股票池
    stocks_all = data_module.get_index_stocks('small', date=DEBUG_DATE)
    print(f"\n股票池大小: {len(stocks_all)}")
    print(f"前10只股票: {stocks_all[:10]}")
    
    # 加载因子
    tester = FactorTester(cfg)
    tester.stocks = stocks_all
    factor_data = tester.get_factors()
    
    if not factor_data:
        print("错误: 未获取到因子数据")
        return
    
    df_factors = pd.DataFrame(factor_data)
    print(f"\n因子数据形状: {df_factors.shape}")
    print(f"因子列: {df_factors.columns.tolist()}")
    
    # 提取调试日期的数据
    target_date = pd.to_datetime(DEBUG_DATE)
    df_today = df_factors.xs(target_date, level='date')
    
    print(f"\n{DEBUG_DATE}当天股票数: {len(df_today)}")
    
    # 调试第一个模型
    factors, coefs, intercept = factor_list[DEBUG_MODEL_IDX]
    print(f"\n{'='*80}")
    print(f"模型{DEBUG_MODEL_IDX}详细信息")
    print(f"{'='*80}")
    print(f"因子: {factors}")
    print(f"系数: {coefs}")
    print(f"截距: {intercept}")
    
    # 检查因子可用性
    df_model = df_today.copy()
    print(f"\n原始数据形状: {df_model.shape}")
    
    for f in factors:
        if f in df_model.columns:
            null_count = df_model[f].isna().sum()
            print(f"  因子 {f}: 缺失值 {null_count}/{len(df_model)}")
            print(f"    统计: min={df_model[f].min():.6f}, max={df_model[f].max():.6f}, mean={df_model[f].mean():.6f}")
        else:
            print(f"  因子 {f}: ❌ 不存在")
    
    # 只保留有所有因子的股票
    df_model = df_model[factors].dropna()
    print(f"\n去除缺失值后: {len(df_model)} 只股票")
    
    if df_model.empty:
        print("所有股票都有缺失值！")
        return
    
    # 计算得分
    df_model['total_score'] = intercept
    for factor, coef in zip(factors, coefs):
        df_model['total_score'] += coef * df_model[factor]
    
    # 显示得分分布
    print(f"\n得分统计:")
    print(f"  min: {df_model['total_score'].min():.6f}")
    print(f"  max: {df_model['total_score'].max():.6f}")
    print(f"  mean: {df_model['total_score'].mean():.6f}")
    print(f"  median: {df_model['total_score'].median():.6f}")
    
    # 排序并显示Top 20
    df_model = df_model.sort_values('total_score', ascending=False)
    print(f"\nTop 20 股票及其得分:")
    print("="*80)
    print(f"{'排名':<6} {'股票代码':<10} {'得分':<15} {' | '.join([f'{f[:20]:<20}' for f in factors])}")
    print("="*80)
    
    for idx, (stock, row) in enumerate(df_model.head(20).iterrows(), 1):
        factor_vals = ' | '.join([f"{row[f]:>20.6f}" for f in factors])
        print(f"{idx:<6} {stock:<10} {row['total_score']:>15.6f} | {factor_vals}")
    
    # 基本面筛选
    print(f"\n{'='*80}")
    print("基本面筛选")
    print(f"{'='*80}")
    
    top_k = max(2, int(0.10 * len(df_model)))
    top_stocks = df_model.head(top_k).index.tolist()
    print(f"Top {top_k} 只股票: {top_stocks[:10]}")
    
    # 检查eps和市值数据
    if 'eps_ttm' in df_today.columns and 'circulating_market_cap' in df_today.columns:
        df_top = df_today.loc[top_stocks].copy()
        print(f"\nTop {top_k} 股票的基本面数据:")
        print(f"{'股票代码':<10} {'eps_ttm':<15} {'流通市值':<20}")
        print("-"*50)
        for stock in top_stocks[:10]:
            if stock in df_top.index:
                eps = df_top.loc[stock, 'eps_ttm']
                mcap = df_top.loc[stock, 'circulating_market_cap']
                print(f"{stock:<10} {eps:>15.6f} {mcap:>20.2f}")
        
        # 应用eps>0筛选
        df_top = df_top[df_top['eps_ttm'] > 0]
        print(f"\neps_ttm > 0 筛选后: {len(df_top)} 只股票")
        
        if not df_top.empty:
            df_top = df_top.sort_values('circulating_market_cap')
            print(f"\n按流通市值排序后的前10只:")
            for idx, stock in enumerate(df_top.head(10).index, 1):
                eps = df_top.loc[stock, 'eps_ttm']
                mcap = df_top.loc[stock, 'circulating_market_cap']
                print(f"{idx}. {stock}: eps={eps:.6f}, 市值={mcap:.2f}")
            
            final_pick = df_top.index[0]
            print(f"\n✅ 最终选中: {final_pick}")
    else:
        print("缺少基本面数据（eps_ttm或circulating_market_cap）")
    
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)

if __name__ == "__main__":
    main()
