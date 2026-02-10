"""
完整迁移notebook策略 - 使用FactorTester从OSS获取因子数据
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

# ==========================================
# 数据过滤函数
# ==========================================
def filter_kcbj_stock(stock_list):
    """过滤科创板北交所股票"""
    return [s for s in stock_list if not (s.startswith('4') or s.startswith('8') or s.startswith('68'))]

# ==========================================
# 因子列表（来自notebook）
# ==========================================
factor_list = [
    (['sales_to_price_ratio', 'share_turnover_monthly', 'natural_log_of_market_cap'], 
     [0.0002666821, -0.0020518674, -0.0023101097], 0.0507803593),
    (['size', 'roe_ttm', 'current_asset_turnover_rate'], 
     [-0.0008979094, -0.0000039691, 0.0002272270], -0.0003337466),
    (['VOL10', 'single_day_VPT_12'], 
     [-0.0006370810, -0.0000001720], 0.0027864796),
    (['adjusted_profit_to_total_profit'], 
     [-0.0000013402], 0.0013302010),
    (['super_quick_ratio', 'cube_of_size', 'cfo_to_ev'], 
     [0.0000357266, -0.0003667557, 0.0130488065], 0.0002890622),
    (['cash_to_current_liability', 'operating_tax_to_operating_revenue_ratio_ttm', 'Price3M'], 
     [-0.0003459985, 0.0010498108, -0.0233277951], 0.0013685457),
    (['liquidity', 'roa_ttm'], 
     [-0.0027426855, -0.0027239563], 0.0013502358),
    (['VSTD10', 'ROC60'], 
     [-0.0000000004, -0.0000823648], 0.0013880176),
]

# ==========================================
# 主函数
# ==========================================
def main():
    print("="*60)
    print("开始运行选股打分程序")
    print("="*60)
    
    # 收集所有需要的因子
    all_factors = set()
    for factors, _, _ in factor_list:
        all_factors.update(factors)
    
    # 添加基本面因子（用于筛选）
    all_factors.add('eps_ttm')
    all_factors.add('circulating_market_cap')
    
    all_factors = list(all_factors)
    
    print(f"\n需要的因子: {len(all_factors)} 个")
    print(f"  - 评分因子: {len(all_factors) - 2} 个")
    print(f"  - 基本面因子: 2 个 (eps_ttm, circulating_market_cap)")
    print(f"模型数量: {len(factor_list)}")
    
    # 配置FactorTester
    class Args:
        start = '2025-11-01'
        end = datetime.datetime.now().strftime('%Y-%m-%d')
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
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    stocks_all = data_module.get_index_stocks('small', date=end_date)
    print(f"获取到 {len(stocks_all)} 只小盘股")

    # 创建FactorTester并获取因子数据（从OSS读取）
    print("\n正在从OSS加载因子数据...")
    tester = FactorTester(cfg)  # 不传custom_factors，让它从OSS读取
    tester.stocks = stocks_all  # 设置股票列表
    factor_data = tester.get_factors()
    
    if not factor_data:
        print("错误: 未获取到因子数据")
        return
    
    # 转换为DataFrame
    df_factors = pd.DataFrame(factor_data)
    print(f"因子数据形状: {df_factors.shape}")
    
    # 获取日期范围
    dates = df_factors.index.get_level_values('date').unique()
    print(f"数据日期范围: {dates.min().date()} 至 {dates.max().date()}")
    print(f"总共 {len(dates)} 个交易日")
    
    # 为每个日期选股
    all_results = []
    stock_num = 1  # 每个模型每天选1只股票
    
    print("\n" + "="*60)
    print("开始按日期处理")
    print("="*60)
    
    # 按日期分组处理
    for date_idx, target_date in enumerate(sorted(dates)):
        print(f"\n处理日期 {date_idx+1}/{len(dates)}: {target_date.date()}")
        
        # 提取当天的数据
        df_today = df_factors.xs(target_date, level='date')
        stock_list = df_today.index.tolist()
        
        # 过滤科创板北交所股票
        stock_list = filter_kcbj_stock(stock_list)
        
        if not stock_list:
            print("  无有效股票")
            continue
        
        # 当天选中的股票
        daily_picks = []
        
        # 遍历每个模型
        for model_idx, (factors, coefs, intercept) in enumerate(factor_list):
            # 筛选当天数据
            df_model = df_today.loc[stock_list].copy()
            
            # 检查因子是否存在
            missing_factors = [f for f in factors if f not in df_model.columns]
            if missing_factors:
                print(f"  模型{model_idx}: 缺失因子 {missing_factors}")
                continue
            
            # 只保留所有因子都有值的股票
            df_model = df_model[factors].dropna()
            
            if df_model.empty:
                print(f"  模型{model_idx}: 所有股票因子都有缺失值")
                continue
            
            # 计算总分
            df_model['total_score'] = intercept
            for factor, coef in zip(factors, coefs):
                df_model['total_score'] += coef * df_model[factor]
            
            # 按分数降序排序
            df_model = df_model.sort_values('total_score', ascending=False)
            
            # 选择前10%的股票
            top_k = max(2, int(0.10 * len(df_model)))
            top_stocks = df_model.head(top_k).index.tolist()
            
            # 基本面筛选：按流通市值排序，选择eps>0的股票
            if 'eps_ttm' in df_today.columns and 'circulating_market_cap' in df_today.columns:
                df_top = df_today.loc[top_stocks].copy()
                df_top = df_top[df_top['eps_ttm'] > 0]
                
                if not df_top.empty:
                    df_top = df_top.sort_values('circulating_market_cap')
                    selected_stocks = df_top.index.tolist()
                else:
                    selected_stocks = top_stocks
            else:
                selected_stocks = top_stocks
            
            # 选取stock_num只股票
            selected_stocks = selected_stocks[:min(stock_num, len(selected_stocks))]
            
            print(f"  模型{model_idx}: 选中 {selected_stocks}")
            
            # 添加到当天选股列表
            for stock in selected_stocks:
                if stock not in daily_picks:
                    daily_picks.append(stock)
                else:
                    print(f"  模型{model_idx}: {stock} 已被其他模型选中，跳过")
        
        # 保存当天的结果
        if daily_picks:
            weight = 1.0 / len(daily_picks)
            for stock in daily_picks:
                all_results.append({
                    'date': target_date.strftime('%Y-%m-%d'),
                    'code': stock,
                    'weight': weight
                })
            print(f"  选中 {len(daily_picks)} 只股票: {daily_picks[:5]}{'...' if len(daily_picks) > 5 else ''}")
    
    # 输出统计
    print("\n" + "="*60)
    print("📋 选股统计")
    print("="*60)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        print(f"总交易日: {df_results['date'].nunique()}")
        print(f"总选股记录: {len(df_results)}")
        print(f"平均每日选股: {len(df_results) / df_results['date'].nunique():.1f} 只")
        
        # 保存预测文件
        output_file = 'data/jq_migration_full_predictions.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n已保存到: {output_file}")
    else:
        print("未生成任何选股结果")
    
    print("="*60)

if __name__ == "__main__":
    main()
