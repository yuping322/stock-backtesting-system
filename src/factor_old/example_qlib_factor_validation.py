#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib因子生成与验证完整示例

流程：
1. 生成最近3个月的Alpha158因子文件
2. 使用生成的因子文件进行Alphalens验证

运行：
    python factor/example_qlib_factor_validation.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

import data
from factor_calculator import create_factor_calculator
from factor import FactorTester, CFG, parse_args


def generate_factors_last_3_months(output_dir='./factors', stock_pool='small'):
    """
    生成最近3个月的因子文件
    
    Args:
        output_dir: 输出目录
        stock_pool: 股票池
    """
    print("=" * 80)
    print("步骤1: 生成最近3个月的Alpha158因子文件")
    print("=" * 80)
    
    # 计算日期范围（最近3个月）
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # 约3个月
    
    print(f"日期范围: {start_date} 到 {end_date}")
    
    # 获取股票池代码（使用默认股票，确保流程可运行）
    # 注意：如果data.get_index_stocks有问题，会使用默认股票列表
    default_stocks = ['000001', '000002', '600000', '600519', '000858', '600036', '000063', '600048']
    stocks = []
    
    try:
        stocks_result = data.get_index_stocks(stock_pool, date=end_date)
        if isinstance(stocks_result, pd.Series):
            stocks = stocks_result.tolist()
        elif isinstance(stocks_result, list):
            stocks = stocks_result
        elif stocks_result is not None:
            stocks = list(stocks_result)
        
        # 过滤空值
        stocks = [s for s in stocks if s and str(s).strip()]
        
        if not stocks:
            print(f"获取到的股票列表为空，使用默认股票")
            stocks = default_stocks
        else:
            # 限制股票数量（用于快速测试）
            if len(stocks) > 20:
                stocks = stocks[:20]
                print(f"使用前20只股票进行测试: {stocks[:5]}...")
            else:
                print(f"使用全部 {len(stocks)} 只股票: {stocks[:5]}...")
    except Exception as e:
        print(f"获取股票池失败: {e}，使用默认股票")
        stocks = default_stocks
    
    if not stocks:
        raise ValueError("无法获取股票列表，无法继续")
    
    print(f"最终使用股票列表: {stocks}")
    
    # 生成因子文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导入生成因子脚本的函数
    from generate_qlib_factors import build_qlib_dataset, extract_factors_from_qlib, save_factors_to_file
    
    # 构建qlib数据集
    qlib_cache_dir = output_dir / 'qlib_cache'
    print(f"\n[1.1] 构建qlib数据集...")
    qlib_data_dir = build_qlib_dataset(
        codes=stocks,
        start_date=str(start_date),
        end_date=str(end_date),
        output_dir=qlib_cache_dir,
        rebuild=False
    )
    
    # 提取因子
    print(f"\n[1.2] 提取Alpha158因子...")
    factors_df = extract_factors_from_qlib(
        qlib_data_dir=qlib_data_dir,
        factor_set='Alpha158',
        codes=stocks,
        start_date=str(start_date),
        end_date=str(end_date)
    )
    
    print(f"[1.2] 提取完成: {len(factors_df.columns)} 个因子")
    print(f"[1.2] 因子示例: {list(factors_df.columns[:10])}")
    
    # 保存因子文件
    print(f"\n[1.3] 保存因子文件...")
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    output_file = output_dir / f'Alpha158_{start_str}_{end_str}.csv'
    
    save_factors_to_file(factors_df, output_file)
    
    print(f"\n[✓] 因子文件生成完成: {output_file}")
    print(f"[✓] 包含 {len(factors_df.columns)} 个因子")
    
    return str(output_file), factors_df.columns.tolist()


def validate_factors_from_file(factor_file, factor_names, output_dir='./factors'):
    """
    使用生成的因子文件进行验证
    
    Args:
        factor_file: 因子文件路径
        factor_names: 因子名称列表
        output_dir: 因子文件目录
    """
    print("\n" + "=" * 80)
    print("步骤2: 使用因子文件进行Alphalens验证")
    print("=" * 80)
    
    # 选择几个常见的因子进行验证
    test_factors = ['ROC5', 'ROC10', 'MA5', 'MA10', 'STD5']
    # 过滤出实际存在的因子
    available_factors = [f for f in test_factors if f in factor_names]
    
    if not available_factors:
        # 如果常用因子都不存在，使用前5个因子
        available_factors = factor_names[:5]
        print(f"常用因子不存在，使用前5个因子: {available_factors}")
    else:
        print(f"验证以下因子: {available_factors}")
    
    # 创建因子计算器字典
    custom_factors = {}
    for factor_name in available_factors:
        try:
            calc = create_factor_calculator(
                factor_name=factor_name,
                factor_dir=str(output_dir)
            )
            custom_factors[factor_name] = calc
            print(f"  ✓ {factor_name}: 已创建因子计算器")
        except Exception as e:
            print(f"  ✗ {factor_name}: 创建失败 - {e}")
    
    if not custom_factors:
        print("错误: 没有成功创建任何因子计算器")
        return
    
    # 计算日期范围（从因子文件）
    try:
        # 读取日期列来判断日期范围
        dates_df = pd.read_csv(factor_file, usecols=['date'])
        dates = pd.to_datetime(dates_df['date'])
        start_date = dates.min().strftime('%Y-%m-%d')
        end_date = dates.max().strftime('%Y-%m-%d')
    except Exception as e:
        print(f"读取日期范围失败: {e}，使用默认值")
        # 默认最近3个月
        end_date_obj = datetime.now().date()
        start_date_obj = end_date_obj - timedelta(days=90)
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')
    
    print(f"\n验证日期范围: {start_date} 到 {end_date}")
    
    # 创建配置对象（模拟命令行参数）
    class Args:
        def __init__(self):
            self.start = start_date
            self.end = end_date
            self.stock_pool = 'HS300'  # 可以从因子文件推断，这里简化
            self.factors = list(custom_factors.keys())
            self.quantiles = 5  # 使用5分位数加快速度
            self.periods = [5, 10]  # 只测试2个周期
            self.fillna = 0
            self.winsorize = 0
            self.neutralize = 0
            self.standardize = 0
            self.roll_win = 30
            self.monitor_csv = 'monitor.csv'
            self.last_only = False
    
    args = Args()
    cfg = CFG(args)
    
    # 创建因子测试器
    print(f"\n创建因子测试器，验证 {len(custom_factors)} 个因子...")
    tester = FactorTester(cfg, custom_factors=custom_factors)
    
    # 获取股票列表
    print("\n获取股票列表...")
    tester.get_stocks()
    
    if not tester.stocks:
        print("警告: 未获取到股票列表，使用默认股票")
        tester.stocks = ['000001', '000002', '600000', '600519', '000858']
    
    # 限制股票数量（加快速度）
    if len(tester.stocks) > 20:
        tester.stocks = tester.stocks[:20]
        print(f"限制使用前20只股票进行验证")
    
    # 运行验证
    print(f"\n开始运行因子验证...")
    try:
        results = tester.run(plot=False)  # plot=False表示不绘图
        print(f"\n验证完成，共验证 {len(results)} 个因子的 {len(cfg.PERIODS)} 个调仓周期")
    except Exception as e:
        print(f"\n验证过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Qlib因子生成与验证完整流程")
    print("=" * 80)
    print()
    
    # 因子文件目录
    factor_dir = Path('./factors')
    factor_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 步骤1: 生成因子文件
        factor_file, factor_names = generate_factors_last_3_months(
            output_dir=str(factor_dir),
            stock_pool='HS300'
        )
        
        print(f"\n共生成 {len(factor_names)} 个因子")
        
        # 步骤2: 验证因子
        validate_factors_from_file(
            factor_file=factor_file,
            factor_names=factor_names,
            output_dir=str(factor_dir)
        )
        
        print("\n" + "=" * 80)
        print("✓ 完整流程执行成功！")
        print("=" * 80)
        print(f"\n因子文件位置: {factor_file}")
        print(f"因子数量: {len(factor_names)}")
        print(f"\n可以单独使用以下命令验证因子:")
        print(f"  python factor/factor.py --factors ROC5 MA10 --factor-dir {factor_dir}")
        
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

