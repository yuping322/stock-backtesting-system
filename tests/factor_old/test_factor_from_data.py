"""
从 data.py 读取因子并运行测试

根据 factor/README.md 的要求：
- 从 data.py 读取因子
- 使用最近3个月的数据
- 不画图
- 保存结果
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入因子模块（使用绝对路径导入）
import importlib.util

# 导入 factor.py
factor_path = os.path.join(project_root, 'src', 'factor', 'factor.py')
spec = importlib.util.spec_from_file_location("factor_module", factor_path)
factor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor_module)

# 导入 factor_calculator.py
factor_calculator_path = os.path.join(project_root, 'src', 'factor', 'factor_calculator.py')
spec2 = importlib.util.spec_from_file_location("factor_calculator_module", factor_calculator_path)
factor_calculator_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(factor_calculator_module)

# 导入数据模块
import data

CFG = factor_module.CFG
FactorTester = factor_module.FactorTester
parse_args = factor_module.parse_args

# --- Added pytest fixture for factor_names to satisfy test invocation ---
import pytest

@pytest.fixture
def factor_names():
    """Provide a minimal default factor list for test.
    Chosen basic factors expected to exist; adjust if repository factors differ.
    """
    return ['VOL10', 'MA_5']


def get_recent_3_months_date_range():
    """获取最近3个月的日期范围"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 最近3个月
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def test_factors_from_data(factor_names, output_dir=None, start_date=None, end_date=None):
    """
    从 data.py 读取因子并运行测试
    
    Args:
        factor_names: 因子名称列表
        output_dir: 输出目录
        start_date: 开始日期
        end_date: 结束日期
    """
    # 1. 获取日期范围
    if start_date is None or end_date is None:
        start_date, end_date = get_recent_3_months_date_range()
    print(f"使用日期范围: {start_date} ~ {end_date}")
    
    # 2. 获取股票池（使用默认指数）
    stock_pool = '000510.XSHG'
    print(f"股票池: {stock_pool}")
    
    # 3. 创建命令行参数
    import argparse
    args = argparse.Namespace(
        start=start_date,
        end=end_date,
        stock_pool=stock_pool,
        factors=factor_names,
        quantiles=10,
        periods=[5, 10, 15],
        fillna=0,
        winsorize=0,
        neutralize=0,
        standardize=0,
        roll_win=60,
        monitor_csv='monitor.csv',
        last_only=False
    )
    
    # 4. 创建配置
    cfg = CFG(args)
    
    # 5. 设置输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'tests/results/factor_test_{timestamp}'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 6. 创建自定义因子计算器（从 data.py 读取）
    custom_factors = {}
    for factor_name in factor_names:
        print(f"\n正在准备因子: {factor_name}")
        
        # 使用 factor_for_al 函数获取因子数据
        try:
            # 先获取股票列表
            import data
            stocks = data.get_index_stocks(stock_pool.split('.')[0])
            print(f"  股票数量: {len(stocks)}")
            
            # 尝试从文件加载（如果存在）
            if os.path.exists(f'data/{factor_name}.csv'):
                print(f"  从文件加载: data/{factor_name}.csv")
                custom_factors[factor_name] = factor_calculator_module.create_factor_calculator(
                    file_path=f'data/{factor_name}.csv',
                    factor_name=factor_name
                )
            else:
                print(f"  使用内置因子计算器")
                # 使用内置因子计算器
                if factor_name in ['VOL10', 'VOL20', 'RSI_14', 'MA_5', 'MA_10', 'MA_20']:
                    custom_factors[factor_name] = factor_calculator_module.create_factor_calculator(
                        factor_name=factor_name
                    )
                else:
                    print(f"  ⚠️  因子 {factor_name} 不可用，跳过")
                    
        except Exception as e:
            print(f"  ❌ 准备因子 {factor_name} 失败: {e}")
    
    # 7. 创建因子测试器
    tester = FactorTester(cfg, custom_factors=custom_factors)
    
    # 8. 运行测试（不画图）
    print("\n开始运行因子测试...")
    test_results = tester.run(plot=False)
    
    # 9. 保存结果
    print("\n保存结果...")
    save_results(output_dir, test_results)
    
    print(f"\n✅ 测试完成！结果保存在: {output_dir}")
    return test_results


def save_results(output_dir, test_results):
    """保存测试结果"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 如果没有结果，保存空报告
    if not test_results:
        print("  ⚠️  没有测试结果，创建空报告")
        with open(Path(output_dir) / 'README.md', 'w', encoding='utf-8') as f:
            f.write("# 因子测试报告\n\n")
            f.write("## 测试结果\n\n")
            f.write("**没有生成任何测试结果**\n\n")
            f.write("可能的原因：\n")
            f.write("- 因子计算失败\n")
            f.write("- 数据加载失败\n")
            f.write("- 日期范围内无数据\n")
        return
    
    # 保存汇总结果
    summary_data = []
    for result in test_results:
        row = {
            'factor_name': result.factor_name,
            'period': result.period,
            'level': result.level,
            'status_flag': result.status_flag,
            'top_turnover': result.top_turnover,
        }
        # 添加各项得分
        for key, (value, passed) in result.scores.items():
            row[f'{key}_value'] = value
            row[f'{key}_passed'] = passed
        
        # 添加滚动监控数据
        if result.rolling_monitor is not None:
            row['roll_ic'] = result.rolling_monitor.roll_ic
            row['roll_ir'] = result.rolling_monitor.roll_ir
            row['roll_t'] = result.rolling_monitor.roll_t
            row['top_std'] = result.rolling_monitor.top_std
            row['neg_day'] = result.rolling_monitor.neg_day
        
        summary_data.append(row)
    
    # 保存汇总 CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Path(output_dir) / 'summary.csv', index=False)
    print(f"  ✓ 汇总结果 CSV: {Path(output_dir) / 'summary.csv'}")
    
    # 保存每个因子的详细数据（按因子分组）
    factors_dict = {}
    for result in test_results:
        if result.factor_name not in factors_dict:
            factors_dict[result.factor_name] = []
        factors_dict[result.factor_name].append(result)
    
    for factor_name, results in factors_dict.items():
        factor_dir = Path(output_dir) / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存该因子的所有周期数据
        for result in results:
            period_dir = factor_dir / f"period_{result.period}"
            period_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 IC 序列
            if result.ic_series is not None and len(result.ic_series) > 0:
                ic_df = pd.DataFrame({'IC': result.ic_series})
                ic_df.to_csv(period_dir / 'ic_series.csv')
            
            # 保存收益序列
            if result.ret_series is not None and len(result.ret_series) > 0:
                ret_df = pd.DataFrame({'return': result.ret_series})
                ret_df.to_csv(period_dir / 'ret_series.csv')
            
            # 保存得分详情
            scores_df = pd.DataFrame([
                {'指标': k, '数值': v[0], '是否通过': '✓' if v[1] else '✗'}
                for k, v in result.scores.items()
            ])
            scores_df.to_csv(period_dir / 'scores.csv', index=False)
        
        print(f"  ✓ 因子 {factor_name}: {factor_dir}")
    
    # 生成 MD 汇总报告
    generate_md_report(output_dir, test_results, summary_df)
    
    print(f"\n✅ 所有结果已保存到: {output_dir}")


def generate_md_report(output_dir, test_results, summary_df):
    """生成 Markdown 格式的汇总报告"""
    md_path = Path(output_dir) / 'README.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 因子测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 测试配置
        f.write("## 测试配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 因子数量 | {len(set(r.factor_name for r in test_results))} |\n")
        f.write(f"| 测试组合数 | {len(test_results)} |\n")
        f.write(f"| 调仓周期 | {', '.join(map(str, sorted(set(r.period for r in test_results))))} 天 |\n")
        f.write("\n")
        
        # 汇总表格
        f.write("## 结果汇总\n\n")
        f.write("| 因子 | 周期 | 等级 | 状态 | 通过指标数 |\n")
        f.write("|------|------|------|------|-----------|\n")
        
        for result in test_results:
            passed_count = sum(1 for _, (_, passed) in result.scores.items() if passed)
            f.write(f"| {result.factor_name} | {result.period}天 | {result.level} | {result.status_flag} | {passed_count}/{len(result.scores)} |\n")
        
        f.write("\n")
        
        # 详细得分
        f.write("## 详细得分\n\n")
        for result in test_results:
            f.write(f"### {result.factor_name} - {result.period}天周期\n\n")
            f.write("| 指标 | 数值 | 状态 |\n")
            f.write("|------|------|------|\n")
            for key, (value, passed) in result.scores.items():
                status = "✓ 通过" if passed else "✗ 未通过"
                if isinstance(value, float):
                    f.write(f"| {key} | {value:.4f} | {status} |\n")
                else:
                    f.write(f"| {key} | {value} | {status} |\n")
            f.write("\n")
            
            # 滚动监控指标
            if result.rolling_monitor is not None:
                f.write("**滚动监控指标**:\n\n")
                f.write(f"- 滚动 IC: {result.rolling_monitor.roll_ic:.4f}\n")
                f.write(f"- 滚动 IR: {result.rolling_monitor.roll_ir:.4f}\n")
                f.write(f"- 滚动 t: {result.rolling_monitor.roll_t:.2f}\n")
                f.write(f"- Top 波动率: {result.rolling_monitor.top_std:.2%}\n")
                f.write(f"- 负收益日占比: {result.rolling_monitor.neg_day:.2%}\n")
                f.write("\n")
        
        # 文件说明
        f.write("## 文件说明\n\n")
        f.write("### 目录结构\n\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── README.md              # 本报告\n")
        f.write("├── summary.csv            # 汇总表格\n")
        for factor_name in sorted(set(r.factor_name for r in test_results)):
            f.write(f"├── {factor_name}/           # {factor_name} 因子详细数据\n")
            for period in sorted(set(r.period for r in test_results if r.factor_name == factor_name)):
                f.write(f"│   └── period_{period}/    # {period}天周期数据\n")
                f.write(f"│       ├── ic_series.csv  # IC序列\n")
                f.write(f"│       ├── ret_series.csv # 收益序列\n")
                f.write(f"│       └── scores.csv     # 得分详情\n")
        f.write("```\n\n")
        
        f.write("### 文件用途\n\n")
        f.write("- **summary.csv**: 所有因子和周期的汇总数据\n")
        f.write("- **ic_series.csv**: IC 序列，用于分析因子与收益的相关性\n")
        f.write("- **ret_series.csv**: Top-Bottom 收益序列，用于分析因子收益\n")
        f.write("- **scores.csv**: 各项指标得分，用于评估因子质量\n")
    
    print(f"  ✓ Markdown 报告: {md_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从 data.py 读取因子并运行测试')
    parser.add_argument('--factors', nargs='+', default=['VOL10', 'single_day_VPT_12'],
                       help='要测试的因子列表')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认: tests/results/factor_test_TIMESTAMP）')
    parser.add_argument('--start', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD)，默认: 最近3个月')
    parser.add_argument('--end', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD)，默认: 今天')
    
    args = parser.parse_args()
    
    # 获取日期范围
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = get_recent_3_months_date_range()
    
    print("=" * 60)
    print("从 data.py 读取因子并运行测试")
    print("=" * 60)
    print(f"因子列表: {args.factors}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print("画图: 否")
    print("=" * 60)
    print()
    
    # 运行测试
    test_results = test_factors_from_data(args.factors, args.output_dir, start_date, end_date)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)
    for result in test_results:
        print(f"\n因子: {result.factor_name}, 周期: {result.period}天")
        print(f"  等级: {result.level}")
        print(f"  状态: {result.status_flag}")
        print(f"  得分:")
        for key, (value, passed) in result.scores.items():
            status = "✅" if passed else "❌"
            print(f"    {key}: {value:.3f} {status}")


if __name__ == '__main__':
    main()

