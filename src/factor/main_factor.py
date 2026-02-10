"""
因子检验主程序 - 命令行入口

提供画图、结果保存等功能
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
import numpy as np

from analyzer.core import FactorAnalyzer
from analyzer import export_analysis_report


def setup_output_dir(output_dir):
    """设置输出目录"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/factor_analysis_{timestamp}'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_single_factor_results(output_dir, factor_name, factor_results, quantiles=10):
    """保存单个因子的结果"""
    if not output_dir or not factor_results:
        return
    
    # 为该因子创建文件夹
    factor_dir = Path(output_dir) / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存该因子的各个周期数据
    for result in factor_results:
        period_dir = factor_dir / f"period_{result.period}"
        period_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存得分详情
        scores_df = pd.DataFrame([
            {'指标': k, '数值': v[0], '是否通过': '✓' if v[1] else '✗'}
            for k, v in result.scores.items()
        ])
        scores_df.to_csv(period_dir / 'scores.csv', index=False)
        
        # 保存 IC 序列
        if hasattr(result, 'ic_series') and result.ic_series is not None:
            ic_df = pd.DataFrame({
                'date': result.ic_series.index,
                'ic_value': result.ic_series.values
            })
            ic_df.to_csv(period_dir / 'ic_series.csv', index=False)
        
        # 保存收益序列
        if hasattr(result, 'ret_series') and result.ret_series is not None:
            ret_df = pd.DataFrame({
                'date': result.ret_series.index,
                'return_value': result.ret_series.values
            })
            ret_df.to_csv(period_dir / 'ret_series.csv', index=False)
        
        # 保存滚动监控数据
        if hasattr(result, 'rolling_monitor') and result.rolling_monitor is not None:
            monitor_data = {
                'roll_ic': getattr(result.rolling_monitor, 'roll_ic', None),
                'roll_ir': getattr(result.rolling_monitor, 'roll_ir', None),
                'roll_t': getattr(result.rolling_monitor, 'roll_t', None),
                'top_std': getattr(result.rolling_monitor, 'top_std', None),
                'neg_day': getattr(result.rolling_monitor, 'neg_day', None)
            }
            monitor_df = pd.DataFrame([monitor_data])
            monitor_df.to_csv(period_dir / 'rolling_monitor.csv', index=False)
        
        # 保存其他重要信息
        metadata = {
            'factor_name': result.factor_name,
            'period': result.period,
            'level': getattr(result, 'level', None),
            'status_flag': getattr(result, 'status_flag', None),
            'top_turnover': getattr(result, 'top_turnover', None),
            'analysis_timestamp': datetime.now().isoformat()
        }
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(period_dir / 'metadata.csv', index=False)
        
        # 保存 Alphalens 统计表（如果有 clean 数据）
        if hasattr(result, 'clean_data') and result.clean_data is not None:
            try:
                import alphalens as al
                from alphalens.performance import (
                    mean_return_by_quantile,
                    factor_returns,
                    factor_information_coefficient,
                    quantile_turnover,
                    factor_alpha_beta,
                    compute_mean_returns_spread,
                    factor_rank_autocorrelation
                )
                from alphalens.plotting import (
                    plot_quantile_statistics_table,
                    plot_information_table,
                    plot_returns_table,
                    plot_turnover_table
                )

                clean = result.clean_data
                
                # 1. Quantiles Statistics (分位数统计)
                try:
                    quantile_stats = plot_quantile_statistics_table(clean, return_df=True)
                    if isinstance(quantile_stats, pd.DataFrame) and not quantile_stats.empty:
                        quantile_stats.to_csv(period_dir / 'quantile_statistics.csv')
                        print(f"  ✓ 已保存分位数统计: {period_dir / 'quantile_statistics.csv'}")
                except Exception as e:
                    print(f"  ⚠️ 分位数统计计算失败: {e}")
                
                # 2. Returns Analysis (收益分析)
                try:
                    # 计算所需的输入数据
                    alpha_beta = factor_alpha_beta(clean)
                    mean_ret_quantile, _ = mean_return_by_quantile(clean, by_group=False)
                    
                    # 提取当前周期的收益数据
                    period_col = f"{result.period}D"
                    if period_col in mean_ret_quantile.columns:
                        mean_ret_by_period = mean_ret_quantile[period_col]
                        
                        # 计算分位数收益差价（最高分位数减最低分位数）
                        if len(mean_ret_by_period) >= 2:
                            max_quantile_ret = mean_ret_by_period.iloc[-1]  # 最高分位数
                            min_quantile_ret = mean_ret_by_period.iloc[0]   # 最低分位数
                            mean_ret_spread = max_quantile_ret - min_quantile_ret
                            
                            returns_table = plot_returns_table(alpha_beta, mean_ret_by_period, mean_ret_spread, return_df=True)
                            if isinstance(returns_table, pd.DataFrame) and not returns_table.empty:
                                returns_table.to_csv(period_dir / 'returns_analysis.csv')
                                print(f"  ✓ 已保存收益分析: {period_dir / 'returns_analysis.csv'}")
                        else:
                            print(f"    ⚠️ 分位数数据不足，无法计算收益差价")
                    else:
                        print(f"    ⚠️ 当前周期 {period_col} 在收益数据中不存在")
                except Exception as e:
                    print(f"  ⚠️ 收益分析计算失败: {e}")
                
                # 3. Information Analysis (信息分析)
                try:
                    ic = factor_information_coefficient(clean)
                    ic_table = plot_information_table(ic, return_df=True)
                    if isinstance(ic_table, pd.DataFrame) and not ic_table.empty:
                        ic_table.to_csv(period_dir / 'information_analysis.csv', index=False)
                        print(f"  ✓ 已保存信息分析: {period_dir / 'information_analysis.csv'}")
                except Exception as e:
                    print(f"  ⚠️ 信息分析计算失败: {e}")
                
                # 4. Turnover Analysis (换手率分析)
                try:
                    # 计算换手率数据 - 使用当前周期的整数值
                    period_int = result.period  # result.period 已经是整数
                    
                    # 计算各分位数的换手率
                    quantile_turnover_data = {}
                    for quantile in range(1, int(clean['factor_quantile'].max()) + 1):
                        try:
                            turnover_series = quantile_turnover(clean['factor_quantile'], quantile=quantile, period=period_int)
                            quantile_turnover_data[quantile] = turnover_series
                        except Exception as e:
                            print(f"    ⚠️ 计算分位数 {quantile} 换手率失败: {e}")
                            continue
                    
                    # 计算因子秩自相关
                    autocorrelation_data = {}
                    try:
                        autocorr_series = factor_rank_autocorrelation(clean, period=period_int)
                        autocorrelation_data[period_int] = autocorr_series
                    except Exception as e:
                        print(f"    ⚠️ 计算因子秩自相关失败: {e}")
                    
                    if quantile_turnover_data and autocorrelation_data:
                        # 重新组织数据格式：quantile_turnover_data 应该是 {period: {quantile: series}}
                        turnover_dict = {period_int: quantile_turnover_data}
                        turnover_table, auto_corr_table = plot_turnover_table(autocorrelation_data, turnover_dict, return_df=True)
                        
                        # 保存换手率表格
                        turnover_table.to_csv(period_dir / 'turnover_analysis.csv')
                        print(f"  ✓ 已保存换手率分析: {period_dir / 'turnover_analysis.csv'}")
                        
                        # 保存自相关表格
                        auto_corr_table.to_csv(period_dir / 'autocorrelation_analysis.csv')
                        print(f"  ✓ 已保存自相关分析: {period_dir / 'autocorrelation_analysis.csv'}")
                except Exception as e:
                    print(f"  ⚠️ 换手率分析计算失败: {e}")
                    
            except ImportError as ie:
                print(f"  ⚠️ 无法导入 alphalens: {ie}")
            except Exception as e:
                print(f"  ⚠️ 保存统计表时出错: {e}")
                import traceback
                traceback.print_exc()


def generate_overall_summary(output_dir, args):
    """生成总体汇总报告，读取所有因子的结果"""
    print("\n📊 生成总体汇总报告...")
    
    all_results = []
    
    # 遍历所有因子文件夹
    for item in Path(output_dir).iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            factor_name = item.name
            factor_dir = item
            
            # 读取该因子的各个周期结果
            for period_dir in factor_dir.glob("period_*"):
                if period_dir.is_dir():
                    period = int(period_dir.name.split("_")[1])
                    
                    # 读取 scores.csv
                    scores_file = period_dir / 'scores.csv'
                    if scores_file.exists():
                        try:
                            scores_df = pd.read_csv(scores_file)
                            
                            # 构建结果对象
                            result = type('FactorResult', (), {})()
                            result.factor_name = factor_name
                            result.period = period
                            result.scores = {}
                            
                            # 解析得分
                            for _, row in scores_df.iterrows():
                                metric = row['指标']
                                value = row['数值']
                                passed = row['是否通过'] == '✓'
                                result.scores[metric] = (value, passed)
                            
                            # 计算等级和状态
                            passed = sum(v[1] for v in result.scores.values())
                            result.level = '优秀' if passed >= 4 else ('良好' if passed >= 3 else '一般')
                            result.status_flag = "🔴 dead"  # 简化处理
                            
                            # 尝试读取其他数据
                            # 读取 metadata
                            metadata_file = period_dir / 'metadata.csv'
                            if metadata_file.exists():
                                try:
                                    metadata_df = pd.read_csv(metadata_file)
                                    if not metadata_df.empty:
                                        row = metadata_df.iloc[0]
                                        result.top_turnover = row.get('top_turnover')
                                        result.level = row.get('level', result.level)
                                        result.status_flag = row.get('status_flag', result.status_flag)
                                except Exception:
                                    pass
                            
                            # 读取滚动监控数据
                            monitor_file = period_dir / 'rolling_monitor.csv'
                            if monitor_file.exists():
                                try:
                                    monitor_df = pd.read_csv(monitor_file)
                                    if not monitor_df.empty:
                                        row = monitor_df.iloc[0]
                                        monitor = type('RollingMonitor', (), {})()
                                        monitor.roll_ic = row.get('roll_ic')
                                        monitor.roll_ir = row.get('roll_ir')
                                        monitor.roll_t = row.get('roll_t')
                                        monitor.top_std = row.get('top_std')
                                        monitor.neg_day = row.get('neg_day')
                                        result.rolling_monitor = monitor
                                except Exception:
                                    pass
                            
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"⚠️  读取 {factor_name} 周期 {period} 结果失败: {e}")
    
    if not all_results:
        print("❌ 没有找到任何因子结果")
        return
    
    # 生成汇总报告
    generate_summary_report(output_dir, all_results, args)
    print(f"✅ 总体汇总报告已生成: {Path(output_dir) / 'README.md'}")


def generate_summary_report(output_dir, all_results, args):
    """生成总体汇总报告"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存汇总 CSV
    summary_data = []
    for result in all_results:
        row = {
            '因子名称': result.factor_name,
            '周期': result.period,
            '等级': result.level,
            '状态': result.status_flag,
            'Top换手率': result.top_turnover,
        }
        # 添加各项得分
        for key, (value, passed) in result.scores.items():
            row[f'{key}_数值'] = value
            row[f'{key}_通过'] = passed
        
        summary_data.append(row)
    
    # 保存汇总 CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = Path(output_dir) / 'factor_analysis_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    
    # 生成 Markdown 报告
    md_path = Path(output_dir) / 'factor_analysis_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 因子分析总体汇总报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 测试配置
        f.write("## 测试配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 回测开始日期 | {args.start} |\n")
        f.write(f"| 回测结束日期 | {args.end} |\n")
        f.write(f"| 股票池 | {args.stock_pool} |\n")
        f.write(f"| 分位数 | {args.quantiles} |\n")
        f.write(f"| 调仓周期 | {', '.join(map(str, args.periods))} 天 |\n")
        f.write(f"| 因子数量 | {len(set(r.factor_name for r in all_results))} |\n")
        f.write(f"| 测试组合数 | {len(all_results)} |\n")
        f.write("\n")
        
        # 汇总表格
        f.write("## 结果汇总\n\n")
        f.write("| 因子 | 周期 | 等级 | 状态 | 通过指标数 |\n")
        f.write("|------|------|------|------|-----------|\n")
        
        for result in all_results:
            passed_count = sum(1 for _, (_, passed) in result.scores.items() if passed)
            f.write(f"| {result.factor_name} | {result.period}天 | {result.level} | {result.status_flag} | {passed_count}/{len(result.scores)} |\n")
        
        f.write("\n")
        
        # 因子表现统计
        f.write("## 因子表现统计\n\n")
        factor_stats = {}
        for result in all_results:
            factor_name = result.factor_name
            if factor_name not in factor_stats:
                factor_stats[factor_name] = {'total': 0, 'excellent': 0, 'good': 0, 'fair': 0, 'dead': 0}
            
            factor_stats[factor_name]['total'] += 1
            if result.level == '优秀':
                factor_stats[factor_name]['excellent'] += 1
            elif result.level == '良好':
                factor_stats[factor_name]['good'] += 1
            elif result.level == '一般':
                factor_stats[factor_name]['fair'] += 1
            else:
                factor_stats[factor_name]['dead'] += 1
        
        f.write("| 因子 | 总周期数 | 优秀 | 良好 | 一般 | 死亡 |\n")
        f.write("|------|---------|------|------|------|------|\n")
        
        for factor_name, stats in factor_stats.items():
            f.write(f"| {factor_name} | {stats['total']} | {stats['excellent']} | {stats['good']} | {stats['fair']} | {stats['dead']} |\n")
        
        f.write("\n")
        
        # 平均指标
        f.write("## 平均指标统计\n\n")
        metrics = ['IC均值', 'IR比率', '多空年化', '单调性', 'Top换手率']
        f.write("| 指标 | 平均值 | 标准差 | 最大值 | 最小值 |\n")
        f.write("|------|--------|--------|--------|--------|\n")
        
        for metric in metrics:
            values = [result.scores[metric][0] for result in all_results if metric in result.scores]
            if values:
                f.write(f"| {metric} | {np.mean(values):.4f} | {np.std(values):.4f} | {np.max(values):.4f} | {np.min(values):.4f} |\n")
        
        f.write("\n")
        
        # 文件说明
        f.write("## 文件说明\n\n")
        f.write("### 目录结构\n\n")
        f.write("```\n")
        f.write(f"{Path(output_dir).name}/\n")
        f.write("├── factor_analysis_summary.csv    # 汇总表格\n")
        f.write("├── factor_analysis_summary.md     # 本报告\n")
        for factor_name in sorted(set(r.factor_name for r in all_results)):
            f.write(f"├── {factor_name}/                   # {factor_name} 因子详细数据\n")
            f.write(f"│   ├── README.md                   # 因子独立报告\n")
            for period in sorted(set(r.period for r in all_results if r.factor_name == factor_name)):
                f.write(f"│   └── period_{period}/            # {period}天周期数据\n")
                f.write(f"│       ├── scores.csv              # 得分详情\n")
                f.write(f"│       ├── ic_series.csv           # IC序列\n")
                f.write(f"│       ├── ret_series.csv          # 收益序列\n")
                f.write(f"│       ├── rolling_monitor.csv     # 滚动监控\n")
                f.write(f"│       ├── metadata.csv            # 元数据\n")
                f.write(f"│       ├── quantile_statistics.csv # 分位数统计\n")
                f.write(f"│       ├── returns_analysis.csv    # 收益分析\n")
                f.write(f"│       ├── information_analysis.csv # 信息分析\n")
                f.write(f"│       └── turnover_analysis.csv   # 换手率分析\n")
        f.write("```\n\n")
        
        f.write("### 文件用途\n\n")
        f.write("- **factor_analysis_summary.csv**: 所有因子和周期的汇总数据\n")
        f.write("- **factor_analysis_summary.md**: 本汇总报告\n")
        f.write("- **因子文件夹/README.md**: 单个因子的详细报告\n")
        f.write("- **period_N/scores.csv**: 各周期的详细得分\n")
        f.write("- **period_N/ic_series.csv**: IC 时间序列数据\n")
        f.write("- **period_N/ret_series.csv**: 多空收益时间序列数据\n")
        f.write("- **period_N/rolling_monitor.csv**: 滚动监控指标数据\n")
        f.write("- **period_N/metadata.csv**: 分析元数据和配置信息\n")
        f.write("- **period_N/quantile_statistics.csv**: 分位数统计表（各分位数组合的平均收益）\n")
        f.write("- **period_N/returns_analysis.csv**: 收益分析表（多空收益的统计特征）\n")
        f.write("- **period_N/information_analysis.csv**: 信息分析表（IC相关统计）\n")
        f.write("- **period_N/turnover_analysis.csv**: 换手率分析表（各分位数组合的换手率）\n")
    
    print(f"\n📊 总体汇总报告已生成:")
    print(f"  CSV: {summary_csv_path}")
    print(f"  MD:  {md_path}")


def parse_main_args():
    """解析主程序命令行参数"""
    parser = argparse.ArgumentParser(
        description='因子检验主程序 - 支持画图和结果保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（不画图）
  python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10

  # 画图并弹窗显示
  python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --plot-mode popup

  # 画图并保存到文件夹
  python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --output-dir results/factor_test
        """
    )

    # 基础参数（传递给 factor.py）
    parser.add_argument('--start', type=str, default='2024-09-25',
                       help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-10-14',
                       help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stock-pool', type=str, default='small',
                       help='股票池：指数代码 或 "stock"（全市场）')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='最大股票数量限制（用于测试，减少计算量）')
    parser.add_argument('--factors', nargs='+', default=['VOL10', 'single_day_VPT_12'],
                       help='要检验的因子列表')
    parser.add_argument('--quantiles', type=int, default=10,
                       help='分组数量')
    parser.add_argument('--periods', nargs='+', type=int, default=[5, 10, 15],
                       help='调仓周期（天）')
    parser.add_argument('--roll-win', type=int, default=60,
                       help='滚动窗口交易日数')
    parser.add_argument('--monitor-csv', type=str, default='monitor.csv',
                       help='监控结果CSV文件路径')

    # 画图和输出参数
    parser.add_argument('--plot', type=str, default='true',
                       choices=['true', 'false'],
                       help='是否画图 (默认: true)')
    parser.add_argument('--plot-mode', type=str, default='save',
                       choices=['popup', 'save'],
                       help='画图模式: popup=弹窗显示, save=保存到文件 (默认: popup)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='结果输出目录（如果 --plot-mode=save，则保存图片和结果到此目录）')

    # 自定义因子参数
    parser.add_argument('--factor-file', type=str, default=None,
                       help='因子数据文件路径（CSV格式，包含date, asset, factor_value列）')
    parser.add_argument('--factor-column', type=str, default=None,
                       help='因子列名（默认为自动检测，通常是weight, factor_value等）')

    return parser.parse_args()


def setup_output_dir(output_dir):
    """设置输出目录"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/factor_analysis_{timestamp}'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir
def main():
    """主函数"""
    # 解析参数
    args = parse_main_args()
    
    # 打印配置信息
    print("=" * 60)
    print("因子分析器")
    print("=" * 60)
    print(f"开始日期: {args.start}")
    print(f"结束日期: {args.end}")
    print(f"股票池: {args.stock_pool}")
    print(f"分位数: {args.quantiles}")
    print(f"调仓周期: {args.periods}")
    print(f"画图: {args.plot}")
    print(f"画图模式: {args.plot_mode}")
    if args.factor_file:
        print(f"因子文件: {args.factor_file}")
        print(f"因子列名: {args.factor_column or '自动检测'}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    print()
    
    # 检查因子文件
    if not args.factor_file:
        print("❌ 请提供因子文件路径 (--factor-file)")
        return
    
    if not os.path.exists(args.factor_file):
        print(f"❌ 因子文件不存在: {args.factor_file}")
        return
    
    try:
        # 读取因子数据
        print(f"📂 读取因子数据: {args.factor_file}")
        factor_df = pd.read_csv(args.factor_file)
        print(f"✓ 成功读取因子数据: {len(factor_df)} 行")
        
        # 检查必要的列
        # 自动检测股票代码列
        possible_asset_cols = ['asset', 'code', 'symbol', 'stock_code', 'ticker']
        asset_column = None
        for col in possible_asset_cols:
            if col in factor_df.columns:
                asset_column = col
                break
        
        if asset_column is None:
            print(f"❌ 无法找到股票代码列")
            print(f"   可用列: {list(factor_df.columns)}")
            return
        
        print(f"✓ 检测到股票代码列: '{asset_column}'")
        
        # 重命名列为标准名称
        factor_df = factor_df.rename(columns={asset_column: 'asset'})
        
        # 自动检测因子列
        if args.factor_column:
            factor_columns = [args.factor_column]
        else:
            # 自动检测可能的因子列名
            possible_factor_cols = ['factor_value', 'weight', 'factor', 'value', 'score']
            factor_columns = []
            for col in possible_factor_cols:
                if col in factor_df.columns:
                    factor_columns.append(col)
            
            if not factor_columns:
                # 如果没找到常见列名，取除了date和asset外的所有数值列
                numeric_cols = factor_df.select_dtypes(include=[np.number]).columns
                factor_columns = [col for col in numeric_cols if col not in ['date', 'asset']]
        
        if not factor_columns:
            print(f"❌ 无法找到因子列")
            print(f"   可用列: {list(factor_df.columns)}")
            print(f"   尝试指定 --factor-column 参数")
            return
        
        # 只分析前5个因子进行测试
        factor_columns = factor_columns
        print(f"✓ 选择前5个因子进行测试: {factor_columns}")
        
        # 设置输出目录
        output_dir = setup_output_dir(args.output_dir)
        
        # 读取已分析的因子列表（通过检查因子文件夹是否存在）
        existing_factors = set()
        for item in Path(output_dir).iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                existing_factors.add(item.name)
        
        if existing_factors:
            print(f"✓ 发现已分析的因子: {len(existing_factors)} 个")
            print(f"  {sorted(existing_factors)}")
        
        print(f"  待分析因子数量: {len([f for f in factor_columns if f not in existing_factors])}")
        print(f"  已跳过因子数量: {len([f for f in factor_columns if f in existing_factors])}")
        print()
        
        # 循环分析每个因子
        all_results = []
        for factor_column in factor_columns:
            if factor_column in existing_factors:
                print(f"⏭️  跳过已分析的因子: {factor_column}")
                continue
                
            print(f"\n🔍 开始分析因子: {factor_column}")
            
            # 为每个因子创建副本并重命名
            factor_df_copy = factor_df.copy()
            factor_df_copy = factor_df_copy.rename(columns={factor_column: 'factor_value'})
            
            # 确保股票代码格式正确
            factor_df_copy['asset'] = factor_df_copy['asset'].astype(str).str.zfill(6)
            
            # 数据质量检查
            print(f"🔍 数据质量检查:")
            print(f"  原始数据行数: {len(factor_df_copy)}")
            print(f"  因子值NaN数量: {factor_df_copy['factor_value'].isna().sum()}")
            print(f"  因子值无穷大数量: {np.isinf(factor_df_copy['factor_value']).sum()}")
            print(f"  因子值有效数量: {factor_df_copy['factor_value'].notna().sum()}")
            print(f"  因子值范围: {factor_df_copy['factor_value'].min():.6f} ~ {factor_df_copy['factor_value'].max():.6f}")
            print(f"  唯一股票数量: {factor_df_copy['asset'].nunique()}")
            print(f"  唯一日期数量: {factor_df_copy['date'].nunique()}")
            
            # 检查是否有重复的 (date, asset) 组合
            duplicates = factor_df_copy.duplicated(['date', 'asset']).sum()
            if duplicates > 0:
                print(f"  ⚠️ 发现重复的 (date, asset) 组合: {duplicates} 个")
                # 保留第一个，去除重复
                factor_df_copy = factor_df_copy.drop_duplicates(['date', 'asset'])
                print(f"  ✓ 去除重复后数据行数: {len(factor_df_copy)}")
            
            # 去除NaN值
            nan_before = len(factor_df_copy)
            factor_df_copy = factor_df_copy.dropna(subset=['factor_value'])
            nan_after = len(factor_df_copy)
            if nan_before != nan_after:
                print(f"  ✓ 去除NaN值: {nan_before} -> {nan_after} 行")
            
            # 去除无穷大值
            inf_before = len(factor_df_copy)
            factor_df_copy = factor_df_copy[~np.isinf(factor_df_copy['factor_value'])]
            inf_after = len(factor_df_copy)
            if inf_before != inf_after:
                print(f"  ✓ 去除无穷大值: {inf_before} -> {inf_after} 行")
            
            if len(factor_df_copy) == 0:
                print(f"❌ 因子 {factor_column} 预处理后无有效数据，跳过")
                continue
            
            print(f"✓ 数据预处理完成")
            print(f"  股票数量: {factor_df_copy['asset'].nunique()}")
            print(f"  日期范围: {factor_df_copy['date'].min()} ~ {factor_df_copy['date'].max()}")
            print(f"  因子值范围: {factor_df_copy['factor_value'].min():.4f} ~ {factor_df_copy['factor_value'].max():.4f}")
            print()
            
            # 创建因子分析器
            # 使用因子数据中的日期范围
            factor_dates = pd.to_datetime(factor_df_copy['date'])
            start_date = factor_dates.min().strftime('%Y-%m-%d')
            end_date = factor_dates.max().strftime('%Y-%m-%d')
            
            print(f"✓ 使用因子数据日期范围: {start_date} ~ {end_date}")
            
            analyzer = FactorAnalyzer(
                factor_df=factor_df_copy,
                start_date=start_date,
                end_date=end_date,
                stock_pool=args.stock_pool,
                quantiles=args.quantiles,
                periods=args.periods,
                output_dir=output_dir
            )
            
            # 决定是否画图
            should_plot = args.plot == 'true'
            
            # 设置 matplotlib 后端（如果需要画图）
            if should_plot:
                import matplotlib
                import sys
                try:
                    if sys.platform == 'darwin':
                        matplotlib.use('macosx', force=True)
                    elif sys.platform.startswith('linux'):
                        matplotlib.use('TkAgg', force=True)
                    else:
                        matplotlib.use('TkAgg', force=True)
                    print("✓ 已设置画图后端")
                except Exception as e:
                    matplotlib.use('Agg', force=True)
                    print(f"⚠️  使用 Agg 后端: {e}")
            
            # 运行因子分析
            print("🔧 开始因子分析...")
            results = analyzer.analyze_factor(factor_name=factor_column, plot=should_plot)
            
            if not results:
                print(f"❌ 因子 {factor_column} 分析失败，无结果")
                continue
            
            print(f"✓ 因子 {factor_column} 分析完成，结果数量: {len(results)}")
            
            # 为结果添加因子名称
            for result in results:
                result.factor_name = factor_column
            
            all_results.extend(results)
            
            # 每分析完一个因子，立即落盘
            save_single_factor_results(output_dir, factor_column, results, args.quantiles)
            generate_single_factor_report(output_dir, factor_column, results, args)
            print(f"💾 因子 {factor_column} 结果已落盘")
        
        if not all_results:
            print("❌ 没有新的因子分析结果")
            return
        
        # 显示结果摘要（只显示新分析的）
        print("\n" + "=" * 60)
        print("新分析结果摘要")
        print("=" * 60)
        
        for result in all_results:
            print(f"\n因子: {result.factor_name}, 周期: {result.period}天")
            print(f"  等级: {result.level}")
            print(f"  状态: {result.status_flag}")
            print(f"  得分详情:")
            for key, (value, passed) in result.scores.items():
                status = "✅" if passed else "❌"
                print(f"    {key}: {value:.3f} {status}")
        
        print(f"\n📁 结果已保存到: {output_dir}")
        print(f"✅ 分析完成！本次分析了 {len(set(r.factor_name for r in all_results))} 个新因子")
        
        # 生成总体汇总报告
        generate_summary_report(output_dir, all_results, args)
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


def generate_single_factor_report(output_dir, factor_name, factor_results, args):
    """为单个因子生成独立报告"""
    from pathlib import Path
    
    factor_dir = Path(output_dir) / factor_name
    md_path = factor_dir / 'README.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {factor_name} 因子测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 测试配置
        f.write("## 测试配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 因子名称 | {factor_name} |\n")
        f.write(f"| 回测开始日期 | {args.start} |\n")
        f.write(f"| 回测结束日期 | {args.end} |\n")
        f.write(f"| 股票池 | {args.stock_pool} |\n")
        f.write(f"| 分位数 | {args.quantiles} |\n")
        f.write(f"| 调仓周期 | {', '.join(map(str, args.periods))} 天 |\n")
        f.write(f"| 测试周期数 | {len(factor_results)} |\n")
        f.write("\n")
        
        # 结果汇总
        f.write("## 结果汇总\n\n")
        f.write("| 周期 | 等级 | 状态 | 通过指标数 |\n")
        f.write("|------|------|------|-----------|\n")
        
        for result in factor_results:
            passed_count = sum(1 for _, (_, passed) in result.scores.items() if passed)
            f.write(f"| {result.period}天 | {result.level} | {result.status_flag} | {passed_count}/{len(result.scores)} |\n")
        
        f.write("\n")
        
        # 详细得分
        f.write("## 详细得分\n\n")
        for result in factor_results:
            f.write(f"### {result.period}天周期\n\n")
            f.write("| 指标 | 数值 | 状态 |\n")
            f.write("|------|------|------|\n")
            for key, (value, passed) in result.scores.items():
                status = "✓ 通过" if passed else "✗ 未通过"
                if isinstance(value, float):
                    f.write(f"| {key} | {value:.4f} | {status} |\n")
                else:
                    f.write(f"| {key} | {value} | {status} |\n")
            f.write("\n")
        
        # 文件说明
        f.write("## 文件说明\n\n")
        f.write("该因子目录包含以下文件：\n\n")
        f.write("- `period_{N}/scores.csv` - 得分详情\n")
        f.write("- `period_{N}/ic_series.csv` - IC 序列数据\n")
        f.write("- `period_{N}/ret_series.csv` - 收益序列数据\n")
        f.write("- `period_{N}/rolling_monitor.csv` - 滚动监控指标\n")
        f.write("- `period_{N}/metadata.csv` - 元数据信息\n")
        f.write("- `period_{N}/quantile_statistics.csv` - 分位数统计表\n")
        f.write("- `period_{N}/returns_analysis.csv` - 收益分析表\n")
        f.write("- `period_{N}/information_analysis.csv` - 信息分析表\n")
        f.write("- `period_{N}/turnover_analysis.csv` - 换手率分析表\n")
        f.write("- `README.md` - 本报告\n")
    
    print(f"  ✓ 已生成: {md_path}")


if __name__ == '__main__':
    main()