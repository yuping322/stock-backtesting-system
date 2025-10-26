"""
因子检验主程序 - 命令行入口

提供画图、结果保存等功能
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
# matplotlib.pyplot 延迟导入，在 main() 中根据模式设置后端

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入因子模块（使用绝对路径）
factor_dir = os.path.join(project_root, 'factor')
sys.path.insert(0, factor_dir)

# 直接导入文件
import importlib.util
spec = importlib.util.spec_from_file_location("factor_module", os.path.join(factor_dir, "factor.py"))
factor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor_module)

spec2 = importlib.util.spec_from_file_location("factor_calculator_module", os.path.join(factor_dir, "factor_calculator.py"))
factor_calculator_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(factor_calculator_module)

CFG = factor_module.CFG
FactorTester = factor_module.FactorTester
FactorTestResult = factor_module.FactorTestResult
quick_score = factor_module.quick_score
rolling_monitor = factor_module.rolling_monitor
create_factor_calculator = factor_calculator_module.create_factor_calculator


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
    parser.add_argument('--plot', type=str, default='false',
                       choices=['true', 'false'],
                       help='是否画图 (默认: false)')
    parser.add_argument('--plot-mode', type=str, default='popup',
                       choices=['popup', 'save'],
                       help='画图模式: popup=弹窗显示, save=保存到文件 (默认: popup)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='结果输出目录（如果 --plot-mode=save，则保存图片和结果到此目录）')
    
    # 自定义因子参数
    parser.add_argument('--custom-factor-file', type=str, default=None,
                       help='自定义因子文件路径')
    parser.add_argument('--custom-factor-name', type=str, default=None,
                       help='自定义因子列名（需要与 --custom-factor-file 一起使用）')
    
    return parser.parse_args()


def setup_output_dir(output_dir):
    """设置输出目录"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'results/factor_test_{timestamp}'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def save_single_factor_results(output_dir, factor_name, factor_results):
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


def save_factor_plots(output_dir, factor_name):
    """保存单个因子的图表"""
    import matplotlib.pyplot as plt
    
    if not output_dir:
        return
    
    factor_dir = Path(output_dir) / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存所有打开的图表
    plot_count = 0
    for i, fig in enumerate(plt.get_fignums()):
        plt.figure(fig)
        save_path = factor_dir / f'plot_{i+1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plot_count += 1
        print(f"  ✓ 已保存图片: {save_path}")
    
    if plot_count > 0:
        print(f"✓ 因子 {factor_name} 已保存 {plot_count} 张图片")
    
    plt.close('all')


def generate_single_factor_report(output_dir, factor_name, factor_results, cfg, args):
    """为单个因子生成独立报告"""
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
        f.write(f"| 回测开始日期 | {cfg.START} |\n")
        f.write(f"| 回测结束日期 | {cfg.END} |\n")
        f.write(f"| 股票池 | {cfg.STOCK_POOL} |\n")
        f.write(f"| 分位数 | {cfg.QUANTILES} |\n")
        f.write(f"| 调仓周期 | {', '.join(map(str, cfg.PERIODS))} 天 |\n")
        f.write(f"| 滚动窗口 | {cfg.ROLL_WIN} 天 |\n")
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
        f.write("该因子目录包含以下文件：\n\n")
        f.write("- `period_{N}/ic_series.csv` - IC 序列数据\n")
        f.write("- `period_{N}/ret_series.csv` - 收益序列数据\n")
        f.write("- `period_{N}/scores.csv` - 得分详情\n")
        f.write("- `plot_{N}.png` - Alphalens 可视化图表（如果启用了 --plot save）\n")
        f.write("- `README.md` - 本报告\n")
    
    print(f"  ✓ 已生成: {md_path}")


def generate_summary_report(output_dir, all_results, cfg, args):
    """生成汇总报告"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存汇总 CSV
    summary_data = []
    for result in all_results:
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
    
    # 生成 Markdown 报告
    md_path = Path(output_dir) / 'README.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 因子测试报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 测试配置
        f.write("## 测试配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 回测开始日期 | {cfg.START} |\n")
        f.write(f"| 回测结束日期 | {cfg.END} |\n")
        f.write(f"| 股票池 | {cfg.STOCK_POOL} |\n")
        f.write(f"| 因子列表 | {', '.join(cfg.FACTORS)} |\n")
        f.write(f"| 分位数 | {cfg.QUANTILES} |\n")
        f.write(f"| 调仓周期 | {', '.join(map(str, cfg.PERIODS))} 天 |\n")
        f.write(f"| 滚动窗口 | {cfg.ROLL_WIN} 天 |\n")
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
        
        # 详细得分
        f.write("## 详细得分\n\n")
        for result in all_results:
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
        f.write(f"{Path(output_dir).name}/\n")
        f.write("├── README.md              # 本报告\n")
        f.write("├── summary.csv            # 汇总表格\n")
        for factor_name in sorted(set(r.factor_name for r in all_results)):
            f.write(f"├── {factor_name}/           # {factor_name} 因子详细数据\n")
            f.write(f"│   ├── README.md           # 因子独立报告\n")
            f.write(f"│   ├── plot_*.png          # 可视化图表（如果启用）\n")
            for period in sorted(set(r.period for r in all_results if r.factor_name == factor_name)):
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
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"汇总文件: {Path(output_dir) / 'summary.csv'}")
    print(f"报告文件: {md_path}")


def main():
    """主函数"""
    # 解析参数
    args = parse_main_args()
    
    # 创建简化的 args 对象用于 CFG
    class CFGArgs:
        start = args.start
        end = args.end
        stock_pool = args.stock_pool
        factors = args.factors
        quantiles = args.quantiles
        periods = args.periods
        fillna = 0
        winsorize = 0
        neutralize = 0
        standardize = 0
        roll_win = args.roll_win
        monitor_csv = args.monitor_csv
        last_only = False
    
    cfg_args = CFGArgs()
    
    # 创建因子配置
    cfg = CFG(cfg_args)
    
    # 打印配置信息
    print("=" * 60)
    print("因子检验配置")
    print("=" * 60)
    print(f"回测区间: {cfg.START} ~ {cfg.END}")
    print(f"股票池: {cfg.STOCK_POOL}")
    print(f"因子列表: {cfg.FACTORS}")
    print(f"分位数: {cfg.QUANTILES}")
    print(f"调仓周期: {cfg.PERIODS}")
    print(f"滚动窗口: {args.roll_win} 天")
    print(f"画图: {args.plot}")
    print(f"画图模式: {args.plot_mode}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    print()
    
    # 设置输出目录（总是创建输出目录）
    output_dir = setup_output_dir(args.output_dir)
    print(f"结果将保存到: {output_dir}")
    print()
    
    # 创建自定义因子计算器（如果有）
    custom_factors = {}
    if args.custom_factor_file and args.custom_factor_name:
        print(f"加载自定义因子: {args.custom_factor_name} from {args.custom_factor_file}")
        custom_factors[args.custom_factor_name] = create_factor_calculator(
            file_path=args.custom_factor_file,
            factor_name=args.custom_factor_name
        )
    
    # 创建因子测试器
    tester = FactorTester(cfg, custom_factors=custom_factors)
    
    # 决定是否画图
    should_plot = args.plot == 'true'
    
    # 如果是保存模式，设置 matplotlib 为非交互后端
    if should_plot and args.plot_mode == 'save':
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端，不会弹窗
        print("✓ 已设置画图为非交互模式（保存到文件）")
        print()
    
    # 逐因子处理，避免内存问题
    all_results = []
    failed_factors = []
    
    for idx, factor_name in enumerate(cfg.FACTORS):
        print(f"\n{'='*60}")
        print(f"处理因子 {idx+1}/{len(cfg.FACTORS)}: {factor_name}")
        print(f"{'='*60}")
        
        try:
            # 运行单个因子测试
            factor_results = tester.run_single_factor(factor_name, plot=should_plot)
            
            if factor_results:
                all_results.extend(factor_results)
            
            # 立即保存该因子的结果
            save_single_factor_results(output_dir, factor_name, factor_results)
            
            # 立即生成该因子的独立报告
            if factor_results:
                generate_single_factor_report(output_dir, factor_name, factor_results, cfg, args)
                print(f"✓ 因子 {factor_name} 报告已生成")
            
            # 保存该因子的图片（如果启用了画图）
            if should_plot and args.plot_mode == 'save':
                save_factor_plots(output_dir, factor_name)
            
            # 清理内存（删除因子数据）
            del factor_results
            
        except Exception as e:
            print(f"\n❌ 因子 {factor_name} 处理失败: {e}")
            print(f"   继续处理下一个因子...")
            failed_factors.append(factor_name)
            # 尝试保存失败标记
            try:
                save_single_factor_results(output_dir, factor_name, None)
            except:
                pass
    
    # 生成最终汇总报告（包含所有因子）
    if all_results:
        generate_summary_report(output_dir, all_results, cfg, args)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)
    print(f"总因子数: {len(cfg.FACTORS)}")
    print(f"成功: {len(set(r.factor_name for r in all_results))}")
    print(f"失败: {len(failed_factors)}")
    if failed_factors:
        print(f"\n失败的因子:")
        for f in failed_factors:
            print(f"  ❌ {f}")
    print("=" * 60)
    
    for result in all_results:
        print(f"\n因子: {result.factor_name}, 周期: {result.period}天")
        print(f"  等级: {result.level}")
        print(f"  状态: {result.status_flag}")
        print(f"  得分详情:")
        for key, (value, passed) in result.scores.items():
            status = "✅" if passed else "❌"
            print(f"    {key}: {value:.3f} {status}")
    
    # 处理剩余图表（popup 模式）
    if should_plot and args.plot_mode == 'popup':
        import matplotlib.pyplot as plt
        print("\n显示图表（关闭窗口继续）...")
        plt.show()
    
    print("\n全部完成！")


if __name__ == '__main__':
    main()

