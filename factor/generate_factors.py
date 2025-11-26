"""
因子数据生成脚本

生成因子数据并保存到文件中，不进行因子检验。

支持：
1. 内置因子（VOL10, RSI_14等）
2. TALIB因子（TALIB_RSI_14等）
3. 从文件加载因子
4. 批量处理多个因子
5. 保存为标准CSV格式（date, code, factor_value）
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
sys.path.insert(0, project_root)

# 导入因子模块
factor_dir = os.path.join(project_root, 'factor')
sys.path.insert(0, factor_dir)

import importlib.util
spec = importlib.util.spec_from_file_location("factor_calculator_module", os.path.join(factor_dir, "factor_calculator.py"))
factor_calculator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor_calculator_module)

create_factor_calculator = factor_calculator_module.create_factor_calculator

# 导入数据模块
import data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='因子数据生成脚本 - 生成因子数据并保存到文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成单个因子
  python generate_factors.py --factors VOL10 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

  # 生成多个因子
  python generate_factors.py --factors VOL10 RSI_14 TALIB_RSI_14 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

  # 从文件加载因子
  python generate_factors.py --factor-file data/factor_values_sample.csv --factor-name my_factor --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

  # 指定股票池
  python generate_factors.py --factors VOL10 --stock-pool small --max-stocks 100 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output
        """
    )

    # 基础参数
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stock-pool', type=str, default='small',
                       help='股票池：指数代码 或 "stock"（全市场）或 "small"（小盘股）')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='最大股票数量限制（用于测试，减少计算量）')

    # 因子参数
    parser.add_argument('--factors', nargs='+', default=['VOL10'],
                       help='要生成的因子列表')
    parser.add_argument('--factor-file', type=str, default=None,
                       help='因子文件路径（当因子来自文件时使用）')
    parser.add_argument('--factor-name', type=str, default=None,
                       help='因子列名（与 --factor-file 一起使用）')
    parser.add_argument('--factor-dir', type=str, default=None,
                       help='因子文件目录，会自动查找包含指定因子的CSV文件')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='factor_data',
                       help='输出目录')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='是否覆盖已存在的文件')

    return parser.parse_args()


def get_stock_list(stock_pool: str, max_stocks: Optional[int] = None) -> List[str]:
    """
    获取股票列表

    Args:
        stock_pool: 股票池标识
        max_stocks: 最大股票数量

    Returns:
        股票代码列表
    """
    try:
        if stock_pool == 'stock':
            # 全市场股票 - 这里需要实现获取全市场股票的逻辑
            # 暂时使用沪深300作为示例
            stocks = data.get_index_stocks('000300')
        elif stock_pool == 'small':
            # 小盘股（市值较小的股票）
            # 暂时使用中证500作为小盘股代表
            stocks = data.get_index_stocks('000905')[:500] if len(data.get_index_stocks('000905')) > 500 else data.get_index_stocks('000905')
        else:
            # 指数成分股
            stocks = data.get_index_stocks(stock_pool)

        if max_stocks and len(stocks) > max_stocks:
            stocks = stocks[:max_stocks]

        print(f"获取到 {len(stocks)} 只股票")
        return stocks

    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def generate_single_factor(factor_name: str, stock_codes: List[str],
                          start_date: str, end_date: str,
                          factor_file: Optional[str] = None,
                          factor_dir: Optional[str] = None) -> pd.DataFrame:
    """
    生成单个因子的数据

    Args:
        factor_name: 因子名称
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        factor_file: 因子文件路径
        factor_dir: 因子文件目录

    Returns:
        包含因子数据的DataFrame (date, code, factor_value)
    """
    print(f"\n开始生成因子: {factor_name}")

    # 创建因子计算器
    try:
        if factor_file and factor_name:
            # 从文件加载因子
            calc = create_factor_calculator(
                file_path=factor_file,
                factor_name=factor_name
            )
        elif factor_dir and factor_name:
            # 从目录查找因子文件
            calc = create_factor_calculator(
                factor_name=factor_name,
                factor_dir=factor_dir
            )
        else:
            # 使用内置因子
            calc = create_factor_calculator(factor_name=factor_name)

        print(f"✓ 创建因子计算器成功: {type(calc).__name__}")

    except Exception as e:
        print(f"❌ 创建因子计算器失败: {e}")
        return pd.DataFrame()

    # 为每只股票计算因子
    all_factor_data = []

    for i, stock_code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i+1}/{len(stock_codes)} 股票")

        try:
            # 计算因子值
            factor_series = calc.calculate(stock_code, start_date, end_date)

            if not factor_series.empty:
                # 删除NaN值，只保留有效的数据点
                factor_series = factor_series.dropna()
                
                if not factor_series.empty:
                    # 转换为标准格式
                    stock_data = pd.DataFrame({
                        'date': factor_series.index,
                        'code': str(stock_code).zfill(6),  # 标准化为6位数字
                        'factor_value': factor_series.values
                    })
                    # 确保code列是字符串类型
                    stock_data['code'] = stock_data['code'].astype(str)
                    all_factor_data.append(stock_data)

        except Exception as e:
            print(f"  ⚠️  计算股票 {stock_code} 因子失败: {e}")
            continue

    # 合并所有股票的数据
    if all_factor_data:
        result_df = pd.concat(all_factor_data, ignore_index=True)

        # 排序并重置索引
        result_df = result_df.sort_values(['date', 'code']).reset_index(drop=True)

        print(f"✓ 因子 {factor_name} 生成完成，共 {len(result_df)} 条记录")
        return result_df
    else:
        print(f"❌ 因子 {factor_name} 生成失败，无有效数据")
        return pd.DataFrame()


def save_factor_data(factor_name: str, factor_data: pd.DataFrame,
                    output_dir: str, start_date: str, end_date: str,
                    overwrite: bool = False) -> bool:
    """
    保存因子数据到文件

    Args:
        factor_name: 因子名称
        factor_data: 因子数据DataFrame
        output_dir: 输出目录
        overwrite: 是否覆盖现有文件

    Returns:
        保存是否成功
    """
    if factor_data.empty:
        print(f"⚠️  因子 {factor_name} 数据为空，跳过保存")
        return False

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    filename = f"{factor_name}_{start_date}_{end_date}.csv"
    filepath = output_path / filename

    # 检查文件是否已存在
    if filepath.exists() and not overwrite:
        print(f"⚠️  文件已存在: {filepath}，使用 --overwrite 覆盖")
        return False

    try:
        # 保存为CSV
        factor_data.to_csv(filepath, index=False, float_format='%.6f')

        # 显示统计信息
        print(f"✓ 因子 {factor_name} 已保存: {filepath}")
        print(f"  数据量: {len(factor_data)} 条记录")
        print(f"  股票数: {factor_data['code'].nunique()}")
        print(f"  日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")

        return True

    except Exception as e:
        print(f"❌ 保存因子 {factor_name} 失败: {e}")
        return False


def generate_summary_report(output_dir: str, factor_results: List[dict],
                          args) -> None:
    """
    生成汇总报告

    Args:
        output_dir: 输出目录
        factor_results: 因子生成结果列表
        args: 命令行参数
    """
    output_path = Path(output_dir)
    summary_file = output_path / 'generation_summary.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("因子数据生成报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 参数信息
        f.write("生成参数:\n")
        f.write(f"  日期范围: {args.start} ~ {args.end}\n")
        f.write(f"  股票池: {args.stock_pool}\n")
        if args.max_stocks:
            f.write(f"  最大股票数: {args.max_stocks}\n")
        f.write(f"  输出目录: {args.output_dir}\n\n")

        # 结果汇总
        f.write("生成结果:\n")
        successful_factors = [r for r in factor_results if r['success']]
        failed_factors = [r for r in factor_results if not r['success']]

        f.write(f"  成功因子数: {len(successful_factors)}\n")
        f.write(f"  失败因子数: {len(failed_factors)}\n\n")

        if successful_factors:
            f.write("成功生成的因子:\n")
            for result in successful_factors:
                f.write(f"  ✓ {result['factor_name']}: {result['records']} 条记录\n")
            f.write("\n")

        if failed_factors:
            f.write("生成失败的因子:\n")
            for result in failed_factors:
                f.write(f"  ❌ {result['factor_name']}: {result.get('error', '未知错误')}\n")
            f.write("\n")

        # 文件列表
        f.write("生成的文件:\n")
        csv_files = list(output_path.glob('*.csv'))
        for csv_file in sorted(csv_files):
            file_size = csv_file.stat().st_size / 1024  # KB
            f.write(f"  {csv_file.name} ({file_size:.1f} KB)\n")

    print(f"\n✓ 汇总报告已保存: {summary_file}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("因子数据生成脚本")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"股票池: {args.stock_pool}")
    print(f"输出目录: {args.output_dir}")

    if args.factor_file and args.factor_name:
        print(f"因子来源: 文件 {args.factor_file}, 列名 {args.factor_name}")
        factor_list = [args.factor_name]
    elif args.factor_dir:
        print(f"因子来源: 目录 {args.factor_dir}")
        factor_list = args.factors
    else:
        print(f"因子列表: {args.factors}")
        factor_list = args.factors

    print("=" * 60)
    print()

    # 获取股票列表
    stock_codes = get_stock_list(args.stock_pool, args.max_stocks)
    if not stock_codes:
        print("❌ 无法获取股票列表，退出")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成因子数据
    factor_results = []

    for factor_name in factor_list:
        try:
            # 生成单个因子数据
            factor_data = generate_single_factor(
                factor_name=factor_name,
                stock_codes=stock_codes,
                start_date=args.start,
                end_date=args.end,
                factor_file=args.factor_file if factor_name == args.factor_name else None,
                factor_dir=args.factor_dir
            )

            # 保存因子数据
            success = save_factor_data(
                factor_name=factor_name,
                factor_data=factor_data,
                output_dir=args.output_dir,
                start_date=args.start,
                end_date=args.end,
                overwrite=args.overwrite
            )

            # 记录结果
            factor_results.append({
                'factor_name': factor_name,
                'success': success,
                'records': len(factor_data) if success else 0,
                'error': None
            })

        except Exception as e:
            print(f"❌ 生成因子 {factor_name} 时发生错误: {e}")
            factor_results.append({
                'factor_name': factor_name,
                'success': False,
                'records': 0,
                'error': str(e)
            })

    # 生成汇总报告
    generate_summary_report(args.output_dir, factor_results, args)

    # 最终统计
    successful_count = sum(1 for r in factor_results if r['success'])
    total_records = sum(r['records'] for r in factor_results)

    print("\n" + "=" * 60)
    print("生成完成")
    print("=" * 60)
    print(f"成功因子数: {successful_count}/{len(factor_list)}")
    print(f"总数据量: {total_records} 条记录")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()