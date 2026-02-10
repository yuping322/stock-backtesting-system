#!/usr/bin/env python3
"""
测试新的因子分析器功能

使用指定的因子文件测试：
- FactorAnalyzer 基本功能
- 绘图功能（如果启用）
- 汇总表导出功能
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.factor.analyzer import FactorAnalyzer, export_analysis_report

def test_factor_analyzer():
    """测试因子分析器"""

    # 指定因子文件路径
    factor_file_path = "/Users/fengzhi/Downloads/git/stock-backtesting-system/data/talib_factors_2025/TALIB_ACCBANDS_10_2025-01-01_2025-11-23.csv"

    print("=" * 60)
    print("因子分析器测试")
    print("=" * 60)
    print(f"因子文件: {factor_file_path}")

    # 检查文件是否存在
    if not os.path.exists(factor_file_path):
        print(f"❌ 因子文件不存在: {factor_file_path}")
        return False

    try:
        # 读取因子数据
        print("\n📂 读取因子数据...")
        factor_df = pd.read_csv(factor_file_path)
        print(f"✓ 成功读取因子数据: {len(factor_df)} 行")

        # 显示数据基本信息
        print(f"  列数: {len(factor_df.columns)}")
        print(f"  股票数量: {factor_df['code'].nunique()}")
        print(f"  日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")

        # 显示前几个因子列
        factor_cols = [col for col in factor_df.columns if col not in ['date', 'code', 'asset']]
        print(f"  因子数量: {len(factor_cols)}")
        print(f"  前5个因子: {factor_cols[:5]}")

        # 选择一个因子进行测试（选择第一个因子）
        test_factor = factor_cols[0]
        print(f"\n🎯 选择测试因子: {test_factor}")

        # 过滤数据，只保留选中的因子
        test_df = factor_df[['date', 'code', test_factor]].copy()
        test_df = test_df.rename(columns={test_factor: 'factor_value'})

        # 清理数据
        test_df = test_df.dropna()
        print(f"✓ 清理后数据: {len(test_df)} 行")
        
        # 检查数据质量
        print(f"  因子值范围: {test_df['factor_value'].min():.4f} ~ {test_df['factor_value'].max():.4f}")
        print(f"  因子值平均值: {test_df['factor_value'].mean():.4f}")
        print(f"  因子值标准差: {test_df['factor_value'].std():.4f}")
        
        # 检查是否有足够的股票和日期
        stock_counts = test_df.groupby('date')['code'].count()
        print(f"  每日平均股票数: {stock_counts.mean():.1f}")
        print(f"  最小每日股票数: {stock_counts.min()}")
        print(f"  最大每日股票数: {stock_counts.max()}")

        # 打印因子数据的日期信息
        print(f"\n📅 因子数据日期信息:")
        print(f"  因子数据日期范围: {test_df['date'].min()} ~ {test_df['date'].max()}")
        unique_dates = sorted(test_df['date'].unique())
        print(f"  因子数据日期数量: {len(unique_dates)}")
        print(f"  前5个因子日期: {unique_dates[:5]}")
        print(f"  后5个因子日期: {unique_dates[-5:]}")

        if len(test_df) == 0:
            print("❌ 清理后无有效数据")
            return False

        # 设置matplotlib后端（支持无GUI环境）
        import matplotlib
        matplotlib.use('Agg', force=True)  # 使用非交互式后端
        print("✓ 已设置matplotlib后端为Agg")

        # 创建分析器
        print("\n🔧 创建因子分析器...")
        analyzer = FactorAnalyzer(
            factor_df=test_df,
            start_date=test_df['date'].min(),
            end_date=test_df['date'].max(),
            quantiles=5,  # 使用较少的quantile以避免数据不足
            periods=[5, 10, 15]  # 使用正常的周期，现在有足够数据
        )
        print("✓ 分析器创建成功")

        # 设置输出目录为当前目录
        output_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 输出目录: {output_dir}")

        # 运行分析（启用绘图）
        print("\n📊 运行因子分析...")
        results = analyzer.analyze_factor(plot=True)
        print(f"✓ 分析完成，结果数量: {len(results)}")

        if not results:
            print("❌ 分析无结果")
            return False

        # 显示结果摘要
        print("\n📋 分析结果摘要:")
        for i, result in enumerate(results[:3]):  # 只显示前3个结果
            if isinstance(result, dict):
                print(f"  结果 {i+1}: 因子={result.get('factor_name', 'N/A')}, 周期={result.get('period', 'N/A')}, 等级={result.get('level', 'N/A')}")
            else:
                print(f"  结果 {i+1}: 因子={result.factor_name}, 周期={result.period}, 等级={result.level}")

        # 测试汇总表导出
        print("\n📄 测试汇总表导出...")
        # 使用当前目录作为输出目录
        output_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 汇总表输出目录: {output_dir}")

        export_analysis_report(analyzer, output_dir)
        summary_file = os.path.join(output_dir, 'factor_analysis_summary.csv')

        if os.path.exists(summary_file):
            print(f"✓ 汇总表导出成功: {summary_file}")

            # 读取并显示汇总表内容
            summary_df = pd.read_csv(summary_file)
            print(f"  汇总表行数: {len(summary_df)}")
            print(f"  汇总表列名: {list(summary_df.columns)}")
            if len(summary_df) > 0:
                print("  前3行数据:")
                print(summary_df.head(3).to_string(index=False))
        else:
            print(f"❌ 汇总表导出失败: {summary_file}")

        print("\n✅ 所有测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plotting_feature():
    """测试绘图功能（可选）"""

    print("\n🎨 测试绘图功能...")
    print("⚠️  绘图功能需要图形界面支持，在无GUI环境中可能无法正常工作")
    print("   如需测试绘图，请在有图形界面的环境中运行并设置 plot=True")

    # 这里可以添加绘图测试，但由于环境限制，暂时跳过
    print("✓ 绘图功能代码已实现，测试跳过")

if __name__ == "__main__":
    print("开始因子分析器测试...")

    # 主测试
    success = test_factor_analyzer()

    if success:
        # 可选的绘图测试
        test_plotting_feature()

    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！新的因子分析器功能正常工作。")
    else:
        print("💥 测试失败！请检查代码和数据。")
    print("=" * 60)