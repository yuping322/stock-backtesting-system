#!/usr/bin/env python3
"""
TALIB转换过程调试脚本
提供详细的调试信息和断点，帮助理解转换过程
"""

import sys
import os
import pandas as pd
import pickle
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live_trading.talib_prediction_loader import TALIBPredictionLoader, create_live_trading_predictions

def debug_original_data():
    """调试原始数据加载"""
    print("=== 调试原始数据加载 ===")

    # 检查文件是否存在
    long_file = project_root / "debug_model_results" / "predictions_long.pkl"
    short_file = project_root / "debug_model_results" / "predictions_short.pkl"

    print(f"Long预测文件: {long_file} (存在: {long_file.exists()})")
    print(f"Short预测文件: {short_file} (存在: {short_file.exists()})")

    if long_file.exists():
        print("\n--- Long预测文件信息 ---")
        with open(long_file, 'rb') as f:
            long_series = pickle.load(f)
        print(f"类型: {type(long_series)}")
        print(f"形状: {long_series.shape}")
        print(f"索引: {long_series.index.names}")
        print(f"数据类型: {long_series.dtype}")
        print(f"数值范围: [{long_series.min():.4f}, {long_series.max():.4f}]")
        print(f"正值比例: {(long_series > 0).mean():.1%}")
        print(f"零值数量: {(long_series == 0).sum()}")
        print(f"负值比例: {(long_series < 0).mean():.1%}")

        print("\n前5个原始数据:")
        print(long_series.head())

        # 重置索引后的数据
        print("\n重置索引后的数据:")
        df_reset = long_series.reset_index()
        print(f"重置后形状: {df_reset.shape}")
        print(f"列名: {df_reset.columns.tolist()}")
        print(df_reset.head())

def debug_loader():
    """调试加载器功能"""
    print("\n=== 调试TALIB加载器 ===")

    loader = TALIBPredictionLoader("debug_model_results")

    print("加载预测数据...")
    success = loader.load_predictions()
    print(f"加载结果: {'成功' if success else '失败'}")

    if success:
        print("\n--- 加载统计 ---")
        for strategy in ['long', 'short']:
            stats = loader.get_prediction_stats(strategy)
            if stats:
                print(f"\n{strategy.upper()}策略:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(".4f")
                    else:
                        print(f"  {key}: {value}")

        print("\n--- 获取最新预测 ---")
        for strategy in ['long', 'short']:
            predictions = loader.get_latest_predictions(strategy)
            if predictions is not None:
                print(f"\n{strategy}策略最新预测:")
                print(f"  形状: {predictions.shape}")
                print(f"  列名: {predictions.columns.tolist()}")
                print("  前3行数据:")
                print(predictions.head(3))
            else:
                print(f"{strategy}策略: 无预测数据")

def debug_conversion():
    """调试转换过程"""
    print("\n=== 调试转换过程 ===")

    loader = TALIBPredictionLoader("debug_model_results")
    if not loader.load_predictions():
        print("加载失败，无法继续调试")
        return

    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    for strategy in ['long', 'short']:
        print(f"\n--- 转换{strategy}策略 ---")

        # 获取原始预测
        raw_predictions = loader.get_latest_predictions(strategy)
        if raw_predictions is None:
            print(f"无法获取{strategy}策略的原始预测")
            continue

        print("原始预测数据:")
        print(f"  形状: {raw_predictions.shape}")
        print(f"  列名: {raw_predictions.columns.tolist()}")
        print("  数据样例:")
        print(raw_predictions.head(3))

        # 执行转换
        print("\n执行转换...")
        filepath = create_live_trading_predictions(
            talib_loader=loader,
            output_dir=str(output_dir),
            strategy=strategy
        )

        if filepath and os.path.exists(filepath):
            print(f"转换成功: {filepath}")

            # 检查转换结果
            df_converted = pd.read_csv(filepath)
            print("转换后数据:")
            print(f"  形状: {df_converted.shape}")
            print(f"  列名: {df_converted.columns.tolist()}")
            print("  数据样例:")
            print(df_converted.head(3))

            # 对比原始和转换数据
            print("\n--- 数据对比 ---")
            print("原始数据样例 (第一行):")
            if len(raw_predictions) > 0:
                first_row = raw_predictions.iloc[0]
                print(f"  date: {first_row['date']}")
                print(f"  code: {first_row['code']}")
                print(".4f")
                print(f"  model: {first_row['model']}")
                print(f"  strategy: {first_row['strategy']}")

            print("转换数据样例 (第一行):")
            if len(df_converted) > 0:
                first_row_conv = df_converted.iloc[0]
                print(f"  model: {first_row_conv['model']}")
                print(f"  date: {first_row_conv['date']}")
                print(f"  code: {first_row_conv['code']}")
                print(".4f")
        else:
            print("转换失败")

def debug_full_process():
    """调试完整流程"""
    print("\n=== 调试完整转换流程 ===")

    # 模拟命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', nargs='+', default=['long'])
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=True)

    args = parser.parse_args(['--strategies', 'long', 'short', '--validate', '--verbose'])

    # 导入并运行转换函数
    from live_trading.convert_talib_for_live_trading import convert_talib_predictions, validate_converted_files

    print("执行转换...")
    success = convert_talib_predictions(
        input_dir="debug_model_results",
        output_dir="data",
        strategies=args.strategies
    )

    if success and args.validate:
        print("\n执行验证...")
        success = validate_converted_files("data", args.strategies)

    print(f"\n最终结果: {'成功' if success else '失败'}")

def main():
    """主调试函数"""
    print("🔍 TALIB转换过程调试工具")
    print("=" * 50)

    try:
        debug_original_data()
        debug_loader()
        debug_conversion()
        debug_full_process()

        print("\n" + "=" * 50)
        print("🎉 调试完成！")

    except Exception as e:
        print(f"\n❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()