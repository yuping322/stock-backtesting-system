#!/usr/bin/env python3
"""
TALIB模型预测格式转换器
将QLib格式的预测结果转换为live_trading模块所需的格式
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live_trading.talib_prediction_loader import TALIBPredictionLoader, create_live_trading_predictions

logger = logging.getLogger(__name__)

def convert_talib_predictions(input_dir: str = "debug_model_results",
                            output_dir: str = "data",
                            strategies: list = ["long", "short"]) -> bool:
    """
    转换TALIB模型预测结果为live_trading格式

    Args:
        input_dir: 输入目录，包含TALIB模型结果
        output_dir: 输出目录
        strategies: 要转换的策略列表

    Returns:
        bool: 转换是否成功
    """
    try:
        logger.info(f"开始转换TALIB预测结果: {input_dir} -> {output_dir}")

        # 处理输入目录路径 - 如果是相对路径，相对于项目根目录
        input_path = Path(input_dir)
        if not input_path.is_absolute():
            # 如果是相对路径，相对于项目根目录
            project_root = Path(__file__).parent.parent
            input_path = project_root / input_dir

        logger.info(f"实际输入路径: {input_path}")

        # 处理输出目录路径
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            project_root = Path(__file__).parent.parent
            output_path = project_root / output_dir

        logger.info(f"实际输出路径: {output_path}")
        output_path.mkdir(exist_ok=True)

        # 创建TALIB预测加载器
        talib_loader = TALIBPredictionLoader(str(input_path))

        # 加载预测数据
        if not talib_loader.load_predictions():
            logger.error("无法加载TALIB预测数据")
            return False

        # 🔍 DEBUG: 检查加载的数据
        logger.info("🔍 DEBUG: 预测数据加载完成")
        for strategy in strategies:
            stats = talib_loader.get_prediction_stats(strategy)
            if stats:
                logger.info(f"  {strategy}策略: {stats['count']} 条记录")

        converted_files = []

        # 为每个策略创建预测文件
        for strategy in strategies:
            logger.info(f"转换{strategy}策略预测...")

            # 🔍 DEBUG: 获取原始预测数据
            raw_predictions = talib_loader.get_latest_predictions(strategy)
            if raw_predictions is not None:
                logger.info(f"🔍 DEBUG: {strategy}策略原始预测数据形状: {raw_predictions.shape}")
                logger.info(f"🔍 DEBUG: 样本数据:\n{raw_predictions.head(3)}")

            # 创建live_trading格式的预测文件
            filepath = create_live_trading_predictions(
                talib_loader=talib_loader,
                output_dir=str(output_path),
                strategy=strategy
            )

            if filepath:
                converted_files.append(filepath)
                logger.info(f"✓ {strategy}策略转换完成: {filepath}")

                # 🔍 DEBUG: 检查转换后的文件
                if os.path.exists(filepath):
                    df_check = pd.read_csv(filepath)
                    logger.info(f"🔍 DEBUG: 转换后文件形状: {df_check.shape}")
                    logger.info(f"🔍 DEBUG: 转换后样本:\n{df_check.head(3)}")
            else:
                logger.warning(f"✗ {strategy}策略转换失败")

        # 显示转换统计
        if converted_files:
            logger.info(f"转换完成，共生成 {len(converted_files)} 个文件:")
            for filepath in converted_files:
                logger.info(f"  - {filepath}")

            # 显示预测统计
            for strategy in strategies:
                stats = talib_loader.get_prediction_stats(strategy)
                if stats:
                    logger.info(f"\n{strategy.upper()}策略统计:")
                    logger.info(f"  预测数量: {stats['count']}")
                    logger.info(f"  股票数量: {stats['unique_stocks']}")
                    logger.info(".4f")
                    logger.info(".4f")
                    logger.info(".4f")
                    logger.info(".1f")
                    logger.info(".1f")

            return True
        else:
            logger.error("没有成功转换任何文件")
            return False

    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_converted_files(output_dir: str, strategies: list = ["long", "short"]) -> bool:
    """
    验证转换后的文件格式是否正确

    Args:
        output_dir: 输出目录
        strategies: 策略列表

    Returns:
        bool: 验证是否通过
    """
    try:
        logger.info("验证转换后的文件格式...")

        # 处理输出目录路径
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            project_root = Path(__file__).parent.parent
            output_path = project_root / output_dir

        logger.info(f"验证输出路径: {output_path}")
        valid_files = 0

        for strategy in strategies:
            # 查找对应的文件
            pattern = f"talib_{strategy}_predictions_*.csv"
            matching_files = list(output_path.glob(pattern))

            if not matching_files:
                logger.warning(f"未找到{strategy}策略的预测文件")
                continue

            # 使用最新的文件
            latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"验证文件: {latest_file}")

            # 读取并验证文件
            df = pd.read_csv(latest_file)

            # 🔍 DEBUG: 文件读取结果
            logger.info(f"🔍 DEBUG: 读取文件成功，形状: {df.shape}")
            logger.info(f"🔍 DEBUG: 列名: {df.columns.tolist()}")
            logger.info(f"🔍 DEBUG: 数据类型:\n{df.dtypes}")

            # 检查必需的列
            required_columns = ['model', 'date', 'code', 'score']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"文件缺少必需列: {missing_columns}")
                return False

            # 验证数据类型和格式
            if len(df) == 0:
                logger.error("文件为空")
                return False

            # 检查日期格式
            try:
                pd.to_datetime(df['date'])
            except:
                logger.error("日期格式不正确")
                return False

            # 检查股票代码格式（应该是以0开头的6位数字）
            codes_valid = df['code'].astype(str).str.match(r'^\d{6}$').all()
            if not codes_valid:
                logger.warning("部分股票代码格式可能不正确")

            logger.info(f"✓ {strategy}策略文件验证通过: {len(df)} 条记录")
            valid_files += 1

        if valid_files == len(strategies):
            logger.info("所有文件验证通过")
            return True
        else:
            logger.error(f"只有 {valid_files}/{len(strategies)} 个文件验证通过")
            return False

    except Exception as e:
        logger.error(f"文件验证出错: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='转换TALIB模型预测结果为live_trading格式')
    parser.add_argument('--input-dir', default='debug_model_results',
                       help='输入目录，包含TALIB模型结果 (默认: debug_model_results)')
    parser.add_argument('--output-dir', default='data',
                       help='输出目录 (默认: data)')
    parser.add_argument('--strategies', nargs='+', default=['long', 'short'],
                       choices=['long', 'short'],
                       help='要转换的策略 (默认: long short)')
    parser.add_argument('--validate', action='store_true',
                       help='转换后验证文件格式')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')

    args = parser.parse_args()

    # 设置日志级别
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("开始TALIB预测格式转换")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"策略: {args.strategies}")

    # 执行转换
    success = convert_talib_predictions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        strategies=args.strategies
    )

    if success and args.validate:
        logger.info("开始验证转换结果...")
        success = validate_converted_files(args.output_dir, args.strategies)

    if success:
        logger.info("🎉 TALIB预测格式转换完成！")
        print(f"\n转换成功！您现在可以使用以下命令运行实盘预测:")
        print(f"python run_live_trading.py --use-talib --talib-strategy {' '.join(args.strategies[:1])}")
    else:
        logger.error("❌ 转换失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    # 🔍 DEBUG: 程序启动
    print("🔍 DEBUG: TALIB转换工具启动")
    print(f"🔍 DEBUG: 当前工作目录: {os.getcwd()}")
    print(f"🔍 DEBUG: 脚本路径: {__file__}")

    main()