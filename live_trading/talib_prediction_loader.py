#!/usr/bin/env python3
"""
TALIB模型预测加载器
专门用于加载和处理TALIB因子模型的预测结果
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TALIBPredictionLoader:
    """TALIB因子模型预测结果加载器"""

    def __init__(self, model_results_dir: str = "debug_model_results", config=None):
        """
        初始化TALIB预测加载器

        Args:
            model_results_dir: 模型结果目录，默认为"debug_model_results"
            config: TALIB模型配置对象（可选）
        """
        self.model_results_dir = Path(model_results_dir)
        self.config = config
        self.predictions_long = None
        self.predictions_short = None
        self._loaded = False

    def load_predictions(self) -> bool:
        """
        加载TALIB模型预测结果

        Returns:
            bool: 是否成功加载
        """
        try:
            # 检查目录是否存在
            if not self.model_results_dir.exists():
                logger.error(f"模型结果目录不存在: {self.model_results_dir}")
                return False

            # 加载long预测
            long_file = self.model_results_dir / "predictions_long.pkl"
            if long_file.exists():
                self.predictions_long = pd.read_pickle(long_file)
                logger.info(f"加载Long预测: {len(self.predictions_long)} 条记录")
            else:
                logger.warning(f"Long预测文件不存在: {long_file}")

            # 加载short预测
            short_file = self.model_results_dir / "predictions_short.pkl"
            if short_file.exists():
                self.predictions_short = pd.read_pickle(short_file)
                logger.info(f"加载Short预测: {len(self.predictions_short)} 条记录")
            else:
                logger.warning(f"Short预测文件不存在: {short_file}")

            if self.predictions_long is None and self.predictions_short is None:
                logger.error("没有找到任何预测文件")
                return False

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"加载预测结果失败: {e}")
            return False

    def get_latest_predictions(self, strategy: str = "long") -> Optional[pd.DataFrame]:
        """
        获取最新的预测结果

        Args:
            strategy: 策略类型，"long" 或 "short"

        Returns:
            pd.DataFrame: 最新的预测结果，包含日期、股票代码和预测分数
        """
        if not self._loaded:
            if not self.load_predictions():
                return None

        # 选择预测数据
        if strategy == "long" and self.predictions_long is not None:
            predictions = self.predictions_long
        elif strategy == "short" and self.predictions_short is not None:
            predictions = self.predictions_short
        else:
            logger.warning(f"未找到{strategy}策略的预测数据")
            return None

        # 转换为标准格式
        try:
            # 重置索引以获取日期和股票代码
            df_reset = predictions.reset_index()

            # 确保列名正确
            if 'datetime' in df_reset.columns and 'instrument' in df_reset.columns:
                result_df = pd.DataFrame({
                    'date': pd.to_datetime(df_reset['datetime']).dt.strftime('%Y%m%d'),
                    'code': df_reset['instrument'].astype(str).str.zfill(6),
                    'score': predictions.values,  # 预测分数
                    'model': f'talib_{strategy}',
                    'strategy': strategy
                })
            else:
                logger.error("预测数据格式不正确，缺少datetime或instrument列")
                return None

            # 按日期排序，取最新日期的数据
            if not result_df.empty:
                latest_date = result_df['date'].max()
                latest_predictions = result_df[result_df['date'] == latest_date].copy()
                logger.info(f"获取{strategy}策略最新预测: {latest_date}, {len(latest_predictions)} 只股票")
                return latest_predictions
            else:
                logger.warning("预测数据为空")
                return None

        except Exception as e:
            logger.error(f"处理预测数据失败: {e}")
            return None

    def get_predictions_for_date(self, date: str, strategy: str = "long") -> Optional[pd.DataFrame]:
        """
        获取指定日期的预测结果

        Args:
            date: 日期字符串，格式为YYYYMMDD
            strategy: 策略类型，"long" 或 "short"

        Returns:
            pd.DataFrame: 指定日期的预测结果
        """
        if not self._loaded:
            if not self.load_predictions():
                return None

        # 选择预测数据
        if strategy == "long" and self.predictions_long is not None:
            predictions = self.predictions_long
        elif strategy == "short" and self.predictions_short is not None:
            predictions = self.predictions_short
        else:
            return None

        try:
            # 重置索引
            df_reset = predictions.reset_index()

            if 'datetime' in df_reset.columns and 'instrument' in df_reset.columns:
                result_df = pd.DataFrame({
                    'date': pd.to_datetime(df_reset['datetime']).dt.strftime('%Y%m%d'),
                    'code': df_reset['instrument'].astype(str).str.zfill(6),
                    'score': predictions.values,
                    'model': f'talib_{strategy}',
                    'strategy': strategy
                })

                # 过滤指定日期
                date_predictions = result_df[result_df['date'] == date].copy()
                logger.info(f"获取{strategy}策略{date}预测: {len(date_predictions)} 只股票")
                return date_predictions
            else:
                return None

        except Exception as e:
            logger.error(f"获取日期预测失败: {e}")
            return None

    def get_available_dates(self, strategy: str = "long") -> List[str]:
        """
        获取可用的预测日期

        Args:
            strategy: 策略类型

        Returns:
            List[str]: 可用的日期列表
        """
        predictions = self.get_latest_predictions(strategy)
        if predictions is not None:
            return sorted(predictions['date'].unique())
        return []

    def get_prediction_stats(self, strategy: str = "long") -> Dict:
        """
        获取预测统计信息

        Args:
            strategy: 策略类型

        Returns:
            Dict: 统计信息
        """
        predictions = self.get_latest_predictions(strategy)
        if predictions is None:
            return {}

        scores = predictions['score']

        stats = {
            'count': len(predictions),
            'unique_dates': predictions['date'].nunique(),
            'unique_stocks': predictions['code'].nunique(),
            'score_mean': scores.mean(),
            'score_std': scores.std(),
            'score_min': scores.min(),
            'score_max': scores.max(),
            'score_median': scores.median(),
            'positive_ratio': (scores > 0).mean(),
            'negative_ratio': (scores < 0).mean(),
            'zero_ratio': (scores == 0).mean()
        }

        return stats

    def export_to_csv(self, output_file: str, strategy: str = "long", date: Optional[str] = None):
        """
        导出预测结果到CSV文件

        Args:
            output_file: 输出文件路径
            strategy: 策略类型
            date: 指定日期，如果为None则导出最新日期
        """
        if date:
            predictions = self.get_predictions_for_date(date, strategy)
        else:
            predictions = self.get_latest_predictions(strategy)

        if predictions is not None:
            predictions.to_csv(output_file, index=False)
            logger.info(f"预测结果已导出到: {output_file}")
        else:
            logger.error("没有预测数据可导出")

def create_live_trading_predictions(talib_loader: TALIBPredictionLoader,
                                  output_dir: str = "data",
                                  strategy: str = "long") -> Optional[str]:
    """
    创建适合live_trading模块的预测文件

    Args:
        talib_loader: TALIB预测加载器
        output_dir: 输出目录
        strategy: 策略类型

    Returns:
        str: 生成的文件路径，如果失败返回None
    """
    try:
        # 获取最新预测
        predictions = talib_loader.get_latest_predictions(strategy)
        if predictions is None:
            return None

        # 转换为live_trading格式
        live_format = pd.DataFrame({
            'model': predictions['model'],
            'date': predictions['date'],
            'code': predictions['code'],
            'score': predictions['score']
        })

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 生成文件名
        latest_date = predictions['date'].iloc[0]
        filename = f"talib_{strategy}_predictions_{latest_date}.csv"
        filepath = output_path / filename

        # 导出文件
        live_format.to_csv(filepath, index=False)
        logger.info(f"Live trading预测文件已生成: {filepath}")

        return str(filepath)

    except Exception as e:
        logger.error(f"创建live trading预测文件失败: {e}")
        return None

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建加载器
    loader = TALIBPredictionLoader()

    # 加载预测
    if loader.load_predictions():
        print("✓ 预测数据加载成功")

        # 显示统计信息
        for strategy in ["long", "short"]:
            stats = loader.get_prediction_stats(strategy)
            if stats:
                print(f"\n{strategy.upper()}策略统计:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(".4f")
                    else:
                        print(f"  {key}: {value}")

        # 创建live trading文件
        for strategy in ["long", "short"]:
            filepath = create_live_trading_predictions(loader, strategy=strategy)
            if filepath:
                print(f"✓ {strategy}策略live trading文件: {filepath}")

    else:
        print("✗ 预测数据加载失败")