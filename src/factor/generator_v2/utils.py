"""
V2 生成器的工具函数

提供数据加载、数据处理、配置管理等支持功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载工具（抽象层）
    
    支持多种数据源，默认从 src.data.data 模块加载
    """
    
    @staticmethod
    def load_ohlcv(
        stock_code: str,
        start_date: str,
        end_date: str,
        fallback_func: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        加载 OHLCV 数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fallback_func: 备用加载函数（如果默认加载失败）
        
        Returns:
            pd.DataFrame: OHLCV 数据
        
        Raises:
            Exception: 数据加载失败
        """
        # 首先尝试使用默认的数据加载方法
        try:
            from src.data.data import load_oss_complex_stocks
            
            data = load_oss_complex_stocks(
                [stock_code],
                start_date,
                end_date
            )
            
            if data is None or data.empty:
                raise ValueError("加载的数据为空")
            
            logger.debug(f"✓ 从 OSS 加载数据: {stock_code}")
            return data
        
        except Exception as e:
            logger.debug(f"⚠️  默认加载失败 ({e})")
            
            # 尝试备用函数
            if fallback_func is not None:
                try:
                    data = fallback_func(stock_code, start_date, end_date)
                    if data is not None and not data.empty:
                        logger.debug(f"✓ 使用备用函数加载数据: {stock_code}")
                        return data
                except Exception as e2:
                    logger.debug(f"⚠️  备用函数也失败了 ({e2})")
            
            raise ValueError(f"无法加载 {stock_code} 的数据: {e}")
    
    @staticmethod
    def load_from_csv(filepath: str, stock_code: Optional[str] = None) -> pd.DataFrame:
        """
        从 CSV 文件加载数据
        
        Args:
            filepath: 文件路径
            stock_code: 可选的股票代码过滤
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        df = pd.read_csv(filepath)
        
        if stock_code and 'stock_code' in df.columns:
            df = df[df['stock_code'] == stock_code]
        
        logger.debug(f"✓ 从 CSV 加载数据: {filepath} ({len(df)} 行)")
        return df


class DataProcessor:
    """
    数据处理工具
    """
    
    @staticmethod
    def normalize_stock_code(code: str) -> str:
        """
        规范化股票代码为 6 位数字格式
        
        Args:
            code: 股票代码
        
        Returns:
            str: 6 位股票代码
        """
        return str(code).zfill(6)
    
    @staticmethod
    def parse_date(date_str: str) -> pd.Timestamp:
        """
        解析日期字符串
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
        
        Returns:
            pd.Timestamp: 时间戳
        """
        return pd.to_datetime(date_str)
    
    @staticmethod
    def fill_na_forward(series: pd.Series, limit: int = 5) -> pd.Series:
        """
        向前填充 NaN 值（最多填充 limit 个）
        
        Args:
            series: 数据序列
            limit: 最大填充数量
        
        Returns:
            pd.Series: 填充后的序列
        """
        return series.fillna(method='ffill', limit=limit)
    
    @staticmethod
    def fill_na_backward(series: pd.Series, limit: int = 5) -> pd.Series:
        """
        向后填充 NaN 值（最多填充 limit 个）
        
        Args:
            series: 数据序列
            limit: 最大填充数量
        
        Returns:
            pd.Series: 填充后的序列
        """
        return series.fillna(method='bfill', limit=limit)
    
    @staticmethod
    def remove_outliers(
        series: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.Series:
        """
        移除异常值
        
        Args:
            series: 数据序列
            method: 方法 ('iqr' 或 'zscore')
            threshold: 阈值
        
        Returns:
            pd.Series: 移除异常值后的序列
        """
        if method == 'iqr':
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            return series[(series >= lower) & (series <= upper)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(series.dropna()))
            return series[np.abs(stats.zscore(series.dropna())) < threshold]
        
        else:
            raise ValueError(f"未知的方法: {method}")
    
    @staticmethod
    def clip_values(
        series: pd.Series,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> pd.Series:
        """
        使用百分位数裁剪值
        
        Args:
            series: 数据序列
            lower_percentile: 下百分位
            upper_percentile: 上百分位
        
        Returns:
            pd.Series: 裁剪后的序列
        """
        lower = series.quantile(lower_percentile / 100)
        upper = series.quantile(upper_percentile / 100)
        return series.clip(lower=lower, upper=upper)
    
    @staticmethod
    def standardize(series: pd.Series, method: str = 'zscore') -> pd.Series:
        """
        标准化序列
        
        Args:
            series: 数据序列
            method: 方法 ('zscore' 或 'minmax')
        
        Returns:
            pd.Series: 标准化后的序列
        """
        if method == 'zscore':
            return (series - series.mean()) / series.std()
        
        elif method == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        
        else:
            raise ValueError(f"未知的方法: {method}")


class ConfigManager:
    """
    配置管理工具
    
    支持从文件加载配置参数
    """
    
    # 内置参数配置
    BUILTIN_PARAMS = {
        'VOL10': {'window': 10},
        'RSI_14': {'period': 14},
        'MA_20': {'window': 20},
        'MACD_12_26_9': {'fast': 12, 'slow': 26, 'signal': 9},
    }
    
    TALIB_PARAMS = {
        'RSI': {'timeperiod': 14},
        'MA': {'timeperiod': 20},
        'BBANDS': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
        'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'STOCH': {'fastk_period': 5, 'slowk_period': 3, 'slowk_matype': 0},
    }
    
    @classmethod
    def get_builtin_params(cls, factor_name: str) -> dict:
        """
        获取内置因子参数
        
        Args:
            factor_name: 因子名称
        
        Returns:
            dict: 参数字典
        """
        if factor_name not in cls.BUILTIN_PARAMS:
            return {}
        
        return cls.BUILTIN_PARAMS[factor_name].copy()
    
    @classmethod
    def get_talib_params(cls, indicator_name: str) -> dict:
        """
        获取 Talib 指标参数
        
        Args:
            indicator_name: 指标名称
        
        Returns:
            dict: 参数字典
        """
        if indicator_name not in cls.TALIB_PARAMS:
            return {}
        
        return cls.TALIB_PARAMS[indicator_name].copy()
    
    @staticmethod
    def load_config_file(filepath: str) -> dict:
        """
        从 YAML/JSON 文件加载配置
        
        Args:
            filepath: 配置文件路径
        
        Returns:
            dict: 配置字典
        """
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"⚠️  配置文件不存在: {filepath}")
            return {}
        
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif filepath.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
                except ImportError:
                    logger.warning("⚠️  pyyaml 未安装，无法加载 YAML 文件")
                    return {}
            
            else:
                logger.warning(f"⚠️  未知的配置文件格式: {filepath.suffix}")
                return {}
        
        except Exception as e:
            logger.warning(f"⚠️  加载配置文件失败: {e}")
            return {}


class ProgressTracker:
    """
    进度跟踪工具
    """
    
    def __init__(self, total: int, name: str = "处理"):
        """
        初始化进度跟踪器
        
        Args:
            total: 总数
            name: 名称
        """
        self.total = total
        self.name = name
        self.current = 0
        self.failures = []
    
    def update(self, increment: int = 1, status: str = ""):
        """
        更新进度
        
        Args:
            increment: 增量
            status: 状态信息
        """
        self.current += increment
        percent = (self.current / self.total) * 100
        
        message = f"  [{self.current}/{self.total}] {percent:.1f}%"
        if status:
            message += f" - {status}"
        
        logger.info(message)
    
    def add_failure(self, item: str, reason: str):
        """
        记录失败
        
        Args:
            item: 项目
            reason: 原因
        """
        self.failures.append({'item': item, 'reason': reason})
    
    def get_summary(self) -> dict:
        """
        获取摘要
        
        Returns:
            dict: 摘要信息
        """
        return {
            'total': self.total,
            'success': self.current - len(self.failures),
            'failed': len(self.failures),
            'failures': self.failures,
        }
