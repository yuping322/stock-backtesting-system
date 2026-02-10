"""
因子计算器 - V2 版本中的统一计算器接口

该模块提供了一个统一的因子计算器接口。所有计算器都实现相同的接口签名：
    calculate(stock_code: str, start_date: str, end_date: str) -> pd.Series

这确保了不同类型的计算器可以互换使用。

支持的计算器类型:
    1. BuiltinFactorCalculator - 内置因子 (VOL10, RSI_14, MA_20, MACD_12_26_9)
    2. TalibFactorCalculator - Talib 因子 (TALIB_RSI_14, TALIB_SMA_20, ...)
    3. CustomFunctionCalculator - 自定义函数
    4. FileFactorCalculator - 从文件加载
"""

import inspect
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, List

from .exceptions import (
    DataNotAvailableError,
    FactorCalculationError,
)

logger = logging.getLogger(__name__)


class FactorCalculator(ABC):
    """
    因子计算器基类
    
    所有具体的因子计算器都应继承此类并实现 calculate() 方法。
    
    接口约定:
        所有计算器都必须实现统一的 3 参数接口:
        calculate(stock_code: str, start_date: str, end_date: str) -> pd.Series
    """
    
    @abstractmethod
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """
        计算因子值
        
        Args:
            stock_code: 股票代码（如 '000001'，必须是 6 位数字）
            start_date: 开始日期（YYYY-MM-DD 格式）
            end_date: 结束日期（YYYY-MM-DD 格式）
        
        Returns:
            pd.Series: 因子值序列
                - 索引: DatetimeIndex（日期）
                - 值: float（因子值）
                - 包含 NaN 值时表示该日期无数据
        
        Raises:
            DataNotAvailableError: 数据不可用
            FactorCalculationError: 计算失败
        """
        pass


class BuiltinFactorCalculator(FactorCalculator):
    """
    内置因子计算器
    
    支持的因子:
        - VOL10: 10 日成交量比值
        - RSI_14: 14 日相对强弱指标
        - MA_20: 20 日移动平均比值
        - MACD_12_26_9: MACD 指标
    
    示例:
        calc = BuiltinFactorCalculator('VOL10')
        result = calc.calculate('000001', '2024-01-01', '2024-12-31')
    """
    
    SUPPORTED_FACTORS = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
    
    def __init__(self, factor_name: str):
        """
        初始化内置因子计算器
        
        Args:
            factor_name: 因子名称，必须在 SUPPORTED_FACTORS 中
        
        Raises:
            ValueError: 因子名称不支持
        """
        if factor_name not in self.SUPPORTED_FACTORS:
            raise ValueError(
                f"不支持的因子: {factor_name}. 支持的因子: {self.SUPPORTED_FACTORS}"
            )
        self.factor_name = factor_name
        self._data_loader = None  # 可注入的数据加载器
    
    def set_data_loader(self, data_loader):
        """
        注入数据加载器（依赖注入）
        
        Args:
            data_loader: 实现 load_ohlcv() 方法的加载器对象
        """
        self._data_loader = data_loader
        return self
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算内置因子"""
        try:
            # 加载 OHLCV 数据
            ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"无法加载 {stock_code} 的 OHLCV 数据")
                raise DataNotAvailableError(stock_code, start_date, end_date, "OHLCV 数据为空")
            
            # 设置日期索引（如果还没有）
            if 'date' in ohlcv.columns:
                ohlcv = ohlcv.set_index('date')
            
            # 使用对应的计算方法
            if self.factor_name == 'VOL10':
                return self._calculate_vol10(ohlcv)
            elif self.factor_name == 'RSI_14':
                return self._calculate_rsi_14(ohlcv)
            elif self.factor_name == 'MA_20':
                return self._calculate_ma_20(ohlcv)
            elif self.factor_name == 'MACD_12_26_9':
                return self._calculate_macd_12_26_9(ohlcv)
            else:
                raise ValueError(f"不支持的因子: {self.factor_name}")
        
        except DataNotAvailableError:
            raise
        except Exception as e:
            raise FactorCalculationError(
                self.factor_name, stock_code, f"计算异常: {str(e)}"
            ) from e
    
    def _load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载 OHLCV 数据
        
        如果设置了数据加载器，使用注入的加载器。
        否则使用默认的加载方式（从 src.data.data 加载）。
        """
        if self._data_loader:
            return self._data_loader.load_ohlcv(stock_code, start_date, end_date)
        
        # 默认加载方式
        try:
            from src.data.data import load_oss_complex_stocks
            
            result = load_oss_complex_stocks(
                codes=[stock_code],
                start=start_date,
                end=end_date,
                fields=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not result:
                return pd.DataFrame()
            
            # 转换格式: Dict[field, DataFrame] -> DataFrame
            dfs = []
            for field, df in result.items():
                if not df.empty and stock_code in df.columns:
                    temp_df = pd.DataFrame({
                        'date': df.index,
                        field: df[stock_code]
                    })
                    dfs.append(temp_df)
            
            if not dfs:
                return pd.DataFrame()
            
            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on='date', how='outer')
            
            return merged
        
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _calculate_vol10(ohlcv: pd.DataFrame) -> pd.Series:
        """计算 VOL10 - 10 日成交量比值"""
        if 'volume' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        volume = ohlcv['volume'].astype(float)
        ma10 = volume.rolling(window=10).mean()
        
        # 避免除以 0
        result = np.where(ma10 != 0, volume / ma10, np.nan)
        return pd.Series(result, index=ohlcv.index, name='VOL10')
    
    @staticmethod
    def _calculate_rsi_14(ohlcv: pd.DataFrame) -> pd.Series:
        """计算 RSI_14 - 14 日相对强弱指标"""
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close'].astype(float)
        delta = close.diff()
        
        # 分离上升和下降
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # 计算 RS 和 RSI
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 * (rs / (1 + rs))
        
        return rsi.rename('RSI_14')
    
    @staticmethod
    def _calculate_ma_20(ohlcv: pd.DataFrame) -> pd.Series:
        """计算 MA_20 - 20 日移动平均比值"""
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close'].astype(float)
        ma20 = close.rolling(window=20).mean()
        
        # 避免除以 0
        result = np.where(ma20 != 0, close / ma20, np.nan)
        return pd.Series(result, index=ohlcv.index, name='MA_20')
    
    @staticmethod
    def _calculate_macd_12_26_9(ohlcv: pd.DataFrame) -> pd.Series:
        """计算 MACD_12_26_9 - MACD 指标"""
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close'].astype(float)
        
        # 计算 EMA
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        # 计算 DIF (MACD Line)
        dif = ema12 - ema26
        
        # 计算 DEA (Signal Line)
        dea = dif.ewm(span=9, adjust=False).mean()
        
        # 计算 MACD Histogram
        macd = 2 * (dif - dea)
        
        return macd.rename('MACD_12_26_9')


class TalibFactorCalculator(FactorCalculator):
    """
    Talib 因子计算器
    
    支持 Talib 库中的所有技术指标函数。
    因子名称格式: 'TALIB_<函数名>_<参数>'
    
    示例:
        calc = TalibFactorCalculator('TALIB_RSI_14')
        result = calc.calculate('000001', '2024-01-01', '2024-12-31')
    
    注意:
        需要安装 TA-Lib: pip install TA-Lib
    """
    
    def __init__(self, factor_name: str, params: Optional[List] = None):
        """
        初始化 Talib 因子计算器
        
        Args:
            factor_name: 因子名称，格式 'TALIB_<函数>_<参数>'
            params: 可选的参数列表（覆盖因子名称中的参数）
        
        Raises:
            ValueError: 因子名称格式错误
        """
        if not factor_name.startswith('TALIB_'):
            raise ValueError(f"Talib 因子应以 'TALIB_' 开头: {factor_name}")
        
        self.factor_name = factor_name
        self.params = params
        
        # 解析因子名称
        self._parse_factor_name(factor_name)
    
    def _parse_factor_name(self, factor_name: str):
        """解析因子名称提取函数名和参数"""
        # 格式: TALIB_RSI_14 -> (RSI, [14])
        parts = factor_name.split('_')[1:]  # 移除 TALIB 前缀
        
        if not parts:
            raise ValueError(f"无效的因子名称格式: {factor_name}")
        
        self.func_name = parts[0]
        
        # 提取参数
        if self.params is None and len(parts) > 1:
            try:
                self.params = [int(p) for p in parts[1:]]
            except ValueError:
                logger.warning(f"无法解析参数: {parts[1:]}")
                self.params = []
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算 Talib 因子"""
        try:
            import talib
        except ImportError:
            raise FactorCalculationError(
                self.factor_name,
                stock_code,
                "TA-Lib 未安装，请执行: pip install TA-Lib"
            )
        
        try:
            # 加载 OHLCV 数据
            ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
            if ohlcv is None or ohlcv.empty:
                raise DataNotAvailableError(stock_code, start_date, end_date, "OHLCV 数据为空")
            
            # 获取函数
            if not hasattr(talib, self.func_name):
                raise ValueError(f"Talib 不支持函数: {self.func_name}")
            
            func = getattr(talib, self.func_name)
            
            # 调用函数计算因子
            # 这里需要根据函数的需求提供对应的 OHLCV 字段
            result = self._call_talib_func(func, ohlcv)
            
            return pd.Series(result, index=ohlcv.index, name=self.factor_name)
        
        except DataNotAvailableError:
            raise
        except Exception as e:
            raise FactorCalculationError(
                self.factor_name, stock_code, str(e)
            ) from e
    
    def _load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载 OHLCV 数据 - 与 Builtin 相同"""
        try:
            from src.data.data import load_oss_complex_stocks
            
            result = load_oss_complex_stocks(
                codes=[stock_code],
                start=start_date,
                end=end_date,
                fields=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not result:
                return pd.DataFrame()
            
            dfs = []
            for field, df in result.items():
                if not df.empty and stock_code in df.columns:
                    temp_df = pd.DataFrame({
                        'date': df.index,
                        field: df[stock_code]
                    })
                    dfs.append(temp_df)
            
            if not dfs:
                return pd.DataFrame()
            
            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on='date', how='outer')
            
            merged['date'] = pd.to_datetime(merged['date'])
            merged = merged.set_index('date').sort_index()
            
            return merged
        
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _call_talib_func(self, func, ohlcv: pd.DataFrame):
        """
        调用 Talib 函数
        
        根据函数的需求提供对应的 OHLCV 字段
        """
        # 这是一个简化的实现
        # 实际的 Talib 函数需要不同的输入
        # 例如: SMA 需要 close, RSI 也需要 close, ATR 需要 high/low/close 等
        
        # 优先使用 close 价格（大多数指标都基于这个）
        if 'close' in ohlcv.columns:
            close = ohlcv['close'].astype(float).values
            
            if self.params:
                result = func(close, *self.params)
            else:
                result = func(close)
            
            return result
        
        raise ValueError(f"缺少必要的数据字段: close")


class CustomFunctionCalculator(FactorCalculator):
    """
    自定义函数计算器
    
    支持两种函数签名:
    1. func(ohlcv: DataFrame) -> Series - 接收完整的 OHLCV DataFrame
    2. func(stock_code, start_date, end_date) -> Series - 自行加载数据
    
    示例:
        def my_factor(ohlcv):
            return ohlcv['close'] / ohlcv['close'].rolling(20).mean()
        
        calc = CustomFunctionCalculator('MY_FACTOR', my_factor)
        result = calc.calculate('000001', '2024-01-01', '2024-12-31')
    """
    
    def __init__(self, factor_name: str, func: Callable):
        """
        初始化自定义函数计算器
        
        Args:
            factor_name: 因子名称
            func: 计算函数（必须是可调用的）
        
        Raises:
            ValueError: 函数不可调用或参数数量不正确
        """
        if not callable(func):
            raise ValueError("提供的函数不是可调用对象")
        
        self.factor_name = factor_name
        self.func = func
        
        # 检测函数的签名类型
        self._func_type = self._detect_func_type(func)
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算自定义因子"""
        try:
            if self._func_type == 'ohlcv':
                # 函数签名: func(ohlcv: DataFrame) -> Series
                ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
                if ohlcv is None or ohlcv.empty:
                    raise DataNotAvailableError(stock_code, start_date, end_date)
                
                result = self.func(ohlcv)
            
            elif self._func_type == 'params':
                # 函数签名: func(stock_code, start_date, end_date) -> Series
                result = self.func(stock_code, start_date, end_date)
            
            else:
                raise ValueError("不支持的函数类型")
            
            # 确保返回 Series
            if not isinstance(result, pd.Series):
                result = pd.Series(result)
            
            return result.rename(self.factor_name)
        
        except DataNotAvailableError:
            raise
        except Exception as e:
            raise FactorCalculationError(
                self.factor_name, stock_code, str(e)
            ) from e
    
    def _detect_func_type(self, func: Callable) -> str:
        """检测函数签名类型"""
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if len(params) == 1:
            # 检查参数名是否是 ohlcv 相关的
            return 'ohlcv'
        elif len(params) == 3:
            return 'params'
        else:
            raise ValueError(
                f"因子函数参数数量不对: {len(params)}. "
                f"应该是 1 个 (DataFrame) 或 3 个 (stock_code, start_date, end_date)"
            )
    
    def _load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载 OHLCV 数据"""
        try:
            from src.data.data import load_oss_complex_stocks
            
            result = load_oss_complex_stocks(
                codes=[stock_code],
                start=start_date,
                end=end_date,
                fields=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not result:
                return pd.DataFrame()
            
            dfs = []
            for field, df in result.items():
                if not df.empty and stock_code in df.columns:
                    temp_df = pd.DataFrame({
                        'date': df.index,
                        field: df[stock_code]
                    })
                    dfs.append(temp_df)
            
            if not dfs:
                return pd.DataFrame()
            
            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on='date', how='outer')
            
            merged['date'] = pd.to_datetime(merged['date'])
            merged = merged.set_index('date').sort_index()
            
            return merged
        
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()


class FileFactorCalculator(FactorCalculator):
    """
    从文件加载因子
    
    从 CSV 文件中加载预先计算好的因子数据。
    文件格式: date, code, factor_value
    
    示例:
        calc = FileFactorCalculator('data/factors.csv', 'MY_FACTOR')
        result = calc.calculate('000001', '2024-01-01', '2024-12-31')
    """
    
    def __init__(self, file_path: str, factor_name: str):
        """
        初始化文件因子计算器
        
        Args:
            file_path: 因子文件路径（CSV 格式）
            factor_name: 文件中的因子列名
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不正确
        """
        self.file_path = file_path
        self.factor_name = factor_name
        self._data = None
        self._load_file()
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """从文件中获取因子值"""
        try:
            # 过滤数据
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            mask = (self._data['code'] == stock_code) & \
                   (self._data['date'] >= start) & \
                   (self._data['date'] <= end)
            
            filtered = self._data[mask]
            
            if filtered.empty:
                raise DataNotAvailableError(
                    stock_code, start_date, end_date,
                    "文件中无数据"
                )
            
            result = filtered.set_index('date')[self.factor_name]
            result = pd.to_numeric(result)
            
            return result
        
        except DataNotAvailableError:
            raise
        except Exception as e:
            raise FactorCalculationError(
                self.factor_name, stock_code, str(e)
            ) from e
    
    def _load_file(self):
        """加载文件"""
        try:
            self._data = pd.read_csv(self.file_path)
            
            # 验证必要的列
            required_cols = ['date', 'code', self.factor_name]
            missing_cols = [col for col in required_cols if col not in self._data.columns]
            
            if missing_cols:
                raise ValueError(f"文件缺少必要的列: {missing_cols}")
            
            # 转换数据类型
            self._data['date'] = pd.to_datetime(self._data['date'])
            self._data['code'] = self._data['code'].astype(str).str.zfill(6)
            self._data[self.factor_name] = pd.to_numeric(self._data[self.factor_name])
        
        except FileNotFoundError:
            raise FileNotFoundError(f"因子文件不存在: {self.file_path}")
        except Exception as e:
            raise ValueError(f"加载因子文件失败: {e}")


def create_factor_calculator(
    factor_name: str = None,
    factor_func: Callable = None,
    file_path: str = None,
    params: Optional[List] = None
) -> FactorCalculator:
    """
    因子计算器工厂函数
    
    优先级:
    1. 文件加载 (file_path)
    2. 内置因子 (factor_name, 包括 TALIB_*)
    3. 自定义函数 (factor_func)
    
    Args:
        factor_name: 因子名称
        factor_func: 自定义计算函数
        file_path: 因子文件路径
        params: Talib 函数的参数
    
    Returns:
        FactorCalculator: 相应的计算器实例
    
    Raises:
        ValueError: 参数不正确或都为空
    
    Examples:
        # 使用内置因子
        calc = create_factor_calculator('VOL10')
        
        # 使用 Talib 因子
        calc = create_factor_calculator('TALIB_RSI_14')
        
        # 使用自定义函数
        def my_factor(ohlcv):
            return ohlcv['close'] / ohlcv['close'].rolling(20).mean()
        calc = create_factor_calculator('MY_FACTOR', factor_func=my_factor)
        
        # 从文件加载
        calc = create_factor_calculator('MY_FACTOR', file_path='factors.csv')
    """
    
    # 优先级 1: 文件加载
    if file_path:
        if not factor_name:
            raise ValueError("使用 file_path 时必须提供 factor_name")
        return FileFactorCalculator(file_path, factor_name)
    
    # 优先级 2: 内置因子或 Talib 因子
    if factor_name:
        if factor_name.startswith('TALIB_'):
            return TalibFactorCalculator(factor_name, params)
        elif factor_name in BuiltinFactorCalculator.SUPPORTED_FACTORS:
            return BuiltinFactorCalculator(factor_name)
        else:
            raise ValueError(f"未知的因子: {factor_name}")
    
    # 优先级 3: 自定义函数
    if factor_func:
        return CustomFunctionCalculator(factor_name or 'CUSTOM', factor_func)
    
    # 都没有提供
    raise ValueError("必须提供 file_path、factor_name 或 factor_func 之一")
