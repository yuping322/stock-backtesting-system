"""
因子计算接口模块

支持：
1. 自定义因子计算函数
2. OHLCV 格式数据接口
3. 从不同数据源读取数据
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod


class FactorCalculator(ABC):
    """因子计算器基类"""
    
    @abstractmethod
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """
        计算因子值
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.Series: 因子值序列，索引为日期
        """
        pass


class OHLCVFactorCalculator(FactorCalculator):
    """基于 OHLCV 数据的因子计算器"""
    
    def __init__(self, factor_func: Callable[[pd.DataFrame], pd.Series], data_loader=None):
        """
        初始化 OHLCV 因子计算器
        
        Args:
            factor_func: 因子计算函数，接受 OHLCV DataFrame，返回因子值 Series
            data_loader: 数据加载器，需要实现 load_ohlcv(code, start, end) 方法
        """
        self.factor_func = factor_func
        self.data_loader = data_loader
        
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """
        计算因子值
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.Series: 因子值序列，索引为日期
        """
        # 加载 OHLCV 数据
        ohlcv_data = self.load_ohlcv(stock_code, start_date, end_date)
        
        if ohlcv_data.empty:
            return pd.Series(dtype=float)
        
        # 计算因子
        factor_values = self.factor_func(ohlcv_data)
        
        return factor_values
    
    def load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载 OHLCV 数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: OHLCV 数据，列包含 open, high, low, close, volume
        """
        if self.data_loader is None:
            # 使用默认数据加载器
            return self._default_load_ohlcv(stock_code, start_date, end_date)
        else:
            # 使用自定义数据加载器
            return self.data_loader.load_ohlcv(stock_code, start_date, end_date)
    
    def _default_load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """默认数据加载方法"""
        # 导入 data 模块
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import data
        
        try:
            # 尝试使用 data.py 的数据接口
            df = data.load_oss_complex_stocks([stock_code], start=start_date, end=end_date, fields='all')
            
            if isinstance(df, dict) and stock_code in df:
                # 如果返回的是字典，提取股票数据
                stock_data = df[stock_code]
                if isinstance(stock_data, pd.DataFrame):
                    return stock_data
            
            return pd.DataFrame()
        except Exception as e:
            print(f"加载数据失败 {stock_code}: {e}")
            return pd.DataFrame()


class FileFactorCalculator(FactorCalculator):
    """从文件加载因子的计算器"""
    
    def __init__(self, file_path: str, factor_name: str):
        """
        初始化文件因子计算器
        
        Args:
            file_path: 因子文件路径
            factor_name: 因子列名
        """
        self.file_path = file_path
        self.factor_name = factor_name
        self._cache = None
        self._file_date_range = None  # 文件的实际日期范围
        self._file_stocks = None  # 文件中的股票列表
    
    def get_file_date_range(self):
        """获取因子文件的实际日期范围"""
        if self._cache is None:
            self._cache = self._load_file()
        if self._file_date_range:
            return self._file_date_range[0], self._file_date_range[1]
        return None, None
    
    def get_file_stocks(self):
        """获取因子文件中的股票列表"""
        if self._cache is None:
            self._cache = self._load_file()
        if self._file_stocks is None and not self._cache.empty:
            self._file_stocks = sorted(self._cache.index.get_level_values('code').unique().tolist())
        return self._file_stocks if self._file_stocks else []
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """从文件加载因子值"""
        # 第一次调用时加载整个文件并缓存
        if self._cache is None:
            self._cache = self._load_file()
        
        if self._cache.empty:
            return pd.Series(dtype=float)
        
        # 使用因子文件中的实际日期范围（如果调用者传入的日期范围超出文件范围）
        if self._file_date_range:
            file_start, file_end = self._file_date_range
            # 如果调用者传入的日期范围超出文件范围，使用文件的实际范围
            actual_start = max(pd.Timestamp(start_date), file_start)
            actual_end = min(pd.Timestamp(end_date), file_end)
        else:
            actual_start = pd.Timestamp(start_date)
            actual_end = pd.Timestamp(end_date)
        
        # 过滤股票代码和日期范围
        try:
            # 标准化股票代码
            code_normalized = self._normalize_code(stock_code)
            
            # 标准化股票代码（补齐6位）
            code_normalized = code_normalized.zfill(6)
            
            # 筛选数据：使用日期比较而不是精确匹配，因为因子文件可能只包含交易日
            # 使用文件的实际日期范围（可能比调用者传入的范围更小）
            filtered = self._cache.loc[
                (self._cache.index.get_level_values('date') >= actual_start) &
                (self._cache.index.get_level_values('date') <= actual_end) &
                (self._cache.index.get_level_values('code') == code_normalized)
            ]
            
            if not filtered.empty:
                # 获取因子值的Series
                factor_series = filtered[self.factor_name]
                # 如果索引是MultiIndex，只保留日期部分作为索引
                if isinstance(factor_series.index, pd.MultiIndex):
                    # 重置索引，只保留日期
                    factor_series = factor_series.reset_index(level='code', drop=True)
                return factor_series
            else:
                return pd.Series(dtype=float)
                
        except Exception as e:
            print(f"从文件加载因子失败 {stock_code}: {e}")
            return pd.Series(dtype=float)
    
    def _load_file(self) -> pd.DataFrame:
        """加载因子文件"""
        try:
            # 读取CSV文件（不指定dtype，让pandas自动推断）
            df = pd.read_csv(self.file_path)
            
            # 检查必要的列
            if 'date' not in df.columns or 'code' not in df.columns:
                raise ValueError(f"文件必须包含 'date' 和 'code' 列")
            
            if self.factor_name not in df.columns:
                raise ValueError(f"文件必须包含因子列 '{self.factor_name}'")
            
            # 日期列转换为 datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # 标准化代码列：先转换为字符串，去掉后缀，再补齐6位
            # 处理数字代码（如1, 2, 63）和字符串代码（如'000001', '000001.XSHG'）
            df['code'] = df['code'].astype(str).str.strip()  # 转为字符串并去除空格
            df['code'] = df['code'].apply(self._normalize_code)  # 去掉.XSHG/.XSHE后缀
            df['code'] = df['code'].str.zfill(6)  # 补齐6位（如 '1' -> '000001', '63' -> '000063'）
            
            # 设置 MultiIndex
            df = df.set_index(['date', 'code']).sort_index()
            
            # 记录文件的实际日期范围和股票列表
            if not df.empty:
                dates = df.index.get_level_values('date')
                self._file_date_range = (dates.min(), dates.max())
                self._file_stocks = sorted(df.index.get_level_values('code').unique().tolist())
            
            return df
            
        except Exception as e:
            print(f"加载因子文件失败 {self.file_path}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _normalize_code(self, code: str) -> str:
        """标准化股票代码"""
        # 转换为字符串，去掉后缀，去除空格
        code = str(code).strip().replace('.XSHG', '').replace('.XSHE', '')
        # 如果代码是纯数字，直接返回（后面会zfill补齐）
        return code


class CustomFactorCalculator(FactorCalculator):
    """自定义因子计算器"""
    
    def __init__(self, calculate_func: Callable[[str, str, str], pd.Series]):
        """
        初始化自定义因子计算器
        
        Args:
            calculate_func: 自定义计算函数，参数为 (stock_code, start_date, end_date)
        """
        self.calculate_func = calculate_func
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算因子值"""
        return self.calculate_func(stock_code, start_date, end_date)


class BuiltinFactorCalculator(FactorCalculator):
    """内置因子计算器"""
    
    BUILTIN_FACTORS = {
        'VOL10': lambda ohlcv: ohlcv['volume'].rolling(10).mean(),
        'VOL20': lambda ohlcv: ohlcv['volume'].rolling(20).mean(),
        'VPT_12': lambda ohlcv: (ohlcv['close'].pct_change() * ohlcv['volume']).rolling(12).sum(),
        'RSI_14': lambda ohlcv: _calculate_rsi(ohlcv['close'], 14),
        'MA_5': lambda ohlcv: ohlcv['close'].rolling(5).mean(),
        'MA_10': lambda ohlcv: ohlcv['close'].rolling(10).mean(),
        'MA_20': lambda ohlcv: ohlcv['close'].rolling(20).mean(),
        'VOLUME_RATIO': lambda ohlcv: ohlcv['volume'] / ohlcv['volume'].rolling(20).mean(),
        'PRICE_CHANGE': lambda ohlcv: ohlcv['close'].pct_change(),
        'HIGH_LOW_RATIO': lambda ohlcv: (ohlcv['high'] - ohlcv['low']) / ohlcv['close'],
    }
    
    def __init__(self, factor_name: str, data_loader=None):
        """
        初始化内置因子计算器
        
        Args:
            factor_name: 因子名称
            data_loader: 数据加载器
        """
        if factor_name not in self.BUILTIN_FACTORS:
            raise ValueError(f"未知的内置因子: {factor_name}")
        
        self.factor_name = factor_name
        self.factor_func = self.BUILTIN_FACTORS[factor_name]
        self.data_loader = data_loader
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算因子值"""
        # 加载 OHLCV 数据
        ohlcv_data = self.load_ohlcv(stock_code, start_date, end_date)
        
        if ohlcv_data.empty:
            return pd.Series(dtype=float)
        
        # 计算因子
        factor_values = self.factor_func(ohlcv_data)
        
        return factor_values
    
    def load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载 OHLCV 数据"""
        if self.data_loader is None:
            return self._default_load_ohlcv(stock_code, start_date, end_date)
        else:
            return self.data_loader.load_ohlcv(stock_code, start_date, end_date)
    
    def _default_load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """默认数据加载方法"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import data
        
        try:
            df_dict = data.load_oss_complex_stocks([stock_code], start=start_date, end=end_date, fields='all')
            
            if isinstance(df_dict, dict):
                # 从字典中提取各字段数据，组合成 DataFrame
                ohlcv_data = {}
                required_fields = ['open', 'high', 'low', 'close', 'volume']
                
                for field in required_fields:
                    if field in df_dict:
                        field_df = df_dict[field]
                        if isinstance(field_df, pd.DataFrame) and stock_code in field_df.columns:
                            ohlcv_data[field] = field_df[stock_code]
                
                if ohlcv_data:
                    result_df = pd.DataFrame(ohlcv_data)
                    return result_df
            
            return pd.DataFrame()
        except Exception as e:
            print(f"加载数据失败 {stock_code}: {e}")
            return pd.DataFrame()


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _find_factor_file_in_dir(factor_dir: str, factor_name: str) -> Optional[str]:
    """
    在目录中查找包含指定因子的CSV文件
    
    Args:
        factor_dir: 因子文件目录
        factor_name: 因子名称
        
    Returns:
        找到的文件路径，如果未找到返回None
    """
    from pathlib import Path
    
    factor_dir = Path(factor_dir)
    if not factor_dir.exists():
        return None
    
    # 查找所有CSV文件
    csv_files = list(factor_dir.glob('*.csv'))
    
    # 按修改时间排序（最新的优先）
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # 检查每个文件是否包含该因子
    for csv_file in csv_files:
        try:
            # 只读取列名，不读取全部数据
            df_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()
            if 'date' in df_columns and 'code' in df_columns and factor_name in df_columns:
                return str(csv_file)
        except Exception:
            continue
    
    return None


# 辅助函数：创建因子计算器
def create_factor_calculator(
    factor_name: Optional[str] = None,
    factor_func: Optional[Callable] = None,
    data_loader=None,
    file_path: Optional[str] = None,
    factor_dir: Optional[str] = None
) -> FactorCalculator:
    """
    创建因子计算器
    
    Args:
        factor_name: 内置因子名称、文件中的因子列名或qlib因子名（如ROC5, MA10等）
        factor_func: 自定义因子计算函数
        data_loader: 数据加载器
        file_path: 因子文件路径（优先使用）
        factor_dir: 因子文件目录，会自动查找包含指定因子的CSV文件
        
    Returns:
        FactorCalculator: 因子计算器实例
        
    Examples:
        # 从目录自动查找因子文件
        calc = create_factor_calculator(factor_name='ROC5', factor_dir='./factors')
        
        # 直接指定文件路径
        calc = create_factor_calculator(factor_name='ROC5', file_path='./factors/Alpha158_20240101_20241231.csv')
        
        # 使用内置因子
        calc = create_factor_calculator(factor_name='VOL10')
    """
    # 优先级1: 直接指定文件路径
    if file_path:
        if not factor_name:
            raise ValueError("使用 file_path 时必须提供 factor_name（因子列名）")
        return FileFactorCalculator(file_path, factor_name)
    
    # 优先级2: 从目录查找因子文件
    if factor_dir and factor_name:
        found_file = _find_factor_file_in_dir(factor_dir, factor_name)
        if found_file:
            return FileFactorCalculator(found_file, factor_name)
        # 如果目录中没找到，继续尝试其他方式
    
    # 优先级3: 使用内置因子
    if factor_name and factor_name in BuiltinFactorCalculator.BUILTIN_FACTORS:
        return BuiltinFactorCalculator(factor_name, data_loader)
    
    # 优先级4: 使用自定义函数
    if factor_func:
        if isinstance(factor_func, Callable):
            # 检查函数签名
            import inspect
            sig = inspect.signature(factor_func)
            if len(sig.parameters) == 1:
                # 接受 DataFrame 的函数
                return OHLCVFactorCalculator(factor_func, data_loader)
            elif len(sig.parameters) == 3:
                # 接受 (code, start, end) 的函数
                return CustomFactorCalculator(factor_func)
            else:
                raise ValueError("因子函数必须接受 1 个参数 (DataFrame) 或 3 个参数 (code, start, end)")
        else:
            raise ValueError("factor_func 必须是可调用对象")
    
    # 如果指定了factor_dir但没找到文件，报错
    if factor_dir and factor_name:
        raise ValueError(f"在目录 {factor_dir} 中未找到包含因子 '{factor_name}' 的文件")
    
    raise ValueError("必须提供 factor_name（配合factor_dir或file_path）或 factor_func")


# 示例用法
if __name__ == '__main__':
    # 示例 1: 使用内置因子
    print("示例 1: 使用内置因子 VOL10")
    calc = create_factor_calculator(factor_name='VOL10')
    print(f"创建成功: {type(calc).__name__}")
    
    # 示例 2: 使用自定义 OHLCV 因子函数
    print("\n示例 2: 使用自定义 OHLCV 因子函数")
    def my_factor(ohlcv):
        return ohlcv['close'] / ohlcv['close'].rolling(20).mean()
    
    calc = create_factor_calculator(factor_func=my_factor)
    print(f"创建成功: {type(calc).__name__}")
    
    # 示例 3: 使用完全自定义的计算函数
    print("\n示例 3: 使用完全自定义的计算函数")
    def custom_calc(code, start, end):
        # 可以在这里自己处理数据
        return pd.Series([1, 2, 3], index=pd.date_range(start, periods=3))
    
    calc = create_factor_calculator(factor_func=custom_calc)
    print(f"创建成功: {type(calc).__name__}")
    
    # 示例 4: 从文件加载因子
    print("\n示例 4: 从文件加载因子")
    # calc = create_factor_calculator(file_path='data/factors.csv', factor_name='MY_FACTOR')
    # print(f"创建成功: {type(calc).__name__}")
    print("需要提供有效的文件路径")
