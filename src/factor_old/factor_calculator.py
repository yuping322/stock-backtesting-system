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

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False


class FactorCalculator(ABC):
    """因子计算器基类"""
    
    @abstractmethod
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """
        计算因子值
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
                    'MINUS_DM': ['high', 'low'],  # 负向运动 - 不需要volume和close
            'PLUS_DM': ['high', 'low'],  # 正向运动 - 不需要volume和close         'MINUS_DM': ['high', 'low'],  # 负向运动 - 不需要volume和close
            'PLUS_DM': ['high', 'low'],  # 正向运动 - 不需要volume和close         'MINUS_DM': ['high', 'low'],  # 负向运动 - 不需要close
            'PLUS_DM': ['high', 'low'],  # 正向运动 - 不需要close    end_date: 结束日期
            
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
        # 检查是否是 TALIB 因子
        if factor_name.startswith('TALIB_'):
            self.factor_name = factor_name
            self.is_talib = True
            self.talib_func_name, self.talib_params = self._parse_talib_factor_name(factor_name)
            self.data_loader = data_loader
        elif factor_name not in self.BUILTIN_FACTORS:
            raise ValueError(f"未知的内置因子: {factor_name}")
        else:
            self.factor_name = factor_name
            self.factor_func = self.BUILTIN_FACTORS[factor_name]
            self.is_talib = False
            self.data_loader = data_loader
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算因子值"""
        # 加载 OHLCV 数据
        ohlcv_data = self.load_ohlcv(stock_code, start_date, end_date)
        
        if ohlcv_data.empty:
            return pd.Series(dtype=float)
        
        # 计算因子
        if self.is_talib:
            factor_values = self._calculate_talib_factor(ohlcv_data)
        else:
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
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"加载数据失败 {stock_code}: {e}")
            return pd.DataFrame()
    
    def _get_talib_min_periods(self) -> int:
        """获取 TALIB 函数的最小周期要求"""
        # 基于 TA-Lib 文档和常见用法定义最小周期
        min_periods_map = {
            # 移动平均类
            'SMA': 2,  # 简单移动平均
            'EMA': 2,  # 指数移动平均
            'DEMA': 2,  # 双重指数移动平均
            'TEMA': 2,  # 三重指数移动平均
            'WMA': 2,  # 加权移动平均
            'KAMA': 30,  # 考夫曼自适应移动平均
            'MAMA': 32,  # MESA自适应移动平均
            'T3': 2,  # T3移动平均
            'TRIX': 2,  # 三重指数平滑移动平均
            'TRIMA': 2,  # 三角移动平均
            'MA': 2,  # 所有移动平均
            
            # 动量指标
            'RSI': 14,  # 相对强弱指数
            'STOCH': 14,  # 随机指标 (K,D周期)
            'STOCHF': 14,  # 快速随机指标
            'STOCHRSI': 14,  # 随机强弱指标
            'WILLR': 14,  # 威廉指标
            'CCI': 20,  # 商品通道指数
            'MOM': 10,  # 动量
            'ROC': 10,  # 变动率
            'ROCP': 10,  # 变动率百分比
            'ROCR': 10,  # 变动率比率
            'ROCR100': 10,  # 变动率比率100
            'CMO': 14,  # 钱德动量振荡器
            'MACD': 26,  # MACD (默认26周期)
            'MACDEXT': 26,  # MACD扩展
            'MACDFIX': 26,  # MACD固定
            'APO': 26,  # 绝对价格震荡器
            'PPO': 26,  # 百分比价格震荡器
            
            # 波动率指标
            'ATR': 14,  # 平均真实波幅
            'NATR': 14,  # 归一化平均真实波幅
            'TRANGE': 2,  # 真实波幅
            'VAR': 5,  # 方差
            'STDDEV': 5,  # 标准差
            'AVGDEV': 2,  # 平均偏差
            
            # 趋向指标
            'ADX': 14,  # 平均趋向指数
            'ADXR': 14 + 14,  # 平均趋向指数评级 (ADX周期 + 额外周期) = 28, 但实际上需要更长
            'DX': 14,  # 趋向指数
            'MINUS_DI': 14,  # 负向指标
            'MINUS_DM': 14,  # 负向运动
            'PLUS_DI': 14,  # 正向指标
            'PLUS_DM': 14,  # 正向运动
            
            # 布林带
            'BBANDS': 20,  # 布林带 (默认20周期)
            
            # 阿隆指标
            'AROON': 14,  # 阿隆指标
            'AROONOSC': 14,  # 阿隆振荡器
            
            # 其他
            'OBV': 2,  # 能量潮
            'AD': 2,  # 累积/派发线
            'ADOSC': 3,  # 震荡指标 (默认3,10)
            'MFI': 14,  # 资金流量指数
            'IMI': 14,  # 内部动量指数
            'BALANCE': 2,  # 平衡量
            'AVGPRICE': 2,  # 平均价格
            'MEDPRICE': 2,  # 中间价格
            'TYPPRICE': 2,  # 典型价格
            'WCLPRICE': 2,  # 加权收盘价
            'MIDPOINT': 2,  # 中点
            'MIDPRICE': 2,  # 中间价格
            'MINMAX': 2,  # 最小最大
            'MINMAXINDEX': 2,  # 最小最大索引
            
            # 希尔伯特变换
            'HT_DCPERIOD': 32,  # 希尔伯特变换-主导周期
            'HT_DCPHASE': 64,  # 希尔伯特变换-主导阶段
            'HT_PHASOR': 32,  # 希尔伯特变换-相量组件
            'HT_SINE': 64,  # 希尔伯特变换-正弦波
            'HT_TRENDMODE': 64,  # 希尔伯特变换-趋势模式
            
            # 线性回归
            'LINEARREG': 2,  # 线性回归
            'LINEARREG_ANGLE': 2,  # 线性回归角度
            'LINEARREG_INTERCEPT': 2,  # 线性回归截距
            'LINEARREG_SLOPE': 2,  # 线性回归斜率
            'TSF': 2,  # 时间序列预测
            
            # 其他高级指标
            'BETA': 30,  # Beta系数
            'CORREL': 30,  # 相关系数
            'ACCBANDS': 14,  # 加速带 - 需要至少 timeperiod 个周期
        }
        
        # 特殊处理 ADXR - 它基于 ADX，需要更长的预热期
        if self.talib_func_name.upper() == 'ADXR':
            # ADXR 是 ADX 的平滑版本，需要 ADX 的周期加上额外的平滑周期
            # 通常 ADX 使用 14 周期，ADXR 使用额外的 14 周期进行平滑
            # 测试显示需要约50个数据点才能开始产生有效值
            return 50  # 基于实际测试结果调整
        
        return min_periods_map.get(self.talib_func_name.upper(), 2)  # 默认最少2个周期
    
    def _parse_talib_factor_name(self, factor_name: str):
        """解析 TALIB 因子名称，返回函数名和参数"""
        if not factor_name.startswith('TALIB_'):
            raise ValueError(f"不是 TALIB 因子: {factor_name}")
        
        # 去掉 TALIB_ 前缀
        talib_part = factor_name[6:]  # 'TALIB_ACCBANDS_14' -> 'ACCBANDS_14'
        
        # 分割函数名和参数
        parts = talib_part.split('_')
        
        # 特殊处理复合函数名
        if talib_part.startswith('HT_'):
            # HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE
            func_name = '_'.join(parts[:2])  # HT_DCPERIOD
            params = parts[2:]
        elif talib_part.startswith('MINUS_'):
            # MINUS_DI, MINUS_DM
            func_name = '_'.join(parts[:2])  # MINUS_DI
            params = parts[2:]
        elif talib_part.startswith('PLUS_'):
            # PLUS_DI, PLUS_DM
            func_name = '_'.join(parts[:2])  # PLUS_DI
            params = parts[2:]
        elif talib_part.startswith('LINEARREG_'):
            # LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE
            func_name = '_'.join(parts[:2])  # LINEARREG_ANGLE
            params = parts[2:]
        else:
            # 普通函数名
            func_name = parts[0]
            params = parts[1:]
        
        # 解析参数
        parsed_params = []
        for part in params:
            try:
                # 尝试转换为数字
                if '.' in part:
                    parsed_params.append(float(part))
                else:
                    parsed_params.append(int(part))
            except ValueError:
                # 如果不是数字，保持字符串
                parsed_params.append(part)
        
        return func_name, parsed_params
    
    def _calculate_talib_factor(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """计算 TALIB 因子"""
        if not TALIB_AVAILABLE:
            raise ImportError("需要安装 TA-Lib: pip install TA-Lib")
        
        # 获取 TALIB 函数
        talib_func = getattr(talib, self.talib_func_name.upper(), None)
        if talib_func is None:
            raise ValueError(f"未知的 TALIB 函数: {self.talib_func_name}")
        
        # 检查数据长度是否足够
        min_periods = self._get_talib_min_periods()
        if len(ohlcv_data) < min_periods:
            print(f"⚠️  警告：数据长度 {len(ohlcv_data)} 不足以计算 {self.talib_func_name} (需要至少 {min_periods} 个数据点)")
            return pd.Series(dtype=float)
        
        # 准备输入数据
        # TALIB 函数通常需要 numpy arrays，根据函数名确定需要的参数
        inputs = {}
        
        # 常见的参数映射 - 指定每个函数需要的参数
        param_mapping = {
            'AD': ['high', 'low', 'close', 'volume'],
            'ADOSC': ['high', 'low', 'close', 'volume'],
            'OBV': ['close', 'volume'],
            'MFI': ['high', 'low', 'close', 'volume'],
            'RSI': ['close'],  # RSI 只需收盘价
            'MACD': ['close'],  # MACD 只需收盘价
            'SMA': ['close'],  # 简单移动平均
            'EMA': ['close'],  # 指数移动平均
            'BBANDS': ['close'],  # 布林带
            'STOCH': ['high', 'low', 'close'],  # 随机指标
            'STOCHF': ['high', 'low', 'close'],  # 快速随机指标
            'STOCHRSI': ['close'],  # 随机强弱指标
            'WILLR': ['high', 'low', 'close'],  # 威廉指标
            'CCI': ['high', 'low', 'close'],  # 商品通道指数
            'ATR': ['high', 'low', 'close'],  # 平均真实波幅
            'NATR': ['high', 'low', 'close'],  # 归一化平均真实波幅
            'ROC': ['close'],  # 变动率
            'ROCP': ['close'],  # 变动率百分比
            'ROCR': ['close'],  # 变动率比率
            'ROCR100': ['close'],  # 变动率比率100
            'MOM': ['close'],  # 动量
            'TSF': ['close'],  # 时间序列预测
            'VAR': ['close'],  # 方差
            'STDDEV': ['close'],  # 标准差
            'BETA': ['close'],  # Beta系数（通常需要基准）
            'CORREL': ['close'],  # 相关系数（通常需要两组数据）
            'LINEARREG': ['close'],  # 线性回归
            'LINEARREG_ANGLE': ['close'],  # 线性回归角度
            'LINEARREG_INTERCEPT': ['close'],  # 线性回归截距
            'LINEARREG_SLOPE': ['close'],  # 线性回归斜率
            'TSF': ['close'],  # 时间序列预测
            'TEMA': ['close'],  # 三重指数移动平均
            'TRIMA': ['close'],  # 三角移动平均
            'WMA': ['close'],  # 加权移动平均
            'DEMA': ['close'],  # 双重指数移动平均
            'KAMA': ['close'],  # 考夫曼自适应移动平均
            'MAMA': ['close'],  # MESA自适应移动平均
            'T3': ['close'],  # T3移动平均
            'TRIX': ['close'],  # 三重指数平滑移动平均
            'ACCBANDS': ['high', 'low', 'close'],  # 加速带 - 不需要volume
            'APO': ['close'],  # 绝对价格震荡器
            'PPO': ['close'],  # 百分比价格震荡器
            'CMO': ['close'],  # 钱德动量振荡器
            'AROON': ['high', 'low'],  # 阿隆指标
            'AROONOSC': ['high', 'low'],  # 阿隆振荡器
            'BALANCE': ['close'],  # 平衡量
            'AVGPRICE': ['open', 'high', 'low', 'close'],  # 平均价格
            'MEDPRICE': ['high', 'low'],  # 中间价格
            'TYPPRICE': ['high', 'low', 'close'],  # 典型价格
            'WCLPRICE': ['high', 'low', 'close'],  # 加权收盘价
            'ADX': ['high', 'low', 'close'],  # 平均趋向指数 - 不需要volume
            'ADXR': ['high', 'low', 'close'],  # 平均趋向指数评级 - 不需要volume
            'DX': ['high', 'low', 'close'],  # 趋向指数 - 不需要volume
            'MINUS_DI': ['high', 'low', 'close'],  # 负向指标 - 不需要volume
            'MINUS_DM': ['high', 'low'],  # 负向运动 - 不需要volume和close
            'PLUS_DI': ['high', 'low', 'close'],  # 正向指标 - 不需要volume
            'PLUS_DM': ['high', 'low'],  # 正向运动 - 不需要volume和close
            'TRANGE': ['high', 'low', 'close'],  # 真实波幅 - 不需要volume
            'HT_DCPERIOD': ['close'],  # 希尔伯特变换-主导周期
            'HT_DCPHASE': ['close'],  # 希尔伯特变换-主导阶段
            'HT_PHASOR': ['close'],  # 希尔伯特变换-相量组件
            'HT_SINE': ['close'],  # 希尔伯特变换-正弦波
            'HT_TRENDMODE': ['close'],  # 希尔伯特变换-趋势模式
            'AVGDEV': ['close'],  # 平均偏差
            'MA': ['close'],  # 所有移动平均
            'MIDPOINT': ['close'],  # 中点
            'MIDPRICE': ['high', 'low'],  # 中间价格
            'MINMAX': ['close'],  # 最小最大
            'MINMAXINDEX': ['close'],  # 最小最大索引
            'IMI': ['close', 'volume'],  # 内部动量指数
            'MACDEXT': ['close'],  # MACD扩展
            'MACDFIX': ['close'],  # MACD固定
            'MAMA': ['close'],  # MESA自适应移动平均
        }
        
        required_params = param_mapping.get(self.talib_func_name.upper(), ['open', 'high', 'low', 'close', 'volume'])
        
        for param in required_params:
            if param in ohlcv_data.columns:
                # 确保数据类型为 float64
                inputs[param] = ohlcv_data[param].values.astype(np.float64)
        
        # 调用 TALIB 函数
        try:
            if self.talib_params:
                # 有参数的调用 - 根据函数类型使用不同的参数传递方式
                func_name_upper = self.talib_func_name.upper()
                
                if func_name_upper in ['STOCH']:
                    # STOCH: fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype
                    if len(self.talib_params) >= 5:
                        result = talib_func(*inputs.values(), 
                                          fastk_period=self.talib_params[0],
                                          slowk_period=self.talib_params[1], 
                                          slowk_matype=self.talib_params[2],
                                          slowd_period=self.talib_params[3],
                                          slowd_matype=self.talib_params[4])
                    elif len(self.talib_params) >= 3:
                        result = talib_func(*inputs.values(), 
                                          fastk_period=self.talib_params[0],
                                          slowk_period=self.talib_params[1], 
                                          slowd_period=self.talib_params[2])
                    else:
                        result = talib_func(*inputs.values(), 
                                          fastk_period=self.talib_params[0],
                                          slowk_period=self.talib_params[1], 
                                          slowd_period=self.talib_params[1])
                elif func_name_upper in ['STOCHF']:
                    # STOCHF: fastk_period, fastd_period, fastd_matype
                    if len(self.talib_params) >= 3:
                        result = talib_func(*inputs.values(), 
                                          fastk_period=self.talib_params[0],
                                          fastd_period=self.talib_params[1],
                                          fastd_matype=self.talib_params[2])
                    else:
                        result = talib_func(*inputs.values(), 
                                          fastk_period=self.talib_params[0],
                                          fastd_period=self.talib_params[1])
                elif func_name_upper in ['STOCHRSI']:
                    # STOCHRSI: timeperiod, fastk_period, fastd_period, fastd_matype
                    if len(self.talib_params) >= 4:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0],
                                          fastk_period=self.talib_params[1],
                                          fastd_period=self.talib_params[2],
                                          fastd_matype=self.talib_params[3])
                    else:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0],
                                          fastk_period=self.talib_params[1],
                                          fastd_period=self.talib_params[2])
                elif func_name_upper in ['T3']:
                    # T3: timeperiod, vfactor
                    if len(self.talib_params) >= 2:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0],
                                          vfactor=self.talib_params[1])
                    else:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0])
                elif func_name_upper in ['BBANDS']:
                    # BBANDS: timeperiod, nbdevup, nbdevdn, matype
                    if len(self.talib_params) >= 4:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0],
                                          nbdevup=self.talib_params[1],
                                          nbdevdn=self.talib_params[2],
                                          matype=self.talib_params[3])
                    elif len(self.talib_params) >= 3:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0],
                                          nbdevup=self.talib_params[1],
                                          nbdevdn=self.talib_params[2])
                    else:
                        result = talib_func(*inputs.values(), 
                                          timeperiod=self.talib_params[0])
                elif func_name_upper in ['ACCBANDS']:
                    # ACCBANDS: timeperiod (只有一个参数)
                    result = talib_func(*inputs.values(), timeperiod=self.talib_params[0])
                elif func_name_upper in ['MACD', 'MACDEXT']:
                    # MACD, MACDEXT: fastperiod, slowperiod, signalperiod
                    if len(self.talib_params) >= 3:
                        result = talib_func(*inputs.values(), 
                                          fastperiod=self.talib_params[0],
                                          slowperiod=self.talib_params[1],
                                          signalperiod=self.talib_params[2])
                    else:
                        result = talib_func(*inputs.values(), 
                                          fastperiod=self.talib_params[0],
                                          slowperiod=self.talib_params[1])
                elif func_name_upper in ['MACDFIX']:
                    # MACDFIX: signalperiod
                    result = talib_func(*inputs.values(), signalperiod=self.talib_params[0])
                elif func_name_upper in ['MAMA']:
                    # MAMA: fastlimit, slowlimit
                    if len(self.talib_params) >= 2:
                        result = talib_func(*inputs.values(), 
                                          fastlimit=self.talib_params[0],
                                          slowlimit=self.talib_params[1])
                    else:
                        result = talib_func(*inputs.values())
                elif func_name_upper in ['MINUS_DM', 'PLUS_DM']:
                    # MINUS_DM, PLUS_DM: high, low, timeperiod (位置参数)
                    result = talib_func(*inputs.values(), self.talib_params[0])
                elif func_name_upper in ['ADX', 'ADXR', 'DX', 'MINUS_DI', 'PLUS_DI']:
                    # ADX系列: high, low, close, timeperiod (位置参数)
                    result = talib_func(*inputs.values(), self.talib_params[0])
                else:
                    # 默认处理：单个参数作为 timeperiod
                    if len(self.talib_params) == 1:
                        result = talib_func(*inputs.values(), timeperiod=self.talib_params[0])
                    else:
                        # 多个参数，尝试位置参数传递
                        all_args = list(inputs.values()) + self.talib_params
                        result = talib_func(*all_args)
            else:
                # 无参数调用
                result = talib_func(*inputs.values())
            
            # 处理返回值
            if isinstance(result, tuple):
                # 特殊处理某些返回多个值的函数
                func_name_upper = self.talib_func_name.upper()
                if func_name_upper.startswith('AROON') and len(result) == 2:
                    # AROON 返回 (AROON_Up, AROON_Down)，我们使用 AROON_Up - AROON_Down 作为因子
                    aroon_up, aroon_down = result
                    result = aroon_up - aroon_down
                elif func_name_upper.startswith('BBANDS') and len(result) == 3:
                    # BBANDS 返回 (Upper, Middle, Lower)，我们使用 Middle Band 作为因子
                    upper, middle, lower = result
                    result = middle
                elif func_name_upper == 'ACCBANDS' and len(result) == 3:
                    # ACCBANDS 返回 (Upper, Middle, Lower)，我们使用 Middle Band 作为因子
                    upper, middle, lower = result
                    result = middle
                elif func_name_upper in ['MACD', 'MACDEXT', 'MACDFIX'] and len(result) == 3:
                    # MACD系列 返回 (MACD, Signal, Histogram)，我们使用 MACD - Signal 作为因子
                    macd, signal, histogram = result
                    result = macd - signal
                elif func_name_upper in ['STOCH', 'STOCHF', 'STOCHRSI'] and len(result) == 2:
                    # STOCH系列 返回 (K, D)，我们使用 K - D 作为因子
                    k_value, d_value = result
                    result = k_value - d_value
                else:
                    # 如果返回多个值，取第一个（通常是主要的指标值）
                    result = result[0]
            
            # 转换为 pandas Series
            factor_series = pd.Series(result, index=ohlcv_data.index)
            
            return factor_series
            
        except Exception as e:
            print(f"TALIB 函数调用失败 {self.talib_func_name}: {e}")
            return pd.Series(dtype=float)


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
    
    # 优先级3: 使用内置因子（包括 TALIB 因子）
    if factor_name and (factor_name in BuiltinFactorCalculator.BUILTIN_FACTORS or factor_name.startswith('TALIB_')):
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
