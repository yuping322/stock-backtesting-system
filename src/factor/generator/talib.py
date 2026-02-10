"""
TA-Lib 因子生成器

支持 TA-Lib 库中的 200+ 技术指标

参考 factor_old/generate_talib_factors.py 的实现方式
"""

import os
import sys
import pandas as pd
import numpy as np
import inspect
from typing import List, Optional, Dict

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False

from src.factor.generator._base import (
    FactorGenerator,
    load_ohlcv_data,
    format_factor_dataframe,
    extend_lookback_start_date,
    clamp_dataframe_to_date_range,
)
from src.factor.utils import validate_factor_names


class TalibFactorListGenerator:
    """TA-Lib 因子列表生成器 - 参考 factor_old 实现"""
    
    @staticmethod
    def get_talib_functions() -> List[str]:
        """获取所有 TA-Lib 指标函数名称，过滤掉不需要的函数"""
        functions = []
        for attr in dir(talib):
            if not attr.startswith('_'):
                obj = getattr(talib, attr)
                if callable(obj):
                    # 过滤掉不需要的函数
                    if not attr.startswith('stream_') and \
                       not attr.startswith('wrapped_') and \
                       not attr.startswith('chain') and \
                       not attr.startswith('get_') and \
                       not attr.startswith('set_') and \
                       attr not in ['wraps']:
                        functions.append(attr)
        return sorted(functions)
    
    @staticmethod
    def get_function_signature(func_name: str) -> Dict[str, any]:
        """获取函数签名信息"""
        func = getattr(talib, func_name)
        try:
            sig = inspect.signature(func)
            params = {}
            for name, param in sig.parameters.items():
                if name != 'price':  # 跳过 price 参数（通常是第一个参数）
                    if param.default != inspect.Parameter.empty:
                        params[name] = param.default
                    else:
                        params[name] = None
            return params
        except:
            return {}
    
    @staticmethod
    def generate_common_parameters(func_name: str) -> List[List[int]]:
        """为指标函数生成常见的参数组合"""
        # 常见的周期参数
        common_periods = [5, 10, 14, 20, 21, 26, 30, 50, 60]

        # 特殊指标的参数组合 - 完全参考 factor_old
        special_params = {
            'SMA': [[p] for p in common_periods],
            'EMA': [[p] for p in common_periods],
            'WMA': [[p] for p in common_periods],
            'DEMA': [[p] for p in common_periods],
            'TEMA': [[p] for p in common_periods],
            'TRIMA': [[p] for p in common_periods],
            'KAMA': [[p] for p in common_periods],
            'MAMA': [[0.5, 0.05]],  # 默认参数
            'T3': [[p, 0.7] for p in common_periods],  # period, vfactor

            # 动量指标
            'RSI': [[p] for p in [6, 14, 21]],
            'STOCHRSI': [[p, 14, 3, 3] for p in [14]],  # timeperiod, fastk_period, fastd_period, fastd_matype
            'MOM': [[p] for p in common_periods],
            'ROC': [[p] for p in common_periods],
            'ROCP': [[p] for p in common_periods],
            'ROCR': [[p] for p in common_periods],
            'ROCR100': [[p] for p in common_periods],
            'TRIX': [[p] for p in common_periods],
            'WILLR': [[p] for p in [14, 21]],
            'CCI': [[p] for p in [14, 20]],
            'CMO': [[p] for p in common_periods],
            'PPO': [[12, 26, 0]],  # fastperiod, slowperiod, matype
            'APO': [[12, 26, 0]],  # fastperiod, slowperiod, matype

            # 波动率指标
            'ATR': [[p] for p in [14, 21]],
            'NATR': [[p] for p in [14, 21]],
            'TRANGE': [[]],  # 无参数
            'ADX': [[p] for p in [14, 21]],
            'ADXR': [[p] for p in [14, 21]],
            'DX': [[p] for p in [14, 21]],
            'PLUS_DI': [[p] for p in [14, 21]],
            'PLUS_DM': [[p] for p in [14, 21]],
            'MINUS_DI': [[p] for p in [14, 21]],
            'MINUS_DM': [[p] for p in [14, 21]],

            # 成交量指标
            'AD': [[]],  # 无参数
            'ADOSC': [[3, 10]],  # fastperiod, slowperiod
            'OBV': [[]],  # 无参数
            'MFI': [[p] for p in [14, 21]],

            # MACD 相关
            'MACD': [[12, 26, 9]],  # fastperiod, slowperiod, signalperiod
            'MACDEXT': [[12, 26, 9, 0, 0, 0]],  # fastperiod, slowperiod, signalperiod, fastmatype, slowmatype, signalmatype
            'MACDFIX': [[9]],  # signalperiod
            'STOCH': [[14, 3, 3]],  # fastk_period, slowk_period, slowd_period
            'STOCHF': [[14, 3]],  # fastk_period, fastd_period

            # 布林带
            'BBANDS': [[p, 2, 2] for p in [5, 10, 20, 21]],  # timeperiod, nbdevup, nbdevdn

            # AROON 指标
            'AROON': [[p] for p in [14, 21]],
            'AROONOSC': [[p] for p in [14, 21]],

            # 其他技术指标
            'HT_DCPERIOD': [[]],  # 无参数
            'HT_DCPHASE': [[]],  # 无参数
            'HT_PHASOR': [[]],  # 无参数
            'HT_SINE': [[]],  # 无参数
            'HT_TRENDMODE': [[]],  # 无参数

            # 价格变换
            'AVGPRICE': [[]],  # 无参数
            'MEDPRICE': [[]],  # 无参数
            'TYPPRICE': [[]],  # 无参数
            'WCLPRICE': [[]],  # 无参数

            # 数学函数（跳过）
            'CEIL': [[]],
            'FLOOR': [[]],
            'COS': [[]],
            'SIN': [[]],
            'TAN': [[]],
            'ACOS': [[]],
            'ASIN': [[]],
            'ATAN': [[]],
            'COSH': [[]],
            'SINH': [[]],
            'TANH': [[]],
            'EXP': [[]],
            'LN': [[]],
            'LOG10': [[]],
            'SQRT': [[]],
            'DIV': [[]],
            'ADD': [[]],
            'SUB': [[]],
            'MULT': [[]],
            'MAX': [[]],
            'MIN': [[]],
            'MAXINDEX': [[]],
            'MININDEX': [[]],
            'SUM': [[]],

            # 其他跳过
            'LINEARREG': [[]],
            'LINEARREG_ANGLE': [[]],
            'LINEARREG_INTERCEPT': [[]],
            'LINEARREG_SLOPE': [[]],
            'TSF': [[]],
            'VAR': [[]],
            'STDDEV': [[]],
            'CORREL': [[]],
            'BETA': [[]],
            'COVAR': [[]],
        }

        # 检查是否是特殊指标
        if func_name in special_params:
            return special_params[func_name]

        # 默认参数：尝试单周期参数
        try:
            sig = TalibFactorListGenerator.get_function_signature(func_name)
            if sig:
                # 如果有 timeperiod 参数，使用常见周期
                if 'timeperiod' in sig:
                    return [[p] for p in common_periods[:3]]  # 只取前3个避免太多
                # 如果有 period 参数
                elif 'period' in sig:
                    return [[p] for p in common_periods[:3]]
                # 如果有 fastperiod 参数（MACD类）
                elif 'fastperiod' in sig:
                    return [[12, 26, 9]]
            else:
                # 无参数函数
                return [[]]
        except:
            return [[]]
    
    @staticmethod
    def generate_talib_factors() -> List[str]:
        """生成所有 TA-Lib 因子名称"""
        functions = TalibFactorListGenerator.get_talib_functions()
        factors = []

        # 完全跳过的函数（数学函数、K线形态等）
        skip_functions = {
            # 数学函数
            'CEIL', 'FLOOR', 'COS', 'SIN', 'TAN', 'ACOS', 'ASIN', 'ATAN',
            'COSH', 'SINH', 'TANH', 'EXP', 'LN', 'LOG10', 'SQRT',
            'DIV', 'ADD', 'SUB', 'MULT', 'MAX', 'MIN', 'MAXINDEX', 'MININDEX', 'SUM',

            # 其他不需要的
            'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
            'TSF', 'VAR', 'STDDEV', 'CORREL', 'BETA', 'COVAR',

            # K线形态识别（太复杂，跳过）
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
            'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
            'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
            'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
            'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
            'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
            'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
            'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',

            # 其他复杂或不需要的函数
            'BOP', 'HT_TRENDLINE', 'MAVP', 'SAR', 'SAREXT', 'ULTOSC',
        }

        for func_name in functions:
            if func_name in skip_functions:
                continue

            try:
                param_combinations = TalibFactorListGenerator.generate_common_parameters(func_name)

                for params in param_combinations:
                    if params:
                        # 有参数的情况
                        param_str = '_'.join(str(p) for p in params)
                        factor_name = f"TALIB_{func_name}_{param_str}"
                    else:
                        # 无参数的情况
                        factor_name = f"TALIB_{func_name}"

                    factors.append(factor_name)

            except Exception as e:
                # 静默跳过有问题的函数
                continue

        return sorted(list(set(factors)))  # 去重并排序


class TalibFactorCalculator:
    """TA-Lib 因子计算器"""
    
    @staticmethod
    def calculate(factor_name: str, ohlcv: pd.DataFrame) -> pd.Series:
        """
        使用 TA-Lib 计算指定的技术指标
        
        因子名称格式：TALIB_{FUNCTION}_{PARAM1}_{PARAM2}_...
        例如：TALIB_RSI_14, TALIB_MACD_12_26_9, TALIB_BBANDS_20_2_2
        
        Args:
            factor_name: 因子名称
            ohlcv: OHLCV DataFrame，包含 open, high, low, close, volume
        
        Returns:
            pd.Series: 因子值，索引为日期
        """
        if not TALIB_AVAILABLE:
            raise Exception("TA-Lib 未安装，请先执行: pip install TA-Lib")
        
        if not factor_name.startswith('TALIB_'):
            raise ValueError(f"因子名称应该以 TALIB_ 开头: {factor_name}")
        
        # 解析因子名称和参数
        # 处理复合函数名（如 HT_DCPERIOD）
        parts = factor_name.split('_')
        if len(parts) < 2:
            raise ValueError(f"无效的因子名称格式: {factor_name}")
        
        # 尝试找到有效的 TA-Lib 函数名
        func_name = None
        params_str = []
        
        # 从后往前尝试构建函数名
        for i in range(len(parts) - 1, 0, -1):
            potential_func = '_'.join(parts[1:i+1])
            if hasattr(talib, potential_func):
                func_name = potential_func
                params_str = parts[i+1:]
                break
        
        # 如果没找到，假设第二个部分是函数名
        if func_name is None:
            func_name = parts[1]
            params_str = parts[2:]
        
        # 特殊处理参数解析，支持浮点数
        params = []
        for param_str in params_str:
            try:
                # 尝试转换为 int
                params.append(int(param_str))
            except ValueError:
                try:
                    # 尝试转换为 float
                    params.append(float(param_str))
                except ValueError:
                    # 如果都不是数字，保持为字符串（虽然很少见）
                    params.append(param_str)
        
        # 获取 TA-Lib 函数
        if not hasattr(talib, func_name):
            raise ValueError(f"TA-Lib 不支持函数: {func_name}")
        
        func = getattr(talib, func_name)
        
        # 提取所需的列
        required_cols = _get_required_columns(func_name)
        
        # 获取参数名映射
        param_names = _get_parameter_names(func_name)
        
        try:
            # 构建参数字典
            kwargs = {}
            for i, param_value in enumerate(params):
                if i < len(param_names):
                    kwargs[param_names[i]] = param_value
            
            # 确保数据是 float64 类型（TA-Lib 要求）
            close_data = ohlcv['close'].values.astype(np.float64)
            high_data = ohlcv['high'].values.astype(np.float64) if 'high' in ohlcv.columns else None
            low_data = ohlcv['low'].values.astype(np.float64) if 'low' in ohlcv.columns else None
            open_data = ohlcv['open'].values.astype(np.float64) if 'open' in ohlcv.columns else None
            volume_data = ohlcv['volume'].values.astype(np.float64) if 'volume' in ohlcv.columns else None
            
            # 调用 TA-Lib 函数
            if required_cols == ['close']:
                result = func(close_data, **kwargs)
            elif required_cols == ['high', 'low']:
                result = func(high_data, low_data, **kwargs)
            elif required_cols == ['high', 'low', 'close']:
                result = func(high_data, low_data, close_data, **kwargs)
            elif required_cols == ['high', 'low', 'close', 'volume']:
                result = func(high_data, low_data, close_data, volume_data, **kwargs)
            elif required_cols == ['open', 'high', 'low', 'close']:
                result = func(open_data, high_data, low_data, close_data, **kwargs)
            elif required_cols == ['open', 'close']:
                result = func(open_data, close_data, **kwargs)  # IMI 需要 open, close
            elif required_cols == ['close', 'volume']:
                result = func(close_data, volume_data, **kwargs)
            else:
                # 默认使用 close
                result = func(close_data, **kwargs)
            
            # 处理返回值（某些函数返回多个值）
            if isinstance(result, tuple):
                # 返回第一个值
                result = result[0]
            
            # 转换为 Series
            return pd.Series(result, index=ohlcv.index)
        
        except Exception as e:
            raise Exception(f"计算 TA-Lib 因子失败 {factor_name}: {e}")


def _get_parameter_names(func_name: str) -> List[str]:
    """
    获取 TA-Lib 函数的参数名称（除第一个价格参数外）
    
    Args:
        func_name: TA-Lib 函数名称
    
    Returns:
        List[str]: 参数名称列表
    """
    # 特殊函数的参数名映射
    param_mappings = {
        'SMA': ['timeperiod'],
        'EMA': ['timeperiod'],
        'WMA': ['timeperiod'],
        'DEMA': ['timeperiod'],
        'TEMA': ['timeperiod'],
        'TRIMA': ['timeperiod'],
        'KAMA': ['timeperiod'],
        'MAMA': ['fastlimit', 'slowlimit'],
        'T3': ['timeperiod', 'vfactor'],
        
        'RSI': ['timeperiod'],
        'STOCHRSI': ['timeperiod', 'fastk_period', 'fastd_period', 'fastd_matype'],
        'MOM': ['timeperiod'],
        'ROC': ['timeperiod'],
        'ROCP': ['timeperiod'],
        'ROCR': ['timeperiod'],
        'ROCR100': ['timeperiod'],
        'TRIX': ['timeperiod'],
        'WILLR': ['timeperiod'],
        'CCI': ['timeperiod'],
        'CMO': ['timeperiod'],
        'PPO': ['fastperiod', 'slowperiod', 'matype'],
        'APO': ['fastperiod', 'slowperiod', 'matype'],
        
        'ATR': ['timeperiod'],
        'NATR': ['timeperiod'],
        'TRANGE': [],
        'ADX': ['timeperiod'],
        'ADXR': ['timeperiod'],
        'DX': ['timeperiod'],
        'PLUS_DI': ['timeperiod'],
        'PLUS_DM': ['timeperiod'],
        'MINUS_DI': ['timeperiod'],
        'MINUS_DM': ['timeperiod'],
        
        'AD': [],
        'ADOSC': ['fastperiod', 'slowperiod'],
        'OBV': [],
        'MFI': ['timeperiod'],
        
        'MACD': ['fastperiod', 'slowperiod', 'signalperiod'],
        'MACDEXT': ['fastperiod', 'slowperiod', 'signalperiod', 'fastmatype', 'slowmatype', 'signalmatype'],
        'MACDFIX': ['signalperiod'],
        'STOCH': ['fastk_period', 'slowk_period', 'slowd_period'],
        'STOCHF': ['fastk_period', 'fastd_period'],
        
        'MAMA': ['fastlimit', 'slowlimit'],
        'T3': ['timeperiod', 'vfactor'],
        
        'RSI': ['timeperiod'],
        'STOCHRSI': ['timeperiod', 'fastk_period', 'fastd_period', 'fastd_matype'],
        'MOM': ['timeperiod'],
        'ROC': ['timeperiod'],
        'ROCP': ['timeperiod'],
        'ROCR': ['timeperiod'],
        'ROCR100': ['timeperiod'],
        'TRIX': ['timeperiod'],
        'WILLR': ['timeperiod'],
        'CCI': ['timeperiod'],
        'CMO': ['timeperiod'],
        'PPO': ['fastperiod', 'slowperiod', 'matype'],
        'APO': ['fastperiod', 'slowperiod', 'matype'],
        
        'ATR': ['timeperiod'],
        'NATR': ['timeperiod'],
        'TRANGE': [],
        'ADX': ['timeperiod'],
        'ADXR': ['timeperiod'],
        'DX': ['timeperiod'],
        'PLUS_DI': ['timeperiod'],
        'PLUS_DM': ['timeperiod'],
        'MINUS_DI': ['timeperiod'],
        'MINUS_DM': ['timeperiod'],
        
        'AD': [],
        'ADOSC': ['fastperiod', 'slowperiod'],
        'OBV': [],
        'MFI': ['timeperiod'],
        
        'MACD': ['fastperiod', 'slowperiod', 'signalperiod'],
        'MACDEXT': ['fastperiod', 'slowperiod', 'signalperiod', 'fastmatype', 'slowmatype', 'signalmatype'],
        'MACDFIX': ['signalperiod'],
        'STOCH': ['fastk_period', 'slowk_period', 'slowd_period'],
        'STOCHF': ['fastk_period', 'fastd_period'],
        
        'BBANDS': ['timeperiod', 'nbdevup', 'nbdevdn'],
        
        'AROON': ['timeperiod'],
        'AROONOSC': ['timeperiod'],
        
        'HT_DCPERIOD': [],
        'HT_DCPHASE': [],
        'HT_PHASOR': [],
        'HT_SINE': [],
        'HT_TRENDMODE': [],
        
        'AVGPRICE': [],
        'MEDPRICE': [],
        'TYPPRICE': [],
        'WCLPRICE': [],
        
        # 修正的参数映射
        'ACCBANDS': ['timeperiod'],
        'IMI': ['timeperiod'],
        'MIDPRICE': ['timeperiod'],
        'MINMAX': ['timeperiod'],
        'MINMAXINDEX': ['timeperiod'],
    }
    
    return param_mappings.get(func_name, [])


def _get_required_columns(func_name: str) -> List[str]:
    """
    获取 TA-Lib 函数所需的列
    
    Args:
        func_name: TA-Lib 函数名称
    
    Returns:
        List[str]: 需要的列名列表
    """
    # 定义常见的参数要求
    price_only_funcs = {
        'RSI', 'ROCP', 'ROC', 'MOM', 'TRIX', 'CCI', 'CMO',
        'STOCHRSI', 'OBV',  # OBV 只用 volume，但我们归类为 close
    }
    
    high_low_funcs = {
        'AROON', 'AROONOSC', 'MEDPRICE', 'MIDPRICE', 'MINUS_DM', 'PLUS_DM',  # MINUS_DM, PLUS_DM 需要 HL
    }
    
    hlc_funcs = {
        'WILLR', 'ADX', 'ADXR', 'DX', 'PLUS_DI', 'MINUS_DI',
        'ACCBANDS', 'CCI', 'TYPPRICE', 'WCLPRICE',  # 移除 IMI，需要 open, close
    }
    
    hlc_with_volume_funcs = {
        'ATR', 'NATR', 'TRANGE', 'STOCH', 'STOCHF', 'MFI', 'AD', 'ADOSC',
    }
    
    ohlc_funcs = {
        'AVGPRICE',  # 只需要 OHLC 的函数
    }
    
    open_close_funcs = {
        'IMI',  # IMI 需要 open, close
    }
    
    # 判断需要的列（先检查明确需要 volume 的）
    if func_name in hlc_with_volume_funcs:
        if func_name in ['AD', 'ADOSC']:
            return ['high', 'low', 'close', 'volume']
        elif func_name == 'MFI':
            return ['high', 'low', 'close', 'volume']
        else:
            return ['high', 'low', 'close']
    elif func_name == 'OBV':
        return ['close', 'volume']  # OBV 需要 close 和 volume
    elif func_name in hlc_funcs:
        return ['high', 'low', 'close']
    elif func_name in open_close_funcs:
        return ['open', 'close']  # IMI 需要 open, close
    elif func_name in high_low_funcs:
        return ['high', 'low']
    elif func_name in ohlc_funcs:
        return ['open', 'high', 'low', 'close']
    elif func_name in price_only_funcs:
        return ['close']
    else:
        # 默认使用 close
        return ['close']


class TalibFactorGenerator(FactorGenerator):
    """TA-Lib 因子生成器"""
    
    def __init__(self, stock_codes: List[str], start_date: str, end_date: str,
                 factor_names: Optional[List[str]] = None,
                 output_dir: str = './data/factor_tasks'):
        """
        初始化 TA-Lib 因子生成器
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factor_names: 因子名称列表，None 表示使用全部因子
            output_dir: 输出目录
        """
        super().__init__(stock_codes, start_date, end_date, output_dir)
        
        # 设置因子列表
        if factor_names is None:
            # 使用全部 TA-Lib 因子
            self.factor_names = TalibFactorListGenerator.generate_talib_factors()
        else:
            self.factor_names = factor_names
            # 验证因子名称
            validate_factor_names(self.factor_names, 'talib')
        
        # 检查 TA-Lib 是否可用
        if not TALIB_AVAILABLE:
            raise Exception("TA-Lib 未安装，请先执行: pip install TA-Lib")
    
    def generate(self) -> pd.DataFrame:
        """
        生成 TA-Lib 因子数据
        
        Returns:
            pd.DataFrame: 包含所有因子的 DataFrame
        """
        # 设置任务
        self.setup_task()
        
        print(f"\n开始生成 TA-Lib 因子...")
        print(f"因子列表: {self.factor_names}")
        
        # 加载 OHLCV 数据
        lookback_start = extend_lookback_start_date(self.start_date)
        print(f"\n加载 OHLCV 数据 ({lookback_start} ~ {self.end_date})...")
        ohlcv_df = load_ohlcv_data(self.stock_codes, lookback_start, self.end_date)
        
        if ohlcv_df.empty:
            print("❌ 未能加载到 OHLCV 数据")
            return pd.DataFrame()
        
        print(f"✓ 加载成功，共 {len(ohlcv_df)} 条记录")
        
        # 为每只股票和每个因子计算因子值
        all_data = []
        
        for i, stock_code in enumerate(self.stock_codes):
            if (i + 1) % 50 == 0:
                print(f"  处理进度: {i+1}/{len(self.stock_codes)} 股票")
            
            # 获取该股票的 OHLCV 数据
            code_col = 'code' if 'code' in ohlcv_df.columns else 'stock_code'
            stock_ohlcv = ohlcv_df[ohlcv_df[code_col] == stock_code]
            
            if stock_ohlcv.empty:
                continue
            
            # 排序日期并设置为索引
            if 'date' in stock_ohlcv.columns:
                stock_ohlcv = stock_ohlcv.sort_values('date')
                stock_ohlcv = stock_ohlcv.set_index('date')
            
            # 计算各个因子
            factor_values = {}
            for factor_name in self.factor_names:
                try:
                    factor_series = TalibFactorCalculator.calculate(factor_name, stock_ohlcv)
                    factor_values[factor_name] = factor_series
                except Exception as e:
                    print(f"  ⚠️  计算股票 {stock_code} 的因子 {factor_name} 失败: {e}")
            
            # 合并该股票的所有因子
            if factor_values:
                stock_data = pd.DataFrame(factor_values)
                stock_data['date'] = stock_data.index
                stock_data['stock_code'] = stock_code
                all_data.append(stock_data)
        
        # 合并所有股票的数据
        if not all_data:
            print("❌ 未能计算任何因子")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 规范化格式
        result = format_factor_dataframe(result)
        result = clamp_dataframe_to_date_range(result, self.start_date, self.end_date)
        
        print(f"✓ 生成完成，共 {len(result)} 条记录")
        
        return result


def generate_talib_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """
    生成 TA-Lib 因子
    
    参数说明：
    - stock_codes: 股票代码列表（必须是股票，不是指数）
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    - factor_names: 因子名称列表，None 表示使用全部因子
    - output_dir: 输出目录
    
    因子命名规范：TALIB_{FUNCTION_NAME}_{PARAM1}_{PARAM2}_...
    
    示例：
    - TALIB_RSI_14: RSI 指标，周期 14
    - TALIB_MACD_12_26_9: MACD，快速 12，慢速 26，信号 9
    - TALIB_BBANDS_20_2_2: 布林带，周期 20，标准差 2
    - TALIB_ATR_14: 真实波幅，周期 14
    
    支持 200+ TA-Lib 指标，详见 TA-Lib 文档
    
    返回值：
    {
        'factor_file': 'path/to/factors_YYYYMMDD_HHMMSS.csv',
        'metadata_file': 'path/to/task_metadata_YYYYMMDD_HHMMSS.json',
        'readme_file': 'path/to/README_task_YYYYMMDD_HHMMSS.md'
    }
    
    使用示例：
    
    # 方式 1: 直接指定股票代码和因子
    result = generate_talib_factors(
        stock_codes=['000001', '000002'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9']
    )
    
    # 方式 2: 如果需要指数的成分股
    from data import load_stock_pool
    index_stocks = load_stock_pool('000001')['code'].tolist()
    result = generate_talib_factors(
        stock_codes=index_stocks,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    """
    # 创建生成器
    generator = TalibFactorGenerator(stock_codes, start_date, end_date, factor_names, output_dir)
    
    # 生成因子
    factor_df = generator.generate()
    
    if factor_df.empty:
        raise Exception("因子生成失败")
    
    # 保存因子
    factor_file = generator.save_factors(factor_df)
    
    # 获取输出路径
    output_paths = generator.get_output_paths()
    
    print(f"\n✅ TA-Lib 因子生成成功!")
    print(f"  因子文件: {output_paths['factor_file']}")
    
    return output_paths
