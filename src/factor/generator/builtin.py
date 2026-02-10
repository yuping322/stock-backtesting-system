"""
内置因子生成器

支持的内置因子：
- VOL10: 10日成交量比值
- RSI_14: 14日相对强弱指标
- MA_20: 20日移动平均比值
- MACD_12_26_9: MACD指标

支持的自定义因子：
- 通过 src/factor/generator/custom/ 目录下的模块自动加载
- 例如：CUSTOM_MA_RATIO
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.factor.generator._base import (
    FactorGenerator,
    load_ohlcv_data,
    merge_factor_dataframes,
    format_factor_dataframe,
    extend_lookback_start_date,
    clamp_dataframe_to_date_range,
)
from src.factor.utils import validate_factor_names


class BuiltinFactorCalculator:
    """内置因子计算器"""
    
    @staticmethod
    def calculate_vol10(ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算 VOL10 - 10日成交量比值
        
        公式：今日成交量 / 10日平均成交量
        
        Args:
            ohlcv: OHLCV DataFrame，必须包含 volume 列，索引为日期
        
        Returns:
            pd.Series: VOL10 值，索引为日期
        """
        if 'volume' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        volume = ohlcv['volume']
        ma10 = volume.rolling(window=10).mean()
        
        # 避免除以 0
        vol10 = np.where(ma10 != 0, volume / ma10, np.nan)
        
        return pd.Series(vol10, index=ohlcv.index)
    
    @staticmethod
    def calculate_rsi_14(ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算 RSI_14 - 14日相对强弱指标
        
        公式：
        RSI = 100 * (上升平均值 / (上升平均值 + 下降平均值))
        
        Args:
            ohlcv: OHLCV DataFrame，必须包含 close 列，索引为日期
        
        Returns:
            pd.Series: RSI_14 值 (0-100)，索引为日期
        """
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close']
        delta = close.diff()
        
        # 分离上升和下降
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # 计算 RSI
        rs = gain / loss
        rsi = 100 * (rs / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_ma_20(ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算 MA_20 - 20日移动平均比值
        
        公式：今日收盘价 / 20日移动平均
        
        Args:
            ohlcv: OHLCV DataFrame，必须包含 close 列，索引为日期
        
        Returns:
            pd.Series: MA_20 值，索引为日期
        """
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close']
        ma20 = close.rolling(window=20).mean()
        
        # 避免除以 0
        ma_ratio = np.where(ma20 != 0, close / ma20, np.nan)
        
        return pd.Series(ma_ratio, index=ohlcv.index)
    
    @staticmethod
    def calculate_macd_12_26_9(ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算 MACD_12_26_9 指标
        
        公式：
        DIF = 12日EMA - 26日EMA
        DEA = DIF的9日EMA
        MACD = 2 * (DIF - DEA)
        
        这里返回 MACD 的柱状图值
        
        Args:
            ohlcv: OHLCV DataFrame，必须包含 close 列，索引为日期
        
        Returns:
            pd.Series: MACD 值，索引为日期
        """
        if 'close' not in ohlcv.columns:
            return pd.Series(dtype=float)
        
        close = ohlcv['close']
        
        # 计算 EMA
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        
        # 计算 DIF (MACD Line)
        dif = ema12 - ema26
        
        # 计算 DEA (Signal Line)
        dea = dif.ewm(span=9, adjust=False).mean()
        
        # 计算 MACD Histogram
        macd = 2 * (dif - dea)
        
        return macd
    
    @staticmethod
    def calculate(factor_name: str, ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算指定的因子
        
        Args:
            factor_name: 因子名称 ('VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9')
            ohlcv: OHLCV DataFrame
        
        Returns:
            pd.Series: 因子值，索引为日期
        """
        if factor_name == 'VOL10':
            return BuiltinFactorCalculator.calculate_vol10(ohlcv)
        elif factor_name == 'RSI_14':
            return BuiltinFactorCalculator.calculate_rsi_14(ohlcv)
        elif factor_name == 'MA_20':
            return BuiltinFactorCalculator.calculate_ma_20(ohlcv)
        elif factor_name == 'MACD_12_26_9':
            return BuiltinFactorCalculator.calculate_macd_12_26_9(ohlcv)
        else:
            raise ValueError(f"不支持的内置因子: {factor_name}")


class BuiltinFactorGenerator(FactorGenerator):
    """内置因子生成器"""
    
    def __init__(self, stock_codes: List[str], start_date: str, end_date: str,
                 factor_names: Optional[List[str]] = None,
                 output_dir: str = './data/factor_tasks'):
        """
        初始化内置因子生成器
        
        Args:
            stock_codes: 股票代码列表（必须是股票，不是指数）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            factor_names: 因子名称列表，None 表示使用全部
            output_dir: 输出目录
        """
        super().__init__(stock_codes, start_date, end_date, output_dir)
        
        # 加载自定义因子
        from src.factor.generator.custom import load_custom_factors
        self.custom_funcs = load_custom_factors()
        
        # 设置因子列表
        if factor_names is None:
            self.factor_names = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
        else:
            self.factor_names = factor_names
            # 验证因子名称
            available_factors = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9'] + list(self.custom_funcs.keys())
            for factor in self.factor_names:
                if factor not in available_factors:
                    raise ValueError(f"无效的因子: {factor}，可用: {available_factors}")
        
        self.factor_data = {}  # 用于存储中间结果
    
    def generate(self) -> pd.DataFrame:
        """
        生成内置因子数据
        
        Returns:
            pd.DataFrame: 包含所有因子的 DataFrame
        """
        # 设置任务
        self.setup_task()
        
        print(f"\n开始生成内置因子...")
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
        
        for stock_code in self.stock_codes:
            # 获取该股票的 OHLCV 数据
            stock_ohlcv = ohlcv_df[ohlcv_df.get('code', ohlcv_df.get('stock_code', pd.Series())) == stock_code]
            
            if stock_ohlcv.empty:
                continue
            
            # 排序日期
            stock_ohlcv = stock_ohlcv.sort_values('date' if 'date' in stock_ohlcv.columns else stock_ohlcv.index)
            
            # 设置日期为索引（某些因子计算需要）
            if 'date' in stock_ohlcv.columns:
                stock_ohlcv = stock_ohlcv.set_index('date')
            
            # 计算各个因子
            factor_values = {}
            for factor_name in self.factor_names:
                try:
                    if factor_name in ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']:
                        # 内置因子
                        factor_series = BuiltinFactorCalculator.calculate(factor_name, stock_ohlcv)
                    else:
                        # 自定义因子
                        factor_func = self.custom_funcs.get(factor_name)
                        if factor_func:
                            factor_series = factor_func(stock_ohlcv)
                        else:
                            raise ValueError(f"未知的因子: {factor_name}")
                    
                    factor_values[factor_name] = factor_series
                except Exception as e:
                    print(f"  ⚠️  计算股票 {stock_code} 的因子 {factor_name} 失败: {e}")
                    factor_values[factor_name] = pd.Series(dtype=float)
            
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


def generate_builtin_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """
    生成内置因子
    
    参数说明：
    - stock_codes: 股票代码列表（必须是股票，不是指数）
    - start_date: 开始日期 (YYYY-MM-DD)
    - end_date: 结束日期 (YYYY-MM-DD)
    - factor_names: 因子名称列表，None 表示使用全部
    - output_dir: 输出目录
    
    支持的因子：
    - VOL10: 10日成交量比值
    - RSI_14: 14日相对强弱指标
    - MA_20: 20日移动平均比值
    - MACD_12_26_9: MACD指标
    - 自定义因子：通过 src/factor/generator/custom/ 目录下的模块自动加载
    
    返回值：
    {
        'factor_file': 'path/to/factors_YYYYMMDD_HHMMSS.csv',
        'metadata_file': 'path/to/task_metadata_YYYYMMDD_HHMMSS.json',
        'readme_file': 'path/to/README_task_YYYYMMDD_HHMMSS.md'
    }
    
    使用示例：
    
    # 方式 1: 直接指定股票代码
    result = generate_builtin_factors(
        stock_codes=['000001', '000002'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        factor_names=['VOL10', 'RSI_14']
    )
    
    # 方式 2: 如果需要指数的成分股，先获取再传入
    from data import load_stock_pool
    index_stocks = load_stock_pool('000001')['code'].tolist()
    result = generate_builtin_factors(
        stock_codes=index_stocks,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    """
    # 创建生成器
    generator = BuiltinFactorGenerator(stock_codes, start_date, end_date, factor_names, output_dir)
    
    # 生成因子
    factor_df = generator.generate()
    
    if factor_df.empty:
        raise Exception("因子生成失败")
    
    # 保存因子
    factor_file = generator.save_factors(factor_df)
    
    # 获取输出路径
    output_paths = generator.get_output_paths()
    
    print(f"\n✅ 因子生成成功!")
    print(f"  因子文件: {output_paths['factor_file']}")
    print(f"  元信息文件: {output_paths['metadata_file']}")
    print(f"  README文件: {output_paths['readme_file']}")
    
    return output_paths
