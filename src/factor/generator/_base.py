"""
因子生成器基类和公共函数

包含所有生成函数共用的逻辑
"""

import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.factor.utils import (
    generate_timestamp,
    create_task_directory,
    normalize_stock_code,
    normalize_stock_codes,
    save_dataframe_to_csv,
    get_factor_output_path,
    validate_all_params,
)


class FactorGenerator(ABC):
    """
    因子生成器基类
    
    所有具体的因子生成器都应继承此类
    """
    
    def __init__(self, stock_codes: List[str], start_date: str, end_date: str,
                 output_dir: str = './data/factor_tasks'):
        """
        初始化因子生成器
        
        Args:
            stock_codes: 股票代码列表（必须是股票，不是指数）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            output_dir: 输出目录
        """
        self.stock_codes = normalize_stock_codes(stock_codes)
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.timestamp = None
        self.task_dir = None
        
        # 验证参数
        self.validate_params()
    
    def validate_params(self):
        """验证输入参数"""
        validate_all_params(self.stock_codes, self.start_date, self.end_date, self.output_dir)
    
    def setup_task(self) -> Tuple[str, str]:
        """
        设置任务（创建时间戳和目录）
        
        Returns:
            Tuple[str, str]: (task_dir, timestamp)
        """
        self.timestamp = generate_timestamp()
        self.task_dir = create_task_directory(self.output_dir, self.timestamp)
        
        print(f"✓ 创建任务目录: {self.task_dir}")
        print(f"  时间戳: {self.timestamp}")
        print(f"  股票数: {len(self.stock_codes)}")
        
        return self.task_dir, self.timestamp
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        生成因子数据
        
        Returns:
            pd.DataFrame: 因子数据，包含 date, stock_code 和因子列
        """
        pass
    
    def save_factors(self, df: pd.DataFrame) -> str:
        """
        保存因子数据到 CSV
        
        Args:
            df: 因子 DataFrame
        
        Returns:
            str: 保存的文件路径
        """
        if self.task_dir is None or self.timestamp is None:
            raise ValueError("必须先调用 setup_task()")
        
        output_file = get_factor_output_path(self.task_dir, self.timestamp)
        
        # 保存到 CSV
        if save_dataframe_to_csv(df, output_file, index=False):
            print(f"✓ 因子文件已保存: {output_file}")
            return output_file
        else:
            raise Exception(f"保存因子文件失败: {output_file}")
    
    def get_output_paths(self) -> Dict[str, str]:
        """
        获取所有输出文件的路径
        
        Returns:
            Dict[str, str]: 输出路径字典，包含 'factor_file', 'metadata_file', 'readme_file'
        """
        if self.task_dir is None or self.timestamp is None:
            raise ValueError("必须先调用 setup_task()")
        
        from ..utils import (
            get_factor_output_path,
            get_metadata_output_path,
            get_readme_output_path,
        )
        
        return {
            'factor_file': get_factor_output_path(self.task_dir, self.timestamp),
            'metadata_file': get_metadata_output_path(self.task_dir, self.timestamp),
            'readme_file': get_readme_output_path(self.task_dir, self.timestamp),
        }


def load_ohlcv_data(stock_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载 OHLCV 数据（来自 data.py）
    
    使用 data.load_oss_complex_stocks 从实际数据源加载市场行情数据
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: OHLCV 数据，包含以下列：
            - date: 交易日期 (DatetimeIndex)
            - 股票代码（列名）
            columns 类型：{字段名: DataFrame}
    """
    try:
        # 添加 data 模块路径
        sys.path.insert(0, project_root)
        from src.data import data
        
        # 使用真实的数据接口加载 OHLCV 数据
        # load_oss_complex_stocks 返回 Dict[字段名, DataFrame]
        # 每个 DataFrame 的 index 是日期，columns 是股票代码
        ohlcv_dict = data.load_oss_complex_stocks(
            codes=stock_codes,
            start=start_date,
            end=end_date,
            fields=['open', 'high', 'low', 'close', 'volume']
        )
        
        if not ohlcv_dict:
            print(f"警告: 未能从 data.load_oss_complex_stocks 加载到数据")
            return pd.DataFrame()
        
        # 转换返回格式：从 Dict[field, DataFrame(index=date, columns=stock)] 
        # 转换为 DataFrame(index=date, columns=['stock_code', 'open', 'high', 'low', 'close', 'volume', ...])
        # 
        # 由于 builtin.py 期望：DataFrame with columns [date, stock_code, open, high, low, close, ...]
        # 我们需要把所有的长表数据合并在一起
        
        all_data = []
        
        # 获取所有日期（从任意一个 field 的 DataFrame）
        first_field = next(iter(ohlcv_dict.keys())) if ohlcv_dict else None
        if not first_field:
            print("警告: 未能获取到任何字段数据")
            return pd.DataFrame()
        
        dates = ohlcv_dict[first_field].index
        
        # 对于每个日期和股票，构建一行记录
        for date in dates:
            for stock_code in stock_codes:
                row_data = {'date': date, 'stock_code': normalize_stock_code(stock_code)}
                
                # 从各个字段的 DataFrame 中提取该日期、该股票的数据
                for field, field_df in ohlcv_dict.items():
                    if stock_code in field_df.columns and date in field_df.index:
                        value = field_df.loc[date, stock_code]
                        # 只保存非 NaN 值
                        if pd.notna(value):
                            row_data[field] = value
                
                # 只有在有至少一个 OHLCV 值的情况下才添加该行
                if len(row_data) > 2:  # date 和 stock_code 加上至少一个数据字段
                    all_data.append(row_data)
        
        if not all_data:
            print("警告: 构建 OHLCV 数据失败")
            return pd.DataFrame()
        
        # 转换为 DataFrame
        result_df = pd.DataFrame(all_data)
        result_df['date'] = pd.to_datetime(result_df['date'])
        
        # 填充缺失字段为 NaN（某些股票可能某些字段缺失）
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field not in result_df.columns:
                result_df[field] = np.nan
        
        print(f"✓ 从 data.load_oss_complex_stocks 加载 OHLCV 数据成功: {len(result_df)} 条记录")
        return result_df
    
    except ImportError as e:
        print(f"导入 data 模块失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载 OHLCV 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def merge_factor_dataframes(all_factors: List[Tuple[str, pd.Series]]) -> pd.DataFrame:
    """
    合并多只股票的因子数据
    
    Args:
        all_factors: 列表，每个元素是 (stock_code, pd.Series) 的元组
                    Series 的索引是日期
    
    Returns:
        pd.DataFrame: 合并后的数据，列: date, stock_code, factor_value, ...
    """
    all_data = []
    
    for stock_code, factor_series in all_factors:
        if factor_series.empty:
            continue
        
        # 将 Series 转换为 DataFrame
        df = pd.DataFrame({
            'date': factor_series.index,
            'stock_code': normalize_stock_code(stock_code),
            'factor_value': factor_series.values
        })
        
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)
    
    # 排序
    result = result.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    return result


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 有正确的 date 列
    
    Args:
        df: DataFrame
    
    Returns:
        pd.DataFrame: 修正后的 DataFrame
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
    
    return df


def ensure_stock_code_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 有正确的 stock_code 列
    
    Args:
        df: DataFrame
    
    Returns:
        pd.DataFrame: 修正后的 DataFrame
    """
    if 'stock_code' in df.columns:
        df['stock_code'] = df['stock_code'].apply(normalize_stock_code)
    elif 'code' in df.columns:
        df.rename(columns={'code': 'stock_code'}, inplace=True)
        df['stock_code'] = df['stock_code'].apply(normalize_stock_code)
    
    return df


def format_factor_dataframe(df: pd.DataFrame, date_col: str = 'date',
                           code_col: str = 'stock_code') -> pd.DataFrame:
    """
    规范化因子 DataFrame 的格式
    
    确保：
    1. 有 date 列（datetime 类型）
    2. 有 stock_code 列（6位数字字符串）
    3. 其他列是因子值（float 类型）
    
    Args:
        df: 原始 DataFrame
        date_col: 日期列名
        code_col: 股票代码列名
    
    Returns:
        pd.DataFrame: 规范化后的 DataFrame
    """
    df = df.copy()
    
    # 确保日期列
    if date_col in df.columns:
        df.rename(columns={date_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # 确保股票代码列
    if code_col in df.columns:
        df.rename(columns={code_col: 'stock_code'}, inplace=True)
    df['stock_code'] = df['stock_code'].apply(normalize_stock_code)
    
    # 确保因子列是 float 类型
    for col in df.columns:
        if col not in ['date', 'stock_code']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 列顺序：date, stock_code, 其他因子列
    factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]
    df = df[['date', 'stock_code'] + sorted(factor_cols)]
    
    return df


def extend_lookback_start_date(start_date: str, months: int = 3) -> str:
    """向前扩展起始日期，便于计算需要的历史数据。"""
    parsed = pd.to_datetime(start_date)
    lookback_start = parsed - pd.DateOffset(months=months)
    return lookback_start.strftime('%Y-%m-%d')


def clamp_dataframe_to_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """将因子/行情数据裁剪到 [start_date, end_date] 区间。"""
    if df.empty:
        return df

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df.loc[mask].reset_index(drop=True)
