"""
辅助函数集合

包含：
- 时间戳生成
- 目录创建
- 数据加载
- 文件操作
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd


def generate_timestamp() -> str:
    """
    生成时间戳 YYYYMMDD_HHMMSS 格式
    
    Returns:
        str: 时间戳，例如 '20250129_153000'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_task_directory(base_dir: str, timestamp: str) -> str:
    """
    创建任务目录
    
    Args:
        base_dir: 基础目录，通常是 './data/factor_tasks'
        timestamp: 时间戳，例如 '20250129_153000'
    
    Returns:
        str: 创建的任务目录路径
    
    Example:
        >>> task_dir = create_task_directory('./data/factor_tasks', '20250129_153000')
        >>> print(task_dir)
        './data/factor_tasks/task_20250129_153000'
    """
    # 创建基础目录
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建任务目录
    task_dir = os.path.join(base_dir, f'task_{timestamp}')
    Path(task_dir).mkdir(parents=True, exist_ok=True)
    
    return task_dir


def get_factor_output_path(task_dir: str, timestamp: str) -> str:
    """
    获取因子输出文件路径
    
    Args:
        task_dir: 任务目录
        timestamp: 时间戳
    
    Returns:
        str: 因子文件路径
    """
    return os.path.join(task_dir, f'factors_{timestamp}.csv')


def get_metadata_output_path(task_dir: str, timestamp: str) -> str:
    """
    获取元信息输出文件路径
    
    Args:
        task_dir: 任务目录
        timestamp: 时间戳
    
    Returns:
        str: 元信息文件路径
    """
    return os.path.join(task_dir, f'task_metadata_{timestamp}.json')


def get_readme_output_path(task_dir: str, timestamp: str) -> str:
    """
    获取 README 输出文件路径
    
    Args:
        task_dir: 任务目录
        timestamp: 时间戳
    
    Returns:
        str: README 文件路径
    """
    return os.path.join(task_dir, f'README_task_{timestamp}.md')


def normalize_stock_code(stock_code: str) -> str:
    """
    标准化股票代码为 6 位数字
    
    Args:
        stock_code: 原始股票代码
    
    Returns:
        str: 标准化后的代码（6位数字）
    """
    return str(stock_code).zfill(6)


def normalize_stock_codes(stock_codes: List[str]) -> List[str]:
    """
    标准化股票代码列表
    
    Args:
        stock_codes: 股票代码列表
    
    Returns:
        List[str]: 标准化后的代码列表
    """
    return [normalize_stock_code(code) for code in stock_codes]


def get_stock_data_from_cache(stock_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    从缓存或数据源获取股票数据 (OHLCV)
    
    注意：此函数会导入 data 模块，确保 data.py 在项目中
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: 股票数据，包含 date, code, open, high, low, close, volume 等列
    """
    try:
        # 动态导入 data 模块
        import sys
        import os as os_module
        project_root = os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import data
        
        # 使用 data 模块加载数据
        df = data.load_ohlcv(stock_codes, start=start_date, end=end_date)
        
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    
    except Exception as e:
        print(f"加载股票数据失败: {e}")
        return pd.DataFrame()


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> bool:
    """
    保存 DataFrame 到 CSV 文件
    
    Args:
        df: DataFrame 对象
        filepath: 输出文件路径
        index: 是否保存索引
    
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        df.to_csv(filepath, index=index)
        return True
    except Exception as e:
        print(f"保存文件失败 {filepath}: {e}")
        return False


def load_csv_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    加载 CSV 文件到 DataFrame
    
    Args:
        filepath: CSV 文件路径
    
    Returns:
        pd.DataFrame: 加载的数据
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"加载文件失败 {filepath}: {e}")
        return pd.DataFrame()
