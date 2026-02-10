"""
参数验证模块

包含所有输入参数的验证逻辑
"""

from typing import List, Optional
from datetime import datetime

from src.data.data import get_index_stocks, _normalize_code_arg


def validate_stock_codes(stock_codes: List[str]) -> bool:
    """
    验证股票代码是否有效
    
    Args:
        stock_codes: 股票代码列表
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果股票代码无效
    """
    # 如果传入的根本不是列表，仍然认为是调用方错误
    if not isinstance(stock_codes, list):
        raise ValueError("stock_codes 必须是列表")

    # 如果是空列表，使用默认的 small 股票池
    if len(stock_codes) == 0:
        try:
            stock_codes[:] = get_index_stocks("small")
        except Exception as e:
            raise ValueError(f"stock_codes 为空且获取默认 small 股票池失败: {e}")
    
    # 使用 data.py 中的归一化函数来处理各种格式的股票代码
    # 这会自动提取6位数字、补齐前缀、去重等
    try:
        normalized_codes = _normalize_code_arg(stock_codes, allow_none=False, deduplicate=True)
        if not normalized_codes:
            raise ValueError("归一化后股票代码列表为空")
        # 将归一化后的代码写回原列表
        stock_codes[:] = normalized_codes
    except Exception as e:
        raise ValueError(f"股票代码归一化失败: {e}")
    
    return True
def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    验证日期范围是否有效
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果日期无效
    """
    try:
        # 验证日期格式
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 验证日期范围
        if start >= end:
            raise ValueError(f"开始日期 {start_date} 应该早于结束日期 {end_date}")
        
        return True
    
    except ValueError as e:
        raise ValueError(f"日期验证失败: {e}")


def validate_factor_names(factor_names: List[str], factor_type: str = 'builtin') -> bool:
    """
    验证因子名称是否有效
    
    Args:
        factor_names: 因子名称列表
        factor_type: 因子类型 ('builtin', 'talib', 'oss', 'file')
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果因子名称无效
    """
    if not isinstance(factor_names, list) or len(factor_names) == 0:
        raise ValueError("factor_names 必须是非空列表")
    
    # 内置因子的可用列表
    BUILTIN_AVAILABLE = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
    
    if factor_type == 'builtin':
        for factor in factor_names:
            if factor not in BUILTIN_AVAILABLE:
                raise ValueError(f"无效的内置因子: {factor}，可用: {BUILTIN_AVAILABLE}")
    
    elif factor_type == 'talib':
        # TA-Lib 因子应该以 TALIB_ 开头
        for factor in factor_names:
            if not factor.startswith('TALIB_'):
                raise ValueError(f"TA-Lib 因子应该以 TALIB_ 开头，得到: {factor}")
    
    elif factor_type == 'oss':
        # OSS 因子应该以 ALPHA 开头
        for factor in factor_names:
            if not (factor.startswith('ALPHA158_') or factor.startswith('ALPHA360_')):
                raise ValueError(f"OSS 因子应该以 ALPHA158_ 或 ALPHA360_ 开头，得到: {factor}")
    
    return True


def validate_output_dir(output_dir: str) -> bool:
    """
    验证输出目录是否有效
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果目录无效
    """
    if not isinstance(output_dir, str) or len(output_dir) == 0:
        raise ValueError("output_dir 必须是非空字符串")
    
    return True


def validate_factor_file_path(file_path: str, factor_name: str) -> bool:
    """
    验证因子文件路径和因子名称
    
    Args:
        file_path: 文件路径
        factor_name: 因子列名
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果文件或因子名称无效
    """
    import os
    
    if not os.path.exists(file_path):
        raise ValueError(f"因子文件不存在: {file_path}")
    
    if not isinstance(factor_name, str) or len(factor_name) == 0:
        raise ValueError("factor_name 必须是非空字符串")
    
    return True


def validate_all_params(stock_codes: List[str], start_date: str, end_date: str,
                       output_dir: str, factor_type: str = 'builtin') -> bool:
    """
    验证所有参数
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        factor_type: 因子类型
    
    Returns:
        bool: 所有参数都有效
    """
    validate_stock_codes(stock_codes)
    validate_date_range(start_date, end_date)
    validate_output_dir(output_dir)
    return True
