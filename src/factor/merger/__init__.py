"""
因子合并层

提供两个核心函数：
- merge_factor_files(): 合并多个因子文件
- merge_factor_directory(): 合并整个目录的因子文件
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

KEY_COLUMNS = ["date", "stock_code"]
MERGE_MODES = {"outer", "inner"}
DATE_COLUMN_CANDIDATES = ["date", "trade_date", "datetime", "timestamp"]
CODE_COLUMN_CANDIDATES = ["stock_code", "asset", "code", "symbol", "ticker"]


def _ensure_valid_mode(how: str) -> str:
    mode = how.lower()
    if mode not in MERGE_MODES:
        raise ValueError(f"how 必须为 'outer' 或 'inner'，收到: {how}")
    return mode


def _standardize_factor_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"文件 {source} 缺少日期列，期待列名之一: {DATE_COLUMN_CANDIDATES}")
    code_col = next((c for c in CODE_COLUMN_CANDIDATES if c in df.columns), None)
    if code_col is None:
        raise ValueError(f"文件 {source} 缺少股票代码列，期待列名之一: {CODE_COLUMN_CANDIDATES}")

    renamed = df.rename(columns={date_col: "date", code_col: "stock_code"})
    renamed["stock_code"] = renamed["stock_code"].astype(str).str.zfill(6)
    return renamed


def _load_factor_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"因子文件不存在: {path}")

    df = pd.read_csv(path)
    df = _standardize_factor_columns(df, path)

    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"文件 {path} 缺少必要列: {missing}")

    return df


def merge_factor_files(
    factor_files: Sequence[str],
    output_file: Optional[str] = None,
    how: str = "outer",
) -> pd.DataFrame:
    """
    合并多个因子文件

    Args:
        factor_files: 因子 CSV 路径列表
        output_file: 可选，保存合并结果的路径
        how: 合并方式，'outer'（默认）或 'inner'

    Returns:
        pd.DataFrame: 合并后的因子数据
    """
    if not isinstance(factor_files, Iterable):
        raise ValueError("factor_files 必须是可迭代对象")

    file_list = [str(Path(p)) for p in factor_files if p]
    if not file_list:
        raise ValueError("factor_files 不能为空")

    mode = _ensure_valid_mode(how)
    merged_df: Optional[pd.DataFrame] = None
    seen_factor_cols: set[str] = set()

    for file_path in file_list:
        df = _load_factor_file(Path(file_path))

        # 避免重复列导致 pandas 自动添加 _x/_y 后缀，默认保留先出现的版本
        factor_cols = [c for c in df.columns if c not in KEY_COLUMNS]
        duplicated = [c for c in factor_cols if c in seen_factor_cols]
        if duplicated:
            df = df.drop(columns=duplicated)

        seen_factor_cols.update([c for c in df.columns if c not in KEY_COLUMNS])

        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.merge(merged_df, df, on=KEY_COLUMNS, how=mode)

    if merged_df is None:
        raise ValueError("未能加载任何因子文件")

    merged_df = merged_df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)

    return merged_df


def merge_factor_directory(
    factor_dir: str = "./data/factor_tasks",
    pattern: str = "factors_*.csv",
    output_file: Optional[str] = None,
    exclude_factors: Optional[Sequence[str]] = None,
    how: str = "outer",
) -> pd.DataFrame:
    """
    合并整个目录的因子文件

    Args:
        factor_dir: 因子目录
        pattern: 文件匹配模式（glob）
        output_file: 可选，保存合并结果的路径
        exclude_factors: 需要排除的因子列
        how: 合并方式，'outer' 或 'inner'

    Returns:
        pd.DataFrame: 合并后的因子数据
    """
    base_dir = Path(factor_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"因子目录不存在: {factor_dir}")

    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"目录 {factor_dir} 中未找到匹配 {pattern} 的文件")

    merged_df = merge_factor_files([str(p) for p in files], output_file=None, how=how)

    if exclude_factors:
        drop_cols = [col for col in exclude_factors if col in merged_df.columns and col not in KEY_COLUMNS]
        merged_df = merged_df.drop(columns=drop_cols, errors="ignore")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)

    return merged_df
