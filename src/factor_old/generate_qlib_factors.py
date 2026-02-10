#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从qlib提取因子并生成因子文件

支持的因子集：
    - Alpha158: 标准Alpha158因子集（158个因子）
    - Alpha360: 标准Alpha360因子集（360个因子）
    - Alpha158vwap: 基于VWAP的Alpha158变体（158个因子）
    - Alpha360vwap: 基于VWAP的Alpha360变体（360个因子）

注意：Alpha158DL和Alpha360DL是DataLoader类型，使用方式不同，暂不支持。
如需自定义配置，可使用Alpha158DL.get_feature_config()获取因子名称，然后使用Handler提取。

用法:
    # 生成Alpha158因子文件
    python factor/generate_qlib_factors.py --factor-set Alpha158 --codes 000001 000002 --start 2024-01-01 --end 2024-12-31 --output ./factors
    
    # 生成Alpha360因子文件
    python factor/generate_qlib_factors.py --factor-set Alpha360 --stock-pool HS300 --start 2024-01-01 --end 2024-12-31 --output ./factors
    
    # 生成Alpha158vwap因子文件
    python factor/generate_qlib_factors.py --factor-set Alpha158vwap --stock-pool HS300 --start 2024-01-01 --end 2024-12-31 --output ./factors
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import datetime as dt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import data
import qlib
from qlib.utils import init_instance_by_config
from qlib.contrib.data.loader import Alpha158DL

# 从alpha_test提取的数据构建逻辑
FIELDS = ["open", "high", "low", "close", "vwap", "volume"]


def build_qlib_dataset(
    codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
    rebuild: bool = False
) -> Path:
    """
    从data.py获取数据并构建qlib数据集
    
    Args:
        codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        rebuild: 是否重建
    
    Returns:
        qlib数据集目录路径
    """
    output_dir = Path(output_dir)
    
    # 检查是否已存在
    marker = output_dir / '.version'
    # 检查数据集是否完整（必须有calendars和instruments目录）
    has_calendars = (output_dir / 'calendars').exists() and (output_dir / 'calendars' / 'day.txt').exists()
    has_instruments = (output_dir / 'instruments').exists() and (output_dir / 'instruments' / 'all.txt').exists()
    dataset_complete = has_calendars and has_instruments
    
    if marker.exists() and dataset_complete and not rebuild:
        print(f'[QLIB] Existing complete dataset found at {output_dir}, skip rebuild.')
        return output_dir
    
    if output_dir.exists():
        if rebuild or not dataset_complete:
            if not dataset_complete:
                print(f'[QLIB] Dataset incomplete at {output_dir}, rebuilding...')
            else:
                print(f'[QLIB] Rebuilding dataset at {output_dir}...')
            shutil.rmtree(output_dir)
        else:
            print(f'[QLIB] Dataset exists at {output_dir}, use --rebuild to rebuild.')
            return output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 从data.py获取数据
    print(f'[DATA] Fetching data from {start_date} to {end_date}...')
    try:
        data_dict = data.load_oss_complex_stocks(
            codes=codes,
            start=start_date,
            end=end_date,
            fields=['open', 'high', 'low', 'close', 'volume']
        )
    except Exception as e:
        raise ValueError(f"加载数据失败: {e}")
    
    if not data_dict:
        raise ValueError(f"Failed to load data from data.py: 返回空字典，股票列表: {codes}")
    
    # 检查是否有实际数据
    has_data = False
    for field_name, field_df in data_dict.items():
        if isinstance(field_df, pd.DataFrame) and not field_df.empty:
            has_data = True
            break
    
    if not has_data:
        raise ValueError(f"数据字典为空或所有DataFrame为空，股票列表: {codes}")
    
    # 转换为长格式DataFrame
    dfs_by_code = {}
    for field_name, field_df in data_dict.items():
        if field_df.empty:
            continue
        for code in codes:
            if code not in field_df.columns:
                continue
            if code not in dfs_by_code:
                dfs_by_code[code] = pd.DataFrame({'date': field_df.index, 'symbol': code})
            dfs_by_code[code][field_name] = field_df[code].values
    
    if not dfs_by_code:
        raise ValueError("No data loaded")
    
    df_list = [df for df in dfs_by_code.values() if not df.empty]
    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # 计算VWAP
    if 'vwap' not in df.columns:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    
    # 确保所有字段存在
    for field in FIELDS:
        if field not in df.columns:
            if field == 'vwap':
                df['vwap'] = df['close']
            else:
                df[field] = 0.0
    
    df = df.dropna(subset=['date', 'symbol']).sort_values(['symbol', 'date'])
    print(f'[DATA] Loaded {len(df)} rows for {df["symbol"].nunique()} stocks')
    
    # 构建qlib数据集
    cal_dir = output_dir / 'calendars'
    ins_dir = output_dir / 'instruments'
    feat_root = output_dir / 'features'
    cal_dir.mkdir(exist_ok=True)
    ins_dir.mkdir(exist_ok=True)
    feat_root.mkdir(exist_ok=True)
    
    all_days = sorted(set(df['date']))
    with (cal_dir / 'day.txt').open('w') as f:
        for d in all_days:
            f.write(f"{pd.Timestamp(d).date()}\n")
    
    lines = []
    for sym, g in df.groupby('symbol'):
        lines.append(f"{sym}\t{g['date'].min().date()}\t{g['date'].max().date()}")
    with (ins_dir / 'all.txt').open('w') as f:
        for line in lines:
            f.write(line + '\n')
    
    # feature bins
    for sym, g in df.groupby('symbol'):
        sym_dir = feat_root / sym.lower()
        sym_dir.mkdir(exist_ok=True)
        g = g.drop_duplicates('date').set_index('date').sort_index()
        for field in FIELDS:
            arr = g[field].to_numpy(dtype='float32')
            out = np.hstack([np.array([0], dtype='float32'), arr])
            with (sym_dir / f'{field}.day.bin').open('wb') as f:
                out.tofile(f)
    
    # 标记版本
    (output_dir / '.version').write_text('1')
    print(f'[QLIB] Dataset built at {output_dir}')
    
    return output_dir


def extract_factors_from_qlib(
    qlib_data_dir: Path,
    factor_set: str,
    codes: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    从qlib数据集提取因子
    
    Args:
        qlib_data_dir: qlib数据集目录
        factor_set: Handler类名，支持：
            - 'Alpha158': 标准Alpha158因子集（158个因子）
            - 'Alpha360': 标准Alpha360因子集（360个因子）
            - 'Alpha158vwap': 基于VWAP的Alpha158变体（158个因子）
            - 'Alpha360vwap': 基于VWAP的Alpha360变体（360个因子）
        codes: 股票代码列表（用于文档，实际使用'instruments: "all"'）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame: MultiIndex (date, code), columns为因子名称
    """
    # 初始化qlib
    qlib.init(provider_uri=str(qlib_data_dir), region='cn')
    
    # 创建handler配置
    # 注意：qlib的Alpha158 handler使用'instruments: "all"'表示使用数据集中的所有股票
    # 如果我们只构建了特定股票的数据集，也可以使用"all"
    handler_conf = {
        'class': factor_set,
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': {
            'start_time': start_date,
            'end_time': end_date,
            'fit_start_time': start_date,
            'fit_end_time': end_date,
            'instruments': 'all',  # 使用'all'而不是codes列表
        },
    }
    
    # 创建dataset配置
    dataset_conf = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': handler_conf,
            'segments': {
                'train': [start_date, end_date],
            },
        },
    }
    
    # 创建dataset
    dataset = init_instance_by_config(dataset_conf)
    
    # 提取特征
    df = dataset.prepare('train', col_set="feature")
    
    # 转换为factor模块格式
    # qlib格式: MultiIndex (datetime, instrument), columns为 ('feature', 'factor_name')
    # 转换为: MultiIndex (date, code), columns为 factor_name
    
    if isinstance(df.columns, pd.MultiIndex):
        # 提取因子名称
        factor_names = [col[1] for col in df.columns if col[0] == 'feature']
        # 只保留feature列
        factors_df = df.loc[:, ('feature', slice(None))].copy()
        factors_df.columns = factor_names
    else:
        factors_df = df.copy()
    
    # 确保index是MultiIndex
    if not isinstance(factors_df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (datetime, instrument)")
    
    # 标准化index
    # qlib的index通常是 (datetime, instrument)，需要转换为 (date, code)
    index_names = factors_df.index.names
    if len(index_names) == 2:
        # 提取日期和代码
        dates = factors_df.index.get_level_values(0)
        instruments = factors_df.index.get_level_values(1)
        
        # 转换日期：datetime -> date
        if len(dates) > 0:
            if isinstance(dates[0], pd.Timestamp):
                # pd.Timestamp可以直接调用.date()
                dates = pd.to_datetime(dates).map(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)
            elif hasattr(dates[0], 'date'):
                # 其他datetime类型
                dates = [d.date() if hasattr(d, 'date') else d for d in dates]
            # 如果已经是date类型，保持不变
        
        # 标准化代码格式（去掉后缀，补齐6位）
        codes_normalized = []
        for code in instruments:
            code_str = str(code).replace('.XSHG', '').replace('.XSHE', '').zfill(6)
            codes_normalized.append(code_str)
        
        # 重建MultiIndex
        factors_df.index = pd.MultiIndex.from_arrays([dates, codes_normalized], names=['date', 'code'])
    else:
        raise ValueError(f"Unexpected index structure: {index_names}")
    
    return factors_df


def save_factors_to_file(factors_df: pd.DataFrame, output_path: Path):
    """
    将因子DataFrame保存到CSV文件
    
    Args:
        factors_df: MultiIndex (date, code), columns为因子名称
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 重置index，将date和code转为普通列
    df_to_save = factors_df.reset_index()
    
    # 确保代码格式统一（6位数字字符串）
    df_to_save['code'] = df_to_save['code'].astype(str).str.replace('.XSHG', '').str.replace('.XSHE', '').str.zfill(6)
    
    # 保存为CSV
    df_to_save.to_csv(output_path, index=False)
    print(f'[SAVE] Factors saved to {output_path}')
    print(f'[SAVE] Shape: {df_to_save.shape}, Factors: {list(df_to_save.columns)[2:]}')  # 前两列是date和code


def main():
    parser = argparse.ArgumentParser(description='从qlib提取因子并生成文件')
    parser.add_argument('--factor-set', 
                       choices=['Alpha158', 'Alpha360', 'Alpha158vwap', 'Alpha360vwap'],
                       default='Alpha158',
                       help='因子集名称：Alpha158(标准158个), Alpha360(标准360个), Alpha158vwap/Alpha360vwap(VWAP版本)')
    parser.add_argument('--codes', nargs='+', help='股票代码列表')
    parser.add_argument('--stock-pool', type=str, help='股票池（如HS300），与--codes二选一')
    parser.add_argument('--start', type=str, required=True, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--output', type=str, default='./factors', help='因子文件输出目录')
    parser.add_argument('--qlib-cache', type=str, help='qlib数据集缓存目录（可选）')
    parser.add_argument('--rebuild', action='store_true', help='重建qlib数据集')
    
    args = parser.parse_args()
    
    # 获取股票代码
    if args.stock_pool:
        end_date = pd.to_datetime(args.end).date()
        try:
            stocks = data.get_index_stocks(args.stock_pool, date=end_date)
            if isinstance(stocks, pd.Series):
                stocks = stocks.tolist()
            elif not isinstance(stocks, list):
                stocks = list(stocks) if stocks else []
            codes = stocks
            print(f'[STOCKS] Using {len(codes)} stocks from {args.stock_pool}')
        except Exception as e:
            print(f'[ERROR] Failed to get stocks from {args.stock_pool}: {e}')
            return
    elif args.codes:
        codes = args.codes
    else:
        print('[ERROR] Must provide either --codes or --stock-pool')
        return
    
    # 构建qlib数据集
    if args.qlib_cache:
        qlib_data_dir = Path(args.qlib_cache)
    else:
        qlib_data_dir = Path(args.output) / 'qlib_data'
    
    print(f'[QLIB] Building dataset at {qlib_data_dir}...')
    build_qlib_dataset(
        codes=codes,
        start_date=args.start,
        end_date=args.end,
        output_dir=qlib_data_dir,
        rebuild=args.rebuild
    )
    
    # 提取因子
    print(f'[FACTOR] Extracting {args.factor_set} factors...')
    factors_df = extract_factors_from_qlib(
        qlib_data_dir=qlib_data_dir,
        factor_set=args.factor_set,
        codes=codes,
        start_date=args.start,
        end_date=args.end
    )
    
    print(f'[FACTOR] Extracted {len(factors_df.columns)} factors: {list(factors_df.columns[:10])}...')
    
    # 保存因子文件
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名：Alpha158_YYYYMMDD_YYYYMMDD.csv
    start_str = pd.to_datetime(args.start).strftime('%Y%m%d')
    end_str = pd.to_datetime(args.end).strftime('%Y%m%d')
    output_file = output_dir / f'{args.factor_set}_{start_str}_{end_str}.csv'
    
    save_factors_to_file(factors_df, output_file)
    
    print(f'\n[SUCCESS] Factor file generated: {output_file}')
    print(f'[INFO] Use in factor module:')
    print(f'  calc = create_factor_calculator(file_path="{output_file}", factor_name="ROC5")')
    print(f'  calc = create_factor_calculator(file_path="{output_file}", factor_name="MA10")')


if __name__ == '__main__':
    main()

