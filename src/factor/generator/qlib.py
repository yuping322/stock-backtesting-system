"""
Qlib 因子生成器

从 Qlib 库提取预定义的因子集合（如 Alpha158、Alpha360 等）
并生成因子数据文件。

支持的因子集：
    - Alpha158: 标准 Alpha158 因子集（158个因子）
    - Alpha360: 标准 Alpha360 因子集（360个因子）
    - Alpha158vwap: 基于 VWAP 的 Alpha158 变体（158个因子）
    - Alpha360vwap: 基于 VWAP 的 Alpha360 变体（360个因子）

用法示例:
    from src.factor.generator.qlib import generate_qlib_factors
    
    # 生成 Alpha158 因子文件
    df = generate_qlib_factors(
        stock_codes=['000001', '000002'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        factor_set='Alpha158',
        output_file='./data/alpha158_factors.csv'
    )
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

try:
    import qlib
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError:
    qlib = None
    init_instance_by_config = None
    QLIB_AVAILABLE = False

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.factor.generator._base import (
    FactorGenerator,
    format_factor_dataframe,
    extend_lookback_start_date,
    clamp_dataframe_to_date_range,
)

FIELDS = ["open", "high", "low", "close", "vwap", "volume"]

from src.data.data import load_oss_complex_stocks


def build_qlib_dataset(codes: List[str], start_date: str, end_date: str, output_dir: Path, rebuild: bool = False) -> Path:
    output_dir = Path(output_dir)
    
    # 检查是否已存在完整数据集
    marker = output_dir / '.version'
    has_calendars = (output_dir / 'calendars').exists() and (output_dir / 'calendars' / 'day.txt').exists()
    has_instruments = (output_dir / 'instruments').exists() and (output_dir / 'instruments' / 'all.txt').exists()
    dataset_complete = has_calendars and has_instruments
    
    if marker.exists() and dataset_complete and not rebuild:
        print(f'[QLIB] 使用现有数据集: {output_dir}')
        return output_dir
    
    if output_dir.exists() and not rebuild:
        print(f'[QLIB] 数据集存在，使用 --rebuild 重建')
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'[DATA] 加载数据 {start_date} ~ {end_date}')
    codes = [str(c).zfill(6) for c in codes]
    data_dict = load_oss_complex_stocks(codes, start=start_date, end=end_date, fields=['open', 'high', 'low', 'close', 'volume'])
    
    if not data_dict:
        raise ValueError('无数据')
    
    # 长格式转换 + VWAP 计算
    dfs_by_code = {}
    for field_name, field_df in data_dict.items():
        if field_df.empty: continue
        for code in codes:
            if code not in field_df.columns: continue
            if code not in dfs_by_code:
                dfs_by_code[code] = pd.DataFrame({'date': field_df.index, 'symbol': code})
            dfs_by_code[code][field_name] = field_df[code].values
    
    df_list = [df for df in dfs_by_code.values() if not df.empty]
    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    if 'vwap' not in df:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Do not fill missing OHLCV fields with 0. Use NaN to preserve missingness.
    # VWAP: compute if missing using (high+low+close)/3 when available; otherwise keep NaN
    for field in FIELDS:
        if field not in df:
            df[field] = np.nan

    if df['vwap'].isna().all():
        # compute VWAP where possible from high/low/close; leave NaN where not computable
        df['vwap'] = np.where(
            df[['high', 'low', 'close']].notna().all(axis=1),
            (df['high'] + df['low'] + df['close']) / 3,
            np.nan,
        )
    
    df = df.dropna(subset=['date', 'symbol']).sort_values(['symbol', 'date'])
    
    # Qlib 目录结构
    cal_dir = output_dir / 'calendars'
    ins_dir = output_dir / 'instruments'
    feat_root = output_dir / 'features'
    cal_dir.mkdir(exist_ok=True)
    ins_dir.mkdir(exist_ok=True)
    feat_root.mkdir(exist_ok=True)
    
    all_days = sorted(set(df['date']))
    with (cal_dir / 'day.txt').open('w') as f:
        for d in all_days:
            f.write(f"{d.date()}\n")
    
    lines = []
    for sym, g in df.groupby('symbol'):
        lines.append(f"{sym}\t{g['date'].min().date()}\t{g['date'].max().date()}")
    with (ins_dir / 'all.txt').open('w') as f:
        for line in lines: f.write(line + '\n')
    
    for sym, g in df.groupby('symbol'):
        sym_dir = feat_root / sym.lower()
        sym_dir.mkdir(exist_ok=True)
        g = g.drop_duplicates('date').set_index('date').sort_index()
        for field in FIELDS:
            arr = g[field].to_numpy(dtype='float32')
            # Do not prefix with a dummy 0 value; write raw array. Missing values (NaN) are preserved.
            out = arr
            with (sym_dir / f'{field}.day.bin').open('wb') as f:
                out.tofile(f)
    
    (output_dir / '.version').write_text('1')
    print(f'[QLIB] 数据集构建完成: {output_dir}')
    return output_dir

def extract_factors_from_qlib(qlib_data_dir: Path, factor_set: str, start_date: str, end_date: str) -> pd.DataFrame:
    import multiprocessing
    import warnings
    
    # 抑制多进程资源跟踪器的警告
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
    
    qlib.init(provider_uri=str(qlib_data_dir), region='cn')

    handler_conf = {
        'class': factor_set,
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': {
            'start_time': start_date, 'end_time': end_date,
            'fit_start_time': start_date, 'fit_end_time': end_date,
            'instruments': 'all',
        },
    }
    
    dataset_conf = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': handler_conf, 
            'segments': {'train': [start_date, end_date]},
        },
    }
    
    try:
        dataset = init_instance_by_config(dataset_conf)
        df = dataset.prepare('train', col_set="feature")
    except TypeError as exc:
        print(f"⚠️ Qlib 数据预处理失败: {exc}")
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        factor_names = [col[1] for col in df.columns if col[0] == 'feature']
        factors_df = df.loc[:, ('feature', slice(None))].copy()
        factors_df.columns = factor_names
    else:
        factors_df = df.copy()
    
    dates = pd.to_datetime(factors_df.index.get_level_values(0))
    instruments = factors_df.index.get_level_values(1)
    
    codes_normalized = [str(code).replace('.XSHG', '').replace('.XSHE', '').zfill(6) for code in instruments]
    
    factors_df.index = pd.MultiIndex.from_arrays([dates, codes_normalized], names=['date', 'code'])
    
    # 尝试清理多进程资源
    try:
        if hasattr(multiprocessing, '_resource_tracker'):
            multiprocessing._resource_tracker._stop()
    except Exception:
        pass
    
    return factors_df

class QlibFactorGenerator(FactorGenerator):
    def __init__(self, stock_codes: List[str], start_date: str, end_date: str,
                 factor_names: Optional[List[str]] = None,
                 factor_set: str = 'Alpha158',
                 output_dir: str = './data/factor_tasks'):
        super().__init__(stock_codes, start_date, end_date, output_dir)
        self.factor_set = factor_set
        self.factor_names = factor_names  # None表示动态取前5个
    
    def generate(self) -> pd.DataFrame:
        self.setup_task()
        
        print(f"\n生成 Qlib {self.factor_set} 因子...")
        
        # 构建数据集 + 提取完整因子集（使用向前扩展的起始日期）
        lookback_start = extend_lookback_start_date(self.start_date)
        task_dir_path = Path(self.task_dir)
        qlib_dir = task_dir_path / 'qlib_data'
        if qlib_dir.exists():
            shutil.rmtree(qlib_dir)
        qlib_dir.mkdir(parents=True, exist_ok=True)
        try:
            build_qlib_dataset(self.stock_codes, lookback_start, self.end_date, qlib_dir, rebuild=True)
            full_factors = extract_factors_from_qlib(qlib_dir, self.factor_set, lookback_start, self.end_date)
        finally:
            if qlib_dir.exists():
                shutil.rmtree(qlib_dir)
        
        if full_factors.empty:
            raise ValueError(f"{self.factor_set} 提取失败")
        
        # 为避免冲突，给 Qlib 因子加前缀
        prefix = f"qlib_{self.factor_set.lower()}_"
        full_factors.columns = [f"{prefix}{col}" for col in full_factors.columns]
        available_factors = full_factors.columns.tolist()
        
        if self.factor_names is None:
            # 默认：所有因子（已前缀）
            selected_cols = available_factors
            print(f"  默认选择所有 {len(selected_cols)} 个因子")
        else:
            # 用户指定原始名 → 加前缀匹配
            prefixed_requested = [f"{prefix}{f}" for f in self.factor_names]
            selected_cols = [f for f in prefixed_requested if f in available_factors]
            if not selected_cols:
                print(f"⚠️ 无匹配因子，可用前10: {available_factors[:10]}...")
                return pd.DataFrame()
        
        factors_df = full_factors[selected_cols]
        
        # 标准格式
        result = factors_df.reset_index()
        result.rename(columns={'code': 'stock_code'}, inplace=True)
        result = format_factor_dataframe(result[['date', 'stock_code'] + selected_cols])
        result = clamp_dataframe_to_date_range(result, self.start_date, self.end_date)
        
        print(f"✓ {len(result)} 条, {len(selected_cols)} 个因子: {selected_cols[:3]}...")
        return result


def generate_qlib_158_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """
    生成 Qlib Alpha158 因子（列名前缀: qlib_alpha158_XXX）
    
    factor_names=None: **所有** ~158 个 (qlib_alpha158_BETA10 等)
    factor_names=['BETA10', 'RSI10']: qlib_alpha158_BETA10 等子集
    """
    generator = QlibFactorGenerator(stock_codes, start_date, end_date, factor_names, 'Alpha158', output_dir)
    df = generator.generate()
    if df.empty: raise Exception("Alpha158 生成失败")
    return generator.save_factors(df)


def generate_qlib_360_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """
    生成 Qlib Alpha360 因子（列名前缀: qlib_alpha360_XXX）
    
    factor_names=None: **所有** ~360 个 (qlib_alpha360_CNTD10 等)
    factor_names=['XXX']: 指定子集
    """
    generator = QlibFactorGenerator(stock_codes, start_date, end_date, factor_names, 'Alpha360', output_dir)
    df = generator.generate()
    if df.empty: raise Exception("Alpha360 生成失败")
    return generator.save_factors(df)


def generate_qlib_158vwap_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """生成 Qlib Alpha158vwap 因子 (VWAP版本)"""
    generator = QlibFactorGenerator(stock_codes, start_date, end_date, factor_names, 'Alpha158vwap', output_dir)
    df = generator.generate()
    if df.empty: raise Exception("Alpha158vwap 生成失败")
    return generator.save_factors(df)


def generate_qlib_360vwap_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """生成 Qlib Alpha360vwap 因子 (VWAP版本)"""
    generator = QlibFactorGenerator(stock_codes, start_date, end_date, factor_names, 'Alpha360vwap', output_dir)
    df = generator.generate()
    if df.empty: raise Exception("Alpha360vwap 生成失败")
    return generator.save_factors(df)
