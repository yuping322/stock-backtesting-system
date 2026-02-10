#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Qlib factor generator test harness.

This script mirrors the helper flow under `src/factor_old` but focuses on
running a single Qlib generator variant and validating that the factor CSV
is emitted and contains the expected columns.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.factor.generator.qlib import (
    generate_qlib_158_factors,
    generate_qlib_360_factors,
    generate_qlib_158vwap_factors,
    generate_qlib_360vwap_factors,
)

try:
    from src.data.data import get_index_stocks
except ImportError:
    get_index_stocks = None

DEFAULT_STOCK_CODES = ['000001', '000002', '000003', '600519']
DEFAULT_START_DATE = '2025-11-25'
DEFAULT_END_DATE = '2025-11-30'

VARIANT_MAP: Dict[str, Callable[..., Dict[str, str]]] = {
    'Alpha158': generate_qlib_158_factors,
    'Alpha360': generate_qlib_360_factors,
    'Alpha158vwap': generate_qlib_158vwap_factors,
    'Alpha360vwap': generate_qlib_360vwap_factors,
}


def choose_stock_codes(preferred_pool: str, limit: int = 4) -> List[str]:
    """Pick a short list of stock codes (falling back to defaults)."""
    if get_index_stocks:
        try:
            codes = get_index_stocks(preferred_pool)
            if isinstance(codes, (list, tuple)):
                filtered = [str(code).zfill(6) for code in codes if code]
            else:
                filtered = []
            if filtered:
                return filtered[:limit]
        except Exception as exc:
            print(f"⚠️ 无法加载指数 {preferred_pool}: {exc}")
    return DEFAULT_STOCK_CODES[:limit]


def verify_factor_csv(factor_file: Path, min_rows: int = 3) -> Tuple[bool, str, Dict[str, str]]:
    """Quick sanity check for the generated factor CSV file."""
    if not factor_file.exists():
        return False, f"因子文件不存在: {factor_file}", {}

    try:
        df = pd.read_csv(factor_file)
    except Exception as exc:
        return False, f"读取因子文件失败: {exc}", {}

    if df.empty:
        return False, "因子文件为空", {}

    required = {'date', 'stock_code'}
    missing = required - set(df.columns)
    if missing:
        return False, f"缺少必要列: {missing}", {}

    stats: Dict[str, str] = {
        'rows': str(len(df)),
        'cols': str(len(df.columns)),
        'codes': str(df['stock_code'].nunique()),
        'date_range': f"{df['date'].min()} ~ {df['date'].max()}",
    }

    if len(df) < min_rows:
        return False, f"因子数据不足: {len(df)} 行 (<{min_rows})", stats

    return True, "因子文件结构看起来正常", stats


def run_variant(
    variant: str,
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
) -> Tuple[bool, str]:
    """Execute a Qlib generator variant and validate the emitted CSV."""
    print(f"\n运行 Qlib 变体: {variant}")
    generator = VARIANT_MAP[variant]

    result = generator(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )

    if isinstance(result, (str, Path)):
        factor_file = Path(result)
    elif isinstance(result, dict):
        factor_file = Path(result.get('factor_file', ''))
    else:
        factor_file = Path('')
    success, message, stats = verify_factor_csv(factor_file)

    print(message)
    if stats:
        print(f"  统计: 行数={stats['rows']}, 列数={stats['cols']}, 股票数={stats['codes']}, 日期={stats['date_range']}")

    if success:
        return True, str(factor_file)
    return False, str(factor_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Standalone Qlib factor generator test')
    parser.add_argument('--variant', choices=list(VARIANT_MAP.keys()), default='Alpha158', help='选择 Qlib 因子集')
    parser.add_argument('--start', default=DEFAULT_START_DATE, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default=DEFAULT_END_DATE, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stock-pool', default='small', help='指数 (默认 small)')
    parser.add_argument('--output', default='./data/factor_tasks', help='因子输出目录')
    return parser.parse_args()


def main():
    args = parse_args()
    stock_codes = choose_stock_codes(args.stock_pool)

    print('Qlib 运行配置:')
    print(f"  变体: {args.variant}")
    print(f"  日期: {args.start} ~ {args.end}")
    print(f"  股票: {stock_codes}")
    print(f"  输出: {args.output}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    success, output = run_variant(
        variant=args.variant,
        stock_codes=stock_codes,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
    )

    if success:
        print(f"\n✅ {args.variant} 因子生成并验证通过: {output}")
        sys.exit(0)
    else:
        print(f"\n❌ {args.variant} 因子生成存在问题，输出文件: {output}")
        sys.exit(1)


if __name__ == '__main__':
    main()
