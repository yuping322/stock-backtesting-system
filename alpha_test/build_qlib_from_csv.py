"""Convert simple per-row symbol CSV into minimal qlib data directory.

Input CSV format (one file or a folder of files):
    date,symbol,open,high,low,close,vwap,volume
    2024-12-30,AAA,10.0,10.5,9.8,10.3,10.25,120000

If vwap column missing, we will substitute close.

Usage:
    python examples/build_qlib_from_csv.py --src examples/data_raw/fake_prices.csv --dest examples/mini_qlib_data
    python examples/build_qlib_from_csv.py --src examples/data_raw --dest examples/mini_qlib_data

Then run:
    python examples/quick_start_alpha_workflows.py --data examples/mini_qlib_data --show-alpha158

Result directory structure:
    mini_qlib_data/
        calendars/day.txt
        instruments/all.txt
        features/<symbol>/open.day.bin ...

This is a minimal implementation inspired by scripts/dump_bin.py but simplified.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

CAL_DIR = "calendars"
INS_DIR = "instruments"
FEAT_DIR = "features"
BIN_SUFFIX = ".day.bin"
FIELDS = ["open", "high", "low", "close", "vwap", "volume"]


def load_source(src: str) -> pd.DataFrame:
    p = Path(src).expanduser()
    if p.is_dir():
        dfs = []
        for f in sorted(p.glob("*.csv")):
            dfs.append(pd.read_csv(f))
        if not dfs:
            raise FileNotFoundError("No csv files found in folder")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    if 'vwap' not in df.columns:
        df['vwap'] = df['close']
    missing_cols = [c for c in FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return df


def build_calendars(dates: pd.Series, dest: Path):
    cal_dir = dest / CAL_DIR
    cal_dir.mkdir(parents=True, exist_ok=True)
    all_days = sorted(set(dates))
    out_path = cal_dir / 'day.txt'
    with out_path.open('w') as f:
        for d in all_days:
            f.write(f"{pd.Timestamp(d).date()}\n")
    return all_days


def build_instruments(df: pd.DataFrame, all_days: list, dest: Path):
    ins_dir = dest / INS_DIR
    ins_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for sym, g in df.groupby('symbol'):
        start = g['date'].min().date()
        end = g['date'].max().date()
        lines.append(f"{sym}\t{start}\t{end}")
    out_path = ins_dir / 'all.txt'
    with out_path.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def build_feature_bins(df: pd.DataFrame, all_days: list, dest: Path):
    feat_root = dest / FEAT_DIR
    feat_root.mkdir(parents=True, exist_ok=True)
    day_index_map = {d: i for i, d in enumerate(all_days)}
    for sym, g in df.groupby('symbol'):
        sym_dir = feat_root / sym.lower()
        sym_dir.mkdir(parents=True, exist_ok=True)
        g = g.drop_duplicates('date').set_index('date').sort_index()
        # reindex to all_days
        full_idx = pd.to_datetime(all_days)
        aligned = g.reindex(full_idx)
        for field in FIELDS:
            arr = aligned[field].to_numpy(dtype='float32')
            # build binary: first number is start index, then values
            if aligned.index.size == 0:
                continue
            start_idx = day_index_map[aligned.index.min()] if aligned.index.min() in day_index_map else 0
            out = np.hstack([np.array([start_idx], dtype='float32'), arr])
            bin_path = sym_dir / f"{field}.{BIN_SUFFIX}"
            with open(bin_path, 'wb') as f:
                out.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Source CSV file or directory containing multiple csv files.')
    ap.add_argument('--dest', required=True, help='Destination qlib style directory.')
    args = ap.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    logger.info('Loading source data...')
    df = load_source(args.src)
    logger.info(f'Source rows: {len(df)}; symbols: {df.symbol.nunique()}')

    logger.info('Building calendars...')
    all_days = build_calendars(df['date'], dest)
    logger.info(f'Calendar days: {len(all_days)}')

    logger.info('Building instruments...')
    build_instruments(df, all_days, dest)

    logger.info('Building feature bins...')
    build_feature_bins(df, all_days, dest)
    logger.info('Done. Mount path ready: %s', dest)
    print(f"Finished building qlib data at {dest}")

if __name__ == '__main__':
    main()
