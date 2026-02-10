"""Diagnostics for large factor CSVs produced by Qlib or other generators.

Usage:
    from src.factor.merger.check import check_factor_file
    check_factor_file(path, out_dir)

CLI:
    python -m src.factor.merger.check --input <csv> --out results/merger_checks
"""
import argparse
import os
from pathlib import Path
import json
import time
import pandas as pd
from typing import Optional


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def check_factor_file(
    csv_path: str,
    out_dir: Optional[str] = None,
    chunksize: int = 200_000,
    sample_rows: int = 5,
):
    """Stream a large factor CSV and produce diagnostics.

    Outputs saved under out_dir/<timestamp>/:
      - columns_summary.csv  (col, non_null_count, pct_non_null, dtype)
      - per_date_counts.csv  (date, rows, non_null_counts for a subset)
      - per_stock_counts.csv (stock_code, rows, non_null_counts)
      - report.json

    Returns the report dict.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)

    ts = int(time.time())
    out_root = Path(out_dir) if out_dir else Path('results') / 'merger_checks'
    out_root = out_root / str(ts)
    _ensure_dir(out_root)

    # read header to get columns
    with p.open('r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')

    # columns of interest heuristics
    has_date = 'date' in header
    has_stock = 'stock_code' in header or 'instrument' in header or 'symbol' in header

    total_rows = 0
    col_notna = None
    col_dtypes = {}
    date_min = None
    date_max = None
    per_date_counts = {}
    per_stock_counts = {}
    sample = []

    reader = pd.read_csv(p, chunksize=chunksize, parse_dates=['date'] if has_date else None)
    for chunk in reader:
        total_rows += len(chunk)
        if col_notna is None:
            col_notna = chunk.notna().sum()
        else:
            col_notna += chunk.notna().sum()

        # capture dtypes from first chunk
        if not col_dtypes:
            for c, t in chunk.dtypes.items():
                col_dtypes[c] = str(t)

        if has_date and 'date' in chunk.columns:
            dmin = chunk['date'].min()
            dmax = chunk['date'].max()
            date_min = dmin if date_min is None or dmin < date_min else date_min
            date_max = dmax if date_max is None or dmax > date_max else date_max
            # per-date counts: number of rows present per date
            per_date = chunk.groupby('date').size()
            for d, cnt in per_date.items():
                per_date_counts[str(d.date())] = per_date_counts.get(str(d.date()), 0) + int(cnt)

        # per-stock counts
        stock_col = None
        for cand in ('stock_code', 'instrument', 'symbol'):
            if cand in chunk.columns:
                stock_col = cand
                break
        if stock_col is not None:
            per_stock = chunk.groupby(stock_col).size()
            for s, cnt in per_stock.items():
                per_stock_counts[str(s)] = per_stock_counts.get(str(s), 0) + int(cnt)

        if len(sample) < sample_rows:
            sample.extend(chunk.head(sample_rows).to_dict('records'))

    if col_notna is None:
        raise ValueError('CSV had no data rows')

    # prepare columns summary DataFrame
    import math

    cols = list(col_notna.index)
    summary = []
    for c in cols:
        nn = int(col_notna[c])
        pct = float(nn) / float(total_rows) if total_rows else 0.0
        dtype = col_dtypes.get(c, '')
        summary.append({'column': c, 'non_null_count': nn, 'pct_non_null': pct, 'dtype': dtype})

    df_cols = pd.DataFrame(summary).sort_values('pct_non_null')
    cols_out = out_root / 'columns_summary.csv'
    df_cols.to_csv(cols_out, index=False)

    # per-date summary
    if per_date_counts:
        df_date = pd.DataFrame([{'date': d, 'rows': c} for d, c in per_date_counts.items()])
        df_date = df_date.sort_values('date')
        df_date.to_csv(out_root / 'per_date_counts.csv', index=False)
    else:
        df_date = None

    # per-stock summary (top offenders)
    if per_stock_counts:
        df_stock = pd.DataFrame([{'stock': s, 'rows': c} for s, c in per_stock_counts.items()])
        df_stock = df_stock.sort_values('rows', ascending=False)
        df_stock.to_csv(out_root / 'per_stock_counts.csv', index=False)
    else:
        df_stock = None

    # JSON report
    report = {
        'input': str(p),
        'out_dir': str(out_root),
        'size_bytes': p.stat().st_size,
        'total_rows': int(total_rows),
        'total_columns': len(cols),
        'date_min': str(date_min) if date_min is not None else None,
        'date_max': str(date_max) if date_max is not None else None,
        'columns_summary_file': str(cols_out),
        'per_date_counts_file': str(out_root / 'per_date_counts.csv') if df_date is not None else None,
        'per_stock_counts_file': str(out_root / 'per_stock_counts.csv') if df_stock is not None else None,
    }

    with (out_root / 'report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # save a tiny sample
    with (out_root / 'sample_rows.json').open('w', encoding='utf-8') as f:
        # Some sample values can be pandas Timestamp; use default=str to serialize them
        json.dump(sample[:sample_rows], f, indent=2, ensure_ascii=False, default=str)

    print('Wrote diagnostics to', out_root)
    print('Total rows', total_rows, 'columns', len(cols))
    print('Date range', report['date_min'], '->', report['date_max'])
    print('Columns summary saved to', cols_out)

    return report


def main():
    parser = argparse.ArgumentParser(description='Check large factor CSV files for missingness and basic stats')
    parser.add_argument('--input', '-i', required=True, help='Path to factor CSV file')
    parser.add_argument('--out', '-o', default='results/merger_checks', help='Output base directory')
    parser.add_argument('--chunksize', type=int, default=200000)
    args = parser.parse_args()

    report = check_factor_file(args.input, out_dir=args.out, chunksize=args.chunksize)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
