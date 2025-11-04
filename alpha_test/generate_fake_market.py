"""Generate a larger synthetic daily market CSV for testing.

Creates symbols S0001..S0100 across ~260 trading days with random walk prices and volumes.

Usage:
    python examples/alpha_test/generate_fake_market.py --dest examples/alpha_test/fake_market.csv --days 260 --symbols 100
Then convert:
    python examples/alpha_test/build_qlib_from_csv.py --src examples/alpha_test/fake_market.csv --dest examples/alpha_test/qlib_big
Run quick start with portfolio:
    python examples/alpha_test/quick_start_alpha_workflows.py --data examples/alpha_test/qlib_big --with-port --show-alpha158
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TRADING_DAY_SKIP = {5,6}  # skip Saturday(5), Sunday(6) using weekday() where Monday=0


def gen_dates(n_days: int, start: str = '2024-01-01'):
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    dates = []
    cur = start_dt
    while len(dates) < n_days:
        if cur.weekday() not in TRADING_DAY_SKIP:
            dates.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)
    return dates


def gen_symbols(n: int):
    return [f'S{str(i+1).zfill(4)}' for i in range(n)]


def synthesize(dates, symbols, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for sym in symbols:
        base_price = rng.uniform(10, 100)
        price = base_price
        for d in dates:
            # random walk
            drift = rng.normal(0, 0.5)
            price = max(1.0, price + drift)
            high = price + rng.uniform(0, 1)
            low = max(0.5, price - rng.uniform(0, 1))
            open_p = price + rng.normal(0, 0.2)
            close = price + rng.normal(0, 0.2)
            vwap = (open_p + close + high + low) / 4.0
            volume = int(rng.uniform(50000, 500000))
            rows.append([d, sym, open_p, high, low, close, vwap, volume])
    df = pd.DataFrame(rows, columns=['date','symbol','open','high','low','close','vwap','volume'])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', required=True, help='Destination CSV path.')
    ap.add_argument('--days', type=int, default=260, help='Approx trading days to generate.')
    ap.add_argument('--symbols', type=int, default=100, help='Number of synthetic symbols.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    dates = gen_dates(args.days)
    symbols = gen_symbols(args.symbols)
    df = synthesize(dates, symbols, seed=args.seed)
    df.to_csv(args.dest, index=False)
    print(f'Generated synthetic market CSV: {args.dest} rows={len(df)} symbols={args.symbols} days={args.days}')

if __name__ == '__main__':
    main()
