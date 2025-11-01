"""Parameter sweep utility for backtesting strategies.

Runs combinations of strategy parameters over a given prediction file (or directory)
and outputs a ranked CSV of performance metrics (Sharpe, annual_return, max_drawdown, etc.).

Usage example:
    python scripts/backtest_param_sweep.py \
        --data-file data/test_sample_predictions.csv \
        --strategy weighted_top_n \
        --top-n 5 10 15 \
        --hold-days 2 3 5 \
        --benchmark sh000300 \
        --out results/param_sweep.csv

Supports random sampling via --random-sample N (samples N random param sets from the cartesian product).

This is a lightweight wrapper around main.run_backtest to evaluate parameter grids.
"""
from __future__ import annotations

import argparse
import sys
import itertools
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ensure repository root on sys.path when executed from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import run_backtest  # type: ignore
from backtest_engine import BacktestResult


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest parameter sweep")
    p.add_argument("--data-file", required=True, help="Prediction CSV file")
    p.add_argument("--strategy", default="weighted_top_n", help="Strategy name")
    p.add_argument("--top-n", nargs="*", type=int, default=[10], help="List of top_n values")
    p.add_argument("--hold-days", nargs="*", type=int, default=[3], help="List of hold_days values")
    p.add_argument("--benchmark", default="sh000300")
    p.add_argument("--commission", type=float, default=0.0002)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--initial-cash", type=float, default=1_000_000)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--random-sample", type=int, default=0, help="If >0 randomly sample that many param combinations")
    p.add_argument("--out", default="results/param_sweep.csv", help="Output CSV path")
    p.add_argument("--rank-metric", default="sharpe_ratio", help="Metric to rank by (must exist in performance_metrics.csv)")
    return p.parse_args()


def build_param_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    grid = list(itertools.product(args.top_n, args.hold_days))
    if args.random_sample and args.random_sample < len(grid):
        random.shuffle(grid)
        grid = grid[: args.random_sample]
    param_dicts: List[Dict[str, Any]] = []
    for top_n, hold_days in grid:
        param_dicts.append({"top_n": top_n, "hold_days": hold_days})
    return param_dicts


def run_single_config(base_args: argparse.Namespace, top_n: int, hold_days: int) -> Dict[str, Any]:
    # clone args for isolation
    class A:  # simple namespace clone
        pass
    cloned = A()
    for k, v in vars(base_args).items():
        setattr(cloned, k, v)
    # override strategy params
    setattr(cloned, "top_n", top_n)
    setattr(cloned, "hold_days", hold_days)

    try:
        system_config, result = run_backtest(cloned, None)
    except Exception as e:
        return {
            "top_n": top_n,
            "hold_days": hold_days,
            "error": str(e),
        }

    row = extract_metrics(result)
    row.update({
        "top_n": top_n,
        "hold_days": hold_days,
    })
    return row


def extract_metrics(result: BacktestResult) -> Dict[str, Any]:
    metrics = {}
    perf_df = result.performance
    for metric, row in perf_df.iterrows():
        metrics[metric] = row.get("value")
    metrics["final_value"] = result.final_value
    return metrics


def main():
    args = parse_args()
    param_grid = build_param_grid(args)
    print(f"Running {len(param_grid)} parameter combinations...")

    rows = []
    for i, param in enumerate(param_grid, 1):
        print(f"[{i}/{len(param_grid)}] top_n={param['top_n']} hold_days={param['hold_days']}")
        row = run_single_config(args, param['top_n'], param['hold_days'])
        rows.append(row)

    df = pd.DataFrame(rows)
    # rank
    if args.rank_metric in df.columns:
        df = df.sort_values(args.rank_metric, ascending=False)
    else:
        print(f"Rank metric {args.rank_metric} not found; skipping ranking.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved sweep results to {out_path}")

    # print top line summary
    if not df.empty:
        print("Top result:")
        print(df.head(1).T)


if __name__ == "__main__":
    main()
