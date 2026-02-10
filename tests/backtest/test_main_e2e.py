import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


if "alphalens" not in sys.modules:
    alphalens_module = types.ModuleType("alphalens")
    sys.modules["alphalens"] = alphalens_module

    performance_module = types.ModuleType("alphalens.performance")
    performance_module.factor_returns = lambda *a, **k: pd.Series(dtype=float)
    performance_module.mean_information_coefficient = lambda *a, **k: pd.Series(dtype=float)
    performance_module.mean_return_by_quantile = lambda *a, **k: (pd.DataFrame(), pd.DataFrame())
    sys.modules["alphalens.performance"] = performance_module

    utils_module = types.ModuleType("alphalens.utils")
    utils_module.get_clean_factor_and_forward_returns = lambda *a, **k: pd.DataFrame()
    sys.modules["alphalens.utils"] = utils_module

if "oss2" not in sys.modules:
    class _DummyAuth:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyBucket:
        def __init__(self, *args, **kwargs):
            pass

    def _dummy_iterator(*args, **kwargs):
        return iter(())

    exceptions_module = types.ModuleType("oss2.exceptions")
    exceptions_module.NoSuchKey = FileNotFoundError
    oss2_module = types.ModuleType("oss2")
    oss2_module.Auth = _DummyAuth
    oss2_module.Bucket = _DummyBucket
    oss2_module.ObjectIterator = _dummy_iterator
    oss2_module.exceptions = exceptions_module
    sys.modules["oss2"] = oss2_module
    sys.modules["oss2.exceptions"] = exceptions_module

import src.backtest.backtest_engine as backtest_engine
from src.backtest.backtest_engine import BacktestResult
from src.backtest.main import (
    _persist_results,
    _write_markdown_report,
    build_markdown_report,
    run_backtest,
)

def test_end_to_end_backtest_with_sample_predictions(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / "data" / "test_sample_predictions.csv"
    assert data_file.exists(), "示例预测数据缺失"

    args = argparse.Namespace(
        data_file=str(data_file),
        strategy="weighted_top_n",
        benchmark="sh000300",
        initial_cash=1_000_000.0,
        commission=0.0002,
        slippage=0.0,
        hold_days=None,
        top_n=None,
        start_date=None,
        end_date=None,
        output_dir=str(tmp_path / "results"),
    )

    def fake_load_bt_stocks(codes, start, end):
        return {}

    def fake_get_index_daily(index_symbol, start, end):
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts
        index = pd.date_range(start_ts, end_ts, freq="B")
        if index.empty:
            index = pd.DatetimeIndex([start_ts])
        nav_values = np.linspace(1.0, 1.05, len(index))
        return pd.Series(nav_values, index=index, name="nav")

    monkeypatch.setattr(backtest_engine, "load_bt_stocks", fake_load_bt_stocks)
    monkeypatch.setattr(backtest_engine, "get_index_daily", fake_get_index_daily)

    system_config, result = run_backtest(args)

    assert isinstance(result, BacktestResult)
    assert len(result.strategy_nav) > 0
    assert result.strategy_nav.index.is_monotonic_increasing
    assert result.valid_stocks > 0
    assert "total_return" in result.performance.index
    assert system_config.benchmark_index == args.benchmark

    persist_dir = tmp_path / "persisted"
    _persist_results(persist_dir, result)

    nav_path = persist_dir / "strategy_nav.csv"
    perf_path = persist_dir / "performance_metrics.csv"
    assert nav_path.exists()
    assert perf_path.exists()
    saved_nav = pd.read_csv(nav_path)
    saved_perf = pd.read_csv(perf_path)
    assert not saved_nav.empty
    assert not saved_perf.empty

    report_path = persist_dir / "analysis_report.md"
    report_content = build_markdown_report(system_config, result)
    _write_markdown_report(report_path, report_content)
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Backtest Report" in report_text
    assert "Performance Metrics" in report_text