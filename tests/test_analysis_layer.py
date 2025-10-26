import sys
import types
from datetime import datetime

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

import backtest_engine
from backtest_engine import AnalysisBuilder, BacktestResult
from config import SystemConfig


@pytest.fixture
def system_config() -> SystemConfig:
    return SystemConfig(
        initial_cash=1_000_000,
        commission_rate=0.0002,
        slippage_rate=0.0,
        show_plots=False,
        save_results=False,
        benchmark_index="sh000300",
    )


@pytest.fixture
def sample_result() -> BacktestResult:
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    strategy_nav = pd.Series([1_000_000, 1_010_000, 1_020_000, 1_005_000, 1_030_000], index=dates, name="strategy_nav")
    benchmark_nav = pd.Series([1_000_000, 1_008_000, 1_012_000, 1_000_000, 1_015_000], index=dates, name="benchmark_nav")

    performance = pd.DataFrame(
        {
            "value": [0.03, 0.18, -0.015],
        },
        index=["total_return", "annual_return", "max_drawdown"],
    )

    drawdown = strategy_nav / strategy_nav.cummax() - 1
    running_max = strategy_nav.cummax()

    monthly = pd.DataFrame(
        {
            "Strategy_Return": [0.012, 0.008],
            "Benchmark_Return": [0.010, 0.006],
            "Excess_Return": [0.002, 0.002],
            "Win": [True, True],
        },
        index=pd.to_datetime(["2023-01-31", "2023-02-28"]),
    )

    yearly = pd.DataFrame(
        {
            "Strategy_Return": [0.15],
            "Benchmark_Return": [0.10],
            "Excess_Return": [0.05],
            "Win": [True],
        },
        index=pd.to_datetime(["2023-12-31"]),
    )

    trade_history = [
        {
            "date": datetime(2023, 1, 3),
            "code": "000001",
            "action": "BUY",
            "size": 100,
            "price": 10.0,
            "value": 1_000.0,
            "portfolio_value": 1_010_000.0,
        },
        {
            "date": datetime(2023, 1, 5),
            "code": "000001",
            "action": "SELL",
            "size": -100,
            "price": 10.5,
            "value": 1_050.0,
            "portfolio_value": 1_030_000.0,
        },
    ]

    daily_holdings = [
        {
            "date": dates[i].date(),
            "holdings": [
                {
                    "code": "000001",
                    "size": 100,
                    "price": 10.0 + i,
                    "value": 1000.0 + i * 10,
                    "weight": 0.5,
                    "buy_date": dates[max(i - 1, 0)].date(),
                },
            ],
            "total_value": float(strategy_nav.iloc[i]),
            "cash": float(strategy_nav.iloc[i] - (1000.0 + i * 10)),
        }
        for i in range(len(dates))
    ]

    return BacktestResult(
        strategy_nav=strategy_nav,
        benchmark_nav=benchmark_nav,
        performance=performance,
        detailed_metrics={"drawdown_series": drawdown, "running_max": running_max},
        monthly_stats=monthly,
        yearly_stats=yearly,
        trade_history=trade_history,
        daily_holdings=daily_holdings,
        final_value=float(strategy_nav.iloc[-1]),
        valid_stocks=3,
        strategy_name="weighted_top_n",
        file_name="sample.csv",
    )


def test_prepare_overview(sample_result, system_config):
    overview = AnalysisBuilder.prepare_overview(sample_result, system_config, ["total_return", "annual_return"])

    assert overview["summary"]["final_value"] == pytest.approx(sample_result.final_value)
    assert overview["summary"]["benchmark"] == system_config.benchmark_index
    assert len(overview["metrics"]) == 2
    assert {m["metric"] for m in overview["metrics"]} == {"total_return", "annual_return"}


def test_prepare_net_value(sample_result):
    net_value = AnalysisBuilder.prepare_net_value(sample_result)

    strategy_nav = net_value["strategy_nav"]
    benchmark_nav = net_value["benchmark_nav"]
    assert pytest.approx(strategy_nav.iloc[0]) == 1.0
    assert pytest.approx(benchmark_nav.iloc[0]) == 1.0
    assert "relative_nav" in net_value


def test_prepare_returns(sample_result):
    returns = AnalysisBuilder.prepare_returns(sample_result)

    assert returns["daily_stats"]["sample_size"] == len(sample_result.strategy_nav) - 1
    assert not returns["daily_returns"].empty
    assert returns["monthly_table"].equals(sample_result.monthly_stats)


def test_prepare_risk(sample_result):
    risk = AnalysisBuilder.prepare_risk(sample_result)

    metrics = risk["risk_metrics"]
    assert "var_95" in metrics
    assert "tracking_error" in metrics
    assert risk["drawdown_series"].equals(sample_result.detailed_metrics["drawdown_series"])


def test_prepare_period_stats(sample_result):
    stats = AnalysisBuilder.prepare_period_stats(sample_result)

    assert stats["monthly"].equals(sample_result.monthly_stats)
    assert stats["yearly"].equals(sample_result.yearly_stats)
    assert stats["monthly_win_rate"] == pytest.approx(1.0)


def test_prepare_holdings(sample_result):
    holdings = AnalysisBuilder.prepare_holdings(sample_result)

    assert "holdings_table" in holdings
    assert not holdings["holdings_table"].empty
    assert holdings["latest_snapshot"]["total_value"] == sample_result.final_value


def test_prepare_trades(sample_result):
    trades = AnalysisBuilder.prepare_trades(sample_result)

    assert trades["trade_count"] == len(sample_result.trade_history)
    assert list(trades["trades_table"]["action"]) == [
        entry["action"] for entry in sample_result.trade_history
    ]
