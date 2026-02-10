import argparse
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
import pytest
import sys
import types

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

from src.backtest import backtest_engine
from src.backtest.backtest_engine import BacktestEngine, BacktestResult
from src.backtest.config import StrategyConfig, SystemConfig
from src.backtest.main import run_backtest


def _make_engine() -> BacktestEngine:
    system_config = SystemConfig(
        initial_cash=1_000_000,
        commission_rate=0.0002,
        slippage_rate=0.0,
    )
    strategy_config = StrategyConfig(strategy_name="weighted_top_n", parameters={})
    return BacktestEngine(system_config, strategy_config)


def test_prepare_prediction_data_normalizes_fields():
    engine = _make_engine()
    raw = pd.DataFrame(
        {
            "code": ["1", "000002"],
            "date": ["2020-01-02", "2020-01-01"],
        }
    )

    prepared = engine._prepare_prediction_data(raw)

    assert list(prepared.columns) == ["code", "date", "weight"]
    assert prepared["code"].tolist() == ["000002", "000001"]
    assert prepared["weight"].tolist() == [1.0, 1.0]
    assert prepared["date"].iloc[0] <= prepared["date"].iloc[1]


@pytest.mark.parametrize(
    "frame,expected",
    [
        (pd.DataFrame({"code": ["1"]}), "预测数据缺少必需的列"),
        (pd.DataFrame({"date": ["2020-01-01"]}), "预测数据缺少必需的列"),
    ],
)
def test_prepare_prediction_data_missing_columns_raises(frame, expected):
    engine = _make_engine()
    with pytest.raises(ValueError) as exc:
        engine._prepare_prediction_data(frame)
    assert expected in str(exc.value)


def test_determine_date_window_valid():
    engine = _make_engine()
    prepared = engine._prepare_prediction_data(
        pd.DataFrame(
            {
                "code": ["1", "2"],
                "date": ["2020-01-02", "2020-01-05"],
            }
        )
    )
    window = engine._determine_date_window(prepared)
    assert window == ("2020-01-02", "2020-01-05")


def test_load_market_feeds_prefers_primary_source(monkeypatch):
    engine = _make_engine()
    dummy_feed = {"000001": "feed"}

    def fake_load_bt_stocks(codes, start, end):
        assert codes == ["000001"]
        assert start == "2020-01-02"
        assert end == "2020-01-05"
        return dummy_feed

    def fake_generate(*args, **kwargs):
        raise AssertionError("fallback should not be called when primary feed exists")

    monkeypatch.setattr(backtest_engine, "load_bt_stocks", fake_load_bt_stocks)
    monkeypatch.setattr(BacktestEngine, "_generate_synthetic_feeds", fake_generate)

    feeds = engine._load_market_feeds(["000001"], "2020-01-02", "2020-01-05", pd.DataFrame())
    assert feeds is dummy_feed


def test_load_market_feeds_falls_back_to_synthetic(monkeypatch):
    engine = _make_engine()
    called = {}

    def fake_load_bt_stocks(codes, start, end):
        return {}

    def fake_generate(self, codes, start, end, pred_df):
        called["args"] = (tuple(codes), start, end)
        return {"synthetic": "feed"}

    monkeypatch.setattr(backtest_engine, "load_bt_stocks", fake_load_bt_stocks)
    monkeypatch.setattr(BacktestEngine, "_generate_synthetic_feeds", fake_generate)

    feeds = engine._load_market_feeds(["000001"], "2020-01-02", "2020-01-05", pd.DataFrame())
    assert feeds == {"synthetic": "feed"}
    assert called["args"] == (("000001",), "2020-01-02", "2020-01-05")


def test_build_strategy_runtime_config_includes_passthrough_fields():
    params = {
        "hold_days": 3,
        "top_n_stocks": 5,
        "commission_rate": 0.001,
        "weight_column": "score",
        "custom_flag": True,
    }
    config_obj = BacktestEngine._build_strategy_runtime_config(params)
    assert config_obj.hold_days == 3
    assert config_obj.top_n_stocks == 5
    assert config_obj.commission_rate == 0.001
    assert config_obj.weight_column == "score"
    assert config_obj.custom_flag is True
    assert config_obj.parameters == params


def test_backtest_result_roundtrip():
    index = pd.date_range("2020-01-02", periods=2, freq="D")
    result = BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1], index=index, name="strategy_nav"),
        benchmark_nav=pd.Series([1.0, 1.05], index=index, name="benchmark_nav"),
        performance=pd.DataFrame({"value": [0.1]}, index=["total_return"]),
        detailed_metrics={"drawdown_series": pd.Series([0, -0.02], index=index)},
        monthly_stats=pd.DataFrame({"Strategy_Return": [0.1]}, index=[pd.Timestamp("2020-01-31")]),
        yearly_stats=pd.DataFrame({"Strategy_Return": [0.1]}, index=[pd.Timestamp("2020-12-31")]),
        trade_history=[{"date": datetime(2020, 1, 2)}],
        daily_holdings=[{"date": datetime(2020, 1, 2)}],
        final_value=123.45,
        valid_stocks=2,
        strategy_name="weighted_top_n",
        file_name="sample.csv",
    )

    roundtrip = BacktestResult.from_dict(result.to_dict())
    assert roundtrip.strategy_name == result.strategy_name
    assert roundtrip.final_value == pytest.approx(result.final_value)
    assert list(roundtrip.strategy_nav.index) == list(result.strategy_nav.index)


def test_metrics_with_nan_nav():
    engine = _make_engine()
    nav = pd.Series([float('nan')] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.array([float('nan')] * 9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0

def test_metrics_with_inf_nav():
    engine = _make_engine()
    nav = pd.Series([float('inf')] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.array([float('inf')] * 9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0

def test_metrics_with_zero_nav():
    engine = _make_engine()
    nav = pd.Series([0.0] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.zeros(9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0
    # 新增行业集中度相关占位指标
    assert 'industry_hhi' in metrics['performance_df'].index
    assert 'top_industry_weight' in metrics['performance_df'].index
    assert 'industry_count' in metrics['performance_df'].index

def test_empty_holdings_and_trades():
    index = pd.date_range("2020-01-02", periods=2, freq="D")
    result = BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1], index=index, name="strategy_nav"),
        benchmark_nav=pd.Series([1.0, 1.05], index=index, name="benchmark_nav"),
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[],
        daily_holdings=[],
        final_value=1.1,
        valid_stocks=2,
        strategy_name="weighted_top_n",
        file_name="sample.csv",
    )
    holdings = backtest_engine.AnalysisBuilder.prepare_holdings(result)
    trades = backtest_engine.AnalysisBuilder.prepare_trades(result)
    assert holdings['holdings_table'].empty
    assert trades['trades_table'].empty

def test_period_stats_with_single_day():
    engine = _make_engine()
    index = pd.date_range("2020-01-01", periods=1, freq="D")
    nav = pd.Series([1.0], index=index)
    bench = pd.Series([1.0], index=index)
    stats = engine.generate_period_stats(nav, bench, 'M')
    assert stats.empty

def test_prepare_prediction_data_extreme_params():
    engine = _make_engine()
    df = pd.DataFrame({
        "code": ["000001"],
        "date": ["2020-01-01"],
        "weight": [1.0],
    })
    # hold_days=0, top_n_stocks=0 should be clamped to >=1
    engine.strategy_config.parameters["hold_days"] = 0
    engine.strategy_config.parameters["top_n_stocks"] = 0
    prepared = engine._prepare_prediction_data(df)
    assert (prepared["weight"] == 1.0).all()


def test_run_backtest_returns_expected_result(monkeypatch, tmp_path):
    args = argparse.Namespace(
        data_file=str(tmp_path / "pred.csv"),
        strategy="weighted_top_n",
        benchmark="sh000300",
        initial_cash=1_000_000.0,
        commission=0.0002,
        slippage=0.0,
        hold_days=None,
        top_n=None,
        start_date="2020-01-02",
        end_date=None,
        output_dir=str(tmp_path / "results"),
    )

    raw_df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "code": ["000001", "000001"],
            "weight": [1.0, 1.0],
        }
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    pred_path = Path(args.data_file)
    pred_path.write_text("code,date,weight\n")

    captured = {}

    def fake_load_prediction(path):
        captured["loaded_path"] = path
        return raw_df

    index = pd.date_range("2020-01-02", periods=2, freq="D")
    fake_result_dict = {
        "strategy_nav": pd.Series([1.0, 1.05], index=index, name="strategy_nav"),
        "benchmark_nav": pd.Series([1.0, 1.02], index=index, name="benchmark_nav"),
        "performance": pd.DataFrame({"value": [0.05]}, index=["total_return"]),
        "detailed_metrics": {"drawdown_series": pd.Series([0.0, -0.01], index=index)},
        "monthly_stats": pd.DataFrame({"Strategy_Return": [0.05]}, index=[pd.Timestamp("2020-01-31")]),
        "yearly_stats": pd.DataFrame({"Strategy_Return": [0.05]}, index=[pd.Timestamp("2020-12-31")]),
        "trade_history": [],
        "daily_holdings": [],
        "final_value": 1_050_000.0,
        "valid_stocks": 1,
        "strategy_name": "weighted_top_n",
        "file_name": "pred.csv",
    }

    def fake_run(self, df, strategy_name, file_name):
        captured["strategy_name"] = strategy_name
        captured["file_name"] = file_name
        captured["dates"] = df["date"].dt.strftime("%Y-%m-%d").tolist()
        return fake_result_dict

    monkeypatch.setattr(backtest_engine.DataLoader, "load_prediction_data", staticmethod(fake_load_prediction))
    monkeypatch.setattr(BacktestEngine, "run_single_backtest", fake_run)

    system_config, result = run_backtest(args)

    assert captured["loaded_path"] == args.data_file
    assert captured["strategy_name"] == args.strategy
    assert captured["file_name"] == "pred.csv"
    assert captured["dates"] == ["2020-01-02"]
    assert isinstance(result, BacktestResult)
    assert result.final_value == pytest.approx(1_050_000.0)
    assert system_config.benchmark_index == args.benchmark