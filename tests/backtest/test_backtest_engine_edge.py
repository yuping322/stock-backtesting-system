import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from src.backtest.backtest_engine import BacktestEngine, BacktestResult
from src.backtest.config import SystemConfig, StrategyConfig

# 构造最小化配置的引擎

def _make_engine():
    system_config = SystemConfig(
        initial_cash=1_000_000,
        commission_rate=0.0002,
        slippage_rate=0.0,
    )
    strategy_config = StrategyConfig(strategy_name="weighted_top_n", parameters={})
    return BacktestEngine(system_config, strategy_config)

# 1. 概览/净值/收益边界

def test_nav_all_nan():
    engine = _make_engine()
    nav = pd.Series([float('nan')] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.array([float('nan')] * 9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0

def test_nav_all_inf():
    engine = _make_engine()
    nav = pd.Series([float('inf')] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.array([float('inf')] * 9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0

def test_nav_all_zero():
    engine = _make_engine()
    nav = pd.Series([0.0] * 10, index=pd.date_range("2020-01-01", periods=10))
    metrics = engine.calculate_detailed_metrics(nav, nav, np.zeros(9))
    for v in metrics['performance_df']['value']:
        assert v == 0.0

# 2. 收益率全负/全正/全零

def test_returns_all_negative():
    engine = _make_engine()
    nav = pd.Series(np.linspace(1, 0.1, 10), index=pd.date_range("2020-01-01", periods=10))
    returns = nav.pct_change().dropna().values
    metrics = engine.calculate_detailed_metrics(nav, nav, returns)
    assert metrics['performance_df'].loc['win_rate', 'value'] == 0.0

def test_returns_all_positive():
    engine = _make_engine()
    nav = pd.Series(np.linspace(1, 2, 10), index=pd.date_range("2020-01-01", periods=10))
    returns = nav.pct_change().dropna().values
    metrics = engine.calculate_detailed_metrics(nav, nav, returns)
    assert metrics['performance_df'].loc['win_rate', 'value'] == 1.0

def test_returns_all_zero():
    engine = _make_engine()
    nav = pd.Series([1.0] * 10, index=pd.date_range("2020-01-01", periods=10))
    returns = nav.pct_change().dropna().values
    metrics = engine.calculate_detailed_metrics(nav, nav, returns)
    assert metrics['performance_df'].loc['win_rate', 'value'] == 0.0

# 3. 持仓/交易为空

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
    from src.backtest.backtest_engine import AnalysisBuilder
    holdings = AnalysisBuilder.prepare_holdings(result)
    trades = AnalysisBuilder.prepare_trades(result)
    assert holdings['holdings_table'].empty
    assert trades['trades_table'].empty

# 4. 极端日期窗口

def test_period_stats_with_single_day():
    engine = _make_engine()
    index = pd.date_range("2020-01-01", periods=1, freq="D")
    nav = pd.Series([1.0], index=index)
    bench = pd.Series([1.0], index=index)
    stats = engine.generate_period_stats(nav, bench, 'M')
    assert stats.empty

# 5. benchmark_nav 与 strategy_nav 长度不一致

def test_nav_length_mismatch():
    engine = _make_engine()
    nav1 = pd.Series([1.0, 1.1, 1.2], index=pd.date_range("2020-01-01", periods=3))
    nav2 = pd.Series([1.0, 1.05], index=pd.date_range("2020-01-01", periods=2))
    returns = nav1.pct_change().dropna().values
    metrics = engine.calculate_detailed_metrics(nav1, nav2, returns)
    # 只要不报错即可
    assert isinstance(metrics['performance_df'], pd.DataFrame)

# 6. 预测数据缺失/格式错误

def test_prepare_prediction_data_missing_columns():
    engine = _make_engine()
    with pytest.raises(ValueError):
        engine._prepare_prediction_data(pd.DataFrame({"code": ["1"]}))
    with pytest.raises(ValueError):
        engine._prepare_prediction_data(pd.DataFrame({"date": ["2020-01-01"]}))
    with pytest.raises(ValueError):
        engine._prepare_prediction_data(pd.DataFrame())

# 7. 合成行情生成失败

def test_generate_synthetic_feeds_empty():
    engine = _make_engine()
    feeds = engine._generate_synthetic_feeds([], "2020-01-01", "2020-01-10", pd.DataFrame())
    assert feeds == {}

# 8. 参数极端值

def test_prepare_prediction_data_extreme_params():
    engine = _make_engine()
    df = pd.DataFrame({
        "code": ["000001"],
        "date": ["2020-01-01"],
        "weight": [1.0],
    })
    engine.strategy_config.parameters["hold_days"] = 0
    engine.strategy_config.parameters["top_n_stocks"] = 0
    prepared = engine._prepare_prediction_data(df)
    assert (prepared["weight"] == 1.0).all()
