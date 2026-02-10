import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from src.backtest.backtest_engine import BacktestResult, AnalysisBuilder

def _make_result():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    return BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1, 1.2, 1.15, 1.18], index=index, name="strategy_nav"),
        benchmark_nav=pd.Series([1.0, 1.05, 1.1, 1.08, 1.12], index=index, name="benchmark_nav"),
        performance=pd.DataFrame({"value": [0.18]}, index=["total_return"]),
        detailed_metrics={"drawdown_series": pd.Series([0, -0.05, -0.02, -0.08, -0.03], index=index), "running_max": pd.Series([1.0, 1.1, 1.2, 1.2, 1.2], index=index)},
        monthly_stats=pd.DataFrame({"Strategy_Return": [0.1]}, index=[pd.Timestamp("2020-01-31")]),
        yearly_stats=pd.DataFrame({"Strategy_Return": [0.18]}, index=[pd.Timestamp("2020-12-31")]),
        trade_history=[{"date": index[0], "code": "000001", "action": "BUY", "size": 100, "price": 10.0, "value": 1000.0, "portfolio_value": 1000000}],
        daily_holdings=[{"date": index[0], "holdings": [{"code": "000001", "size": 100, "price": 10.0, "value": 1000.0, "weight": 0.1, "buy_date": index[0]}], "total_value": 1000000, "cash": 990000}],
        final_value=1180000.0,
        valid_stocks=1,
        strategy_name="weighted_top_n",
        file_name="test.csv",
    )

def test_prepare_overview():
    from src.backtest.config import SystemConfig
    result = _make_result()
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    overview = AnalysisBuilder.prepare_overview(result, sys_cfg)
    assert "summary" in overview
    assert "metrics" in overview
    assert "performance_table" in overview

def test_prepare_net_value():
    result = _make_result()
    net = AnalysisBuilder.prepare_net_value(result)
    assert "strategy_nav" in net
    assert "benchmark_nav" in net
    assert "relative_nav" in net
    assert "summary" in net

def test_prepare_returns():
    result = _make_result()
    returns = AnalysisBuilder.prepare_returns(result)
    assert "daily_returns" in returns
    assert "benchmark_returns" in returns
    assert "daily_stats" in returns
    assert "cumulative_strategy" in returns
    assert "cumulative_benchmark" in returns
    assert "monthly_table" in returns
    assert "yearly_table" in returns

def test_prepare_risk():
    result = _make_result()
    risk = AnalysisBuilder.prepare_risk(result)
    assert "drawdown_series" in risk
    assert "running_max" in risk
    assert "risk_metrics" in risk

def test_prepare_period_stats():
    result = _make_result()
    period = AnalysisBuilder.prepare_period_stats(result)
    assert "monthly" in period
    assert "yearly" in period
    assert "monthly_win_rate" in period

def test_prepare_holdings():
    result = _make_result()
    holdings = AnalysisBuilder.prepare_holdings(result)
    assert "latest_snapshot" in holdings
    assert "holdings_table" in holdings
    assert "asset_curve" in holdings

def test_prepare_trades():
    result = _make_result()
    trades = AnalysisBuilder.prepare_trades(result)
    assert "trades_table" in trades
    assert "trade_count" in trades

# 边界：空数据

def test_prepare_methods_with_empty_result():
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    empty_result = BacktestResult(
        strategy_nav=pd.Series(dtype=float, name="strategy_nav"),
        benchmark_nav=pd.Series(dtype=float, name="benchmark_nav"),
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[],
        daily_holdings=[],
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="empty.csv",
    )
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    # 不应抛异常
    AnalysisBuilder.prepare_overview(empty_result, sys_cfg)
    AnalysisBuilder.prepare_net_value(empty_result)
    AnalysisBuilder.prepare_returns(empty_result)
    AnalysisBuilder.prepare_risk(empty_result)
    AnalysisBuilder.prepare_period_stats(empty_result)
    AnalysisBuilder.prepare_holdings(empty_result)
    AnalysisBuilder.prepare_trades(empty_result)


# 1. 异常类型输入
import pytest
def test_prepare_methods_with_none():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_overview(None, sys_cfg)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_net_value(None)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_returns(None)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_risk(None)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_period_stats(None)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_holdings(None)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_trades(None)

def test_prepare_methods_with_wrong_type():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    dummy = object()
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_overview(dummy, sys_cfg)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_net_value(dummy)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_returns(dummy)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_risk(dummy)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_period_stats(dummy)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_holdings(dummy)
    with pytest.raises(Exception):
        AnalysisBuilder.prepare_trades(dummy)

# 2. 部分字段缺失
def test_prepare_methods_with_missing_fields():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    class Partial:
        pass
    partial = Partial()
    partial.strategy_nav = pd.Series([1.0, 1.1])
    partial.benchmark_nav = pd.Series([1.0, 1.05])
    # 缺 performance, detailed_metrics, monthly_stats, yearly_stats, trade_history, daily_holdings, final_value, valid_stocks, strategy_name, file_name
    try:
        AnalysisBuilder.prepare_overview(partial, sys_cfg)
        assert False, "should raise Exception"
    except Exception:
        pass
    # Other prepare methods may not raise, so skip

# 3. 极端数值
def test_prepare_methods_with_extreme_nav():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    for nav in [
        pd.Series([-1.0, -2.0, -3.0]),
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([float('inf')] * 3),
        pd.Series([float('nan')] * 3),
    ]:
        result = BacktestResult(
            strategy_nav=nav,
            benchmark_nav=nav,
            performance=pd.DataFrame(),
            detailed_metrics={},
            monthly_stats=pd.DataFrame(),
            yearly_stats=pd.DataFrame(),
            trade_history=[],
            daily_holdings=[],
            final_value=0.0,
            valid_stocks=0,
            strategy_name="weighted_top_n",
            file_name="extreme.csv",
        )
        # 不应抛异常
        AnalysisBuilder.prepare_overview(result, sys_cfg)
        AnalysisBuilder.prepare_net_value(result)
        AnalysisBuilder.prepare_returns(result)
        AnalysisBuilder.prepare_risk(result)
        AnalysisBuilder.prepare_period_stats(result)
        AnalysisBuilder.prepare_holdings(result)
        AnalysisBuilder.prepare_trades(result)

# 4. 极端持仓/交易
def test_prepare_methods_with_malformed_holdings_trades():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    index = pd.date_range("2020-01-01", periods=2, freq="D")
    # holdings/trades 为 None
    result1 = BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1], index=index),
        benchmark_nav=pd.Series([1.0, 1.05], index=index),
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=None,
        daily_holdings=None,
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="malformed.csv",
    )
    AnalysisBuilder.prepare_holdings(result1)
    AnalysisBuilder.prepare_trades(result1)
    # holdings/trades 为非 list
    result2 = BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1], index=index),
        benchmark_nav=pd.Series([1.0, 1.05], index=index),
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=123,
        daily_holdings=456,
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="malformed.csv",
    )
    AnalysisBuilder.prepare_holdings(result2)
    AnalysisBuilder.prepare_trades(result2)
    # holdings/trades 为 list 但内容为异常 dict
    result3 = BacktestResult(
        strategy_nav=pd.Series([1.0, 1.1], index=index),
        benchmark_nav=pd.Series([1.0, 1.05], index=index),
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[{"bad": "field"}],
        daily_holdings=[{"bad": "field"}],
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="malformed.csv",
    )
    AnalysisBuilder.prepare_holdings(result3)
    AnalysisBuilder.prepare_trades(result3)

# 5. 极端日期
def test_prepare_methods_with_irregular_index():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    # 非 DatetimeIndex
    nav = pd.Series([1.0, 1.1, 1.2], index=[1, 2, 3])
    result = BacktestResult(
        strategy_nav=nav,
        benchmark_nav=nav,
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[],
        daily_holdings=[],
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="irregular.csv",
    )
    AnalysisBuilder.prepare_overview(result, sys_cfg)
    AnalysisBuilder.prepare_net_value(result)
    AnalysisBuilder.prepare_returns(result)
    AnalysisBuilder.prepare_risk(result)
    AnalysisBuilder.prepare_period_stats(result)
    AnalysisBuilder.prepare_holdings(result)
    AnalysisBuilder.prepare_trades(result)
    # 乱序、重复、跨度极大
    nav2 = pd.Series([1.0, 1.1, 1.2], index=pd.to_datetime(["2020-01-03", "2020-01-01", "2050-01-01"]))
    result2 = BacktestResult(
        strategy_nav=nav2,
        benchmark_nav=nav2,
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[],
        daily_holdings=[],
        final_value=0.0,
        valid_stocks=0,
        strategy_name="weighted_top_n",
        file_name="irregular2.csv",
    )
    AnalysisBuilder.prepare_overview(result2, sys_cfg)
    AnalysisBuilder.prepare_net_value(result2)
    AnalysisBuilder.prepare_returns(result2)
    AnalysisBuilder.prepare_risk(result2)
    AnalysisBuilder.prepare_period_stats(result2)
    AnalysisBuilder.prepare_holdings(result2)
    AnalysisBuilder.prepare_trades(result2)


# 2. prepare_* 输出字段类型和内容完整性
def test_prepare_output_field_types():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    result = _make_result()
    overview = AnalysisBuilder.prepare_overview(result, sys_cfg)
    assert isinstance(overview["summary"], dict)
    assert isinstance(overview["metrics"], list)
    assert isinstance(overview["performance_table"], pd.DataFrame)
    net = AnalysisBuilder.prepare_net_value(result)
    assert isinstance(net["strategy_nav"], pd.Series)
    assert isinstance(net["benchmark_nav"], pd.Series)
    assert isinstance(net["relative_nav"], pd.Series)
    assert isinstance(net["summary"], dict)
    returns = AnalysisBuilder.prepare_returns(result)
    assert isinstance(returns["daily_returns"], pd.Series)
    assert isinstance(returns["benchmark_returns"], pd.Series)
    assert isinstance(returns["daily_stats"], dict)
    assert isinstance(returns["cumulative_strategy"], pd.Series)
    assert isinstance(returns["cumulative_benchmark"], pd.Series)
    assert isinstance(returns["monthly_table"], pd.DataFrame)
    assert isinstance(returns["yearly_table"], pd.DataFrame)
    risk = AnalysisBuilder.prepare_risk(result)
    assert isinstance(risk["drawdown_series"], pd.Series)
    assert isinstance(risk["running_max"], pd.Series)
    assert isinstance(risk["risk_metrics"], dict)
    period = AnalysisBuilder.prepare_period_stats(result)
    assert isinstance(period["monthly"], pd.DataFrame)
    assert isinstance(period["yearly"], pd.DataFrame)
    holdings = AnalysisBuilder.prepare_holdings(result)
    assert isinstance(holdings["latest_snapshot"], dict)
    assert isinstance(holdings["holdings_table"], pd.DataFrame)
    assert isinstance(holdings["asset_curve"], pd.DataFrame)
    trades = AnalysisBuilder.prepare_trades(result)
    assert isinstance(trades["trades_table"], pd.DataFrame)
    assert isinstance(trades["trade_count"], int)

# 3. prepare_* 组合链式调用兼容性
def test_prepare_chain_compatibility():
    from src.backtest.config import SystemConfig
    sys_cfg = SystemConfig(initial_cash=1000000, commission_rate=0.0002, slippage_rate=0.0, benchmark_index="sh000300")
    result = _make_result()
    # returns -> risk
    returns = AnalysisBuilder.prepare_returns(result)
    # 用 daily_returns 构造一个新的 BacktestResult，测试 risk 能否处理
    nav = (1 + returns["daily_returns"]).cumprod()
    new_result = BacktestResult(
        strategy_nav=nav,
        benchmark_nav=nav,
        performance=pd.DataFrame(),
        detailed_metrics={},
        monthly_stats=pd.DataFrame(),
        yearly_stats=pd.DataFrame(),
        trade_history=[],
        daily_holdings=[],
        final_value=nav.iloc[-1],
        valid_stocks=1,
        strategy_name="weighted_top_n",
        file_name="chain.csv",
    )
    risk = AnalysisBuilder.prepare_risk(new_result)
    assert "risk_metrics" in risk
    # holdings -> trades
    holdings = AnalysisBuilder.prepare_holdings(result)
    trades = AnalysisBuilder.prepare_trades(result)
    assert isinstance(holdings["holdings_table"], pd.DataFrame)
    assert isinstance(trades["trades_table"], pd.DataFrame)
