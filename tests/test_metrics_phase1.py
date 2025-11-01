import numpy as np
import pandas as pd
import pytest

from metrics.portfolio_structure import structure_metrics
from metrics.trade_metrics import rebuild_round_trips, trading_metrics
from metrics.risk_extension import extended_risk_metrics


def _sample_holdings():
    return [
        {
            "date": "2023-01-02",
            "total_value": 1_000_000,
            "holdings": [
                {"code": "000001", "weight": 0.6, "value": 600_000},
                {"code": "000002", "weight": 0.4, "value": 400_000},
            ],
        },
        {
            "date": "2023-01-03",
            "total_value": 1_010_000,
            "holdings": [
                {"code": "000001", "weight": 0.5, "value": 505_000},
                {"code": "000063", "weight": 0.5, "value": 505_000},
            ],
        },
    ]


def _sample_trades():
    return [
        {
            "date": "2023-01-02",
            "code": "000001",
            "action": "BUY",
            "size": 100,
            "price": 10.0,
            "value": 1_000.0,
            "portfolio_value": 1_000_000.0,
        },
        {
            "date": "2023-01-05",
            "code": "000001",
            "action": "SELL",
            "size": -100,
            "price": 11.0,
            "value": 1_100.0,
            "portfolio_value": 1_010_000.0,
        },
        {
            "date": "2023-01-03",
            "code": "000002",
            "action": "BUY",
            "size": 200,
            "price": 5.0,
            "value": 1_000.0,
            "portfolio_value": 1_005_000.0,
        },
        {
            "date": "2023-01-07",
            "code": "000002",
            "action": "SELL",
            "size": -200,
            "price": 4.5,
            "value": 900.0,
            "portfolio_value": 1_002_000.0,
        },
    ]


def test_structure_metrics_outputs_industry_and_diversification():
    metrics = structure_metrics(_sample_holdings())

    assert metrics["industry_count"] >= 1
    assert 0 <= metrics["industry_hhi"] <= 1
    assert 0 <= metrics["normalized_entropy"] <= 1
    assert metrics["industry_weights"]
    assert metrics["max_single_weight"] <= 1


def test_trading_metrics_turnover_and_round_trips():
    trades = _sample_trades()
    nav = pd.Series(
        [1_000_000, 1_005_000, 1_010_000, 1_012_000, 1_015_000],
        index=pd.date_range("2023-01-02", periods=5, freq="B"),
    )

    metrics = trading_metrics(trades, nav_series=nav)

    assert metrics["trade_count"] == len(trades)
    assert metrics["round_trip_count"] == 2
    assert metrics["avg_holding_days"] >= 0
    assert 0 <= metrics["win_rate"] <= 1


def test_round_trip_rebuild_matches_expected_pnl():
    trips = rebuild_round_trips(_sample_trades())
    pnls = sorted([round(trip.pnl, 2) for trip in trips])
    assert pnls == [-100.0, 100.0]


def test_extended_risk_metrics_basic_values():
    nav = pd.Series(
        [1_000_000, 1_010_000, 1_005_000, 1_025_000, 1_030_000],
        index=pd.date_range("2023-01-02", periods=5, freq="B"),
    )

    metrics = extended_risk_metrics(nav)

    assert metrics["return_count"] == 4
    assert metrics["ulcer_index"] >= 0
    assert metrics["downside_deviation"] >= 0
    assert "skewness" in metrics and "kurtosis" in metrics


def test_structure_metrics_handles_empty_input():
    metrics = structure_metrics([])

    assert metrics["industry_count"] == 0
    assert metrics["industry_weights"] == {}
    assert metrics["industry_rotation"] is None
    assert metrics["max_single_weight"] == 0.0
    assert metrics["snapshot_date"] is None
    assert metrics["effective_positions"] == 0.0


def test_structure_metrics_infers_weights_from_values(monkeypatch):
    from metrics import portfolio_structure as ps

    def _fake_industry_lookup(codes):
        if isinstance(codes, (str, int)):
            return "SINGLE"
        labels = ["TECH", "FIN"]
        return {code: labels[i % len(labels)] for i, code in enumerate(codes)}

    monkeypatch.setattr(ps, "get_industry_category", _fake_industry_lookup)

    snapshot = {
        "date": "2023-01-04",
        "total_value": 1_000_000,
        "holdings": [
            {"code": "123456", "value": 250_000},
            {"code": "654321", "value": 750_000},
        ],
    }

    metrics = structure_metrics([snapshot])

    assert metrics["max_single_weight"] == pytest.approx(0.75)
    assert metrics["industry_weights"]["FIN"] == pytest.approx(0.75)
    assert metrics["industry_weights"]["TECH"] == pytest.approx(0.25)
    assert metrics["snapshot_date"] == pd.Timestamp("2023-01-04")


def test_trading_metrics_handles_no_trades():
    metrics = trading_metrics([])

    assert metrics["trade_count"] == 0
    assert metrics["total_turnover"] == 0.0
    assert metrics["round_trip_count"] == 0
    assert metrics["avg_holding_days"] == 0.0


def test_rebuild_round_trips_handles_partial_closure():
    trades = [
        {"date": "2023-01-02", "code": "000001", "action": "BUY", "size": 100, "price": 10.0},
        {"date": "2023-01-03", "code": "000001", "action": "SELL", "size": -40, "price": 12.0},
    ]

    trips = rebuild_round_trips(trades)

    assert len(trips) == 1
    assert trips[0].shares == pytest.approx(40)
    assert trips[0].pnl == pytest.approx(80.0)


def test_rebuild_round_trips_ignores_invalid_entries():
    trades = [
        {"date": "2023-01-02", "code": "000001", "action": "BUY", "size": 100, "price": 10.0},
        {"date": "2023-01-03", "code": "000001", "action": "SELL", "size": -100, "price": 0.0},
    ]

    trips = rebuild_round_trips(trades)

    assert trips == []


def test_extended_risk_metrics_with_empty_nav():
    metrics = extended_risk_metrics(pd.Series(dtype=float))

    assert metrics["return_count"] == 0
    for key, value in metrics.items():
        if key != "return_count":
            assert value == 0.0


def test_extended_risk_metrics_handles_constant_and_negative_nav():
    constant_nav = pd.Series(
        [1_000_000] * 5,
        index=pd.date_range("2023-02-01", periods=5, freq="B"),
    )
    negative_nav = pd.Series(
        [-1_000_000, -1_010_000, -1_015_000, -1_012_000],
        index=pd.date_range("2023-03-01", periods=4, freq="B"),
    )

    constant_metrics = extended_risk_metrics(constant_nav)
    negative_metrics = extended_risk_metrics(negative_nav)

    assert constant_metrics["return_count"] == 4
    assert constant_metrics["downside_deviation"] == 0.0
    assert constant_metrics["sortino_ratio"] == 0.0
    assert constant_metrics["tail_ratio"] == 0.0

    assert negative_metrics["return_count"] == 3
    assert negative_metrics["ulcer_index"] >= 0.0
    assert not np.isnan(negative_metrics["ulcer_index"])