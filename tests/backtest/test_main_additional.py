import argparse
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Mock external dependencies
import sys
if "alphalens" not in sys.modules:
    alphalens_module = type(sys)("alphalens")
    sys.modules["alphalens"] = alphalens_module

    performance_module = type(sys)("alphalens.performance")
    performance_module.factor_returns = lambda *a, **k: pd.Series(dtype=float)
    performance_module.mean_information_coefficient = lambda *a, **k: pd.Series(dtype=float)
    performance_module.mean_return_by_quantile = lambda *a, **k: (pd.DataFrame(), pd.DataFrame())
    sys.modules["alphalens.performance"] = performance_module

    utils_module = type(sys)("alphalens.utils")
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

    exceptions_module = type(sys)("oss2.exceptions")
    exceptions_module.NoSuchKey = FileNotFoundError
    oss2_module = type(sys)("oss2")
    oss2_module.Auth = _DummyAuth
    oss2_module.Bucket = _DummyBucket
    oss2_module.ObjectIterator = _dummy_iterator
    oss2_module.exceptions = exceptions_module
    sys.modules["oss2"] = oss2_module
    sys.modules["oss2.exceptions"] = exceptions_module

from src.backtest.main import (
    _build_configs,
    _filter_predictions,
    _parse_args,
    _resolve_output_dir,
    build_markdown_report,
    run_backtest,
)


class TestMainModule:
    """Test cases for main.py functions"""

    def test_parse_args_default_values(self):
        """Test argument parsing with default values"""
        with patch("sys.argv", ["main.py"]):
            args = _parse_args()
            assert args.data_file == "data/test_sample_predictions.csv"
            assert args.strategy == "weighted_top_n"
            assert args.benchmark == "sh000300"
            assert args.initial_cash == 1_000_000.0
            assert args.commission == 0.0002
            assert args.slippage == 0.0
            assert args.hold_days == 3
            assert args.top_n == 10

    def test_parse_args_custom_values(self):
        """Test argument parsing with custom values"""
        test_args = [
            "main.py",
            "--data-file", "custom_data.csv",
            "--strategy", "equal_weight",
            "--benchmark", "sh000001",
            "--initial-cash", "500000",
            "--commission", "0.0003",
            "--slippage", "0.001",
            "--hold-days", "5",
            "--top-n", "20",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31",
            "--output-dir", "/tmp/test"
        ]
        with patch("sys.argv", test_args):
            args = _parse_args()
            assert args.data_file == "custom_data.csv"
            assert args.strategy == "equal_weight"
            assert args.benchmark == "sh000001"
            assert args.initial_cash == 500000.0
            assert args.commission == 0.0003
            assert args.slippage == 0.001
            assert args.hold_days == 5
            assert args.top_n == 20
            assert args.start_date == "2023-01-01"
            assert args.end_date == "2023-12-31"
            assert args.output_dir == "/tmp/test"

    def test_filter_predictions_no_filters(self):
        """Test prediction filtering with no date filters"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-05"),
            "code": ["000001", "000002", "000003", "000004", "000005"],
            "weight": [0.2, 0.2, 0.2, 0.2, 0.2]
        })
        filtered = _filter_predictions(df, None, None)
        pd.testing.assert_frame_equal(filtered, df)

    def test_filter_predictions_with_start_date(self):
        """Test prediction filtering with start date"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-05"),
            "code": ["000001", "000002", "000003", "000004", "000005"],
            "weight": [0.2, 0.2, 0.2, 0.2, 0.2]
        })
        filtered = _filter_predictions(df, "2023-01-03", None)
        expected = df[df["date"] >= pd.to_datetime("2023-01-03")].reset_index(drop=True)
        # Reset index for filtered result too
        filtered = filtered.reset_index(drop=True)
        pd.testing.assert_frame_equal(filtered, expected)

    def test_filter_predictions_with_end_date(self):
        """Test prediction filtering with end date"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-05"),
            "code": ["000001", "000002", "000003", "000004", "000005"],
            "weight": [0.2, 0.2, 0.2, 0.2, 0.2]
        })
        filtered = _filter_predictions(df, None, "2023-01-03")
        expected = df[df["date"] <= pd.to_datetime("2023-01-03")].reset_index(drop=True)
        pd.testing.assert_frame_equal(filtered, expected)

    def test_filter_predictions_with_both_dates(self):
        """Test prediction filtering with both start and end dates"""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-10"),
            "code": [f"00000{i}" for i in range(1, 11)],
            "weight": [0.1] * 10
        })
        filtered = _filter_predictions(df, "2023-01-03", "2023-01-07")
        expected = df[(df["date"] >= pd.to_datetime("2023-01-03")) &
                     (df["date"] <= pd.to_datetime("2023-01-07"))].reset_index(drop=True)
        # Reset index for filtered result too
        filtered = filtered.reset_index(drop=True)
        pd.testing.assert_frame_equal(filtered, expected)

    def test_build_configs_basic(self):
        """Test config building with basic parameters"""
        args = argparse.Namespace(
            data_file="data/test.csv",
            initial_cash=1000000.0,
            commission=0.0002,
            slippage=0.0,
            start_date=None,
            end_date=None,
            benchmark="sh000300",
            strategy="weighted_top_n",
            hold_days=None,
            top_n=None
        )
        output_dir = Path("/tmp/test")

        system_config, strategy_config = _build_configs(args, output_dir)

        assert system_config.initial_cash == 1000000.0
        assert system_config.commission_rate == 0.0002
        assert system_config.slippage_rate == 0.0
        assert system_config.benchmark_index == "sh000300"
        assert strategy_config.strategy_name == "weighted_top_n"
        assert strategy_config.parameters == {"hold_days": 2, "top_n_stocks": 10}

    def test_build_configs_with_overrides(self):
        """Test config building with parameter overrides"""
        args = argparse.Namespace(
            data_file="data/test.csv",
            initial_cash=2000000.0,
            commission=0.0003,
            slippage=0.001,
            start_date="2023-01-01",
            end_date="2023-12-31",
            benchmark="sh000001",
            strategy="weighted_top_n",
            hold_days=5,
            top_n=15
        )
        output_dir = Path("/tmp/test")

        system_config, strategy_config = _build_configs(args, output_dir)

        assert system_config.initial_cash == 2000000.0
        assert system_config.commission_rate == 0.0003
        assert system_config.slippage_rate == 0.001
        assert system_config.start_date == "2023-01-01"
        assert system_config.end_date == "2023-12-31"
        assert system_config.benchmark_index == "sh000001"
        assert strategy_config.parameters == {"hold_days": 5, "top_n_stocks": 15}

    def test_resolve_output_dir_with_user_dir(self):
        """Test output directory resolution with user-specified directory"""
        user_dir = "/custom/output/dir"
        data_file = "data/test.csv"
        result = _resolve_output_dir(user_dir, data_file)
        assert str(result) == user_dir

    def test_resolve_output_dir_default(self):
        """Test output directory resolution with default naming"""
        data_file = "/path/to/data/test_predictions.csv"
        with patch("main.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231231_120000"
            result = _resolve_output_dir(None, data_file)
            expected = Path("/path/to/data/test_predictions_20231231_120000")
            assert result == expected

    def test_run_backtest_file_not_found(self):
        """Test run_backtest with non-existent file"""
        args = argparse.Namespace(
            data_file="non_existent_file.csv",
            strategy="weighted_top_n",
            benchmark="sh000300",
            initial_cash=1000000.0,
            commission=0.0002,
            slippage=0.0,
            start_date=None,
            end_date=None
        )

        with pytest.raises(FileNotFoundError, match="数据文件不存在"):
            run_backtest(args)

    def test_run_backtest_empty_predictions(self, tmp_path):
        """Test run_backtest with empty prediction file"""
        # Create empty CSV file
        csv_file = tmp_path / "empty.csv"
        pd.DataFrame(columns=["date", "code", "weight"]).to_csv(csv_file, index=False)

        args = argparse.Namespace(
            data_file=str(csv_file),
            strategy="weighted_top_n",
            benchmark="sh000300",
            initial_cash=1000000.0,
            commission=0.0002,
            slippage=0.0,
            start_date=None,
            end_date=None
        )

        with pytest.raises(ValueError, match="预测数据为空"):
            run_backtest(args)

    def test_run_backtest_filtered_empty(self, tmp_path):
        """Test run_backtest with predictions that get filtered to empty"""
        # Create CSV with data outside filter range
        csv_file = tmp_path / "filtered_empty.csv"
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", "2023-01-05"),
            "code": ["000001"] * 5,
            "weight": [1.0] * 5
        })
        df.to_csv(csv_file, index=False)

        args = argparse.Namespace(
            data_file=str(csv_file),
            strategy="weighted_top_n",
            benchmark="sh000300",
            initial_cash=1000000.0,
            commission=0.0002,
            slippage=0.0,
            start_date="2024-01-01",  # Future date
            end_date="2024-12-31"
        )

        with pytest.raises(ValueError, match="按指定日期过滤后预测数据为空"):
            run_backtest(args)

    def test_build_markdown_report_basic(self):
        """Test markdown report building with basic data"""
        # Mock system config
        class MockSystemConfig:
            benchmark_index = "sh000300"
            initial_cash = 1000000.0

        # Mock backtest result
        mock_result = type('MockResult', (), {
            'strategy_name': 'weighted_top_n',
            'file_name': 'test_data.csv',
            'final_value': 950000.0,
            'valid_stocks': 10,
            'strategy_nav': pd.Series([1.0, 0.98, 0.95], index=pd.date_range('2023-01-01', periods=3)),
            'benchmark_nav': pd.Series([1.0, 0.97, 0.94], index=pd.date_range('2023-01-01', periods=3)),
            'performance': pd.DataFrame({
                'value': [-0.05, -0.05, -0.05, -1.0, 0.02, 0.5, -0.01, 0.02],
                'description': ['总收益率', '年化收益率', '最大回撤', '夏普比率', '波动率', '胜率', 'Alpha', 'Beta']
            }, index=['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'volatility', 'win_rate', 'alpha', 'beta']),
            'trade_history': [],
            'daily_holdings': [],
            'monthly_stats': pd.DataFrame(),
            'yearly_stats': pd.DataFrame(),
            'detailed_metrics': {}
        })()

        system_config = MockSystemConfig()
        report = build_markdown_report(system_config, mock_result)

        # Check basic structure
        assert "# Backtest Report: weighted_top_n" in report
        assert "## Overview" in report
        assert "## Performance Metrics" in report
        assert "test_data.csv" in report
        assert "sh000300" in report

    def test_build_markdown_report_with_selected_metrics(self):
        """Test markdown report with selected metrics"""
        # Similar to above but with selected_metrics parameter
        class MockSystemConfig:
            benchmark_index = "sh000300"
            initial_cash = 1000000.0

        mock_result = type('MockResult', (), {
            'strategy_name': 'weighted_top_n',
            'file_name': 'test_data.csv',
            'final_value': 950000.0,
            'valid_stocks': 10,
            'strategy_nav': pd.Series([1.0, 0.98, 0.95], index=pd.date_range('2023-01-01', periods=3)),
            'benchmark_nav': pd.Series([1.0, 0.97, 0.94], index=pd.date_range('2023-01-01', periods=3)),
            'performance': pd.DataFrame({
                'value': [-0.05, -0.05, -0.05, -1.0, 0.02, 0.5, -0.01, 0.02],
                'description': ['总收益率', '年化收益率', '最大回撤', '夏普比率', '波动率', '胜率', 'Alpha', 'Beta']
            }, index=['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'volatility', 'win_rate', 'alpha', 'beta']),
            'trade_history': [],
            'daily_holdings': [],
            'monthly_stats': pd.DataFrame(),
            'yearly_stats': pd.DataFrame(),
            'detailed_metrics': {}
        })()

        system_config = MockSystemConfig()
        selected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        report = build_markdown_report(system_config, mock_result, selected_metrics=selected_metrics)

        assert "# Backtest Report: weighted_top_n" in report
        # Should contain selected metrics
        assert "总收益率" in report
        assert "夏普比率" in report
        assert "最大回撤" in report