"""Standalone runner for the backtesting engine.

This script lets you execute a backtest without launching the Streamlit UI.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from numbers import Number

import pandas as pd

from src.backtest.backtest_engine import (
    BacktestEngine,
    DataLoader,
    StrategyFactory,
    AnalysisBuilder,
    BacktestResult,
)
from src.backtest.config import SystemConfig, StrategyConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone backtest")
    parser.add_argument(
        "--data-file",
        default="data/test_sample_predictions.csv",
        help="CSV file with prediction data (columns: date, code, weight)",
    )
    parser.add_argument(
        "--strategy",
        default="weighted_top_n",
        choices=StrategyFactory.list_strategies(),
        help="Strategy to apply",
    )
    parser.add_argument(
        "--benchmark",
        default="sh000300",
        help="Benchmark index code (e.g. sh000300)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="Initial cash for the backtest",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0002,
        help="Commission rate per trade",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Slippage rate",
    )
    parser.add_argument(
        "--hold-days",
        type=int,
        default=3,
        help="Override hold_days parameter for the strategy",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Override top_n_stocks parameter for the strategy",
    )
    parser.add_argument(
        "--start-date",
        help="Filter predictions on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="Filter predictions on or before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for all generated artifacts. If omitted, a folder named "
            "'<data-file-stem>_<timestamp>' will be created alongside the data file."
        ),
    )
    return parser.parse_args()


def _filter_predictions(pred_df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    filtered = pred_df.copy()

    if start:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start)]
    if end:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end)]

    return filtered


def _build_configs(args: argparse.Namespace, output_dir: Path) -> tuple[SystemConfig, StrategyConfig]:
    data_dir = Path(args.data_file).resolve().parent

    system_config = SystemConfig(
        data_dir=str(data_dir),
        initial_cash=args.initial_cash,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        start_date=args.start_date,
        end_date=args.end_date,
        show_plots=False,
        save_results=False,
        result_cache_dir=str(output_dir),
    )
    system_config.benchmark_index = args.benchmark

    parameters = {}
    if args.hold_days is not None:
        parameters["hold_days"] = args.hold_days
    if args.top_n is not None:
        parameters["top_n_stocks"] = args.top_n

    strategy_config = StrategyConfig(strategy_name=args.strategy, parameters=parameters)
    return system_config, strategy_config


def main() -> None:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir, args.data_file)

    system_config, result = run_backtest(args, output_dir)
    _print_summary(system_config, result)

    _persist_results(output_dir, result)

    report_path = output_dir / "analysis_report.md"
    report_content = build_markdown_report(system_config, result)
    _write_markdown_report(report_path, report_content)
    print(f"\n分析报告已生成: {report_path}")


def run_backtest(args: argparse.Namespace, output_dir: Optional[Path] = None) -> tuple[SystemConfig, BacktestResult]:
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")

    pred_df = DataLoader.load_prediction_data(args.data_file)
    if pred_df.empty:
        raise ValueError("预测数据为空，请提供有效的CSV文件")

    pred_df = _filter_predictions(pred_df, args.start_date, args.end_date)
    if pred_df.empty:
        raise ValueError("按指定日期过滤后预测数据为空")

    resolved_output = output_dir or _resolve_output_dir(getattr(args, "output_dir", None), args.data_file)
    system_config, strategy_config = _build_configs(args, resolved_output)
    engine = BacktestEngine(system_config, strategy_config)
    raw_result = engine.run_single_backtest(pred_df, args.strategy, os.path.basename(args.data_file))
    result = BacktestResult.from_dict(raw_result)
    return system_config, result


def _print_summary(system_config: SystemConfig, result: BacktestResult) -> None:
    print("\n=== Backtest Summary ===")
    print(f"Strategy: {result.strategy_name}")
    print(f"Benchmark: {system_config.benchmark_index}")
    print(f"Initial cash: {system_config.initial_cash:,.2f}")
    print(f"Final portfolio value: {result.final_value:,.2f}")

    if system_config.initial_cash:
        total_return = (result.final_value - system_config.initial_cash) / system_config.initial_cash
    else:
        total_return = 0.0
    print(f"Total return: {total_return:.2%}")

    strategy_nav = result.strategy_nav
    if not strategy_nav.empty:
        first_date = strategy_nav.index.min().strftime("%Y-%m-%d")
        last_date = strategy_nav.index.max().strftime("%Y-%m-%d")
        print(f"Date range: {first_date} -> {last_date}")

    def _is_number(val: object) -> bool:
        return isinstance(val, Number)

    def _fmt_percent(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "n/a"
        return f"{float(val):.2%}"

    def _fmt_ratio(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "n/a"
        return f"{float(val):.3f}"

    performance_df = result.performance if isinstance(result.performance, pd.DataFrame) else pd.DataFrame()
    if performance_df.empty:
        print("\nPerformance metrics: <none>")
    else:
        print("\nPerformance metrics:")
        percent_keys = {
            "total_return",
            "annual_return",
            "max_drawdown",
            "volatility",
            "win_rate",
            "benchmark_annual_return",
            "industry_hhi",
            "top_industry_weight",
            "industry_rotation",
            "total_turnover",
            "average_daily_turnover",
            "round_trip_win_rate",
            "downside_deviation",
            "ulcer_index",
        }
        ratio_keys = {
            "sharpe_ratio",
            "alpha",
            "beta",
            "payoff_ratio",
            "expectancy",
            "sortino_ratio",
            "tail_ratio",
            "gini_coefficient",
        }
        integer_keys = {"trade_count", "round_trip_count", "return_count"}

        for metric, row in performance_df.iterrows():
            value = row.get("value")
            if not _is_number(value):
                print(f"  {metric:>24}: {value}")
                continue

            if metric in percent_keys:
                formatted = _fmt_percent(value)
            elif metric in ratio_keys:
                formatted = _fmt_ratio(value)
            elif metric in integer_keys:
                formatted = f"{int(value)}"
            else:
                formatted = f"{float(value):.3f}" if not pd.isna(value) else "n/a"

            print(f"  {metric:>24}: {formatted}")

    structure_metrics = result.detailed_metrics.get("structure_metrics", {}) if isinstance(result.detailed_metrics, dict) else {}
    trading_metrics = result.detailed_metrics.get("trading_metrics", {}) if isinstance(result.detailed_metrics, dict) else {}
    extended_risk_metrics = result.detailed_metrics.get("extended_risk_metrics", {}) if isinstance(result.detailed_metrics, dict) else {}

    def _print_section(title: str):
        print(f"\n{title}:")

    if structure_metrics:
        _print_section("Structure metrics")
        for key in [
            "effective_positions",
            "max_single_weight",
            "normalized_entropy",
            "industry_hhi",
            "top_industry_weight",
            "industry_rotation",
            "industry_count",
        ]:
            if key not in structure_metrics:
                continue
            val = structure_metrics.get(key)
            if key in {"max_single_weight", "industry_hhi", "top_industry_weight", "industry_rotation"}:
                formatted = _fmt_percent(val)
            elif key == "normalized_entropy":
                formatted = _fmt_ratio(val)
            else:
                formatted = f"{val}" if not pd.isna(val) else "n/a"
            print(f"  {key:>24}: {formatted}")

        weights = structure_metrics.get("industry_weights") if isinstance(structure_metrics.get("industry_weights"), dict) else {}
        if weights:
            top_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
            weight_str = ", ".join(f"{code}:{_fmt_percent(w)}" for code, w in top_weights)
            print(f"  {'industry_weights':>24}: {weight_str}")

    if trading_metrics:
        _print_section("Trading metrics")
        for key in [
            "total_turnover",
            "average_daily_turnover",
            "trade_count",
            "round_trip_count",
            "round_trip_win_rate",
            "avg_holding_days",
            "median_holding_days",
            "max_holding_days",
            "payoff_ratio",
            "expectancy",
        ]:
            val = trading_metrics.get(key)
            if val is None and key == "round_trip_win_rate":
                val = trading_metrics.get("win_rate")
            if val is None:
                continue
            if key in {"total_turnover", "average_daily_turnover", "round_trip_win_rate"}:
                formatted = _fmt_percent(val)
            elif key in {"payoff_ratio", "expectancy"}:
                formatted = _fmt_ratio(val)
            elif key in {"trade_count", "round_trip_count"}:
                formatted = f"{int(val)}"
            else:
                formatted = f"{float(val):.2f}" if not pd.isna(val) else "n/a"
            print(f"  {key:>24}: {formatted}")

    if extended_risk_metrics:
        _print_section("Extended risk metrics")
        for key in [
            "sortino_ratio",
            "downside_deviation",
            "tail_ratio",
            "ulcer_index",
            "skewness",
            "kurtosis",
            "return_count",
        ]:
            if key not in extended_risk_metrics:
                continue
            val = extended_risk_metrics.get(key)
            if key in {"downside_deviation", "ulcer_index"}:
                formatted = _fmt_percent(val)
            elif key in {"sortino_ratio", "tail_ratio", "skewness", "kurtosis"}:
                formatted = _fmt_ratio(val)
            elif key == "return_count":
                formatted = f"{int(val)}"
            else:
                formatted = f"{val}"
            print(f"  {key:>24}: {formatted}")


def _persist_results(result_dir: Path, result: BacktestResult) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    nav_path = result_dir / "strategy_nav.csv"
    perf_path = result_dir / "performance_metrics.csv"
    result.strategy_nav.to_csv(nav_path)
    result.performance.to_csv(perf_path)
    extra_metrics = {}
    if isinstance(result.detailed_metrics, dict):
        extra_metrics = {
            key: result.detailed_metrics.get(key, {})
            for key in [
                "structure_metrics",
                "trading_metrics",
                "extended_risk_metrics",
            ]
        }
    if extra_metrics:
        extra_path = result_dir / "additional_metrics.json"
        with extra_path.open("w", encoding="utf-8") as fh:
            json.dump(extra_metrics, fh, ensure_ascii=False, indent=2)
    else:
        extra_path = None
    print(f"\n结果已保存至: {nav_path} 和 {perf_path}")
    if extra_path:
        print(f"附加指标已保存至: {extra_path}")


def _resolve_output_dir(user_dir: Optional[str], data_file: str) -> Path:
    if user_dir:
        return Path(user_dir)

    data_path = Path(data_file).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = data_path.parent / f"{data_path.stem}_{timestamp}"
    return default_dir


def build_markdown_report(
    system_config: SystemConfig,
    result: BacktestResult,
    *,
    selected_metrics: Optional[list[str]] = None,
    top_holdings: int = 5,
) -> str:
    overview = AnalysisBuilder.prepare_overview(result, system_config, selected_metrics)
    net_value = AnalysisBuilder.prepare_net_value(result)
    returns = AnalysisBuilder.prepare_returns(result)
    risk = AnalysisBuilder.prepare_risk(result)
    period_stats = AnalysisBuilder.prepare_period_stats(result)
    holdings = AnalysisBuilder.prepare_holdings(result)
    trades = AnalysisBuilder.prepare_trades(result)

    # result已经是BacktestResult对象，直接使用

    def _fmt_pct(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{value:.2%}"

    def _fmt_float(value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}"

    summary = overview["summary"]
    metrics_rows = overview["metrics"]

    lines: list[str] = []
    lines.append(f"# Backtest Report: {summary.get('strategy_name', 'N/A')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Data file: `{summary.get('file_name', 'N/A')}`")
    lines.append(f"- Benchmark: `{summary.get('benchmark', 'N/A')}`")
    start_date = summary.get("start_date")
    end_date = summary.get("end_date")
    if start_date and end_date:
        lines.append(f"- Date range: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} ({summary.get('date_span')} days)")
    lines.append(f"- Initial cash: {summary.get('initial_cash'):,.2f}")
    lines.append(f"- Final portfolio value: {summary.get('final_value'):,.2f}")
    total_return = summary.get("total_return")
    lines.append(f"- Total return: {_fmt_pct(total_return)}")
    lines.append(f"- Valid stocks: {summary.get('valid_stocks', 0)}")
    lines.append("")

    if metrics_rows:
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("| Metric | Value | Description |")
        lines.append("| --- | --- | --- |")
        for row in metrics_rows:
            metric = row.get("metric")
            formatted = row.get("formatted") or row.get("value")
            meta = row.get("meta") or {}
            desc = meta.get("description", "")
            lines.append(f"| {meta.get('name', metric)} | {formatted} | {desc} |")
        lines.append("")

    lines.append("## Net Value Snapshot")
    lines.append("")
    net_summary = net_value["summary"]
    lines.append(f"- Strategy NAV (final): {_fmt_float(net_summary.get('strategy_return'))}")
    lines.append(f"- Benchmark NAV (final): {_fmt_float(net_summary.get('benchmark_return'))}")
    lines.append(f"- Excess return: {_fmt_pct(net_summary.get('excess_return'))}")
    lines.append("")

    lines.append("## Daily Return Stats")
    lines.append("")
    daily_stats = returns["daily_stats"]
    lines.append(f"- Mean daily return: {_fmt_pct(daily_stats.get('mean'))}")
    lines.append(f"- Daily volatility: {_fmt_pct(daily_stats.get('std'))}")
    lines.append(f"- Positive / Negative days: {daily_stats.get('positive_days')} / {daily_stats.get('negative_days')}")
    win_rate = daily_stats.get("win_rate")
    lines.append(f"- Win rate vs benchmark: {_fmt_pct(win_rate)}" if win_rate is not None else "- Win rate vs benchmark: -")
    lines.append("")

    lines.append("## Risk Metrics")
    lines.append("")
    risk_metrics = risk["risk_metrics"]
    lines.append(f"- Max drawdown: {_fmt_pct(risk_metrics.get('max_drawdown'))}")
    lines.append(f"- Annual return: {_fmt_pct(risk_metrics.get('annual_return'))}")
    lines.append(f"- Calmar ratio: {_fmt_float(risk_metrics.get('calmar_ratio'))}")
    lines.append(f"- Tracking error: {_fmt_pct(risk_metrics.get('tracking_error'))}")
    lines.append(f"- Information ratio: {_fmt_float(risk_metrics.get('information_ratio'))}")
    lines.append(f"- VaR 95%: {_fmt_pct(risk_metrics.get('var_95'))}")
    lines.append(f"- CVaR 95%: {_fmt_pct(risk_metrics.get('cvar_95'))}")
    lines.append("")

    structure_metrics = {}
    trading_metrics = {}
    extended_risk_metrics = {}
    if isinstance(result.detailed_metrics, dict):
        structure_metrics = result.detailed_metrics.get("structure_metrics", {}) or {}
        trading_metrics = result.detailed_metrics.get("trading_metrics", {}) or {}
        extended_risk_metrics = result.detailed_metrics.get("extended_risk_metrics", {}) or {}

    if structure_metrics:
        lines.append("## Portfolio Structure")
        lines.append("")
        structure_rows = [
            ("Effective positions", structure_metrics.get("effective_positions")),
            ("Max single weight", _fmt_pct(structure_metrics.get("max_single_weight"))),
            ("Industry HHI", _fmt_pct(structure_metrics.get("industry_hhi"))),
            ("Top industry weight", _fmt_pct(structure_metrics.get("top_industry_weight"))),
            ("Industry rotation", _fmt_pct(structure_metrics.get("industry_rotation"))),
            ("Industry count", structure_metrics.get("industry_count")),
            ("Normalized entropy", _fmt_float(structure_metrics.get("normalized_entropy"))),
            ("Gini coefficient", _fmt_float(structure_metrics.get("gini_coefficient"))),
        ]
        for label, value in structure_rows:
            if value is None or (isinstance(value, str) and not value):
                continue
            lines.append(f"- {label}: {value}")

        weights = structure_metrics.get("industry_weights")
        if isinstance(weights, dict) and weights:
            lines.append("- Industry weights:")
            top_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
            for code, weight in top_weights[:10]:
                lines.append(f"  - {code}: {_fmt_pct(weight)}")
        lines.append("")

    if trading_metrics:
        lines.append("## Trading Activity")
        lines.append("")
        trading_rows = [
            ("Total turnover", _fmt_pct(trading_metrics.get("total_turnover"))),
            ("Average daily turnover", _fmt_pct(trading_metrics.get("average_daily_turnover"))),
            ("Trade count", trading_metrics.get("trade_count")),
            ("Round trip count", trading_metrics.get("round_trip_count")),
            ("Round trip win rate", _fmt_pct(trading_metrics.get("round_trip_win_rate"))),
            ("Average holding days", trading_metrics.get("avg_holding_days")),
            ("Median holding days", trading_metrics.get("median_holding_days")),
            ("Max holding days", trading_metrics.get("max_holding_days")),
            ("Payoff ratio", _fmt_float(trading_metrics.get("payoff_ratio"))),
            ("Expectancy", _fmt_float(trading_metrics.get("expectancy"))),
        ]
        for label, value in trading_rows:
            if value is None or (isinstance(value, str) and not value):
                continue
            lines.append(f"- {label}: {value}")
        lines.append("")

    if extended_risk_metrics:
        lines.append("## Extended Risk Metrics")
        lines.append("")
        extended_rows = [
            ("Sortino ratio", _fmt_float(extended_risk_metrics.get("sortino_ratio"))),
            ("Downside deviation", _fmt_pct(extended_risk_metrics.get("downside_deviation"))),
            ("Tail ratio", _fmt_float(extended_risk_metrics.get("tail_ratio"))),
            ("Ulcer index", _fmt_pct(extended_risk_metrics.get("ulcer_index"))),
            ("Skewness", _fmt_float(extended_risk_metrics.get("skewness"))),
            ("Kurtosis", _fmt_float(extended_risk_metrics.get("kurtosis"))),
            ("Return samples", extended_risk_metrics.get("return_count")),
        ]
        for label, value in extended_rows:
            if value is None or (isinstance(value, str) and not value):
                continue
            lines.append(f"- {label}: {value}")
        lines.append("")

    lines.append("## Period Performance")
    lines.append("")
    monthly = period_stats["monthly"]
    if not monthly.empty:
        lines.append("### Monthly Returns (last 6)")
        lines.append("")
        preview = monthly.tail(6).copy()
        preview.index = preview.index.strftime("%Y-%m")
        lines.extend(_dataframe_to_markdown(preview, float_fmt="{:.2%}"))
        lines.append("")
    yearly = period_stats["yearly"]
    if not yearly.empty:
        lines.append("### Yearly Returns")
        lines.append("")
        preview = yearly.copy()
        preview.index = preview.index.strftime("%Y")
        lines.extend(_dataframe_to_markdown(preview, float_fmt="{:.2%}"))
        lines.append("")

    lines.append("## Latest Holdings Snapshot")
    lines.append("")
    holdings_table = holdings["holdings_table"].head(top_holdings)
    if holdings_table.empty:
        lines.append("_No holdings recorded._")
    else:
        display_cols = [col for col in ["code", "name", "size", "price", "value", "weight"] if col in holdings_table.columns]
        lines.extend(_dataframe_to_markdown(holdings_table[display_cols], float_fmt="{:.4f}"))
    lines.append("")

    # 添加每日持仓历史
    lines.append("## Daily Holdings History")
    lines.append("")

    # 使用AnalysisBuilder生成每日持仓历史
    daily_holdings_df = AnalysisBuilder.prepare_daily_holdings_history(result)

    if daily_holdings_df.empty:
        lines.append("_No daily holdings data available._")
    else:
        # 选择要显示的列
        display_cols = ['date', 'total_value']
        # 添加股票相关的列（只显示有持仓的股票）
        stock_cols = [col for col in daily_holdings_df.columns if col.endswith('_size') or col.endswith('_weight')]
        display_cols.extend(stock_cols)

        # 只显示前10条记录，避免报告过长
        display_df = daily_holdings_df[display_cols].head(10).copy()

        # 格式化列名
        display_df = display_df.rename(columns=lambda x: x.replace('_size', '_Size').replace('_weight', '_Weight'))

        # 格式化数值
        for col in display_df.columns:
            if col == 'date':
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
            elif col == 'total_value':
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
            elif col.endswith('_Size'):
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
            elif col.endswith('_Weight'):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

        lines.extend(_dataframe_to_markdown(display_df, float_fmt="{:.4f}"))

        if len(daily_holdings_df) > 10:
            lines.append(f"_... and {len(daily_holdings_df) - 10} more days_")

    lines.append("")

    lines.append("## Trade Summary")
    lines.append("")
    trade_count = trades["trade_count"]
    lines.append(f"- Total trades executed: {trade_count}")
    trades_table = trades["trades_table"]
    if not trades_table.empty:
        last_trade = trades_table.iloc[-1]
        lines.append(
            "- Last trade: {date:%Y-%m-%d} {action} {code} @ {price:.2f} ({value:.2f})".format(
                date=last_trade.get("date"),
                action=last_trade.get("action"),
                code=last_trade.get("code"),
                price=last_trade.get("price", 0.0),
                value=last_trade.get("value", 0.0),
            )
        )
    else:
        lines.append("- No trades recorded.")

    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame, *, float_fmt: str = "{:.4f}") -> list[str]:
    if df.empty:
        return ["_No data available._"]

    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].apply(lambda x: float_fmt.format(x) if pd.notna(x) else "-")

    header = "| " + " | ".join(["index"] + df_display.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * (len(df_display.columns) + 1)) + " |"
    rows = [header, separator]
    for idx, row in df_display.iterrows():
        row_values = [str(idx)] + [str(row[col]) for col in df_display.columns]
        rows.append("| " + " | ".join(row_values) + " |")
    return rows


def _write_markdown_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
