"""
股票回测系统 - Streamlit Web界面
简化版界面，专注于UI展示和用户交互
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data.data import code2name

# 导入核心回测引擎
from src.backtest.backtest_engine import BacktestEngine, DataLoader, StrategyFactory, AnalysisBuilder, BacktestResult

# 导入配置模块
try:
    from src.backtest.config import (
        SystemConfig, StrategyConfig, 
        get_benchmark_info, get_strategy_info, get_metric_info,
        list_benchmark_indices, list_strategies, list_metrics
    )
    CONFIG_AVAILABLE = True
except Exception as e:
    print(f"配置模块加载失败: {e}")
    CONFIG_AVAILABLE = False

# 设置页面配置
st.set_page_config(
    page_title="股票回测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置matplotlib中文支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = True
sns.set_style("whitegrid")

def main():
    """主函数"""
    st.title("📈 股票回测系统")
    st.sidebar.title("⚙️ 回测配置")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("📂 数据配置")
        
        # 数据目录
        data_dir = st.text_input("数据目录", value="data", help="存放CSV数据文件的目录")
        
        # 系统参数
        st.header("💰 系统参数")
        initial_cash = st.number_input("初始资金", min_value=10000, max_value=100000000, value=1000000, step=100000)
        commission_rate = st.number_input("手续费率", min_value=0.0, max_value=0.01, value=0.0002, step=0.0001, format="%.4f")
        slippage_rate = st.number_input("滑点率", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")
        
        # 基准指数选择
        st.header("📊 基准设置")
        if CONFIG_AVAILABLE:
            benchmark_options = list_benchmark_indices()
            benchmark_labels = [f"{code} - {get_benchmark_info(code)['name']}" for code in benchmark_options]
            selected_benchmark = st.selectbox("基准指数", benchmark_labels)
            benchmark_index = selected_benchmark.split(" - ")[0]
        else:
            benchmark_options = ['sh000300', 'sh000001', 'sz399001']
            benchmark_index = st.selectbox("基准指数", benchmark_options)
        
        # 策略选择
        st.header("🎯 策略配置")
        if CONFIG_AVAILABLE:
            strategy_options = list_strategies()
            strategy_names = [f"{strategy} - {get_strategy_info(strategy)['name']}" for strategy in strategy_options]
            selected_strategy = st.selectbox("选择策略", strategy_names)
            strategy_name = selected_strategy.split(" - ")[0]
            
            # 获取策略参数配置
            strategy_info = get_strategy_info(strategy_name)
            strategy_params = {}
            if strategy_info:
                st.subheader("策略参数")
                for param_name, param_spec in strategy_info["parameters"].items():
                    if param_spec["type"] == "int":
                        strategy_params[param_name] = st.number_input(
                            param_spec["name"],
                            min_value=param_spec["min"],
                            max_value=param_spec["max"],
                            value=param_spec["default"],
                            step=param_spec.get("step", 1)
                        )
                    elif param_spec["type"] == "float":
                        strategy_params[param_name] = st.number_input(
                            param_spec["name"],
                            min_value=param_spec["min"],
                            max_value=param_spec["max"],
                            value=param_spec["default"],
                            step=param_spec.get("step", 0.1),
                            format="%.3f"
                        )
        else:
            strategy_name = st.selectbox("选择策略", ["weighted_top_n", "equal_weight", "momentum"])
            strategy_params = {
                'hold_days': st.number_input("持仓天数", min_value=1, max_value=30, value=2),
                'top_n_stocks': st.number_input("股票数量", min_value=1, max_value=50, value=10)
            }
        
        # 分析选项
        st.header("📈 分析选项")
        show_plots = st.checkbox("显示图表", value=True)
        save_results = st.checkbox("保存结果", value=False)
        st.subheader("结果展示")
        show_risk_metrics = st.checkbox("显示风险分析", value=True)
        show_period_stats = st.checkbox("显示期间统计", value=True)
        show_daily_holdings = st.checkbox("显示每日持仓", value=True)
        show_trade_history = st.checkbox("显示交易记录", value=True)
        
        # 指标选择
        if CONFIG_AVAILABLE:
            available_metrics = list_metrics()
            default_metrics = [
                'total_return',
                'annual_return',
                'max_drawdown',
                'sharpe_ratio',
                'win_rate',
                'industry_hhi',
                'total_turnover',
                'round_trip_win_rate',
                'sortino_ratio',
            ]
            selected_metrics = st.multiselect(
                "选择显示指标",
                available_metrics,
                default=[metric for metric in default_metrics if metric in available_metrics]
            )
        else:
            selected_metrics = ["total_return", "annual_return", "max_drawdown", "sharpe_ratio"]
    
    # 文件选择
    available_files = DataLoader.scan_prediction_files(data_dir)
    selected_files = st.multiselect(
        "选择回测文件",
        available_files,
        default=available_files[:1] if available_files else []
    )
    
    # 主界面
    if st.sidebar.button("开始回测") and selected_files:
        # 创建系统配置
        system_config = SystemConfig(
            data_dir=data_dir,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            show_plots=show_plots,
            save_results=save_results
        )
        # 设置基准指数
        system_config.benchmark_index = benchmark_index
        
        # 创建策略配置
        strategy_config = StrategyConfig(
            strategy_name=strategy_name,
            parameters=strategy_params
        )
        
        # 创建回测引擎
        engine = BacktestEngine(system_config, strategy_config)
        
        # 运行回测
        with st.spinner('正在运行回测...'):
            results = {}
            progress_bar = st.progress(0)
            
            for i, file_name in enumerate(selected_files):
                try:
                    st.write(f"🔄 处理文件: {file_name}")
                    
                    # 加载数据
                    file_path = os.path.join(data_dir, file_name)
                    pred_df = DataLoader.load_prediction_data(file_path)
                    
                    if pred_df.empty:
                        st.warning(f"文件 {file_name} 数据为空，跳过")
                        continue
                    
                    # 运行回测
                    result = engine.run_single_backtest(pred_df, strategy_name, file_name)
                    results[file_name] = result
                    
                    progress_bar.progress((i + 1) / len(selected_files))
                    st.success(f"✅ {file_name} 回测完成")
                    
                except Exception as e:
                    st.error(f"❌ {file_name} 回测失败: {e}")
                    continue
        
        # 显示结果
        if results:
            display_results(
                results,
                selected_metrics,
                system_config,
                show_risk_metrics=show_risk_metrics,
                show_period_stats=show_period_stats,
                show_daily_holdings=show_daily_holdings,
                show_trade_history=show_trade_history,
            )
        else:
            st.error("没有成功完成的回测结果")
    
    elif not available_files:
        st.warning(f"在目录 '{data_dir}' 中没有找到CSV文件")
    else:
        st.info("请在侧边栏配置参数并选择文件后开始回测")

def display_results(
    results: dict,
    selected_metrics: list,
    system_config,
    *,
    show_risk_metrics: bool,
    show_period_stats: bool,
    show_daily_holdings: bool,
    show_trade_history: bool,
):
    """显示回测结果"""
    def _fmt_pct(value):
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.2%}"

    def _fmt_ratio(value, decimals: int = 3):
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"

    st.header("📊 回测结果")

    st.subheader("📋 综合摘要")
    summary_rows = []
    for file_name, result in results.items():
        strategy_name = result.get("strategy_name", "unknown")
        final_value = result.get("final_value", 0.0)
        initial_cash = getattr(system_config, "initial_cash", 0)
        return_rate = (final_value - initial_cash) / initial_cash if initial_cash else 0

        perf_df = result.get("performance", pd.DataFrame())
        annual_return = perf_df.loc["annual_return", "value"] if "annual_return" in perf_df.index else 0
        max_drawdown = perf_df.loc["max_drawdown", "value"] if "max_drawdown" in perf_df.index else 0
        sharpe_ratio = perf_df.loc["sharpe_ratio", "value"] if "sharpe_ratio" in perf_df.index else 0

        industry_hhi = perf_df.loc["industry_hhi", "value"] if "industry_hhi" in perf_df.index else None
        total_turnover = perf_df.loc["total_turnover", "value"] if "total_turnover" in perf_df.index else None
        round_trip_win = None
        if "round_trip_win_rate" in perf_df.index:
            round_trip_win = perf_df.loc["round_trip_win_rate", "value"]
        elif "win_rate" in perf_df.index:
            round_trip_win = perf_df.loc["win_rate", "value"]
        sortino_ratio = perf_df.loc["sortino_ratio", "value"] if "sortino_ratio" in perf_df.index else None

        summary_rows.append(
            {
                "文件名": file_name,
                "策略": strategy_name,
                "初始资金": f"{initial_cash:,.0f}",
                "最终资金": f"{final_value:,.0f}",
                "总收益率": _fmt_pct(return_rate),
                "年化收益率": _fmt_pct(annual_return),
                "最大回撤": _fmt_pct(max_drawdown),
                "夏普比率": _fmt_ratio(sharpe_ratio),
                "行业HHI": _fmt_pct(industry_hhi),
                "总换手率": _fmt_pct(total_turnover),
                "RoundTrip胜率": _fmt_pct(round_trip_win),
                "Sortino": _fmt_ratio(sortino_ratio),
                "有效股票数": result.get("valid_stocks", 0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    tabs = st.tabs(list(results.keys()))
    for tab, (file_name, result) in zip(tabs, results.items()):
        with tab:
            st.subheader(f"📈 详细分析: {file_name}")
            display_detailed_analysis(
                result,
                selected_metrics,
                system_config,
                show_risk_metrics=show_risk_metrics,
                show_period_stats=show_period_stats,
                show_daily_holdings=show_daily_holdings,
                show_trade_history=show_trade_history,
            )

def display_detailed_analysis(
    result: dict,
    selected_metrics: list,
    system_config,
    *,
    show_risk_metrics: bool,
    show_period_stats: bool,
    show_daily_holdings: bool,
    show_trade_history: bool,
):
    """显示详细分析"""
    # 转换为BacktestResult对象以使用新的方法
    result_obj = BacktestResult.from_dict(result)

    def fmt_pct(value, decimals: int = 2):
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}%}"

    def fmt_ratio(value, decimals: int = 3):
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"

    strategy_nav: pd.Series = result.get("strategy_nav")
    benchmark_nav: pd.Series = result.get("benchmark_nav")

    if strategy_nav is None or strategy_nav.empty:
        st.warning("当前结果缺少净值数据，无法展示详细分析。")
        return

    daily_returns = strategy_nav.pct_change().dropna()
    detail_tabs = st.tabs([
        "📋 概览",
        "📊 净值曲线",
        "📈 收益分析",
        "⚠️ 风险分析",
        "📅 期间统计",
        "💼 持仓分析",
        "🔄 交易记录",
    ])

    # 概览
    with detail_tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        initial_cash = getattr(system_config, "initial_cash", 0)
        final_value = result.get("final_value", 0.0)
        return_rate = (final_value - initial_cash) / initial_cash if initial_cash else 0

        with col1:
            st.metric("初始资金", f"{initial_cash:,.0f}")
        with col2:
            st.metric("最终资金", f"{final_value:,.0f}", f"{return_rate:.2%}")
        with col3:
            st.metric("有效股票数", result.get("valid_stocks", 0))
        with col4:
            if CONFIG_AVAILABLE:
                benchmark_info = get_benchmark_info(system_config.benchmark_index)
                st.metric("基准指数", benchmark_info['name'] if benchmark_info else system_config.benchmark_index)
            else:
                st.metric("基准指数", system_config.benchmark_index)

        perf_df = result.get("performance", pd.DataFrame())
        if perf_df.empty:
            st.info("暂无性能指标数据。")
        else:
            st.subheader("关键性能指标")
            if selected_metrics:
                display_metrics = [m for m in selected_metrics if m in perf_df.index]
            else:
                display_metrics = perf_df.index.tolist()
            display_metrics = display_metrics or perf_df.index.tolist()

            cols = st.columns(min(len(display_metrics), 4) or 1)
            for idx, metric_name in enumerate(display_metrics):
                value = perf_df.loc[metric_name, "value"]
                with cols[idx % len(cols)]:
                    if CONFIG_AVAILABLE:
                        info = get_metric_info(metric_name)
                        if info:
                            st.metric(info["name"], info["format"].format(value), info.get("description", ""))
                        else:
                            st.metric(metric_name, f"{value:.3f}")
                    else:
                        st.metric(metric_name, f"{value:.3f}")

        structure_metrics = {}
        trading_metrics = {}
        extended_metrics = {}
        if isinstance(result_obj.detailed_metrics, dict):
            structure_metrics = result_obj.detailed_metrics.get("structure_metrics", {}) or {}
            trading_metrics = result_obj.detailed_metrics.get("trading_metrics", {}) or {}
            extended_metrics = result_obj.detailed_metrics.get("extended_risk_metrics", {}) or {}

        if structure_metrics:
            with st.expander("🧱 结构指标", expanded=False):
                struct_rows = [
                    {"指标": "有效持仓数", "数值": fmt_ratio(structure_metrics.get("effective_positions"), 2)},
                    {"指标": "最大单一权重", "数值": fmt_pct(structure_metrics.get("max_single_weight"))},
                    {"指标": "行业HHI", "数值": fmt_pct(structure_metrics.get("industry_hhi"))},
                    {"指标": "Top行业权重", "数值": fmt_pct(structure_metrics.get("top_industry_weight"))},
                    {"指标": "行业轮动", "数值": fmt_pct(structure_metrics.get("industry_rotation"))},
                    {"指标": "行业数量", "数值": structure_metrics.get("industry_count", "-")},
                    {"指标": "归一化熵", "数值": fmt_ratio(structure_metrics.get("normalized_entropy"), 3)},
                    {"指标": "基尼系数", "数值": fmt_ratio(structure_metrics.get("gini_coefficient"), 3)},
                ]
                st.table(pd.DataFrame(struct_rows))

                weights = structure_metrics.get("industry_weights")
                if isinstance(weights, dict) and weights:
                    weight_df = pd.DataFrame(
                        sorted(weights.items(), key=lambda kv: kv[1], reverse=True),
                        columns=["行业", "权重"],
                    )
                    weight_df["权重"] = weight_df["权重"].apply(fmt_pct)
                    st.write("**行业权重分布**")
                    st.dataframe(weight_df, use_container_width=True)

        if trading_metrics:
            with st.expander("🔄 交易指标", expanded=False):
                win_rate = trading_metrics.get("round_trip_win_rate")
                if win_rate is None:
                    win_rate = trading_metrics.get("win_rate")
                trading_rows = [
                    {"指标": "总换手率", "数值": fmt_pct(trading_metrics.get("total_turnover"))},
                    {"指标": "日均换手率", "数值": fmt_pct(trading_metrics.get("average_daily_turnover"))},
                    {"指标": "交易笔数", "数值": trading_metrics.get("trade_count", "-")},
                    {"指标": "RoundTrip数量", "数值": trading_metrics.get("round_trip_count", "-")},
                    {"指标": "RoundTrip胜率", "数值": fmt_pct(win_rate)},
                    {"指标": "平均持有天数", "数值": fmt_ratio(trading_metrics.get("avg_holding_days"), 2)},
                    {"指标": "中位持有天数", "数值": fmt_ratio(trading_metrics.get("median_holding_days"), 2)},
                    {"指标": "最长持有天数", "数值": fmt_ratio(trading_metrics.get("max_holding_days"), 2)},
                    {"指标": "盈亏比", "数值": fmt_ratio(trading_metrics.get("payoff_ratio"), 3)},
                    {"指标": "期望收益", "数值": fmt_ratio(trading_metrics.get("expectancy"), 3)},
                ]
                st.table(pd.DataFrame(trading_rows))

        if extended_metrics:
            with st.expander("⚖️ 扩展风险指标", expanded=False):
                extended_rows = [
                    {"指标": "Sortino比率", "数值": fmt_ratio(extended_metrics.get("sortino_ratio"), 3)},
                    {"指标": "下行波动率", "数值": fmt_pct(extended_metrics.get("downside_deviation"))},
                    {"指标": "尾部比率", "数值": fmt_ratio(extended_metrics.get("tail_ratio"), 3)},
                    {"指标": "Ulcer指数", "数值": fmt_pct(extended_metrics.get("ulcer_index"))},
                    {"指标": "偏度", "数值": fmt_ratio(extended_metrics.get("skewness"), 3)},
                    {"指标": "峰度", "数值": fmt_ratio(extended_metrics.get("kurtosis"), 3)},
                    {"指标": "样本数量", "数值": extended_metrics.get("return_count", "-")},
                ]
                st.table(pd.DataFrame(extended_rows))

    # 净值曲线
    with detail_tabs[1]:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            strategy_cumret = strategy_nav / strategy_nav.iloc[0]
            benchmark_series = benchmark_nav if benchmark_nav is not None else pd.Series()
            if benchmark_series is not None and not benchmark_series.empty:
                benchmark_cumret = benchmark_series / benchmark_series.iloc[0]
            else:
                benchmark_cumret = pd.Series(index=strategy_cumret.index, data=1.0)

            ax1.plot(strategy_cumret.index, strategy_cumret.values, label="Strategy NAV", linewidth=2, color="blue")
            if not benchmark_cumret.empty:
                ax1.plot(benchmark_cumret.index, benchmark_cumret.values, label="Benchmark NAV", linestyle="--", alpha=0.8, color="red")
            ax1.set_title("Net Asset Value Curve")
            ax1.set_ylabel("Cumulative Return")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            relative_ret = strategy_cumret - benchmark_cumret
            color = "green" if relative_ret.iloc[-1] > 0 else "red"
            ax2.fill_between(relative_ret.index, relative_ret.values, alpha=0.3, color=color)
            ax2.plot(relative_ret.index, relative_ret.values, color=color, linewidth=1)
            ax2.set_title("Relative to Benchmark Return")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Relative Return")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("### 📊 关键指标")
            metrics = result.get("performance", pd.DataFrame())
            for metric_key in ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "volatility"]:
                if metric_key in metrics.index:
                    value = metrics.loc[metric_key, "value"]
                    if CONFIG_AVAILABLE:
                        info = get_metric_info(metric_key)
                        label = info["name"] if info else metric_key
                        fmt = info["format"] if info else "{:.3f}"
                        st.write(f"**{label}:** `{fmt.format(value)}`")
                    else:
                        st.write(f"**{metric_key}:** `{value:.3f}`")

    # 收益分析
    with detail_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 📊 日收益分布")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(daily_returns, bins=20, alpha=0.7, color="blue", edgecolor="black")
            ax1.axvline(daily_returns.mean(), color="red", linestyle="--", label=f"Mean: {daily_returns.mean():.4f}")
            ax1.set_title("Daily Return Distribution")
            ax1.set_xlabel("Return Rate")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            cumulative_returns = (1 + daily_returns).cumprod()
            ax2.plot(cumulative_returns.index, cumulative_returns.values, color="green", linewidth=2)
            ax2.set_title("Cumulative Return")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Cumulative Return")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("### 📈 收益统计")
            stats_data = {
                "指标": ["日均收益", "日收益标准差", "正收益天数", "负收益天数", "胜率"],
                "数值": [
                    f"{daily_returns.mean():.4f}",
                    f"{daily_returns.std():.4f}",
                    f"{(daily_returns > 0).sum()}",
                    f"{(daily_returns < 0).sum()}",
                    f"{(daily_returns > 0).mean():.2%}",
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

            st.write("### 📅 月度收益")
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            benchmark_monthly = pd.Series(dtype=float)
            if benchmark_nav is not None and not benchmark_nav.empty:
                benchmark_daily = benchmark_nav.pct_change().dropna()
                benchmark_monthly = benchmark_daily.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if not monthly_returns.empty:
                monthly_df = pd.DataFrame({"策略收益": monthly_returns})
                if not benchmark_monthly.empty:
                    monthly_df["基准收益"] = benchmark_monthly.reindex(monthly_df.index)
                    monthly_df["超额收益"] = monthly_df["策略收益"] - monthly_df["基准收益"]
                st.dataframe(monthly_df.tail(6), use_container_width=True)
            else:
                st.info("暂无月度收益数据。")

    # 风险分析
    with detail_tabs[3]:
        if not show_risk_metrics:
            st.info("已关闭风险分析展示。")
        else:
            drawdown = strategy_nav / strategy_nav.cummax() - 1
            running_max = strategy_nav.cummax()
            col1, col2 = st.columns(2)
            with col1:
                st.write("### ⚠️ 回撤分析")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                ax1.plot(strategy_nav.index, strategy_nav.values, label='Strategy NAV', color='blue', linewidth=2)
                ax1.plot(running_max.index, running_max.values, label='Historical High', color='green', linestyle='--', alpha=0.8)
                ax1.set_ylabel('NAV')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax2.fill_between(drawdown.index, drawdown.values, color='red', alpha=0.3)
                ax2.set_ylabel('Drawdown')
                ax2.set_xlabel('Date')
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.write("### 风险指标")
                metrics = result.get("performance", pd.DataFrame())
                risk_metrics = [
                    "max_drawdown",
                    "volatility",
                    "var_95",
                    "cvar_95",
                    "information_ratio",
                    "calmar_ratio",
                    "sortino_ratio",
                    "downside_deviation",
                    "tail_ratio",
                    "ulcer_index",
                ]
                rows = []
                for metric_name in risk_metrics:
                    if metric_name in metrics.index:
                        value = metrics.loc[metric_name, "value"]
                        if CONFIG_AVAILABLE:
                            info = get_metric_info(metric_name)
                            label = info["name"] if info else metric_name
                            fmt = info["format"] if info else "{:.3f}"
                        else:
                            label, fmt = metric_name, "{:.3f}"
                        rows.append({"指标": label, "数值": fmt.format(value)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("暂无额外风险指标。")

    # 期间统计
    with detail_tabs[4]:
        if not show_period_stats:
            st.info("已关闭期间统计展示。")
        else:
            st.write("### 📅 月度统计")
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            benchmark_monthly = pd.Series(dtype=float)
            if benchmark_nav is not None and not benchmark_nav.empty:
                benchmark_daily = benchmark_nav.pct_change().dropna()
                benchmark_monthly = benchmark_daily.resample('M').apply(lambda x: (1 + x).prod() - 1)
            if not monthly_returns.empty:
                monthly_df = pd.DataFrame({"策略收益": monthly_returns})
                if not benchmark_monthly.empty:
                    monthly_df["基准收益"] = benchmark_monthly.reindex(monthly_df.index)
                    monthly_df["超额收益"] = monthly_df["策略收益"] - monthly_df["基准收益"]
                monthly_df["胜率"] = (monthly_df["策略收益"] > 0).astype(float)
                st.dataframe(monthly_df.tail(12), use_container_width=True)
            else:
                st.info("暂无月度统计数据。")

            st.write("### 📆 年度统计")
            yearly_returns = daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            if benchmark_nav is not None and not benchmark_nav.empty:
                benchmark_yearly = benchmark_nav.pct_change().dropna().resample('Y').apply(lambda x: (1 + x).prod() - 1)
            else:
                benchmark_yearly = pd.Series(dtype=float)
            if not yearly_returns.empty:
                yearly_df = pd.DataFrame({"策略收益": yearly_returns})
                if not benchmark_yearly.empty:
                    yearly_df["基准收益"] = benchmark_yearly.reindex(yearly_df.index)
                    yearly_df["超额收益"] = yearly_df["策略收益"] - yearly_df["基准收益"]
                st.dataframe(yearly_df, use_container_width=True)
            else:
                st.info("暂无年度统计数据。")

    # 持仓分析
    with detail_tabs[5]:
        daily_holdings = result.get('daily_holdings', []) if show_daily_holdings else []
        if not daily_holdings:
            st.info("暂无持仓记录或已关闭展示。")
        else:
            # 显示当日持仓明细
            st.write("### 当日持仓明细")
            latest_record = next((rec for rec in reversed(daily_holdings) if rec.get('holdings')), None)
            if latest_record:
                holdings_data = latest_record.get('holdings') or []
                rows = []
                if isinstance(holdings_data, dict):
                    iterable = holdings_data.items()
                else:
                    iterable = [(item.get('code'), item) for item in holdings_data]

                for code, info in iterable:
                    if code is None or info is None:
                        continue
                    stock_name = code2name.get(code, '') if code2name else ''
                    rows.append({
                        '代码': code,
                        '名称': stock_name,
                        '持仓数量': info.get('size', 0),
                        '现价': info.get('price', 0),
                        '持仓市值': info.get('value', 0),
                        '权重': f"{info.get('weight', 0):.2%}" if info.get('weight') is not None else '',
                        '买入日期': info.get('buy_date')
                    })
                if rows:
                    holdings_df = pd.DataFrame(rows)
                    for col in ['现价', '持仓市值']:
                        if col in holdings_df.columns:
                            holdings_df[col] = holdings_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float, np.number)) else x)
                    st.dataframe(holdings_df, use_container_width=True)
                else:
                    st.info("暂无持仓数据。")
            else:
                st.info("暂无持仓数据。")

            # 显示每日持仓历史
            st.write("### 每日持仓历史")

            # 使用AnalysisBuilder生成每日持仓历史
            daily_holdings_df = AnalysisBuilder.prepare_daily_holdings_history(result_obj)

            if daily_holdings_df.empty:
                st.info("暂无每日持仓历史数据。")
            else:
                # 创建显示用的DataFrame
                display_df = daily_holdings_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

                # 重命名列
                rename_dict = {'date': '日期', 'total_value': '总资产', 'cash': '现金'}
                for col in display_df.columns:
                    if col.endswith('_size'):
                        code = col.replace('_size', '')
                        rename_dict[col] = f"{code}_数量"
                    elif col.endswith('_value'):
                        code = col.replace('_value', '')
                        rename_dict[col] = f"{code}_市值"
                    elif col.endswith('_weight'):
                        code = col.replace('_weight', '')
                        rename_dict[col] = f"{code}_权重"

                display_df = display_df.rename(columns=rename_dict)

                # 格式化数值列
                numeric_cols = [col for col in display_df.columns if col != '日期']
                for col in numeric_cols:
                    if col == '总资产' or col == '现金':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else '')
                    elif '_数量' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if x != 0 else '')
                    elif '_市值' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if x != 0 else '')
                    elif '_权重' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if x != 0 else '')

                st.dataframe(display_df, use_container_width=True)

                # 显示持仓权重变化图
                weight_cols = [col for col in display_df.columns if '_权重' in col and not display_df[col].str.contains('').all()]
                if weight_cols:
                    st.write("#### 持仓权重变化")
                    weight_data = daily_holdings_df[[col.replace('_权重', '_weight') for col in weight_cols]].copy()
                    weight_data.columns = [col.replace('_weight', '') for col in weight_data.columns]
                    weight_data.index = daily_holdings_df['date'].dt.strftime('%Y-%m-%d')
                    st.line_chart(weight_data)

            # 显示资产曲线
            timeline = [
                {
                    'date': rec.get('date'),
                    'total_value': rec.get('total_value'),
                    'cash': rec.get('cash')
                }
                for rec in daily_holdings if rec.get('date') is not None
            ]
            if timeline:
                timeline_df = pd.DataFrame(timeline).set_index('date')
                st.write("### 资产曲线")
                st.line_chart(timeline_df)

    # 交易记录
    with detail_tabs[6]:
        trade_history = result.get('trade_history', []) if show_trade_history else []
        if not trade_history:
            st.info("暂无交易记录或已关闭展示。")
        else:
            st.write("### 🔄 交易历史")
            trade_df = pd.DataFrame(trade_history)
            if not trade_df.empty:
                if 'date' in trade_df.columns:
                    trade_df['date'] = pd.to_datetime(trade_df['date']).dt.strftime('%Y-%m-%d')
                for col in ['value', 'price', 'portfolio_value']:
                    if col in trade_df.columns:
                        trade_df[col] = trade_df[col].apply(
                            lambda x: f"{float(x):,.2f}" if isinstance(x, (int, float, np.number)) else x
                        )
                if 'size' in trade_df.columns:
                    trade_df['size'] = trade_df['size'].apply(
                        lambda x: f"{int(x)}" if isinstance(x, (int, np.integer)) else x
                    )
                st.dataframe(trade_df, use_container_width=True)
            else:
                st.info("暂无交易记录。")

if __name__ == "__main__":
    main()