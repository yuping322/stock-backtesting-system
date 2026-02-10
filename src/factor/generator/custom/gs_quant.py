#!/usr/bin/env python3
"""
GS-Quant 独有指标生成器 (本地实现版本)
使用纯 Python 实现 gs-quant 独有的 34 个指标因子

作者: GitHub Copilot
日期: 2025年12月1日
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


def generate_fake_data(n_days=252, start_price=100.0, volatility=0.02):
    """
    生成假的股票价格数据

    参数:
    - n_days: 数据天数 (默认252个交易日)
    - start_price: 起始价格
    - volatility: 日波动率
    """
    # 生成日期序列
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # 生成随机收益率 (正态分布)
    np.random.seed(42)  # 固定种子保证可重复
    returns_data = np.random.normal(0.0001, volatility, n_days)  # 轻微正向漂移

    # 计算价格序列
    prices = [start_price]
    for ret in returns_data:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # 创建价格序列
    price_series = pd.Series(prices[:-1], index=dates, name='price')

    # 生成基准收益率 (更稳定的)
    benchmark_returns = np.random.normal(0.00005, volatility * 0.5, n_days)
    benchmark_prices = [start_price]
    for ret in benchmark_returns:
        new_price = benchmark_prices[-1] * (1 + ret)
        benchmark_prices.append(new_price)

    benchmark_series = pd.Series(benchmark_prices[:-1], index=dates, name='benchmark')

    # 添加一些缺失值和异常值
    price_with_missing = price_series.copy()
    # 随机设置一些值为 NaN
    missing_indices = np.random.choice(len(price_with_missing), size=int(len(price_with_missing)*0.05), replace=False)
    price_with_missing.iloc[missing_indices] = np.nan

    # 添加一些异常值
    outlier_indices = np.random.choice(len(price_with_missing), size=int(len(price_with_missing)*0.03), replace=False)
    price_with_missing.iloc[outlier_indices] = price_with_missing.iloc[outlier_indices] * np.random.choice([3, 0.3], len(outlier_indices))

    return price_series, benchmark_series, price_with_missing


# ==================== 风险指标 ====================

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, annualization_factor=252):
    """
    计算夏普比率
    Sharpe Ratio = (Expected Return - Risk Free Rate) / Volatility
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    if len(excess_returns) < 2:
        return np.nan
    return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns, risk_free_rate=0.02, annualization_factor=252):
    """
    计算索提诺比率
    Sortino Ratio = (Expected Return - Risk Free Rate) / Downside Deviation
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return np.nan
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return np.nan
    return np.sqrt(annualization_factor) * excess_returns.mean() / downside_deviation


def calculate_max_drawdown(prices):
    """
    计算最大回撤
    Max Drawdown = (Peak - Trough) / Peak
    """
    if len(prices) < 2:
        return np.nan

    peak = prices.iloc[0]
    max_dd = 0

    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def calculate_information_ratio(returns, benchmark_returns):
    """
    计算信息比率
    Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
    """
    excess_returns = returns - benchmark_returns
    if len(excess_returns) < 2:
        return np.nan
    tracking_error = excess_returns.std()
    if tracking_error == 0:
        return np.nan
    return excess_returns.mean() / tracking_error


def calculate_alpha(returns, benchmark_returns, beta):
    """
    计算 Alpha
    Alpha = Portfolio Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
    """
    # 简化的 Alpha 计算 (假设无风险利率为0)
    return returns.mean() - beta * benchmark_returns.mean()


def calculate_treynor_ratio(returns, beta, risk_free_rate=0.02):
    """
    计算特雷诺比率
    Treynor Ratio = (Portfolio Return - Risk Free Rate) / Beta
    """
    if beta == 0:
        return np.nan
    return (returns.mean() - risk_free_rate/252) / beta


# ==================== 数据处理指标 ====================

def fill_missing_values(series, method='interpolate'):
    """
    填充缺失值
    """
    if method == 'interpolate':
        return series.interpolate(method='linear')
    elif method == 'ffill':
        return series.fillna(method='ffill')
    elif method == 'bfill':
        return series.fillna(method='bfill')
    else:
        return series.fillna(series.mean())


def remove_outliers(series, method='iqr', threshold=1.5):
    """
    移除异常值
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    return series


def normalize_series(series, method='zscore'):
    """
    标准化序列
    """
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    return series


def smooth_spikes(series, threshold=2.0):
    """
    平滑尖峰值
    """
    mean_val = series.mean()
    std_val = series.std()
    z_scores = np.abs((series - mean_val) / std_val)

    # 将超过阈值的点替换为附近值的平均
    smoothed = series.copy()
    spike_indices = z_scores > threshold

    for i, is_spike in enumerate(spike_indices):
        if is_spike:
            # 使用前后值的平均
            neighbors = []
            if i > 0:
                neighbors.append(series.iloc[i-1])
            if i < len(series) - 1:
                neighbors.append(series.iloc[i+1])

            if neighbors:
                smoothed.iloc[i] = np.mean(neighbors)

    return smoothed


def repeat(series, n=1):
    """
    重复序列
    """
    repeated = series.copy()
    for i in range(n):
        repeated = pd.concat([repeated, series])
    return repeated


def interpolate_series(series):
    """
    插值序列
    """
    return series.interpolate(method='linear')


def lag(series, periods=1):
    """
    滞后序列
    """
    return series.shift(periods)


def diff(series, periods=1):
    """
    差分序列
    """
    return series.diff(periods)


# ==================== 统计分析指标 ====================

def calculate_autocorrelation(series, lags=5):
    """
    计算自相关
    """
    try:
        return acf(series.dropna(), nlags=lags)
    except:
        return np.array([np.nan] * (lags + 1))


def calculate_partial_autocorrelation(series, lags=5):
    """
    计算偏自相关
    """
    try:
        return pacf(series.dropna(), nlags=lags)
    except:
        return np.array([np.nan] * (lags + 1))


def is_stationary(series, significance_level=0.05):
    """
    平稳性检验 (ADF检验)
    返回: True 如果是平稳的, False 如果不是
    """
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        return p_value < significance_level
    except:
        return False


def seasonal_decompose_series(series, model='additive', period=30):
    """
    季节性分解
    """
    try:
        decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    except:
        return {'trend': None, 'seasonal': None, 'residual': None}


def calculate_skewness(series):
    """
    计算偏度
    """
    return stats.skew(series.dropna())


def calculate_kurtosis(series):
    """
    计算峰度
    """
    return stats.kurtosis(series.dropna())


def calculate_excess_kurtosis(series):
    """
    计算超额峰度 (减去正态分布的峰度 3)
    """
    return stats.kurtosis(series.dropna()) - 3


# ==================== 收益和波动率指标 ====================

def calculate_returns(prices):
    """
    计算收益率
    """
    return prices.pct_change().dropna()


def calculate_rolling_returns(prices, window=20):
    """
    计算滚动收益率
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).mean()


def calculate_rolling_log_returns(prices, window=20):
    """
    计算滚动对数收益率
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).mean()


def calculate_rolling_volatility(prices, window=20, annualization_factor=252):
    """
    计算滚动波动率
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(annualization_factor)


def calculate_cumulative_returns(prices):
    """
    计算累计收益率
    """
    return (prices / prices.iloc[0] - 1)


def calculate_excess_returns(prices, benchmark_prices):
    """
    计算超额收益
    """
    portfolio_returns = prices.pct_change()
    benchmark_returns = benchmark_prices.pct_change()
    return portfolio_returns - benchmark_returns


def calculate_trailing_return(prices, window=60):
    """
    计算尾随收益
    """
    return (prices / prices.shift(window) - 1).dropna()


def calculate_calendar_spread(prices, start_month=1, end_month=12):
    """
    计算日历价差 (简化的实现)
    """
    # 简化为计算年初到年底的收益差
    yearly_returns = prices.groupby(prices.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    return yearly_returns.mean() if len(yearly_returns) > 0 else np.nan


def calculate_volatility(prices, window=20, annualization_factor=252):
    """
    计算波动率
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(annualization_factor)


def calculate_correlation(series1, series2, window=20):
    """
    计算相关性
    """
    return series1.rolling(window=window).corr(series2)


def calculate_beta(series, benchmark, window=20):
    """
    计算 Beta
    """
    returns = series.pct_change()
    benchmark_returns = benchmark.pct_change()
    covariance = returns.rolling(window=window).cov(benchmark_returns)
    variance = benchmark_returns.rolling(window=window).var()
    return covariance / variance


# ==================== 新增：风险质量与分布类指标 ====================

def cagr(prices, annualization_factor=252):
    """复合年化增长率 (CAGR)"""
    n = len(prices)
    if n < 2:
        return np.nan
    total_return = prices.iloc[-1] / prices.iloc[0]
    return total_return ** (annualization_factor / n) - 1


def calmar_ratio(prices, annualization_factor=252):
    """Calmar 比率 = CAGR / 最大回撤"""
    dd = calculate_max_drawdown(prices)
    if dd == 0:
        return np.nan
    return cagr(prices, annualization_factor) / dd


def mar_ratio(prices, annualization_factor=252):
    """MAR 比率（常与 Calmar 同口径，这里等同实现）"""
    return calmar_ratio(prices, annualization_factor)


def omega_ratio(returns, threshold=0.0):
    """Omega 比率：收益分布相对于阈值的正负面积比"""
    r = returns.dropna()
    gain = np.sum(np.clip(r - threshold, 0, None))
    loss = np.sum(np.clip(threshold - r, 0, None))
    return np.nan if loss == 0 else gain / loss


def downside_deviation_series(returns, threshold=0.0, window=20):
    """下行偏差（序列版）"""
    r = returns - threshold
    downside = np.where(r < 0, r, 0.0)
    downside = pd.Series(downside, index=returns.index)
    return downside.pow(2).rolling(window=window).mean().pow(0.5)


def rolling_sharpe(returns, window=60, annualization_factor=252, threshold=0.0):
    """滚动 Sharpe（阈值可作为近似无风险利率/252）"""
    er = returns - threshold
    mu = er.rolling(window=window).mean()
    sd = er.rolling(window=window).std()
    return np.sqrt(annualization_factor) * (mu / sd)


def rolling_sortino(returns, window=60, annualization_factor=252, threshold=0.0):
    """滚动 Sortino"""
    er = returns - threshold
    mu = er.rolling(window=window).mean()
    dd = downside_deviation_series(returns, threshold=threshold, window=window)
    return np.sqrt(annualization_factor) * (mu / dd)


def tracking_error_series(port_returns, bench_returns, window=60):
    """跟踪误差（序列）：滚动 std(port - bench)"""
    ar = (port_returns - bench_returns)
    return ar.rolling(window=window).std()


def var_historical(returns, alpha=0.95):
    """历史法 VaR：alpha 分位数（损失为正值）"""
    r = returns.dropna().sort_values()
    q = r.quantile(1 - alpha)
    return -q


def es_historical(returns, alpha=0.95):
    """历史法 ES：低于 VaR 阈值的条件均值"""
    r = returns.dropna()
    var = r.quantile(1 - alpha)
    tail = r[r <= var]
    return -tail.mean() if len(tail) > 0 else np.nan


# ==================== 新增：回撤时长与恢复 ====================

def drawdown_series(prices):
    """返回从峰值的回撤序列"""
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def drawdown_duration(prices):
    """最长回撤时长（天数）"""
    dd = drawdown_series(prices)
    durations = []
    current = 0
    for v in dd:
        if v < 0:
            current += 1
        else:
            durations.append(current)
            current = 0
    durations.append(current)
    return int(np.max(durations)) if len(durations) else 0


def average_drawdown(prices):
    """平均回撤深度（只统计回撤期）"""
    dd = drawdown_series(prices)
    negatives = dd[dd < 0]
    return float(negatives.mean()) if len(negatives) else 0.0


def average_recovery_time(prices):
    """平均恢复时间（从谷底回到新高的平均天数，粗略估计）"""
    dd = drawdown_series(prices)
    idx = prices.index
    recovery_times = []
    in_dd = False
    trough_i = None
    peak_val = prices.iloc[0]
    for i in range(len(prices)):
        price = prices.iloc[i]
        peak_val = max(peak_val, price)
        if price < peak_val:
            # in drawdown
            if not in_dd:
                in_dd = True
                trough_i = i
        else:
            # recovered
            if in_dd and trough_i is not None:
                recovery_times.append(i - trough_i)
            in_dd = False
            trough_i = None
    return float(np.mean(recovery_times)) if recovery_times else 0.0


def main():
    """
    主函数：生成所有 gs-quant 独有指标
    """
    print("🚀 开始生成 GS-Quant 独有指标 (本地实现)...")
    print("=" * 50)

    # 1. 生成假数据
    print("\n📅 生成假数据...")
    price_series, benchmark_series, price_with_missing = generate_fake_data(
        n_days=252,  # 一年交易日
        start_price=100.0,
        volatility=0.02
    )

    # 计算收益率
    returns_series = calculate_returns(price_series)

    print(f"✅ 生成数据完成: {len(price_series)} 个观测值")
    print(f"💰 起始价格: ${price_series.iloc[0]:.2f}")
    print(f"💰 结束价格: ${price_series.iloc[-1]:.2f}")
    print(f"📊 缺失值数量: {price_with_missing.isna().sum()}")

    # 2. 计算各类指标
    all_results = {}

    # 风险指标
    print("\n📊 计算风险指标...")
    all_results['sharpe_ratio'] = calculate_sharpe_ratio(returns_series)
    all_results['sortino_ratio'] = calculate_sortino_ratio(returns_series)
    all_results['max_drawdown'] = calculate_max_drawdown(price_series)
    all_results['information_ratio'] = calculate_information_ratio(returns_series, calculate_returns(benchmark_series))
    beta_val = calculate_beta(price_series, benchmark_series).iloc[-1]
    all_results['alpha'] = calculate_alpha(returns_series, calculate_returns(benchmark_series), beta_val)
    all_results['treynor_ratio'] = calculate_treynor_ratio(returns_series, beta_val)
    print("✅ 风险指标计算完成")

    # 数据处理指标
    print("\n🔧 计算数据处理指标...")
    all_results['fill_missing_values'] = fill_missing_values(price_with_missing)
    all_results['remove_outliers'] = remove_outliers(price_with_missing)
    all_results['normalize_series'] = normalize_series(price_series)
    all_results['smooth_spikes'] = smooth_spikes(price_series)
    all_results['repeat'] = repeat(price_series, n=2)
    all_results['interpolate'] = interpolate_series(price_with_missing)
    all_results['lag'] = lag(price_series)
    all_results['diff'] = diff(price_series)
    print("✅ 数据处理指标计算完成")

    # 统计分析指标
    print("\n📈 计算统计分析指标...")
    all_results['autocorrelation'] = calculate_autocorrelation(returns_series)
    all_results['partial_autocorrelation'] = calculate_partial_autocorrelation(returns_series)
    all_results['is_stationary'] = is_stationary(returns_series)
    all_results['seasonal_decompose'] = seasonal_decompose_series(price_series)
    all_results['skewness'] = calculate_skewness(returns_series)
    all_results['kurtosis'] = calculate_kurtosis(returns_series)
    all_results['excess_kurtosis'] = calculate_excess_kurtosis(returns_series)
    print("✅ 统计分析指标计算完成")

    # 收益和波动率指标
    print("\n💰 计算收益和波动率指标...")
    all_results['rolling_returns'] = calculate_rolling_returns(price_series)
    all_results['rolling_log_returns'] = calculate_rolling_log_returns(price_series)
    all_results['rolling_volatility'] = calculate_rolling_volatility(price_series)
    all_results['cumulative_returns'] = calculate_cumulative_returns(price_series)
    all_results['excess_returns'] = calculate_excess_returns(price_series, benchmark_series)
    all_results['trailing_return'] = calculate_trailing_return(price_series)
    all_results['calendar_spread'] = calculate_calendar_spread(price_series)
    all_results['volatility'] = calculate_volatility(price_series)
    all_results['correlation'] = calculate_correlation(price_series, benchmark_series)
    all_results['beta'] = calculate_beta(price_series, benchmark_series)
    print("✅ 收益和波动率指标计算完成")

    # 3. 输出结果摘要
    print("\n" + "=" * 50)
    print("📋 计算完成摘要")
    print("=" * 50)

    total_indicators = len(all_results)
    print(f"🎯 成功计算指标数量: {total_indicators}/34")

    if total_indicators > 0:
        print("\n📊 各类型指标计算结果:")

        # 风险指标
        risk_indicators = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'information_ratio', 'alpha', 'treynor_ratio']
        risk_count = sum(1 for ind in risk_indicators if ind in all_results)
        print(f"  💰 风险指标: {risk_count}/6")

        # 数据处理指标
        data_indicators = ['fill_missing_values', 'remove_outliers', 'normalize_series', 'smooth_spikes',
                          'repeat', 'interpolate', 'lag', 'diff']
        data_count = sum(1 for ind in data_indicators if ind in all_results)
        print(f"  🔧 数据处理: {data_count}/8")

        # 统计分析指标
        stat_indicators = ['autocorrelation', 'partial_autocorrelation', 'is_stationary', 'seasonal_decompose',
                          'skewness', 'kurtosis', 'excess_kurtosis']
        stat_count = sum(1 for ind in stat_indicators if ind in all_results)
        print(f"  📈 统计分析: {stat_count}/7")

        # 收益波动率指标
        return_indicators = ['rolling_returns', 'rolling_log_returns', 'rolling_volatility', 'cumulative_returns',
                           'excess_returns', 'trailing_return', 'calendar_spread', 'volatility', 'correlation', 'beta']
        return_count = sum(1 for ind in return_indicators if ind in all_results)
        print(f"  📊 收益波动率: {return_count}/10")

        print("\n💾 所有指标已保存到内存中")
        print("💡 提示: 你可以访问 all_results 字典来查看具体数值")

        # 显示几个关键指标的示例值
        print("\n📈 示例指标值:")
        if 'sharpe_ratio' in all_results:
            print(f"  📊 Sharpe Ratio: {all_results['sharpe_ratio']:.3f}")
        if 'max_drawdown' in all_results:
            print(f"  📉 Max Drawdown: {all_results['max_drawdown']:.1%}")
        if 'volatility' in all_results:
            latest_vol = all_results['volatility'].iloc[-1] if hasattr(all_results['volatility'], 'iloc') else all_results['volatility']
            print(f"  📊 Volatility: {latest_vol:.3f}")
        if 'beta' in all_results:
            latest_beta = all_results['beta'].iloc[-1] if hasattr(all_results['beta'], 'iloc') else all_results['beta']
            print(f"  📊 Beta: {latest_beta:.3f}")
        # 新增指标示例
        print("\n🔎 新增指标示例:")
        print(f"  🚀 CAGR: {cagr(price_series):.2%}")
        print(f"  🧮 Calmar Ratio: {calmar_ratio(price_series):.3f}")
        print(f"  🧮 MAR Ratio: {mar_ratio(price_series):.3f}")
        print(f"  ⚖️  Omega Ratio: {omega_ratio(returns_series):.3f}")
        print(f"  📉 Max DD Duration: {drawdown_duration(price_series)} days")
        print(f"  📉 Avg Drawdown: {average_drawdown(price_series):.2%}")
        print(f"  🔄 Avg Recovery Time: {average_recovery_time(price_series):.1f} days")
    else:
        print("❌ 没有成功计算任何指标")

    print("\n🎉 GS-Quant 独有指标生成完成！")
    print("=" * 50)

    return all_results


if __name__ == "__main__":
    # 运行主函数
    results = main()

    # 可选：保存结果到文件
    import pickle
    try:
        with open('gs_quant_unique_indicators_local.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("💾 结果已保存到 gs_quant_unique_indicators_local.pkl")
    except Exception as e:
        print(f"⚠️ 保存结果失败: {e}")
