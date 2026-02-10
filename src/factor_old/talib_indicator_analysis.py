#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TA-Lib 技术指标检验程序
用于检验和可视化各种技术分析指标
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import talib
import akshare as ak


def load_market_data():
    """加载市场数据"""
    print("📥 下载上证指数数据...")
    try:
        # 使用 AKShare 获取上证指数数据
        index_data = ak.stock_zh_index_daily(symbol="sh000001")
        index_data['date'] = pd.to_datetime(index_data['date'])
        index_data.set_index('date', inplace=True)
        index_data = index_data.sort_index()

        # 取最近一年的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = index_data.loc[start_date:end_date].copy()

        print(f"✅ 获取到 {len(data)} 条数据 ({data.index.min().date()} 到 {data.index.max().date()})")
        return data

    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        print("🔄 生成模拟数据...")
        # 生成模拟数据
        dates = pd.date_range(start='2024-01-01', end=datetime.now().date(), freq='D')
        np.random.seed(42)
        price = 3000 + np.cumsum(np.random.randn(len(dates)) * 20)
        data = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.01),
            'high': price * (1 + np.random.randn(len(dates)) * 0.015),
            'low': price * (1 - np.random.randn(len(dates)) * 0.015),
            'close': price,
            'volume': np.random.randint(1000000, 50000000, len(dates))
        }, index=dates)
        return data


def calculate_indicators(data):
    """计算技术指标"""
    print("🔬 计算技术指标...")

    indicators = {
        'SMA': {'func': talib.SMA, 'params': {'timeperiod': 20}, 'desc': '简单移动平均线'},
        'EMA': {'func': talib.EMA, 'params': {'timeperiod': 20}, 'desc': '指数移动平均线'},
        'RSI': {'func': talib.RSI, 'params': {'timeperiod': 14}, 'desc': '相对强弱指数'},
        'MACD': {'func': talib.MACD, 'params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}, 'desc': 'MACD指标'},
        'BBANDS': {'func': talib.BBANDS, 'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2}, 'desc': '布林带'},
        'STOCH': {'func': talib.STOCH, 'params': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}, 'desc': '随机震荡指标'},
        'WILLR': {'func': talib.WILLR, 'params': {'timeperiod': 14}, 'desc': '威廉指标'},
        'CCI': {'func': talib.CCI, 'params': {'timeperiod': 20}, 'desc': '顺势指标'},
        'MFI': {'func': talib.MFI, 'params': {'timeperiod': 14}, 'desc': '资金流量指标'},
        'ROC': {'func': talib.ROC, 'params': {'timeperiod': 10}, 'desc': '变动率指标'}
    }

    results = {}

    for name, config in indicators.items():
        try:
            func = config['func']
            params = config['params']

            if name == 'MACD':
                macd, macdsignal, macdhist = func(data['close'], **params)
                results[name] = {
                    'macd': macd,
                    'signal': macdsignal,
                    'hist': macdhist,
                    'desc': config['desc']
                }
            elif name == 'BBANDS':
                upper, middle, lower = func(data['close'], **params)
                results[name] = {
                    'upper': upper,
                    'middle': middle,
                    'lower': lower,
                    'desc': config['desc']
                }
            elif name == 'STOCH':
                slowk, slowd = func(data['high'], data['low'], data['close'], **params)
                results[name] = {
                    'slowk': slowk,
                    'slowd': slowd,
                    'desc': config['desc']
                }
            else:
                # 处理需要多个价格参数的指标
                if name in ['WILLR', 'CCI']:
                    result = func(data['high'], data['low'], data['close'], **params)
                elif name == 'MFI':
                    result = func(data['high'], data['low'], data['close'], data['volume'], **params)
                else:
                    result = func(data['close'], **params)
                results[name] = {
                    'value': result,
                    'desc': config['desc']
                }

            print(f"✅ {name}: {config['desc']}")

        except Exception as e:
            print(f"❌ {name}: 计算失败 - {e}")
            results[name] = {'error': str(e), 'desc': config['desc']}

    return results


def print_statistics(results):
    """打印指标统计信息"""
    print()
    print("📊 技术指标统计信息:")
    print("-" * 50)

    for name, result in results.items():
        if 'error' in result:
            print(f"{name:8}: ❌ 计算失败")
            continue

        desc = result['desc']
        print(f"{name:8}: {desc}")

        if name == 'MACD':
            macd, signal, hist = result['macd'], result['signal'], result['hist']
            valid_count = macd.dropna().shape[0]
            print(f"- 有效数据点: {valid_count}")
            print(f"- MACD 均值: {macd.mean():.4f}")
            print(f"- Signal 均值: {signal.mean():.4f}")
            print()
        elif name == 'BBANDS':
            upper, middle, lower = result['upper'], result['middle'], result['lower']
            bandwidth = (upper - lower) / middle
            valid_count = upper.dropna().shape[0]
            print(f"- 有效数据点: {valid_count}")
            print(f"- 平均带宽: {bandwidth.mean():.4f}")
            print()
        elif name == 'STOCH':
            slowk, slowd = result['slowk'], result['slowd']
            valid_count = slowk.dropna().shape[0]
            print(f"- 有效数据点: {valid_count}")
            print(f"- %K 均值: {slowk.mean():.4f}")
            print(f"- %D 均值: {slowd.mean():.4f}")
            print()
        else:
            values = result['value']
            valid_count = values.dropna().shape[0]
            mean_val = values.mean()
            std_val = values.std()
            print(f"- 有效数据点: {valid_count}")
            print(f"- 均值: {mean_val:.4f}")
            print(f"- 标准差: {std_val:.4f}")
            print()


def create_visualization(data, results, output_dir):
    """创建可视化图表"""
    print("📈 生成可视化图表...")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图 - 只创建我们确定能绘制的图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TA-Lib 技术指标检验报告', fontsize=16, fontweight='bold')

    plot_count = 0
    max_plots = 4

    # 1. 价格走势和移动平均线
    if plot_count < max_plots and 'SMA' in results and 'value' in results['SMA']:
        ax = axes.flat[plot_count]
        ax.plot(data.index, data['close'], label='收盘价', color='black', linewidth=1)
        ax.plot(data.index, results['SMA']['value'], label='SMA(20)', color='blue', linewidth=1.5)
        if 'EMA' in results and 'value' in results['EMA']:
            ax.plot(data.index, results['EMA']['value'], label='EMA(20)', color='red', linewidth=1.5)
        ax.set_title('价格走势与移动平均线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_count += 1

    # 2. RSI指标
    if plot_count < max_plots and 'RSI' in results and 'value' in results['RSI']:
        ax = axes.flat[plot_count]
        rsi = results['RSI']['value']
        ax.plot(data.index, rsi, color='purple', linewidth=1.5)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='超买线')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='超卖线')
        ax.fill_between(data.index, 30, 70, alpha=0.1, color='gray', label='中性区')
        ax.set_title('RSI 相对强弱指数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_count += 1

    # 3. MACD指标
    if plot_count < max_plots and 'MACD' in results:
        ax = axes.flat[plot_count]
        macd_data = results['MACD']
        ax.plot(data.index, macd_data['macd'], label='MACD', color='blue', linewidth=1.5)
        ax.plot(data.index, macd_data['signal'], label='Signal', color='red', linewidth=1.5)
        ax.bar(data.index, macd_data['hist'], label='Histogram', color='gray', alpha=0.7, width=1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('MACD 指标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_count += 1

    # 4. 布林带
    if plot_count < max_plots and 'BBANDS' in results:
        ax = axes.flat[plot_count]
        bb_data = results['BBANDS']
        ax.plot(data.index, data['close'], label='收盘价', color='black', linewidth=1)
        ax.plot(data.index, bb_data['upper'], label='上轨', color='red', linewidth=1.5)
        ax.plot(data.index, bb_data['middle'], label='中轨', color='blue', linewidth=1.5)
        ax.plot(data.index, bb_data['lower'], label='下轨', color='green', linewidth=1.5)
        ax.fill_between(data.index, bb_data['lower'], bb_data['upper'], alpha=0.1, color='blue')
        ax.set_title('布林带 (BBANDS)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_count += 1

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'talib_indicators_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存: {chart_path}")


def generate_report(data, results, output_dir):
    """生成详细报告"""
    print("📝 生成检验报告...")

    report = f"""# TA-Lib 技术指标检验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据概览
- 数据源: 上证指数 (SH000001)
- 时间范围: {data.index.min().date()} 至 {data.index.max().date()}
- 数据条数: {len(data)}

## 检验的技术指标

"""

    for name, result in results.items():
        report += f"### {name}: {result['desc']}\n\n"
        if 'error' in result:
            report += f"❌ 计算失败: {result['error']}\n\n"
        else:
            if name == 'MACD':
                macd = result['macd']
                report += f"- 有效数据点: {macd.dropna().shape[0]}\n"
                report += f"- MACD 均值: {macd.mean():.4f}\n"
                report += f"- Signal 均值: {result['signal'].mean():.4f}\n\n"
            elif name == 'BBANDS':
                upper, middle, lower = result['upper'], result['middle'], result['lower']
                bandwidth = (upper - lower) / middle
                report += f"- 有效数据点: {upper.dropna().shape[0]}\n"
                report += f"- 平均带宽: {bandwidth.mean():.4f}\n\n"
            elif name == 'STOCH':
                slowk = result['slowk']
                report += f"- 有效数据点: {slowk.dropna().shape[0]}\n"
                report += f"- %K 均值: {slowk.mean():.4f}\n"
                report += f"- %D 均值: {result['slowd'].mean():.4f}\n\n"
            else:
                values = result['value']
                report += f"- 有效数据点: {values.dropna().shape[0]}\n"
                report += f"- 均值: {values.mean():.4f}\n"
                report += f"- 标准差: {values.std():.4f}\n"
                report += f"- 最小值: {values.min():.4f}\n"
                report += f"- 最大值: {values.max():.4f}\n\n"

    report += """
## 图表说明
- **价格走势与移动平均线**: 展示收盘价及SMA/EMA趋势
- **RSI指标**: 相对强弱指数，显示超买超卖状态
- **MACD指标**: 趋势-following动量指标
- **布林带**: 价格波动区间分析

## 使用说明
此报告使用 TA-Lib 库计算了10个常见技术指标，用于检验技术分析的有效性。
各指标的具体含义和使用方法请参考技术分析相关文献。

---
*报告由 TA-Lib 技术指标检验程序自动生成*
"""

    report_path = os.path.join(output_dir, 'README.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 报告已保存: {report_path}")


def main():
    """主函数"""
    print("📊 TA-Lib 技术指标检验程序")
    print("=" * 50)

    # 创建输出目录
    output_dir = f"results/talib_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 加载数据
        data = load_market_data()
        data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        data = data.dropna()

        print(f"📈 数据预览:")
        print(data.head())
        print()

        # 2. 计算指标
        results = calculate_indicators(data)

        # 3. 打印统计信息
        print_statistics(results)

        # 4. 创建可视化
        create_visualization(data, results, output_dir)

        # 5. 生成报告
        generate_report(data, results, output_dir)

        print()
        print("🎉 TA-Lib 技术指标检验完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 图表文件: talib_indicators_chart.png")
        print(f"📝 报告文件: README.md")

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())