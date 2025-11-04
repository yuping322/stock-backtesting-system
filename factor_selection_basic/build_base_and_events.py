"""构建 Base 动量分数与稀疏事件信号示例脚本

功能: 
1. 初始化 qlib 数据环境 (需要用户已准备好 ~/.qlib/qlib_data/cn_data)
2. 选定股票池 (例如: csi300) 与时间区间
3. 计算 Base 分数: 20 日动量 (简化实现)
4. 构造稀疏事件信号: 成交额/成交量相对过去均值激增 且 当日涨幅不超过阈值, 生成事件标记
5. 输出两个文件:
   - base_scores.csv: columns=[date,instrument,base_score]
   - events.csv: columns=[date,instrument,event_flag]

注意:
- 这是最小示例, 未做严格的市场数据质量过滤与异常检测。
- 事件逻辑可根据实际需要替换。
- 输出为 csv 方便后续快速查看; 可改为 parquet。

运行示例:
python examples/factor_selection_basic/build_base_and_events.py 

后续: 使用 run_event_backtest.py 进行回测与增量分析。
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import List

import qlib
from qlib.data import D
from qlib.constant import REG_CN

# -------------------- 可调参数 --------------------
MOUNT_PATH = os.path.expanduser("~/.qlib/qlib_data/cn_data")  # 数据路径
UNIVERSE = "csi300"  # 股票池名称 (qlib 内置)
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
MOMENTUM_WINDOW = 20
EVENT_VOLUME_SHORT = 5  # 短期窗口
EVENT_VOLUME_LONG = 20  # 长期窗口
VOLUME_SPIKE_RATIO = 2.5  # 成交量激增阈值 (短期均值 / 长期均值 > 该值)
DAILY_RETURN_CAP = 0.03  # 当日涨幅不超过该值 (避免追涨一字板)
MIN_EVENT_STOCKS = 3  # 若当天事件过少 可选是否放宽 (此处仅统计不强制)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# -------------------- 帮助函数 --------------------
def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def init_qlib():
    if not qlib.is_initialized():
        qlib.init(mount_path=MOUNT_PATH, region=REG_CN, expression_cache=True, dataset_cache=True)


def get_universe_list() -> List[str]:
    insts = D.instruments(UNIVERSE)
    return D.list_instruments(insts, start_time=START_DATE, end_time=END_DATE, as_list=True)


def load_feature(instruments: List[str], fields: List[str]) -> pd.DataFrame:
    df = D.features(instruments, fields, start_time=START_DATE, end_time=END_DATE, freq="day")
    # MultiIndex (date, instrument) -> columns
    df = df.reset_index().rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_base_momentum(raw_df: pd.DataFrame) -> pd.DataFrame:
    # 需要 close 与其滞后
    pivot_close = raw_df.pivot(index="date", columns="instrument", values="$close")
    momentum = pivot_close / pivot_close.shift(MOMENTUM_WINDOW) - 1.0
    momentum = momentum.stack().rename("base_score").reset_index()
    return momentum.dropna(subset=["base_score"])  # 去掉前期缺失


def compute_events(raw_df: pd.DataFrame) -> pd.DataFrame:
    pivot_vol = raw_df.pivot(index="date", columns="instrument", values="$volume")
    pivot_close = raw_df.pivot(index="date", columns="instrument", values="$close")
    # 简单日收益率 (收盘对前收)
    daily_ret = pivot_close.pct_change()

    short_ma = pivot_vol.rolling(EVENT_VOLUME_SHORT).mean()
    long_ma = pivot_vol.rolling(EVENT_VOLUME_LONG).mean()
    spike_ratio = short_ma / long_ma

    # 事件条件: 成交量激增 + 涨幅不超过阈值 + 有效数据
    cond = (spike_ratio > VOLUME_SPIKE_RATIO) & (daily_ret < DAILY_RETURN_CAP) & (daily_ret.notna())

    events = cond.stack().rename("event_flag").reset_index()
    events = events[events["event_flag"]]  # 仅保留 True
    events = events.drop(columns=["event_flag"])  # 标记为存在事件
    events["event_flag"] = 1
    return events


def main():
    ensure_output()
    init_qlib()

    instruments = get_universe_list()
    # 基础字段: 收盘价、成交量
    fields = ["$close", "$volume"]
    raw_df = load_feature(instruments, fields)

    # Base 动量分数
    base_scores = compute_base_momentum(raw_df)

    # 稀疏事件 (成交量激增 + 涨幅不过度)
    events = compute_events(raw_df)

    # 保存
    base_path = os.path.join(OUTPUT_DIR, "base_scores.csv")
    event_path = os.path.join(OUTPUT_DIR, "events.csv")
    base_scores.to_csv(base_path, index=False)
    events.to_csv(event_path, index=False)

    # 简要统计
    event_counts = events.groupby("date")["instrument"].count()
    summary = {
        "dates": int(event_counts.shape[0]),
        "total_events": int(events.shape[0]),
        "avg_events_per_day": float(event_counts.mean()) if not event_counts.empty else 0.0,
        "momentum_stock_days": int(base_scores.shape[0]),
    }
    print("[Summary]", summary)
    print(f"Saved base scores -> {base_path}")
    print(f"Saved events -> {event_path}")


if __name__ == "__main__":
    main()
