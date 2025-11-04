"""验证稀疏事件信号的数据有效性与增量提升

输入:
- output/base_scores.csv  (date,instrument,base_score)
- output/events.csv       (date,instrument,event_flag=1)
依赖: qlib 日线数据 (收盘价)

验证逻辑分两部分:
1. 数据可用性 (Data Usability)
   - 覆盖度: 有事件的交易日比例 / 总交易日; 当日事件股票占Universe比例
   - 稳定性: 每日事件数量的均值/标准差/变异系数
   - 稀疏性: 事件股票在全时期去重计数 / 总股票数; 单股票事件次数分布(均值/中位数/最大值)
   - 延续性: 事件后1~N日累计收益均值; 命中率 (事件后第1日正收益比率)
   - 随机对照差异: 与随机抽样对照窗口收益差异及t检验
2. 增量对 Base 的提升 (Incremental Lift)
   - Base TopK 等权组合每日收益与事件过滤/加权策略的差异 (在已有回测中, 此处简化只比较事件集合 vs Base)
   - 计算: 日度差值序列 -> 年化增量收益、信息比率(IR = mean(diff)/std(diff)*sqrt(252))

输出:
- validation_report.json
- validation_daily_diff.csv (可选)

运行示例:
python examples/factor_selection_basic/validate_event_signal.py --topk 50 --start 2023-01-01 --end 2024-12-31 --control_seed 42
"""
from __future__ import annotations
import os
import argparse
import json
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.constant import REG_CN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BASE_FILE = os.path.join(OUTPUT_DIR, "base_scores.csv")
EVENT_FILE = os.path.join(OUTPUT_DIR, "events.csv")

DEFAULT_MOUNT = os.path.expanduser("~/.qlib/qlib_data/cn_data")

# ----------------- 统计函数 -----------------

def annualize_return(daily_ret: pd.Series) -> float:
    return daily_ret.mean() * 252

def annualize_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std() * np.sqrt(252)

def info_ratio(diff_ret: pd.Series) -> float:
    vol = diff_ret.std()
    return diff_ret.mean() / vol * np.sqrt(252) if vol != 0 else np.nan

# ----------------- 核心流程 -----------------

def init_qlib(mount_path: str):
    if not qlib.is_initialized():
        qlib.init(mount_path=mount_path, region=REG_CN)


def load_base_events(start: str, end: str):
    base = pd.read_csv(BASE_FILE)
    events = pd.read_csv(EVENT_FILE)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    events["date"] = pd.to_datetime(events["date"]).dt.date
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    base = base[(base["date"] >= start_d) & (base["date"] <= end_d)]
    events = events[(events["date"] >= start_d) & (events["date"] <= end_d)]
    return base, events


def load_prices(instruments, start, end):
    df = D.features(instruments, ["$close"], start_time=start, end_time=end, freq="day")
    df = df.reset_index().rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    pivot = df.pivot(index="date", columns="instrument", values="$close")
    return pivot


def forward_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.shift(-1) / price_df - 1.0


def compute_event_usability(events_df: pd.DataFrame, universe: list, fwd_ret_df: pd.DataFrame, windows=(1,3,5), control_seed: int = 42):
    dates = sorted(fwd_ret_df.index.tolist())
    event_days = sorted(events_df["date"].unique().tolist())
    coverage_day_ratio = len(event_days) / len(dates) if dates else 0
    daily_evt_counts = events_df.groupby("date")["instrument"].count()
    mean_evt = float(daily_evt_counts.mean()) if not daily_evt_counts.empty else 0.0
    std_evt = float(daily_evt_counts.std()) if not daily_evt_counts.empty else 0.0
    cv_evt = std_evt / mean_evt if mean_evt > 0 else 0.0

    unique_event_stocks = events_df["instrument"].unique().tolist()
    sparse_ratio = len(unique_event_stocks) / len(universe) if universe else 0
    per_stock_counts = events_df.groupby("instrument")["date"].count()
    per_stock_stats = {
        "mean": float(per_stock_counts.mean()) if not per_stock_counts.empty else 0.0,
        "median": float(per_stock_counts.median()) if not per_stock_counts.empty else 0.0,
        "max": int(per_stock_counts.max()) if not per_stock_counts.empty else 0,
    }

    # 窗口收益与命中率
    window_stats = {}
    for w in windows:
        cum_list = []
        hits = 0
        total = 0
        for d, group in events_df.groupby("date"):
            if d not in fwd_ret_df.index:
                continue
            idx = dates.index(d)
            seq_dates = dates[idx: idx + w]
            if len(seq_dates) < w:
                continue
            insts = group["instrument"].tolist()
            for ins in insts:
                series = fwd_ret_df.loc[seq_dates, ins]
                cum_ret = (series + 1).prod() - 1
                cum_list.append(cum_ret)
                if series.iloc[0] > 0:
                    hits += 1
                total += 1
        window_stats[f"window_{w}_avg_cum_ret"] = float(np.mean(cum_list)) if cum_list else 0.0
        window_stats[f"window_{w}_hit_ratio_day1"] = hits / total if total > 0 else 0.0

    # 随机对照: 保持每日日事件股票数量, 随机抽样 (未行业匹配)
    rng = np.random.default_rng(control_seed)
    control_stats = {}
    for w in windows:
        cum_list = []
        for d, group in events_df.groupby("date"):
            k = len(group)
            if d not in fwd_ret_df.index:
                continue
            candidates = [ins for ins in universe if ins not in group["instrument"].tolist()]
            if len(candidates) < k:
                continue
            idx = dates.index(d)
            seq_dates = dates[idx: idx + w]
            if len(seq_dates) < w:
                continue
            sample = rng.choice(candidates, size=k, replace=False)
            for ins in sample:
                series = fwd_ret_df.loc[seq_dates, ins]
                cum_ret = (series + 1).prod() - 1
                cum_list.append(cum_ret)
        control_stats[f"control_window_{w}_avg_cum_ret"] = float(np.mean(cum_list)) if cum_list else 0.0

    # t检验 (简单近似，使用事件与对照累计收益样本)
    from math import sqrt
    t_stats = {}
    for w in windows:
        # 重建样本列表用于 t 值估计
        event_samples = []
        control_samples = []
        for d, group in events_df.groupby("date"):
            if d not in fwd_ret_df.index:
                continue
            idx = dates.index(d)
            seq_dates = dates[idx: idx + w]
            if len(seq_dates) < w:
                continue
            insts = group["instrument"].tolist()
            for ins in insts:
                series = fwd_ret_df.loc[seq_dates, ins]
                event_samples.append((series + 1).prod() - 1)
            k = len(insts)
            candidates = [ins for ins in universe if ins not in insts]
            if len(candidates) >= k:
                sample = rng.choice(candidates, size=k, replace=False)
                for ins in sample:
                    series = fwd_ret_df.loc[seq_dates, ins]
                    control_samples.append((series + 1).prod() - 1)
        # 计算 t
        if event_samples and control_samples:
            e_arr = np.array(event_samples)
            c_arr = np.array(control_samples)
            diff_mean = e_arr.mean() - c_arr.mean()
            se = sqrt(e_arr.var(ddof=1)/len(e_arr) + c_arr.var(ddof=1)/len(c_arr))
            t_val = diff_mean / se if se > 0 else np.nan
        else:
            t_val = np.nan
        t_stats[f"t_value_window_{w}"] = float(t_val) if not np.isnan(t_val) else None

    usability = {
        "coverage_day_ratio": coverage_day_ratio,
        "avg_event_count_per_day": mean_evt,
        "std_event_count_per_day": std_evt,
        "cv_event_count": cv_evt,
        "sparse_stock_ratio": sparse_ratio,
        "per_stock_event_distribution": per_stock_stats,
        "window_stats": window_stats,
        "control_stats": control_stats,
        "t_test_stats": t_stats,
    }
    return usability


def build_base_portfolio(base_df: pd.DataFrame, topk: int) -> dict:
    portfolios = {}
    for d, grp in base_df.groupby("date"):
        sel = grp.sort_values("base_score", ascending=False).head(topk)
        w = 1.0 / sel.shape[0] if sel.shape[0] > 0 else 0
        portfolios[d] = pd.DataFrame({"instrument": sel["instrument"], "weight": w})
    return portfolios


def build_event_portfolio(events_df: pd.DataFrame) -> dict:
    portfolios = {}
    for d, grp in events_df.groupby("date"):
        insts = grp["instrument"].tolist()
        if not insts:
            portfolios[d] = pd.DataFrame({"instrument": [], "weight": []})
            continue
        w = 1.0 / len(insts)
        portfolios[d] = pd.DataFrame({"instrument": insts, "weight": w})
    return portfolios


def simulate(portfolios: dict, fwd_ret_df: pd.DataFrame) -> pd.Series:
    daily_ret = []
    for d, wdf in portfolios.items():
        if wdf.empty or d not in fwd_ret_df.index:
            daily_ret.append((d, 0.0))
            continue
        rets = fwd_ret_df.loc[d]
        merged = wdf.merge(rets.rename("r").reset_index(), on="instrument", how="left")
        merged["r"].fillna(0.0, inplace=True)
        daily_ret.append((d, (merged["weight"] * merged["r"]).sum()))
    return pd.Series({d: r for d, r in daily_ret}).sort_index()


def incremental_lift(base_ret: pd.Series, event_ret: pd.Series):
    # 对齐日期; 事件为空日收益设0
    idx = base_ret.index
    evt_aligned = event_ret.reindex(idx).fillna(0.0)
    diff = evt_aligned - base_ret  # 简化: 直接比较事件组合与Base组合
    return {
        "annual_return_event": annualize_return(evt_aligned),
        "annual_return_base": annualize_return(base_ret),
        "annual_return_diff": annualize_return(diff),
        "info_ratio_diff": info_ratio(diff),
        "mean_daily_diff": float(diff.mean()),
        "std_daily_diff": float(diff.std()),
        "days": int(diff.shape[0]),
    }, diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount", default=DEFAULT_MOUNT)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--control_seed", type=int, default=42)
    args = parser.parse_args()

    init_qlib(args.mount)
    base_df, events_df = load_base_events(args.start, args.end)
    universe = sorted(base_df["instrument"].unique().tolist())
    price_df = load_prices(universe, args.start, args.end)
    fwd_ret_df = forward_returns(price_df)

    usability = compute_event_usability(events_df, universe, fwd_ret_df, control_seed=args.control_seed)

    # 增量提升: Base vs 事件组合
    base_port = build_base_portfolio(base_df, args.topk)
    event_port = build_event_portfolio(events_df)
    base_ret = simulate(base_port, fwd_ret_df)
    event_ret = simulate(event_port, fwd_ret_df)
    lift_stats, diff_ser = incremental_lift(base_ret, event_ret)

    report = {
        "params": {
            "topk": args.topk,
            "start": args.start,
            "end": args.end,
            "control_seed": args.control_seed,
        },
        "usability": usability,
        "incremental_lift": lift_stats,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    diff_path = os.path.join(OUTPUT_DIR, "validation_daily_diff.csv")
    pd.DataFrame({"date": diff_ser.index, "diff_event_minus_base": diff_ser.values}).to_csv(diff_path, index=False)

    print(f"Saved validation report -> {json_path}")
    print(f"Saved daily diff -> {diff_path}")


if __name__ == "__main__":
    main()
