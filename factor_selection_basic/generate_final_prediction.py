"""生成最终预测结果: 将 Base 排序与事件信号融合得到 final_score / final_weight

输入:
- output/base_scores.csv (date,instrument,base_score)
- output/events.csv      (date,instrument,event_flag=1)
策略融合模式(选择一种):
1. boost: 事件股票得分乘以 (1+alpha)
2. filter: 若当日事件数>=min_events 仅事件股票入选, 不足补齐Base高分
3. extend_hold: 事件股票在后续 N 天保持加权 (延长持仓), 基础得分 + bonus_beta
4. two_layer: Base 得分归一化为 (1-sleeve_ratio), 事件等权分配 sleeve_ratio

输出:
- final_prediction.csv: date,instrument,final_score,final_weight,meta (JSON字符串 描述来源)

运行示例:
python examples/factor_selection_basic/generate_final_prediction.py --mode boost --alpha 0.3 --topk 200 --start 2023-01-01 --end 2024-12-31
"""
from __future__ import annotations
import os
import argparse
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BASE_FILE = os.path.join(OUTPUT_DIR, "base_scores.csv")
EVENT_FILE = os.path.join(OUTPUT_DIR, "events.csv")

# ----------------- 加载 -----------------

def load_data(start: str, end: str):
    base = pd.read_csv(BASE_FILE)
    events = pd.read_csv(EVENT_FILE)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    events["date"] = pd.to_datetime(events["date"]).dt.date
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    base = base[(base["date"] >= s) & (base["date"] <= e)]
    events = events[(events["date"] >= s) & (events["date"] <= e)]
    return base, events

# ----------------- 各模式融合 -----------------

def build_base_rank(base_df: pd.DataFrame, topk: int):
    result = {}
    for d, grp in base_df.groupby("date"):
        sel = grp.sort_values("base_score", ascending=False).head(topk)
        result[d] = sel[["instrument", "base_score"]]
    return result


def mode_boost(base_rank: dict, events_df: pd.DataFrame, alpha: float) -> dict:
    events_map = events_df.groupby("date")["instrument"].apply(set).to_dict()
    out = {}
    for d, df in base_rank.items():
        evt_set = events_map.get(d, set())
        df_local = df.copy()
        df_local["final_score"] = np.where(df_local["instrument"].isin(evt_set), df_local["base_score"] * (1 + alpha), df_local["base_score"])
        # 归一化为权重 (正值得分); 若有负值可平移
        min_val = df_local["final_score"].min()
        if min_val < 0:
            df_local["final_score"] = df_local["final_score"] - min_val
        total = df_local["final_score"].sum()
        if total > 0:
            df_local["final_weight"] = df_local["final_score"] / total
        else:
            df_local["final_weight"] = 0.0
        df_local["meta"] = json.dumps({"mode": "boost", "alpha": alpha, "event_count": len(evt_set)})
        out[d] = df_local[["instrument", "final_score", "final_weight", "meta"]]
    return out


def mode_filter(base_rank: dict, events_df: pd.DataFrame, topk: int, min_events: int) -> dict:
    events_map = events_df.groupby("date")["instrument"].apply(list).to_dict()
    out = {}
    for d, df in base_rank.items():
        evt_list = events_map.get(d, [])
        if len(evt_list) >= min_events:
            sel_evt = df[df["instrument"].isin(evt_list)]
            if sel_evt.shape[0] < topk:
                # 补齐其他股票
                others = df[~df["instrument"].isin(evt_list)].head(topk - sel_evt.shape[0])
                merged = pd.concat([sel_evt, others], ignore_index=True)
            else:
                merged = sel_evt.head(topk)
        else:
            merged = df
        merged = merged.copy()
        merged["final_score"] = merged["base_score"]
        min_val = merged["final_score"].min()
        if min_val < 0:
            merged["final_score"] = merged["final_score"] - min_val
        total = merged["final_score"].sum()
        merged["final_weight"] = merged["final_score"] / total if total > 0 else 0.0
        merged["meta"] = json.dumps({"mode": "filter", "min_events": min_events, "events_today": len(evt_list)})
        out[d] = merged[["instrument", "final_score", "final_weight", "meta"]]
    return out


def mode_extend_hold(base_rank: dict, events_df: pd.DataFrame, hold_days: int, bonus_beta: float, topk: int) -> dict:
    dates = sorted(base_rank.keys())
    events_map = events_df.groupby("date")["instrument"].apply(list).to_dict()
    active = {}  # instrument -> expiry_date
    out = {}
    for idx, d in enumerate(dates):
        # remove expired
        expired = [ins for ins, exp in active.items() if exp < d]
        for ins in expired:
            active.pop(ins, None)
        # add new events
        new_events = events_map.get(d, [])
        for ins in new_events:
            exp_idx = min(idx + hold_days - 1, len(dates) - 1)
            active[ins] = dates[exp_idx]
        base_df = base_rank[d].copy()
        # 加入延长持仓股票 (可能不在TopK中, 用最近base得分替换?) 简化: 若不在TopK则忽略
        active_stocks = set(active.keys())
        base_df["final_score"] = base_df["base_score"]
        if bonus_beta > 0:
            base_df["final_score"] = np.where(base_df["instrument"].isin(active_stocks), base_df["final_score"] * (1 + bonus_beta), base_df["final_score"])
        # 归一化
        min_val = base_df["final_score"].min()
        if min_val < 0:
            base_df["final_score"] = base_df["final_score"] - min_val
        total = base_df["final_score"].sum()
        base_df["final_weight"] = base_df["final_score"] / total if total > 0 else 0.0
        base_df["meta"] = json.dumps({"mode": "extend_hold", "hold_days": hold_days, "bonus_beta": bonus_beta, "active_events": len(active_stocks)})
        out[d] = base_df[["instrument", "final_score", "final_weight", "meta"]]
    return out


def mode_two_layer(base_rank: dict, events_df: pd.DataFrame, sleeve_ratio: float) -> dict:
    events_map = events_df.groupby("date")["instrument"].apply(list).to_dict()
    out = {}
    for d, df in base_rank.items():
        evt_list = events_map.get(d, [])
        df_local = df.copy()
        # Base层分配 (1 - sleeve_ratio) 按得分归一化
        min_val = df_local["base_score"].min()
        score = df_local["base_score"] - min_val if min_val < 0 else df_local["base_score"]
        total = score.sum()
        base_weight = (score / total * (1 - sleeve_ratio)) if total > 0 else 0.0
        # 事件 sleeve 等权
        if evt_list and sleeve_ratio > 0:
            w_evt = sleeve_ratio / len(evt_list)
            evt_df = pd.DataFrame({"instrument": evt_list, "evt_weight": w_evt})
            merged = df_local.merge(evt_df, on="instrument", how="left")
            merged["evt_weight"].fillna(0.0, inplace=True)
            merged["final_weight"] = base_weight + merged["evt_weight"]
        else:
            merged = df_local.copy()
            merged["final_weight"] = base_weight
        # final_score 用 final_weight(可乘以规模因子)
        merged["final_score"] = merged["final_weight"]
        merged["meta"] = json.dumps({"mode": "two_layer", "sleeve_ratio": sleeve_ratio, "event_count": len(evt_list)})
        out[d] = merged[["instrument", "final_score", "final_weight", "meta"]]
    return out

# ----------------- 主入口 -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["boost", "filter", "extend_hold", "two_layer"], required=True)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    # 模式参数
    parser.add_argument("--alpha", type=float, default=0.3, help="boost 模式提升系数")
    parser.add_argument("--min_events", type=int, default=5, help="filter 模式阈值")
    parser.add_argument("--hold_days", type=int, default=3, help="extend_hold 模式延长天数")
    parser.add_argument("--bonus_beta", type=float, default=0.5, help="extend_hold 模式加权系数")
    parser.add_argument("--sleeve_ratio", type=float, default=0.2, help="two_layer 模式事件层资金占比")
    args = parser.parse_args()

    base_df, events_df = load_data(args.start, args.end)
    base_rank = build_base_rank(base_df, args.topk)

    if args.mode == "boost":
        fused = mode_boost(base_rank, events_df, args.alpha)
    elif args.mode == "filter":
        fused = mode_filter(base_rank, events_df, args.topk, args.min_events)
    elif args.mode == "extend_hold":
        fused = mode_extend_hold(base_rank, events_df, args.hold_days, args.bonus_beta, args.topk)
    elif args.mode == "two_layer":
        fused = mode_two_layer(base_rank, events_df, args.sleeve_ratio)
    else:
        raise ValueError("Unsupported mode")

    rows = []
    for d, df in fused.items():
        for _, row in df.iterrows():
            rows.append({
                "date": d,
                "instrument": row["instrument"],
                "final_score": row["final_score"],
                "final_weight": row["final_weight"],
                "meta": row["meta"],
            })
    out_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "final_prediction.csv")
    out_df.to_csv(path, index=False)
    print(f"Saved final prediction -> {path}")

if __name__ == "__main__":
    main()
