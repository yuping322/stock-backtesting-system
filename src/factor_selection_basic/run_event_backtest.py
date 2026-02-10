"""事件信号与 Base 选股增强回测示例

读取 build_base_and_events.py 生成的:
- base_scores.csv: date,instrument,base_score
- events.csv: date,instrument,event_flag(=1)

构造多类组合:
1. Base 等权 TopK (按 base_score 排序)
2. 过滤融合(Filter): Base TopK 中若当日事件触发则保留其权重, 非触发仍保留; *演示增量效果计算*
   (可改为仅保留事件股票, 不足补齐 Base 其它高分)
3. 加权提升(Boost): 事件触发股票权重提升 (乘以 1+ALPHA), 然后重新归一化
4. 事件独立 sleeve (仅事件集合等权)  -- 用于对比
5. 两层组合(Two-layer): Base 与事件 sleeve 资金分配 (参数 sleeve_ratio)
6. 延长持仓(ExtendHold): 事件触发股票持有 event_hold_days 天 (合并到每日 Base 选股集合)
7. 延长持仓加仓(ExtendHoldBoost): 延长周期内事件股票权重乘 (1+extend_boost_beta)

指标:
- 每日组合收益 (forward return: next close / today close - 1)
- 年化收益、年化波动、Sharpe、最大回撤（含成本前/后）
- 平均换手率 (turnover) 与交易成本影响 (成本 = turnover * cost_bps)
- 事件后 N 日窗口平均累计收益与命中率
- 对照组随机匹配窗口收益 (diff 与显著性粗略对比)

依赖: qlib 数据 (收盘价、开盘价)

运行示例:
python examples/factor_selection_basic/run_event_backtest.py \
    --topk 50 --alpha 0.3 --sleeve_ratio 0.2 \
    --event_hold_days 3 --extend_boost_beta 0.5 \
    --filter_min_events 5 --cost_bps 5 --start 2023-01-01 --end 2024-12-31

注意:
- 交易成本为简单线性模型: 当日净换手 * cost_bps。
- 未考虑滑点及冲击成本，可后续扩展。
- 对照组为随机行业/市值未匹配的简化版本，仅作参考。
"""
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.constant import REG_CN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BASE_FILE = os.path.join(OUTPUT_DIR, "base_scores.csv")
EVENT_FILE = os.path.join(OUTPUT_DIR, "events.csv")

DEFAULT_MOUNT = os.path.expanduser("~/.qlib/qlib_data/cn_data")

@dataclass
class PortfolioResult:
    daily_return: pd.Series  # index=date
    weights: dict  # date -> DataFrame(instrument, weight)

# ----------------- 工具函数 -----------------

def init_qlib(mount_path: str):
    if not qlib.is_initialized():
        qlib.init(mount_path=mount_path, region=REG_CN)


def load_prices(instruments, start, end):
    df = D.features(instruments, ["$close"], start_time=start, end_time=end, freq="day")
    df = df.reset_index().rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    pivot = df.pivot(index="date", columns="instrument", values="$close")
    return pivot


def compute_simple_forward_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    # forward return: next_day_close / today_close - 1
    return price_df.shift(-1) / price_df - 1.0


def annualize_return(daily_ret: pd.Series) -> float:
    return daily_ret.mean() * 252


def annualize_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std() * np.sqrt(252)


def sharpe(daily_ret: pd.Series, rf: float = 0.0) -> float:
    ar = annualize_return(daily_ret) - rf
    av = annualize_vol(daily_ret)
    return ar / av if av != 0 else np.nan


def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()


def load_base_and_events(start: str, end: str):
    base = pd.read_csv(BASE_FILE)
    events = pd.read_csv(EVENT_FILE)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    events["date"] = pd.to_datetime(events["date"]).dt.date
    base = base[(base["date"] >= pd.to_datetime(start).date()) & (base["date"] <= pd.to_datetime(end).date())]
    events = events[(events["date"] >= pd.to_datetime(start).date()) & (events["date"] <= pd.to_datetime(end).date())]
    return base, events

# ----------------- 组合构建 -----------------

def build_base_portfolio(base_df: pd.DataFrame, topk: int) -> dict:
    portfolios = {}
    for d, group in base_df.groupby("date"):
        sel = group.sort_values("base_score", ascending=False).head(topk)
        w = 1.0 / sel.shape[0] if sel.shape[0] > 0 else 0
        portfolios[d] = pd.DataFrame({"instrument": sel["instrument"], "weight": w})
    return portfolios


def apply_filter_strategy(base_portfolios: dict, events_df: pd.DataFrame, topk: int, min_events: int) -> dict:
    """过滤策略: 当日事件数>=min_events时优先事件股票, 不足补齐Base其余高分; 否则使用原Base组合"""
    events_map = events_df.groupby("date")["instrument"].apply(list).to_dict()
    result = {}
    for d, df_w in base_portfolios.items():
        evt_list = events_map.get(d, [])
        if len(evt_list) >= min_events:
            # 仅事件, 补齐逻辑
            sel = evt_list
            if len(sel) < topk:
                # 补齐 base 里非事件股票
                base_extra = [ins for ins in df_w["instrument"].tolist() if ins not in sel]
                need = topk - len(sel)
                sel += base_extra[:need]
        else:
            sel = df_w["instrument"].tolist()
        w = 1.0 / len(sel) if sel else 0
        result[d] = pd.DataFrame({"instrument": sel, "weight": w})
    return result


def apply_boost_strategy(base_portfolios: dict, events_df: pd.DataFrame, alpha: float) -> dict:
    events_map = events_df.groupby("date")["instrument"].apply(set).to_dict()
    result = {}
    for d, df_w in base_portfolios.items():
        evt_set = events_map.get(d, set())
        df_local = df_w.copy()
        # 提升事件权重
        df_local["raw_weight"] = df_local["weight"]
        df_local["boost_weight"] = np.where(df_local["instrument"].isin(evt_set), df_local["raw_weight"] * (1 + alpha), df_local["raw_weight"])
        # 归一化
        total = df_local["boost_weight"].sum()
        if total > 0:
            df_local["weight"] = df_local["boost_weight"] / total
        else:
            df_local["weight"] = df_local["boost_weight"]
        result[d] = df_local[["instrument", "weight"]]
    return result


def build_event_sleeve(events_df: pd.DataFrame) -> dict:
    portfolios = {}
    for d, group in events_df.groupby("date"):
        insts = group["instrument"].tolist()
        if not insts:
            portfolios[d] = pd.DataFrame({"instrument": [], "weight": []})
            continue
        w = 1.0 / len(insts)
        portfolios[d] = pd.DataFrame({"instrument": insts, "weight": w})
    return portfolios

# ----------------- 回测执行 -----------------

def simulate(portfolios: dict, forward_ret_df: pd.DataFrame) -> PortfolioResult:
    daily_ret = []
    for d, wdf in portfolios.items():
        if wdf.empty:
            daily_ret.append((d, 0.0))
            continue
        # 使用forward_ret_df的当日收益(对应下一日的实际变化) -> 简化示例
        if d not in forward_ret_df.index:
            daily_ret.append((d, 0.0))
            continue
        rets = forward_ret_df.loc[d]
        merged = wdf.merge(rets.rename("r").reset_index(), on="instrument", how="left")
        merged["r"].fillna(0.0, inplace=True)
        port_ret = (merged["weight"] * merged["r"]).sum()
        daily_ret.append((d, port_ret))
    ser = pd.Series({d: r for d, r in daily_ret})
    return PortfolioResult(daily_return=ser.sort_index(), weights=portfolios)

# ----------------- 事件后窗口分析 -----------------

def event_window_stats(events_df: pd.DataFrame, forward_ret_df: pd.DataFrame, windows=(1,3,5)):
    stats = {}
    # 转换 events 为按日期列出股票
    for w in windows:
        rs = []
        hit = 0
        total = 0
        for d, group in events_df.groupby("date"):
            # 累计窗口收益: 从 d 开始的 w 日 forward_ret (近似)
            # 简化: 使用 forward_ret_df 的第 d 起连续 w 日的收益链条累乘 - 1
            dates_seq = forward_ret_df.index
            if d not in dates_seq:
                continue
            idx = list(dates_seq).index(d)
            seq_dates = dates_seq[idx: idx + w]
            if len(seq_dates) < w:
                continue
            insts = group["instrument"].tolist()
            # 对每个股票累积收益
            cum_list = []
            for ins in insts:
                series = forward_ret_df.loc[seq_dates, ins]
                cum_ret = (series + 1.0).prod() - 1.0
                cum_list.append(cum_ret)
                if series.iloc[0] > 0:  # 第1日正收益作为命中
                    hit += 1
                total += 1
            if cum_list:
                rs.append(np.mean(cum_list))
        stats[f"window_{w}_avg_cum_ret"] = float(np.mean(rs)) if rs else 0.0
        stats[f"window_{w}_hit_ratio_day1"] = hit / total if total > 0 else 0.0
    return stats

# ----------------- 主流程 -----------------

def build_two_layer_portfolio(base_portfolios: dict, event_sleeve: dict, sleeve_ratio: float) -> dict:
    """将 Base 与事件 sleeve 按资金比例合成。
    若当日无事件 sleeve, 则使用 Base 原组合。
    权重归一化: base 权重 * (1 - sleeve_ratio), 事件权重 * sleeve_ratio。
    """
    result = {}
    for d in base_portfolios.keys():
        base_df = base_portfolios[d]
        evt_df = event_sleeve.get(d, pd.DataFrame({"instrument": [], "weight": []}))
        if evt_df.empty or sleeve_ratio <= 0:
            # 直接缩放 base (实际不需要缩放, 保持为满仓)
            result[d] = base_df.copy()
            continue
        # 缩放权重
        base_scaled = base_df.copy()
        base_scaled["weight"] = base_scaled["weight"] * (1 - sleeve_ratio)
        evt_scaled = evt_df.copy()
        evt_scaled["weight"] = evt_scaled["weight"] * sleeve_ratio
        merged = pd.concat([base_scaled, evt_scaled], ignore_index=True)
        # 若有重复股票(可能事件股票也在Base中), 合并权重
        merged = merged.groupby("instrument", as_index=False)["weight"].sum()
        # 归一化(避免计算误差导致总和!=1)
        total = merged["weight"].sum()
        if total > 0:
            merged["weight"] = merged["weight"] / total
        result[d] = merged
    return result


def build_extended_hold_portfolio(base_portfolios: dict, events_df: pd.DataFrame, event_hold_days: int, extend_boost_beta: float = 0.0) -> dict:
    """事件延长持仓+加仓模式:
    - 延长持仓: 事件股票在后续 event_hold_days 天保持在组合中
    - 加仓: 延长周期或当日内事件股票权重乘 (1+extend_boost_beta)
    简化: 全部股票等权基础上再加权, 然后归一化
    """
    if event_hold_days <= 1:
        if extend_boost_beta > 0:
            boosted = {}
            events_map = events_df.groupby("date")["instrument"].apply(set).to_dict()
            for d, df in base_portfolios.items():
                evt_set = events_map.get(d, set())
                df_local = df.copy()
                df_local["raw_weight"] = df_local["weight"]
                df_local["weight"] = np.where(df_local["instrument"].isin(evt_set), df_local["raw_weight"] * (1 + extend_boost_beta), df_local["raw_weight"])
                tot = df_local["weight"].sum()
                if tot > 0:
                    df_local["weight"] = df_local["weight"] / tot
                boosted[d] = df_local[["instrument", "weight"]]
            return boosted
        return {d: df.copy() for d, df in base_portfolios.items()}
    # 构建日期序列
    dates = sorted(base_portfolios.keys())
    events_map = events_df.groupby("date")["instrument"].apply(list).to_dict()
    active_events = {}  # instrument -> expiry_date
    result = {}
    for idx, d in enumerate(dates):
        # 移除过期事件
        to_remove = [ins for ins, exp in active_events.items() if exp < d]
        for ins in to_remove:
            active_events.pop(ins, None)
        # 新事件添加
        new_events = events_map.get(d, [])
        # 计算到期日期: 在日期列表中向后 event_hold_days-1 天的日期
        for ins in new_events:
            # 找到当前索引 + event_hold_days -1 对应日期
            exp_idx = min(idx + event_hold_days - 1, len(dates) - 1)
            expiry_date = dates[exp_idx]
            active_events[ins] = expiry_date
        # 当前组合股票集合
        base_df = base_portfolios[d]
        base_stocks = set(base_df["instrument"].tolist())
        event_stocks = set(active_events.keys())
        all_stocks = sorted(base_stocks.union(event_stocks))
        if not all_stocks:
            result[d] = pd.DataFrame({"instrument": [], "weight": []})
            continue
        df_weights = pd.DataFrame({"instrument": all_stocks})
        df_weights["weight"] = 1.0
        if extend_boost_beta > 0 and event_stocks:
            df_weights["weight"] = np.where(df_weights["instrument"].isin(event_stocks), df_weights["weight"] * (1 + extend_boost_beta), df_weights["weight"])
        tot = df_weights["weight"].sum()
        if tot > 0:
            df_weights["weight"] = df_weights["weight"] / tot
        result[d] = df_weights
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount", default=DEFAULT_MOUNT)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.3, help="加权提升系数")
    parser.add_argument("--sleeve_ratio", type=float, default=0.2, help="事件 sleeve 资金占比 (0~0.5 建议)")
    parser.add_argument("--event_hold_days", type=int, default=1, help="事件股票延长持仓天数 (>=1)")
    parser.add_argument("--extend_boost_beta", type=float, default=0.0, help="延长持仓加仓系数 beta")
    parser.add_argument("--filter_min_events", type=int, default=5, help="过滤策略事件数阈值")
    parser.add_argument("--cost_bps", type=float, default=0.0, help="单边交易成本(bps) 应用于换手")
    parser.add_argument("--control_seed", type=int, default=42, help="对照组随机种子")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    init_qlib(args.mount)

    base_df, events_df = load_base_and_events(args.start, args.end)
    instruments = base_df["instrument"].unique().tolist()
    price_df = load_prices(instruments, args.start, args.end)
    forward_ret_df = compute_simple_forward_returns(price_df)

    # 构造各类组合
    base_port = build_base_portfolio(base_df, args.topk)
    filter_port = apply_filter_strategy(base_port, events_df, args.topk, args.filter_min_events)
    boost_port = apply_boost_strategy(base_port, events_df, args.alpha)
    event_sleeve = build_event_sleeve(events_df)
    two_layer_port = build_two_layer_portfolio(base_port, event_sleeve, args.sleeve_ratio)
    extend_hold_port = build_extended_hold_portfolio(base_port, events_df, args.event_hold_days, args.extend_boost_beta)

    # 回测
    res_base = simulate(base_port, forward_ret_df)
    res_filter = simulate(filter_port, forward_ret_df)
    res_boost = simulate(boost_port, forward_ret_df)
    res_event = simulate(event_sleeve, forward_ret_df)
    res_two_layer = simulate(two_layer_port, forward_ret_df)
    res_extend_hold = simulate(extend_hold_port, forward_ret_df)

    # 指标
    def compute_turnover(portfolios: dict) -> pd.Series:
        prev = None
        records = []
        for d in sorted(portfolios.keys()):
            df = portfolios[d]
            if prev is None:
                records.append((d, 0.0))
                prev = df
                continue
            merged = prev.merge(df, on="instrument", how="outer", suffixes=("_prev", "_cur"))
            merged["weight_prev"].fillna(0.0, inplace=True)
            merged["weight_cur"].fillna(0.0, inplace=True)
            t = (merged["weight_prev"] - merged["weight_cur"]).abs().sum() / 2.0
            records.append((d, t))
            prev = df
        return pd.Series({d: v for d, v in records})

    def summarize(name, ser, turnover_ser, cost_bps):
        cum = (ser + 1).cumprod()
        avg_turnover = turnover_ser.mean() if not turnover_ser.empty else 0.0
        daily_cost = turnover_ser * (cost_bps / 10000.0)
        ser_net = ser - daily_cost
        cum_net = (ser_net + 1).cumprod()
        return {
            "portfolio": name,
            "annual_return": annualize_return(ser),
            "annual_vol": annualize_vol(ser),
            "sharpe": sharpe(ser),
            "max_drawdown": max_drawdown(cum),
            "annual_return_net": annualize_return(ser_net),
            "sharpe_net": sharpe(ser_net),
            "max_drawdown_net": max_drawdown(cum_net),
            "avg_turnover": avg_turnover,
            "cost_bps": cost_bps,
            "days": int(ser.shape[0]),
        }

    turnover_map = {
        "base": compute_turnover(base_port),
        "filter": compute_turnover(filter_port),
        "boost": compute_turnover(boost_port),
        "event": compute_turnover(event_sleeve),
        "two_layer": compute_turnover(two_layer_port),
        "extend_hold": compute_turnover(extend_hold_port),
    }
    reports = [
        summarize("base", res_base.daily_return, turnover_map["base"], args.cost_bps),
        summarize("filter", res_filter.daily_return, turnover_map["filter"], args.cost_bps),
        summarize("boost", res_boost.daily_return, turnover_map["boost"], args.cost_bps),
        summarize("event", res_event.daily_return, turnover_map["event"], args.cost_bps),
        summarize("two_layer", res_two_layer.daily_return, turnover_map["two_layer"], args.cost_bps),
        summarize("extend_hold", res_extend_hold.daily_return, turnover_map["extend_hold"], args.cost_bps),
    ]

    # 事件窗口统计
    evt_stats = event_window_stats(events_df, forward_ret_df, windows=(1,3,5))
    # 对照组随机匹配(简化): 不匹配行业/市值，仅排除事件股票
    rng = np.random.default_rng(args.control_seed)
    unique_insts = base_df["instrument"].unique().tolist()
    control_stats = {}
    for w in (1,3,5):
        cum_list = []
        for d, group in events_df.groupby("date"):
            k = len(group)
            if k == 0:
                continue
            candidates = [ins for ins in unique_insts if ins not in group["instrument"].tolist()]
            if len(candidates) < k:
                continue
            sample = rng.choice(candidates, size=k, replace=False)
            dates_seq = forward_ret_df.index
            if d not in dates_seq:
                continue
            idx = list(dates_seq).index(d)
            seq_dates = dates_seq[idx: idx + w]
            if len(seq_dates) < w:
                continue
            for ins in sample:
                series = forward_ret_df.loc[seq_dates, ins]
                cum_ret = (series + 1).prod() - 1
                cum_list.append(cum_ret)
        control_stats[f"control_window_{w}_avg_cum_ret"] = float(np.mean(cum_list)) if cum_list else 0.0
    diff_stats = {f"diff_{w}": evt_stats.get(f"window_{w}_avg_cum_ret",0.0) - control_stats.get(f"control_window_{w}_avg_cum_ret",0.0) for w in (1,3,5)}

    report_df = pd.DataFrame(reports)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "report_portfolios.csv")
    report_df.to_csv(report_path, index=False)

    # 汇总 JSON
    import json
    summary = {
        "params": {
            "topk": args.topk,
            "alpha": args.alpha,
            "sleeve_ratio": args.sleeve_ratio,
            "event_hold_days": args.event_hold_days,
            "extend_boost_beta": args.extend_boost_beta,
            "filter_min_events": args.filter_min_events,
            "cost_bps": args.cost_bps,
            "start": args.start,
            "end": args.end,
        },
        "portfolio_stats": reports,
        "event_window_stats": evt_stats,
        "control_stats": control_stats,
        "diff_stats": diff_stats,
    }
    json_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved portfolio summary -> {report_path}")
    print(f"Saved json summary -> {json_path}")


if __name__ == "__main__":
    main()
