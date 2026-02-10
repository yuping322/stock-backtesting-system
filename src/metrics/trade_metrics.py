"""Trading activity and round-trip analysis metrics."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SMALL_VALUE = 1e-12


@dataclass
class RoundTrip:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares: float
    entry_price: float
    exit_price: float
    pnl: float

    @property
    def holding_days(self) -> float:
        if pd.isna(self.entry_date) or pd.isna(self.exit_date):
            return float("nan")
        return float((self.exit_date - self.entry_date).days or 0)

    @property
    def return_ratio(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


def _to_timestamp(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if value is None:
        return pd.NaT
    return pd.Timestamp(value)


def compute_turnover(trades: Iterable[dict], nav_series: Optional[pd.Series] = None) -> Dict[str, float]:
    trades_list = list(trades or [])
    if not trades_list:
        return {
            "total_turnover": 0.0,
            "average_daily_turnover": 0.0,
            "trade_count": 0,
        }

    total_traded = float(sum(abs(t.get("value", 0.0)) for t in trades_list))

    if nav_series is not None and not nav_series.empty:
        average_equity = float(nav_series.astype(float).mean())
        trading_days = max(1, len(nav_series) - 1)
    else:
        portfolio_values = [t.get("portfolio_value") for t in trades_list if t.get("portfolio_value")]
        average_equity = float(np.mean(portfolio_values)) if portfolio_values else total_traded
        unique_dates = {pd.Timestamp(t.get("date")) for t in trades_list if t.get("date")}
        trading_days = max(1, len(unique_dates))

    if average_equity <= SMALL_VALUE:
        average_equity = SMALL_VALUE

    total_turnover = total_traded / average_equity
    average_daily_turnover = total_turnover / trading_days

    return {
        "total_turnover": float(total_turnover),
        "average_daily_turnover": float(average_daily_turnover),
        "trade_count": len(trades_list),
    }


def rebuild_round_trips(trades: Iterable[dict]) -> List[RoundTrip]:
    trips: List[RoundTrip] = []
    trade_queue: Dict[str, Deque] = {}

    for trade in sorted(trades or [], key=lambda t: _to_timestamp(t.get("date"))):
        code = str(trade.get("code", "")).zfill(6)
        size = float(trade.get("size", 0.0))
        price = float(trade.get("price", 0.0))
        date = _to_timestamp(trade.get("date"))
        if not code or price <= 0 or size == 0:
            continue

        queue = trade_queue.setdefault(code, deque())

        if size > 0:  # buy
            queue.append({"shares": size, "price": price, "date": date})
            continue

        shares_to_close = abs(size)
        while shares_to_close > SMALL_VALUE and queue:
            entry = queue[0]
            matched = min(shares_to_close, entry["shares"])
            pnl = matched * (price - entry["price"])

            trips.append(
                RoundTrip(
                    code=code,
                    entry_date=entry["date"],
                    exit_date=date,
                    shares=matched,
                    entry_price=entry["price"],
                    exit_price=price,
                    pnl=pnl,
                )
            )

            shares_to_close -= matched
            entry["shares"] -= matched
            if entry["shares"] <= SMALL_VALUE:
                queue.popleft()

    return trips


def holding_period_stats(round_trips: Iterable[RoundTrip]) -> Dict[str, float]:
    trips = [trip for trip in round_trips if not np.isnan(trip.holding_days)]
    if not trips:
        return {
            "avg_holding_days": 0.0,
            "median_holding_days": 0.0,
            "max_holding_days": 0.0,
        }

    durations = np.array([trip.holding_days for trip in trips], dtype=float)
    return {
        "avg_holding_days": float(durations.mean()),
        "median_holding_days": float(np.median(durations)),
        "max_holding_days": float(durations.max()),
    }


def payoff_stats(round_trips: Iterable[RoundTrip]) -> Dict[str, float]:
    trips = list(round_trips or [])
    if not trips:
        return {
            "round_trip_count": 0,
            "win_rate": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
            "average_gain": 0.0,
            "average_loss": 0.0,
        }

    pnls = np.array([trip.pnl for trip in trips], dtype=float)
    gains = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = float(len(gains) / len(trips)) if trips else 0.0
    avg_gain = float(gains.mean()) if len(gains) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff_ratio = float(avg_gain / abs(avg_loss)) if avg_gain > 0 and avg_loss < 0 else 0.0
    expectancy = float(pnls.mean()) if len(pnls) else 0.0

    return {
        "round_trip_count": len(trips),
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "average_gain": avg_gain,
        "average_loss": avg_loss,
    }


def trading_metrics(trades: Iterable[dict], nav_series: Optional[pd.Series] = None) -> Dict[str, float]:
    turnover = compute_turnover(trades, nav_series=nav_series)
    round_trips = rebuild_round_trips(trades)
    holding = holding_period_stats(round_trips)
    payoff = payoff_stats(round_trips)

    return {
        **turnover,
        **holding,
        **payoff,
    }
