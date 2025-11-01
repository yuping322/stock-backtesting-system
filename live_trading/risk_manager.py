"""Risk management utilities for live trading.

Computes drawdown, volatility, concentration, and applies circuit-breaker style rules.
Maintains a small rolling window of NAVs for fast checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

from .live_config import RiskConfig


@dataclass
class RiskStatus:
    drawdown: float
    rolling_vol: float
    hhi: float
    circuit_break: bool
    de_risk: bool
    meta: Dict[str, float]


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self._nav_history: list[float] = []

    def update_nav(self, nav: float):
        self._nav_history.append(nav)
        # keep last 200 entries to avoid memory growth
        if len(self._nav_history) > 200:
            self._nav_history = self._nav_history[-200:]

    def evaluate(self, portfolio_weights: pd.DataFrame) -> RiskStatus:
        nav_series = pd.Series(self._nav_history)
        if nav_series.empty:
            drawdown = 0.0
            rolling_vol = 0.0
        else:
            cum_max = nav_series.cummax()
            dd_series = (nav_series - cum_max) / cum_max
            drawdown = float(dd_series.min())
            # daily returns approximation
            returns = nav_series.pct_change().dropna()
            if returns.empty:
                rolling_vol = 0.0
            else:
                # annualize assuming ~250 trading days
                rolling_vol = float(returns[-self.config.vol_lookback:].std() * np.sqrt(250))

        # concentration (HHI)
        if portfolio_weights is not None and not portfolio_weights.empty:
            hhi = float((portfolio_weights['weight'] ** 2).sum())
        else:
            hhi = 0.0

        de_risk = drawdown < -self.config.max_drawdown_limit or hhi > self.config.concentration_hhi_limit
        circuit_break = drawdown < -self.config.circuit_break_drawdown

        meta = {
            'nav_points': float(len(nav_series)),
            'latest_nav': float(nav_series.iloc[-1]) if not nav_series.empty else 0.0,
        }
        return RiskStatus(drawdown=drawdown,
                          rolling_vol=rolling_vol,
                          hhi=hhi,
                          circuit_break=circuit_break,
                          de_risk=de_risk,
                          meta=meta)
