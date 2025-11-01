"""Model/strategy drift detection.

Computes simple IC (Information Coefficient) between prediction weights and realized returns.
Triggers flags when rolling IC falls below threshold or consecutive negatives occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

from .live_config import RiskConfig


@dataclass
class DriftStatus:
    latest_ic: float
    rolling_ic: float
    consecutive_negative: int
    trigger_retrain: bool
    meta: Dict[str, float]


class DriftDetector:
    def __init__(self, config: RiskConfig):
        self.config = config
        self._history: pd.DataFrame = pd.DataFrame(columns=['date', 'code', 'weight', 'return'])

    def update(self, pred_df: pd.DataFrame, realized_returns: pd.DataFrame):
        """Merge today's predictions with realized next-day returns.
        realized_returns columns: date, code, return (return can be daily pct change for code)
        """
        if pred_df.empty or realized_returns.empty:
            return
        # assume realized_returns.date aligns to prediction date (or next day). Adjust if needed.
        merged = pred_df.merge(realized_returns, on=['date', 'code'], how='inner')
        if merged.empty:
            return
        self._history = pd.concat([self._history, merged[['date', 'code', 'weight', 'return']]], ignore_index=True)

    def evaluate(self) -> Optional[DriftStatus]:
        if self._history.empty:
            return None
        # compute daily ICs
        daily_groups = self._history.groupby('date')
        ic_list = []
        consecutive_negative = 0
        last_ic = 0.0
        for dt, grp in daily_groups:
            if grp['weight'].nunique() <= 1 or grp['return'].nunique() <= 1:
                ic = 0.0
            else:
                ic = grp[['weight', 'return']].corr().iloc[0, 1]
            ic_list.append((dt, ic))
            last_ic = ic
        # count consecutive negatives from end
        for _, ic in reversed(ic_list):
            if ic < 0:
                consecutive_negative += 1
            else:
                break
        ic_series = pd.Series([ic for _, ic in ic_list])
        window = self.config.ic_rolling_window
        rolling_ic = ic_series.tail(window).mean()
        # Trigger retrain if rolling IC below threshold, or consecutive negatives large,
        # or immediate latest IC is strongly negative (defensive early warning).
        trigger = (
            rolling_ic < self.config.min_ic_threshold
            or consecutive_negative >= 3
            or last_ic < 0
        )
        meta = {
            'ic_count': float(len(ic_series)),
            'rolling_window': float(window),
        }
        return DriftStatus(latest_ic=last_ic,
                           rolling_ic=float(rolling_ic),
                           consecutive_negative=float(consecutive_negative),
                           trigger_retrain=bool(trigger),
                           meta=meta)
