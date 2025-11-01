"""State persistence for live trading.

Stores positions, NAV history, audit events, and drift metrics using simple CSV / log files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from .live_config import PersistenceConfig


@dataclass
class LiveState:
    positions: pd.DataFrame  # columns: code, weight, avg_price
    nav_history: pd.DataFrame  # columns: date, nav


class StateStore:
    def __init__(self, config: PersistenceConfig):
        self.config = config
        os.makedirs(self.config.state_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.config.state_dir, name)

    def load_state(self) -> LiveState:
        pos_path = self._path(self.config.position_file)
        nav_path = self._path(self.config.nav_file)
        if os.path.exists(pos_path):
            positions = pd.read_csv(pos_path)
        else:
            positions = pd.DataFrame(columns=['code', 'weight', 'avg_price'])
        if os.path.exists(nav_path):
            nav_history = pd.read_csv(nav_path, parse_dates=['date'])
        else:
            nav_history = pd.DataFrame(columns=['date', 'nav'])
        return LiveState(positions=positions, nav_history=nav_history)

    def save_positions(self, df: pd.DataFrame):
        df.to_csv(self._path(self.config.position_file), index=False)

    def append_nav(self, date: datetime, nav: float):
        nav_path = self._path(self.config.nav_file)
        row = pd.DataFrame([[date, nav]], columns=['date', 'nav'])
        if os.path.exists(nav_path):
            row.to_csv(nav_path, mode='a', header=False, index=False)
        else:
            row.to_csv(nav_path, index=False)

    def audit(self, message: str, **kwargs: Any):
        audit_path = self._path(self.config.audit_file)
        ts = datetime.now().isoformat()
        kv = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        line = f"{ts} | {message} {kv}\n"
        with open(audit_path, 'a', encoding='utf-8') as f:
            f.write(line)

    def save_drift_metrics(self, df: pd.DataFrame):
        df.to_csv(self._path(self.config.drift_file), index=False)

    def load_drift_metrics(self) -> Optional[pd.DataFrame]:
        path = self._path(self.config.drift_file)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
