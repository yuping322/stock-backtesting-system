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

    def save_target_weights(self, df: pd.DataFrame):
        """Save target weights (code, weight)."""
        df.to_csv(self._path(self.config.target_weights_file), index=False)

    def load_target_weights(self) -> pd.DataFrame:
        """Load target weights (code, weight)."""
        path = self._path(self.config.target_weights_file)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=['code', 'weight'])

    def save_orders(self, df: pd.DataFrame):
        """Save orders (code, side, shares)."""
        df.to_csv(self._path(self.config.orders_file), index=False)

    def load_orders(self) -> pd.DataFrame:
        """Load orders (code, side, shares)."""
        path = self._path(self.config.orders_file)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=['code', 'side', 'shares'])

    def save_model_ic(self, df: pd.DataFrame):
        """Save per-model IC metrics (date, model, ic)."""
        path = self._path(self.config.model_ic_file)
        if os.path.exists(path):
            existing = pd.read_csv(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['date', 'model'], keep='last')
        df.to_csv(path, index=False)

    def load_model_ic(self) -> Optional[pd.DataFrame]:
        """Load per-model IC metrics."""
        path = self._path(self.config.model_ic_file)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def set_retrain_flag(self):
        """Create retrain.flag file."""
        flag_path = self._path(self.config.retrain_flag_file)
        with open(flag_path, 'w') as f:
            f.write(f"{datetime.now().isoformat()}\n")

    def clear_retrain_flag(self):
        """Remove retrain.flag file."""
        flag_path = self._path(self.config.retrain_flag_file)
        if os.path.exists(flag_path):
            os.remove(flag_path)

    def has_retrain_flag(self) -> bool:
        """Check if retrain.flag exists."""
        return os.path.exists(self._path(self.config.retrain_flag_file))
