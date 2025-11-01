"""Execution engine abstraction (placeholder).

In live trading this would connect to a broker or trading API. Here we simulate orders
and track slippage & fill status for monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import random

from .live_config import ExecutionConfig


@dataclass
class Order:
    code: str
    target_weight: float
    action: str  # BUY / SELL / HOLD
    est_price: float
    est_value: float
    filled: bool = False
    slippage_bp: float = 0.0


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self._orders: List[Order] = []

    def generate_orders(self, current_positions: pd.DataFrame, target_weights: pd.DataFrame, total_equity: float) -> List[Order]:
        """Diff current vs target to create orders. current_positions columns: code, weight, avg_price.
        target_weights columns: code, weight.
        """
        cur_map = {row.code: row.weight for row in current_positions.itertuples()} if current_positions is not None and not current_positions.empty else {}
        orders: List[Order] = []
        for row in target_weights.itertuples():
            code = row.code
            tgt_w = row.weight
            cur_w = cur_map.get(code, 0.0)
            delta = tgt_w - cur_w
            if abs(delta) < 1e-6:
                continue
            action = 'BUY' if delta > 0 else 'SELL'
            est_price = self._mock_price(code)
            est_value = abs(delta) * total_equity
            orders.append(Order(code=code, target_weight=tgt_w, action=action, est_price=est_price, est_value=est_value))
        # exit positions not in target
        for code, cur_w in cur_map.items():
            if code not in target_weights['code'].values:
                est_price = self._mock_price(code)
                est_value = cur_w * total_equity
                orders.append(Order(code=code, target_weight=0.0, action='SELL', est_price=est_price, est_value=est_value))
        self._orders = orders
        return orders

    def execute(self) -> List[Order]:
        executed: List[Order] = []
        for o in self._orders:
            if self.config.simulate:
                # random slippage within tolerance
                slippage_bp = random.uniform(0, self.config.max_slippage_bp)
                o.slippage_bp = slippage_bp
                o.filled = True
            else:
                # integrate actual broker API
                pass
            executed.append(o)
        return executed

    def _mock_price(self, code: str) -> float:
        # placeholder deterministic pseudo-price for repeatability
        seed = int(code) % 1000
        return 10 + (seed / 1000.0) * 30  # price between 10 and 40

    def summary(self) -> Dict[str, float]:
        if not self._orders:
            return {}
        avg_slippage = sum(o.slippage_bp for o in self._orders) / len(self._orders)
        buy_count = sum(1 for o in self._orders if o.action == 'BUY')
        sell_count = sum(1 for o in self._orders if o.action == 'SELL')
        return {'order_count': len(self._orders), 'avg_slippage_bp': avg_slippage, 'buy_orders': buy_count, 'sell_orders': sell_count}
