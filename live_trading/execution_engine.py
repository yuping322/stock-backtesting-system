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

    def generate_orders_dataframe(self, current_positions: pd.DataFrame, target_weights: pd.DataFrame, total_equity: float) -> pd.DataFrame:
        """Generate orders DataFrame with columns: code, side, shares.
        
        Args:
            current_positions: DataFrame with columns: code, weight, avg_price (optional)
            target_weights: DataFrame with columns: code, weight
            total_equity: Total equity value for calculating shares
            
        Returns:
            DataFrame with columns: code, side, shares
        """
        cur_map = {}
        if current_positions is not None and not current_positions.empty:
            for row in current_positions.itertuples():
                code = str(row.code).zfill(6)
                weight = row.weight
                avg_price = getattr(row, 'avg_price', self._mock_price(code))
                cur_map[code] = {'weight': weight, 'price': avg_price}
        
        orders_rows = []
        
        # Process target positions
        for row in target_weights.itertuples():
            code = str(row.code).zfill(6)
            tgt_w = row.weight
            
            if code in cur_map:
                cur_w = cur_map[code]['weight']
                cur_price = cur_map[code]['price']
                delta_w = tgt_w - cur_w
            else:
                cur_w = 0.0
                cur_price = self._mock_price(code)
                delta_w = tgt_w
            
            if abs(delta_w) < 1e-6:
                continue
            
            if delta_w > 0:
                side = 'buy'
                target_value = delta_w * total_equity
                shares = int(target_value / cur_price / self.config.lot_size) * self.config.lot_size
            else:
                side = 'sell'
                target_value = abs(delta_w) * total_equity
                # Use current position price for selling
                if code in cur_map:
                    shares = int(target_value / cur_map[code]['price'] / self.config.lot_size) * self.config.lot_size
                else:
                    shares = 0
            
            if shares > 0:
                orders_rows.append({'code': code, 'side': side, 'shares': shares})
        
        # Exit positions not in target
        for code, pos_info in cur_map.items():
            if code not in target_weights['code'].values:
                side = 'sell'
                target_value = pos_info['weight'] * total_equity
                shares = int(target_value / pos_info['price'] / self.config.lot_size) * self.config.lot_size
                if shares > 0:
                    orders_rows.append({'code': code, 'side': side, 'shares': shares})
        
        if orders_rows:
            return pd.DataFrame(orders_rows)
        else:
            return pd.DataFrame(columns=['code', 'side', 'shares'])

    def summary(self) -> Dict[str, float]:
        if not self._orders:
            return {}
        avg_slippage = sum(o.slippage_bp for o in self._orders) / len(self._orders)
        buy_count = sum(1 for o in self._orders if o.action == 'BUY')
        sell_count = sum(1 for o in self._orders if o.action == 'SELL')
        return {'order_count': len(self._orders), 'avg_slippage_bp': avg_slippage, 'buy_orders': buy_count, 'sell_orders': sell_count}
