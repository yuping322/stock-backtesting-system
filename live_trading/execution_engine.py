"""execution_engine: advanced order management with splitting, priority, slippage control, and volume limits
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    order_id: str
    code: str
    side: str  # 'buy' or 'sell'
    quantity: int
    filled_quantity: int = 0
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher number = higher priority
    slippage_limit: Optional[float] = None
    volume_limit: Optional[int] = None
    parent_order_id: Optional[str] = None  # For split orders
    tags: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    def __init__(self, broker_adapter, max_order_size: int = 100000, max_daily_volume: int = 1000000):
        self.broker = broker_adapter
        self.max_order_size = max_order_size
        self.max_daily_volume = max_daily_volume
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.daily_volume: Dict[str, int] = {}  # code -> volume today
        self._lock = threading.RLock()
        self._order_counter = 0

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        with self._lock:
            self._order_counter += 1
            return f"ORD_{int(time.time())}_{self._order_counter}"

    def compute_diff(self, target: pd.DataFrame, current: pd.DataFrame, ignore_minor: float = 0.001) -> List[Dict[str, Any]]:
        """Compute portfolio differences with advanced features"""
        cur = current.set_index("code").weight if not current.empty else pd.Series(dtype=float)
        tgt = target.set_index("code").target_weight
        codes = sorted(set(cur.index.tolist()) | set(tgt.index.tolist()))
        orders = []

        for c in codes:
            w_cur = float(cur.get(c, 0.0))
            w_tgt = float(tgt.get(c, 0.0))
            delta = w_tgt - w_cur
            if abs(delta) < ignore_minor:
                continue
            orders.append({
                "code": c,
                "delta_weight": delta,
                "current_weight": w_cur,
                "target_weight": w_tgt
            })

        return orders

    def submit_orders(self, orders: List[Dict[str, Any]], portfolio_value: float = 1000000.0):
        """Submit orders with advanced order management"""
        with self._lock:
            for order_spec in orders:
                self._submit_single_order(order_spec, portfolio_value)

    def _submit_single_order(self, order_spec: Dict[str, Any], portfolio_value: float):
        """Submit a single order with splitting and optimization"""
        code = order_spec['code']
        delta_weight = order_spec['delta_weight']
        side = 'buy' if delta_weight > 0 else 'sell'

        # Calculate target quantity
        target_quantity = int(abs(delta_weight) * portfolio_value / self._get_current_price(code))

        if target_quantity == 0:
            return

        # Check volume limits
        if not self._check_volume_limits(code, target_quantity):
            logger.warning(f"Volume limit exceeded for {code}, skipping order")
            return

        # Split large orders
        order_specs = self._split_large_order(code, target_quantity, side, order_spec)

        for spec in order_specs:
            order = self._create_order(spec)
            self.active_orders[order.order_id] = order

            # Submit to broker
            try:
                self.broker.send_order({
                    'order_id': order.order_id,
                    'code': order.code,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price': order.price,
                    'order_type': order.order_type.value
                })
                logger.info(f"Submitted order {order.order_id} for {order.code}")
            except Exception as e:
                logger.error(f"Failed to submit order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED

    def _split_large_order(self, code: str, total_quantity: int, side: str, order_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split large orders into smaller chunks"""
        if total_quantity <= self.max_order_size:
            return [{
                'code': code,
                'quantity': total_quantity,
                'side': side,
                'priority': self._calculate_priority(order_spec),
                'slippage_limit': self._calculate_slippage_limit(code, side),
                'volume_limit': self._calculate_volume_limit(code)
            }]

        # Split into chunks
        chunks = []
        remaining = total_quantity
        chunk_size = min(self.max_order_size, total_quantity // 3 + 1)  # Split into roughly equal chunks

        while remaining > 0:
            quantity = min(chunk_size, remaining)
            chunks.append({
                'code': code,
                'quantity': quantity,
                'side': side,
                'priority': self._calculate_priority(order_spec),
                'slippage_limit': self._calculate_slippage_limit(code, side),
                'volume_limit': self._calculate_volume_limit(code)
            })
            remaining -= quantity

        return chunks

    def _calculate_priority(self, order_spec: Dict[str, Any]) -> int:
        """Calculate order priority based on various factors"""
        priority = 0

        # Higher priority for larger weight changes
        delta_weight = abs(order_spec.get('delta_weight', 0))
        if delta_weight > 0.05:  # 5% change
            priority += 2
        elif delta_weight > 0.02:  # 2% change
            priority += 1

        # Higher priority for selling (reduce risk)
        if order_spec.get('delta_weight', 0) < 0:
            priority += 1

        return priority

    def _calculate_slippage_limit(self, code: str, side: str) -> float:
        """Calculate slippage limit for order"""
        # Simple implementation - in practice would use volatility, liquidity data
        base_slippage = 0.005  # 0.5%
        return base_slippage

    def _calculate_volume_limit(self, code: str) -> int:
        """Calculate volume limit for order"""
        # Simple implementation - in practice would use market data
        return self.max_order_size

    def _check_volume_limits(self, code: str, quantity: int) -> bool:
        """Check if order quantity is within volume limits"""
        current_daily = self.daily_volume.get(code, 0)
        return current_daily + quantity <= self.max_daily_volume

    def _get_current_price(self, code: str) -> float:
        """Get current price for security (mock implementation)"""
        # In practice, this would query market data
        return 10.0  # Mock price

    def _create_order(self, spec: Dict[str, Any]) -> Order:
        """Create Order object from specification"""
        return Order(
            order_id=self._generate_order_id(),
            code=spec['code'],
            side=spec['side'],
            quantity=spec['quantity'],
            priority=spec.get('priority', 0),
            slippage_limit=spec.get('slippage_limit'),
            volume_limit=spec.get('volume_limit')
        )

    def update_order_status(self, order_id: str, status: OrderStatus, filled_quantity: int = 0, price: float = None):
        """Update order status from broker feedback"""
        with self._lock:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = status
                order.filled_quantity = filled_quantity
                if price:
                    order.price = price

                # Update daily volume
                if status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                    self.daily_volume[order.code] = self.daily_volume.get(order.code, 0) + filled_quantity

                # Move to history if completed
                if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    self.order_history.append(order)
                    del self.active_orders[order_id]

                logger.info(f"Updated order {order_id} status to {status.value}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        with self._lock:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                try:
                    self.broker.cancel_order(order_id)
                    order.status = OrderStatus.CANCELLED
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    logger.info(f"Cancelled order {order_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")
                    return False
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        with self._lock:
            if order_id in self.active_orders:
                return self.active_orders[order_id]

            # Check history
            for order in self.order_history:
                if order.order_id == order_id:
                    return order
            return None

    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        with self._lock:
            return list(self.active_orders.values())

    def get_order_history(self, code: str = None, limit: int = 100) -> List[Order]:
        """Get order history, optionally filtered by code"""
        with self._lock:
            history = self.order_history[-limit:]  # Most recent
            if code:
                history = [o for o in history if o.code == code]
            return history

    def optimize_execution(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize order execution for minimal market impact"""
        # Sort by priority (highest first)
        orders.sort(key=lambda x: x.get('priority', 0), reverse=True)

        # Add time delays between orders to reduce market impact
        for i, order in enumerate(orders):
            order['execution_delay'] = i * 5  # 5 seconds between orders

        return orders

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        with self._lock:
            total_orders = len(self.order_history)
            filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
            fill_rate = filled_orders / total_orders if total_orders > 0 else 0

            return {
                'total_orders': total_orders,
                'active_orders': len(self.active_orders),
                'fill_rate': fill_rate,
                'daily_volume': dict(self.daily_volume)
            }
