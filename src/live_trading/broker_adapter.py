"""broker_adapter: simple adapter interface and a xueqiu placeholder implementation
"""
from __future__ import annotations
from typing import Dict, Any, List


class BrokerAdapter:
    def send_order(self, order: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError()

    def poll_fills(self) -> List[Dict[str, Any]]:
        raise NotImplementedError()


class XueqiuAdapter(BrokerAdapter):
    def __init__(self, portfolio_code: str = None, cookies: str = None):
        self.portfolio_code = portfolio_code
        self.cookies = cookies

    def send_order(self, order: Dict[str, Any]) -> str:
        # Placeholder: log and return fake id
        print(f"XueqiuAdapter: send_order {order}")
        return "ORDER_FAKE_1"

    def cancel_order(self, order_id: str) -> bool:
        print(f"XueqiuAdapter: cancel {order_id}")
        return True

    def poll_fills(self) -> List[Dict[str, Any]]:
        return []
