import pandas as pd
from live_trading.execution_engine import ExecutionEngine
from live_trading.live_config import ExecutionConfig


def test_execution_engine_order_diff_and_slippage(monkeypatch):
    cfg = ExecutionConfig(simulate=True, max_slippage_bp=10)
    engine = ExecutionEngine(cfg)
    current = pd.DataFrame({'code': ['000001', '000002'], 'weight': [0.5, 0.5], 'avg_price': [10, 12]})
    target = pd.DataFrame({'code': ['000001', '000003'], 'weight': [0.6, 0.4]})

    orders = engine.generate_orders(current_positions=current, target_weights=target, total_equity=1_000_000)
    # Expect: BUY 000001 (increase 0.1), SELL 000002 (removed), BUY 000003 (new 0.4)
    actions = {o.code: o.action for o in orders}
    assert actions['000001'] == 'BUY'
    assert actions['000002'] == 'SELL'
    assert actions['000003'] == 'BUY'

    executed = engine.execute()
    # All filled & slippage within range
    for o in executed:
        assert o.filled is True
        assert 0 <= o.slippage_bp <= cfg.max_slippage_bp

    summary = engine.summary()
    assert summary['order_count'] == 3
