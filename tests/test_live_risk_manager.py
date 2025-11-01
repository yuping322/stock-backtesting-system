import pandas as pd
from live_trading.risk_manager import RiskManager
from live_trading.live_config import RiskConfig


def test_risk_manager_evaluate_drawdown_and_flags():
    cfg = RiskConfig(max_drawdown_limit=0.05, circuit_break_drawdown=0.10, concentration_hhi_limit=0.30)
    rm = RiskManager(cfg)
    # simulate nav path with drawdown > limit but < circuit breaker
    navs = [100, 102, 101, 99, 95]  # peak 102 -> dd at 95 is (95-102)/102 ≈ -0.0686
    for n in navs:
        rm.update_nav(n)
    weights = pd.DataFrame({'code': ['000001', '000002'], 'weight': [0.6, 0.4]})
    status = rm.evaluate(weights)
    assert status.de_risk is True  # drawdown beyond limit
    assert status.circuit_break is False  # not beyond circuit break threshold


def test_risk_manager_circuit_break():
    cfg = RiskConfig(max_drawdown_limit=0.05, circuit_break_drawdown=0.10)
    rm = RiskManager(cfg)
    # larger drawdown
    navs = [100, 110, 105, 90]  # peak 110 -> dd at 90 is (90-110)/110 = -0.1818
    for n in navs:
        rm.update_nav(n)
    status = rm.evaluate(pd.DataFrame({'code': ['000001'], 'weight': [1.0]}))
    assert status.circuit_break is True
