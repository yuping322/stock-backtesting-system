"""Main orchestration script for live trading workflow.

Stages implemented (simplified):
1. Pre-market: load predictions & build target portfolio.
2. Before open: risk evaluation & potential de-risk adjustments.
3. Open/Intraday (simulated): generate & execute orders, monitor slippage.
4. Close: append NAV, run drift detection, audit.

This is a deterministic, single-run script for prototype purposes; in real deployment,
these stages would be scheduled at different times with persistent state across them.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd

from .live_config import DEFAULT_LIVE_CONFIG, LiveConfig
from .prediction_loader import PredictionLoader
from .portfolio_builder import PortfolioBuilder
from .risk_manager import RiskManager
from .execution_engine import ExecutionEngine
from .state_store import StateStore
from .drift_detector import DriftDetector


def run_live(config: LiveConfig = DEFAULT_LIVE_CONFIG, total_equity: float = 1_000_000):
    # load previous state
    store = StateStore(config.persistence)
    state = store.load_state()

    # 1. Pre-market: load predictions & portfolio construction
    loader = PredictionLoader(config.data)
    pred_df = loader.load_latest()
    portfolio_builder = PortfolioBuilder(config.portfolio)
    portfolio_result = portfolio_builder.build(pred_df)

    if portfolio_result is None:
        store.audit("No portfolio built (empty predictions)")
        return

    # 2. Risk evaluation before open
    risk_mgr = RiskManager(config.risk)
    # use last NAV if exists, else initial
    if not state.nav_history.empty:
        for nav in state.nav_history['nav'].tail(50):
            risk_mgr.update_nav(nav)
    else:
        risk_mgr.update_nav(total_equity)
    risk_status = risk_mgr.evaluate(portfolio_result.target_weights)

    adjusted_weights = portfolio_result.target_weights.copy()
    if risk_status.circuit_break:
        # full cash: zero out weights
        adjusted_weights['weight'] = 0.0
        store.audit("Circuit break triggered: liquidating", drawdown=risk_status.drawdown)
    elif risk_status.de_risk:
        # scale down proportionally (e.g., 50% exposure)
        adjusted_weights['weight'] = adjusted_weights['weight'] * 0.5
        # renormalize (preserve cash portion implicitly)
        adjusted_weights['weight'] = adjusted_weights['weight'] / adjusted_weights['weight'].sum() * 0.5
        store.audit("De-risk scaling applied", drawdown=risk_status.drawdown, hhi=risk_status.hhi)

    # 3. Intraday: generate & execute orders
    exec_engine = ExecutionEngine(config.execution)
    current_positions = state.positions
    orders = exec_engine.generate_orders(current_positions=current_positions, target_weights=adjusted_weights, total_equity=total_equity)
    executed = exec_engine.execute()
    exec_summary = exec_engine.summary()
    store.audit("Orders executed", **exec_summary)

    # update positions snapshot (simplified: assume perfect fills at est_price)
    new_pos = adjusted_weights.copy()
    # assign mock avg_price
    new_pos['avg_price'] = new_pos['code'].apply(lambda c: exec_engine._mock_price(c))
    store.save_positions(new_pos[['code', 'weight', 'avg_price']])

    # 4. Close: compute mock NAV (sum weights * equity assumed constant for proto)
    nav = total_equity  # placeholder; real nav would revalue positions with closing prices
    store.append_nav(datetime.now(), nav)

    # Drift detection (requires realized returns; here we simulate random returns for placeholder)
    drift_detector = DriftDetector(config.risk)
    # simulate realized returns for the same date with small noise
    sim_returns = new_pos[['code']].copy()
    sim_returns['date'] = portfolio_result.date
    sim_returns['return'] = 0.001  # constant small return placeholder
    drift_detector.update(pred_df[pred_df['date'] == portfolio_result.date], sim_returns[['date', 'code', 'return']])
    drift_status = drift_detector.evaluate()
    if drift_status:
        drift_df = pd.DataFrame([[portfolio_result.date, drift_status.latest_ic, drift_status.rolling_ic, drift_status.consecutive_negative, drift_status.trigger_retrain]],
                                columns=['date', 'latest_ic', 'rolling_ic', 'consecutive_negative', 'trigger_retrain'])
        store.save_drift_metrics(drift_df)
        store.audit("Drift evaluated", latest_ic=drift_status.latest_ic, rolling_ic=drift_status.rolling_ic, trigger=int(drift_status.trigger_retrain))

    # final audit summary
    store.audit("Run complete", order_count=exec_summary.get('order_count', 0))


if __name__ == "__main__":
    run_live()
