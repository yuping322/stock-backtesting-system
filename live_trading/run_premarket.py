"""Pre-market script: Load ensemble predictions, build target portfolio, generate orders.

This script runs at 07:00 before market open to:
1. Load ensemble predictions from multiple models
2. Apply risk checks (single stock ≤ 10%, industry ≤ 30%)
3. Generate target weights (target_w.csv)
4. Generate orders (orders.csv)

Usage:
    python -m live_trading.run_premarket 20250603
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
import pandas as pd

from .live_config import DEFAULT_LIVE_CONFIG, LiveConfig
from .prediction_loader import PredictionLoader
from .portfolio_builder import PortfolioBuilder
from .risk_manager import RiskManager
from .execution_engine import ExecutionEngine
from .state_store import StateStore


def run_premarket(date: str, config: LiveConfig = None, total_equity: float = 1_000_000):
    """Run pre-market workflow for a specific date.
    
    Args:
        date: Date string in YYYYMMDD format (e.g., '20250603')
        config: LiveConfig instance (defaults to DEFAULT_LIVE_CONFIG)
        total_equity: Total equity value for calculating shares
    """
    if config is None:
        config = DEFAULT_LIVE_CONFIG
    
    store = StateStore(config.persistence)
    
    # Load current positions
    state = store.load_state()
    current_positions = state.positions
    
    # 1. Load ensemble predictions (07:00)
    loader = PredictionLoader(config.data)
    target_weights = loader.load_ensemble(date, config.data.models, config.portfolio.top_n)
    
    if target_weights.empty:
        store.audit(f"No ensemble predictions loaded for {date}")
        print(f"Warning: No predictions found for {date}")
        return
    
    # Save target weights before risk checks
    store.save_target_weights(target_weights)
    store.audit(f"Ensemble loaded for {date}", stock_count=len(target_weights))
    
    # 2. Apply risk checks (07:01)
    # Portfolio builder already applies max_stock_weight and max_industry_weight constraints
    portfolio_builder = PortfolioBuilder(config.portfolio)
    # Convert target_weights to pred_df format for portfolio builder
    pred_df = pd.DataFrame({
        'date': [pd.Timestamp(date[:4] + '-' + date[4:6] + '-' + date[6:8])] * len(target_weights),
        'code': target_weights['code'],
        'weight': target_weights['weight']
    })
    
    portfolio_result = portfolio_builder.build(pred_df)
    
    if portfolio_result is None:
        store.audit(f"Portfolio build failed for {date}")
        return
    
    # Get risk-adjusted weights
    adjusted_weights = portfolio_result.target_weights
    
    # Risk manager checks (drawdown, HHI, etc.)
    risk_mgr = RiskManager(config.risk)
    if not state.nav_history.empty:
        for nav in state.nav_history['nav'].tail(50):
            risk_mgr.update_nav(nav)
    else:
        risk_mgr.update_nav(total_equity)
    
    risk_status = risk_mgr.evaluate(adjusted_weights)
    
    # Apply risk adjustments
    if risk_status.circuit_break:
        adjusted_weights['weight'] = 0.0
        store.audit("Circuit break triggered: liquidating", drawdown=risk_status.drawdown)
    elif risk_status.de_risk:
        adjusted_weights['weight'] = adjusted_weights['weight'] * 0.5
        total_w = adjusted_weights['weight'].sum()
        if total_w > 0:
            adjusted_weights['weight'] = adjusted_weights['weight'] / total_w * 0.5
        store.audit("De-risk scaling applied", drawdown=risk_status.drawdown, hhi=risk_status.hhi)
    
    # Enforce max_stock_weight constraint if violated
    max_w = config.portfolio.max_stock_weight
    if (adjusted_weights['weight'] > max_w).any():
        adjusted_weights['weight'] = adjusted_weights['weight'].clip(upper=max_w)
        total_w = adjusted_weights['weight'].sum()
        if total_w > 0:
            adjusted_weights['weight'] = adjusted_weights['weight'] / total_w
    
    # Save final target weights
    store.save_target_weights(adjusted_weights)
    store.audit("Risk checks applied", 
                stock_count=len(adjusted_weights),
                max_weight=float(adjusted_weights['weight'].max()),
                hhi=float((adjusted_weights['weight'] ** 2).sum()))
    
    # 3. Generate orders (07:02)
    exec_engine = ExecutionEngine(config.execution)
    orders_df = exec_engine.generate_orders_dataframe(
        current_positions=current_positions,
        target_weights=adjusted_weights,
        total_equity=total_equity
    )
    
    store.save_orders(orders_df)
    store.audit("Orders generated", order_count=len(orders_df))
    
    print(f"Pre-market workflow completed for {date}")
    print(f"  Target stocks: {len(adjusted_weights)}")
    print(f"  Orders: {len(orders_df)} (buy: {len(orders_df[orders_df['side'] == 'buy']) if not orders_df.empty else 0}, "
          f"sell: {len(orders_df[orders_df['side'] == 'sell']) if not orders_df.empty else 0})")


def main():
    parser = argparse.ArgumentParser(description="Pre-market workflow: load ensemble and generate orders")
    parser.add_argument('date', type=str, help='Date in YYYYMMDD format (e.g., 20250603)')
    parser.add_argument('--config', type=str, help='Path to config file (future: JSON/YAML)')
    parser.add_argument('--equity', type=float, default=1_000_000, help='Total equity value')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y%m%d')
    except ValueError:
        print(f"Error: Invalid date format. Expected YYYYMMDD, got: {args.date}")
        sys.exit(1)
    
    run_premarket(args.date, config=None, total_equity=args.equity)


if __name__ == "__main__":
    main()

