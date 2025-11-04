"""Trading script: Execute orders from orders.csv during market hours.

This script runs during market hours (e.g., 09:30) to:
1. Load orders from orders.csv
2. Execute orders (simulated or real broker API)
3. Update positions and cash after fills

Usage:
    python -m live_trading.run_trade 20250603
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
import pandas as pd

from .live_config import DEFAULT_LIVE_CONFIG, LiveConfig
from .execution_engine import ExecutionEngine
from .state_store import StateStore


def run_trade(date: str, config: LiveConfig = None, total_equity: float = 1_000_000):
    """Execute orders for a specific date.
    
    Args:
        date: Date string in YYYYMMDD format (e.g., '20250603')
        config: LiveConfig instance (defaults to DEFAULT_LIVE_CONFIG)
        total_equity: Total equity value for calculating position values
    """
    if config is None:
        config = DEFAULT_LIVE_CONFIG
    
    store = StateStore(config.persistence)
    
    # Load orders
    orders_df = store.load_orders()
    
    if orders_df.empty:
        store.audit(f"No orders found for {date}")
        print(f"No orders to execute for {date}")
        return
    
    store.audit(f"Loading orders for {date}", order_count=len(orders_df))
    
    # Load current positions
    state = store.load_state()
    current_positions = state.positions.copy()
    
    # Execute orders (simulated or real)
    exec_engine = ExecutionEngine(config.execution)
    
    # Update positions based on executed orders
    # For now, simulate execution at mock prices
    new_positions = []
    position_map = {}
    
    # Initialize position map from current positions
    if not current_positions.empty:
        for row in current_positions.itertuples():
            code = str(row.code).zfill(6)
            weight = row.weight
            avg_price = getattr(row, 'avg_price', exec_engine._mock_price(code))
            position_map[code] = {
                'weight': weight,
                'price': avg_price,
                'shares': int((weight * total_equity) / avg_price / config.execution.lot_size) * config.execution.lot_size
            }
    
    # Process orders
    executed_orders = []
    for row in orders_df.itertuples():
        code = str(row.code).zfill(6)
        side = row.side.lower()
        shares = int(row.shares)
        
        # Get current position or initialize
        if code not in position_map:
            position_map[code] = {'weight': 0.0, 'price': exec_engine._mock_price(code), 'shares': 0}
        
        current_shares = position_map[code]['shares']
        price = exec_engine._mock_price(code)
        
        # Execute order
        if side == 'buy':
            new_shares = current_shares + shares
            # Update average price
            if current_shares == 0:
                new_price = price
            else:
                new_price = (current_shares * position_map[code]['price'] + shares * price) / new_shares
            position_map[code]['shares'] = new_shares
            position_map[code]['price'] = new_price
        elif side == 'sell':
            new_shares = max(0, current_shares - shares)
            position_map[code]['shares'] = new_shares
            if new_shares == 0:
                position_map[code]['weight'] = 0.0
        
        executed_orders.append({
            'code': code,
            'side': side,
            'shares': shares,
            'price': price,
            'status': 'filled'
        })
    
    # Convert position_map to DataFrame
    for code, pos_info in position_map.items():
        if pos_info['shares'] > 0:
            weight = (pos_info['shares'] * pos_info['price']) / total_equity
            new_positions.append({
                'code': code,
                'weight': weight,
                'avg_price': pos_info['price']
            })
    
    new_positions_df = pd.DataFrame(new_positions)
    
    # Save updated positions
    store.save_positions(new_positions_df)
    
    # Log execution summary
    exec_summary = {
        'order_count': len(executed_orders),
        'buy_orders': len([o for o in executed_orders if o['side'] == 'buy']),
        'sell_orders': len([o for o in executed_orders if o['side'] == 'sell']),
        'position_count': len(new_positions_df)
    }
    
    store.audit("Orders executed", **exec_summary)
    
    print(f"Trading workflow completed for {date}")
    print(f"  Executed orders: {len(executed_orders)}")
    print(f"  Buy: {exec_summary['buy_orders']}, Sell: {exec_summary['sell_orders']}")
    print(f"  Positions after execution: {len(new_positions_df)}")


def main():
    parser = argparse.ArgumentParser(description="Execute orders during market hours")
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
    
    run_trade(args.date, config=None, total_equity=args.equity)


if __name__ == "__main__":
    main()

