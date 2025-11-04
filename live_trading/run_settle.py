"""Settlement script: Update NAV, calculate model ICs, trigger retraining if needed.

This script runs at market close (15:15) to:
1. Load closing prices and update NAV
2. Calculate per-model IC metrics
3. Check if retraining is needed (5-day avg IC < 0.02)
4. Generate retrain.flag if trigger conditions met

Usage:
    python -m live_trading.run_settle 20250603
"""
from __future__ import annotations

import sys
import argparse
from datetime import datetime
import pandas as pd

from .live_config import DEFAULT_LIVE_CONFIG, LiveConfig
from .prediction_loader import PredictionLoader
from .drift_detector import DriftDetector
from .state_store import StateStore


def get_closing_prices(date: str, codes: list, mock: bool = True) -> pd.DataFrame:
    """Get closing prices for codes on date.
    
    Args:
        date: Date string in YYYYMMDD format
        codes: List of stock codes
        mock: If True, use mock prices (placeholder for real data source)
        
    Returns:
        DataFrame with columns: date, code, price
    """
    if mock:
        # Mock prices (placeholder - replace with real data source)
        prices = []
        for code in codes:
            # Ensure code is string and zfill to 6 digits
            code_str = str(code).zfill(6)
            # Deterministic mock price based on code
            try:
                seed = int(code_str) % 1000
            except (ValueError, TypeError):
                seed = hash(code_str) % 1000
            price = 10 + (seed / 1000.0) * 30
            prices.append({
                'date': pd.Timestamp(date[:4] + '-' + date[4:6] + '-' + date[6:8]),
                'code': code_str,
                'price': price
            })
        return pd.DataFrame(prices)
    else:
        # TODO: Integrate real price data source (e.g., akshare, tushare, etc.)
        # For now, return empty DataFrame
        return pd.DataFrame(columns=['date', 'code', 'price'])


def get_realized_returns(date: str, codes: list, positions_df: pd.DataFrame, closing_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate realized returns for positions.
    
    Args:
        date: Date string in YYYYMMDD format
        codes: List of stock codes
        positions_df: DataFrame with columns: code, weight, avg_price (entry price)
        closing_prices: DataFrame with columns: date, code, price (closing price)
        
    Returns:
        DataFrame with columns: date, code, return (daily return as pct change)
    """
    if positions_df.empty or closing_prices.empty:
        return pd.DataFrame(columns=['date', 'code', 'return'])
    
    # Merge positions with closing prices
    merged = positions_df.merge(closing_prices, on='code', how='inner')
    
    # Calculate returns: (close_price - avg_price) / avg_price
    merged['return'] = (merged['price'] - merged['avg_price']) / merged['avg_price']
    
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(merged['date']):
        merged['date'] = pd.to_datetime(merged['date']).dt.normalize()
    
    return merged[['date', 'code', 'return']]


def calculate_nav(positions_df: pd.DataFrame, closing_prices: pd.DataFrame, total_equity: float = 1_000_000, initial_cash: float = 0.0) -> float:
    """Calculate NAV from positions and closing prices.
    
    Args:
        positions_df: DataFrame with columns: code, weight, avg_price
        closing_prices: DataFrame with columns: date, code, price
        total_equity: Total equity value for calculating position values
        initial_cash: Initial cash amount (not invested)
        
    Returns:
        NAV value (float)
    """
    if positions_df.empty:
        return initial_cash
    
    # Ensure code columns are strings for proper merging
    positions_df = positions_df.copy()
    positions_df['code'] = positions_df['code'].astype(str).str.zfill(6)
    closing_prices = closing_prices.copy()
    closing_prices['code'] = closing_prices['code'].astype(str).str.zfill(6)
    
    # Merge positions with closing prices
    merged = positions_df.merge(closing_prices, on='code', how='inner')
    
    if merged.empty:
        return initial_cash
    
    # Calculate position values from weights
    # Value = weight * total_equity * (closing_price / entry_price)
    # This gives us the current value based on price appreciation/depreciation
    if 'avg_price' in merged.columns:
        merged['price_ratio'] = merged['price'] / merged['avg_price']
        merged['value'] = merged['weight'] * total_equity * merged['price_ratio']
    else:
        # Fallback: just use weights
        merged['value'] = merged['weight'] * total_equity
    
    # Total equity value
    total_value = merged['value'].sum()
    
    return total_value + initial_cash


def run_settle(date: str, config: LiveConfig = None, total_equity: float = 1_000_000):
    """Run settlement workflow for a specific date.
    
    Args:
        date: Date string in YYYYMMDD format (e.g., '20250603')
        config: LiveConfig instance (defaults to DEFAULT_LIVE_CONFIG)
        total_equity: Total equity value for NAV calculation
    """
    if config is None:
        config = DEFAULT_LIVE_CONFIG
    
    store = StateStore(config.persistence)
    
    # Load current positions
    state = store.load_state()
    positions_df = state.positions
    
    # 1. Get closing prices and update NAV
    if not positions_df.empty:
        codes = positions_df['code'].tolist()
        closing_prices = get_closing_prices(date, codes, mock=True)
        
        # Calculate NAV
        nav = calculate_nav(positions_df, closing_prices, total_equity=total_equity, initial_cash=0.0)
        
        # For mock, use a simple approximation
        # In production, NAV should be calculated from actual closing prices and positions
        if nav == 0.0 or nav is None:
            # Fallback: use last NAV or initial equity
            if not state.nav_history.empty:
                nav = state.nav_history['nav'].iloc[-1]
            else:
                nav = total_equity
        
        # Append NAV
        settle_date = pd.Timestamp(date[:4] + '-' + date[4:6] + '-' + date[6:8])
        store.append_nav(settle_date, nav)
        store.audit(f"NAV updated for {date}", nav=nav)
    
    # 2. Calculate per-model IC
    loader = PredictionLoader(config.data)
    
    # Load predictions for this date (with model info)
    pred_dfs = []
    for model in config.data.models:
        model_path = f"{config.data.data_dir}/{model}/{date}.csv"
        try:
            df = pd.read_csv(model_path, dtype={'code': str})
            if 'code' not in df.columns or 'score' not in df.columns:
                continue
            df['code'] = df['code'].astype(str).str.zfill(6)
            df['model'] = model
            df['date'] = pd.Timestamp(date[:4] + '-' + date[4:6] + '-' + date[6:8])
            pred_dfs.append(df[['date', 'code', 'score', 'model']])
        except Exception:
            continue
    
    if not pred_dfs:
        store.audit(f"No predictions found for IC calculation on {date}")
        print(f"Warning: No predictions found for {date}, skipping IC calculation")
        return
    
    pred_df = pd.concat(pred_dfs, ignore_index=True)
    
    # Get realized returns
    if not positions_df.empty:
        codes = positions_df['code'].tolist()
        closing_prices = get_closing_prices(date, codes, mock=True)
        # Ensure positions_df code is string type
        positions_df_str = positions_df.copy()
        positions_df_str['code'] = positions_df_str['code'].astype(str).str.zfill(6)
        realized_returns = get_realized_returns(date, codes, positions_df_str, closing_prices)
    else:
        realized_returns = pd.DataFrame(columns=['date', 'code', 'return'])
    
    # Calculate per-model IC
    drift_detector = DriftDetector(config.risk)
    model_ic_df = drift_detector.compute_per_model_ic(pred_df, realized_returns, date)
    
    if not model_ic_df.empty:
        store.save_model_ic(model_ic_df)
        store.audit(f"Model IC calculated for {date}", 
                   model_count=len(model_ic_df),
                   avg_ic=float(model_ic_df['ic'].mean()))
    
    # 3. Check retraining trigger (5-day avg IC < 0.02)
    ic_history = store.load_model_ic()
    if ic_history is not None and not ic_history.empty:
        # Filter to last 5 days
        ic_history['date'] = pd.to_datetime(ic_history['date'])
        latest_dates = sorted(ic_history['date'].unique())[-5:]
        recent_ic = ic_history[ic_history['date'].isin(latest_dates)]
        
        if len(latest_dates) >= 5:
            # Calculate 5-day average IC per model
            avg_ic_by_model = recent_ic.groupby('model')['ic'].mean()
            
            # Check if any model has avg IC < threshold
            threshold = config.risk.min_ic_threshold
            low_ic_models = avg_ic_by_model[avg_ic_by_model < threshold]
            
            if len(low_ic_models) > 0:
                store.set_retrain_flag()
                store.audit("Retrain flag triggered", 
                           low_ic_models=list(low_ic_models.index),
                           avg_ics=dict(low_ic_models),
                           threshold=threshold)
                print(f"Retrain flag set: Models with low IC: {list(low_ic_models.index)}")
            else:
                # Clear flag if conditions no longer met
                if store.has_retrain_flag():
                    store.clear_retrain_flag()
                    store.audit("Retrain flag cleared: IC conditions improved")
    
    print(f"Settlement workflow completed for {date}")
    if not model_ic_df.empty:
        print(f"  Model ICs calculated: {len(model_ic_df)} models")
        for _, row in model_ic_df.iterrows():
            print(f"    {row['model']}: IC = {row['ic']:.4f}")
    
    retrain_flag = store.has_retrain_flag()
    print(f"  Retrain flag: {'SET' if retrain_flag else 'NOT SET'}")


def main():
    parser = argparse.ArgumentParser(description="Settlement workflow: update NAV, calculate IC, check retrain")
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
    
    run_settle(args.date, config=None, total_equity=args.equity)


if __name__ == "__main__":
    main()

