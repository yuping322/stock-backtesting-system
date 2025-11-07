#!/usr/bin/env python3
"""Run live trading pipeline with real data from data/ directory.

This script loads prediction data from CSV files in the data/ directory,
converts them to the format expected by live_trading, and runs the full pipeline.
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import live_trading
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.data_provider import DataProvider, ProviderConfig
from live_trading.prediction_loader import PredictionLoader
from live_trading.portfolio_builder import PortfolioBuilder
from live_trading.risk_manager import CompositeRiskEngine, BlacklistRule, MarketCapRule, LiquidityRule
from live_trading.broker_adapter import XueqiuAdapter
from live_trading.execution_engine import ExecutionEngine
from live_trading.state_store import StateStore
from live_trading.drift_detector import AdvancedDriftDetector
from live_trading.pipeline_runner import ResilientPipelineRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_predictions(data_dir: str = "data", file_pattern: str = "*.csv") -> pd.DataFrame:
    """Load prediction data from CSV files in data directory.

    Converts from backtest format (date, code, weight) to live_trading format (model, date, code, score).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Find all CSV files
    csv_files = list(data_path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    # Sort by modification time, use latest
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = csv_files[0]

    logger.info(f"Loading predictions from: {latest_file}")

    # Load CSV
    df = pd.read_csv(latest_file)

    # Validate required columns
    required_cols = ['date', 'code']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    # Convert to live_trading format
    predictions_df = pd.DataFrame({
        'model': 'default',  # Use default model name
        'date': pd.to_datetime(df['date']).dt.strftime('%Y%m%d'),  # Ensure YYYYMMDD format
        'code': df['code'].astype(str).str.zfill(6),  # Ensure 6-digit codes
        'score': df.get('weight', 1.0)  # Use weight as score, default to 1.0
    })

    logger.info(f"Loaded {len(predictions_df)} predictions for {predictions_df['date'].nunique()} dates")
    return predictions_df

def create_components():
    """Create all pipeline components with default configurations."""

    # Data provider
    provider_config = ProviderConfig(
        universe_mode='hs300',  # Use HS300 for smaller universe in demo
        allow_offline_fallback=True
    )
    data_provider = DataProvider(provider_config)

    # Prediction loader
    prediction_loader = PredictionLoader()

    # Portfolio builder
    portfolio_builder = PortfolioBuilder()

    # Risk manager with default rules
    risk_rules = [
        BlacklistRule(),
        MarketCapRule(min_market_cap=1e9),  # 1 billion RMB minimum
        LiquidityRule(min_volume=1e6)       # 1 million shares minimum volume
    ]
    risk_engine = CompositeRiskEngine(rules=risk_rules)

    # Execution engine with broker adapter
    broker_adapter = XueqiuAdapter()  # Placeholder adapter
    execution_engine = ExecutionEngine(broker_adapter=broker_adapter)

    # State store
    state_store = StateStore()

    # Drift detector
    drift_detector = AdvancedDriftDetector()

    return data_provider, prediction_loader, portfolio_builder, risk_engine, execution_engine, state_store, drift_detector

def main():
    """Main execution function."""
    try:
        logger.info("Starting live trading pipeline with real data")

        # Load real predictions
        predictions_df = load_real_predictions()

        # Get unique dates and use the latest
        dates = sorted(predictions_df['date'].unique())
        if not dates:
            raise ValueError("No dates found in predictions")

        trading_date = dates[-1]  # Use latest date
        logger.info(f"Using trading date: {trading_date}")

        # Filter predictions for this date
        date_predictions = predictions_df[predictions_df['date'] == trading_date].copy()
        logger.info(f"Found {len(date_predictions)} predictions for {trading_date}")

        # Create components
        components = create_components()
        data_provider, prediction_loader, portfolio_builder, risk_engine, execution_engine, state_store, drift_detector = components

        # Create pipeline runner
        runner = ResilientPipelineRunner(
            data_provider=data_provider,
            prediction_loader=prediction_loader,
            portfolio_builder=portfolio_builder,
            risk_engine=risk_engine,
            execution_engine=execution_engine,
            state_store=state_store,
            drift_detector=drift_detector,
            max_retries=3,
            enable_degraded_mode=True
        )

        # Run pipeline
        logger.info("Running pipeline...")
        result = runner.run(trading_date, date_predictions)

        # Print results
        print("\n" + "="*50)
        print("LIVE TRADING PIPELINE RESULTS")
        print("="*50)

        print(f"Pipeline Status: {result.get('pipeline_status', 'unknown')}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f} seconds")

        recovery_info = result.get('recovery_info', {})
        print(f"Errors Encountered: {len(recovery_info.get('errors_encountered', []))}")
        print(f"Recoveries Attempted: {len(recovery_info.get('recoveries_attempted', []))}")

        # Print target weights if available
        pipeline_data = result.get('data', {})
        if 'target_weights' in pipeline_data and not pipeline_data['target_weights'].empty:
            print(f"\nTarget Weights ({len(pipeline_data['target_weights'])} positions):")
            weights_df = pipeline_data['target_weights']
            if 'code' in weights_df.columns and 'weight' in weights_df.columns:
                for _, row in weights_df.head(10).iterrows():
                    print(".4f")
        else:
            print("\nNo target weights generated")

        # Print filtered weights if available
        if 'filtered_weights' in pipeline_data and not pipeline_data['filtered_weights'].empty:
            print(f"\nFiltered Weights ({len(pipeline_data['filtered_weights'])} positions):")
            filtered_df = pipeline_data['filtered_weights']
            if 'code' in filtered_df.columns and 'weight' in filtered_df.columns:
                for _, row in filtered_df.head(10).iterrows():
                    print(".4f")
        else:
            print("No filtered weights generated")

        # Print aggregated predictions info
        if 'aggregated_predictions' in pipeline_data:
            agg_pred = pipeline_data['aggregated_predictions']
            if not agg_pred.empty:
                print(f"\nAggregated Predictions ({len(agg_pred)} stocks):")
                print(agg_pred.head(5))
            else:
                print("No aggregated predictions")

        # Print orders if available
        if 'orders' in pipeline_data:
            orders = pipeline_data['orders']
            print(f"\nOrders Generated: {len(orders)}")
            for order in orders[:5]:  # Show first 5
                print(f"  {order}")
        else:
            print("No orders generated")

        # Print health status
        health = runner.get_health_status()
        print("\nSystem Health:")
        print(f"  Status: {health.get('system_status', 'unknown')}")
        print(f"  Degraded Components: {len(health.get('degraded_components', []))}")
        print(f"  Recent Errors: {health.get('recent_errors', 0)}")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()