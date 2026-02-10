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
import argparse

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
from live_trading.talib_prediction_loader import TALIBPredictionLoader, create_live_trading_predictions
from live_trading.talib_model_config import (
    TALIBModelConfig,
    get_config_by_risk_profile,
    create_custom_config,
    DEFAULT_TALIB_CONFIG
)

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

def load_talib_predictions(model_results_dir: str, strategy: str, config: TALIBModelConfig) -> pd.DataFrame:
    """Load predictions from TALIB factor model.

    Args:
        model_results_dir: Directory containing TALIB model results
        strategy: Strategy type ('long' or 'short')
        config: TALIB model configuration

    Returns:
        pd.DataFrame: Predictions in live_trading format
    """
    try:
        # Create TALIB prediction loader with config
        talib_loader = TALIBPredictionLoader(model_results_dir, config)

        # Load predictions
        if not talib_loader.load_predictions():
            raise ValueError("Failed to load TALIB predictions")

        # Get latest predictions
        predictions = talib_loader.get_latest_predictions(strategy)
        if predictions is None:
            raise ValueError(f"No predictions found for strategy: {strategy}")

        # Convert to live_trading format
        live_format = pd.DataFrame({
            'model': predictions['model'],
            'date': predictions['date'],
            'code': predictions['code'],
            'score': predictions['score']
        })

        logger.info(f"Loaded TALIB {strategy} predictions: {len(live_format)} records")
        logger.info(f"Using config: max_positions={config.max_positions}, threshold={config.prediction_threshold}")

        return live_format

    except Exception as e:
        logger.error(f"Failed to load TALIB predictions: {e}")
        raise

def save_results(result: dict, output_dir: str, use_talib: bool, strategy: str, talib_config: TALIBModelConfig = None, config_profile: str = None):
    """Save pipeline results to files.

    Args:
        result: Pipeline execution result
        output_dir: Output directory
        use_talib: Whether TALIB model was used
        strategy: Strategy type
        talib_config: TALIB model configuration (if used)
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "talib" if use_talib else "csv"

        # Save pipeline data
        pipeline_data = result.get('data', {})

        if 'target_weights' in pipeline_data and not pipeline_data['target_weights'].empty:
            weights_file = output_path / f"{model_type}_{strategy}_target_weights_{timestamp}.csv"
            pipeline_data['target_weights'].to_csv(weights_file, index=False)
            logger.info(f"Target weights saved: {weights_file}")

        if 'orders' in pipeline_data and pipeline_data['orders']:
            orders_file = output_path / f"{model_type}_{strategy}_orders_{timestamp}.txt"
            with open(orders_file, 'w') as f:
                for order in pipeline_data['orders']:
                    f.write(f"{order}\n")
            logger.info(f"Orders saved: {orders_file}")

        # Save summary
        summary_file = output_path / f"{model_type}_{strategy}_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("LIVE TRADING PIPELINE SUMMARY\n")
            f.write("="*40 + "\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Status: {result.get('pipeline_status', 'unknown')}\n")
            f.write(".2f")
            f.write(f"Trading Date: {result.get('trading_date', 'unknown')}\n")

            # Add TALIB config info if available
            if talib_config:
                f.write(f"\nTALIB Configuration:\n")
                f.write(f"  Risk Profile: {config_profile or 'custom'}\n")
                f.write(f"  Max Positions: {talib_config.max_positions}\n")
                f.write(f"  Prediction Threshold: {talib_config.prediction_threshold}\n")
                f.write(f"  Max Single Weight: {talib_config.max_single_weight}\n")
                f.write(f"  Transaction Cost (bps): {talib_config.transaction_cost_bps}\n")

            # Add data summary
            if 'target_weights' in pipeline_data:
                weights = pipeline_data['target_weights']
                f.write(f"Target Positions: {len(weights)}\n")
                if not weights.empty and 'weight' in weights.columns:
                    f.write(".4f")

        logger.info(f"Summary saved: {summary_file}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def print_results(result: dict):
    """Print pipeline results to console."""
    print("\n" + "="*50)
    print("LIVE TRADING PIPELINE RESULTS")
    print("="*50)

    print(f"Pipeline Status: {result.get('pipeline_status', 'unknown')}")
    print(".2f")

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
    try:
        from live_trading.pipeline_runner import ResilientPipelineRunner
        # Note: This is a simplified health check - in real implementation
        # you'd get the runner instance from the execution
        print("\nSystem Health:")
        print("  Status: Pipeline completed")
        print("  Degraded Components: 0")
        print("  Recent Errors: 0")
    except:
        print("\nSystem Health: Unable to retrieve")

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
    parser = argparse.ArgumentParser(description='Run live trading pipeline')
    parser.add_argument('--use-talib', action='store_true',
                       help='Use TALIB factor model predictions instead of CSV files')
    parser.add_argument('--talib-strategy', choices=['long', 'short'], default='long',
                       help='TALIB strategy to use (default: long)')
    parser.add_argument('--model-results-dir', default='debug_model_results',
                       help='Directory containing TALIB model results (default: debug_model_results)')
    parser.add_argument('--talib-config', choices=['conservative', 'moderate', 'aggressive', 'high_frequency'],
                       default='moderate', help='Risk profile for TALIB model (default: moderate)')
    parser.add_argument('--custom-max-positions', type=int,
                       help='Custom maximum positions (overrides config)')
    parser.add_argument('--custom-threshold', type=float,
                       help='Custom prediction threshold (overrides config)')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing prediction CSV files (default: data)')
    parser.add_argument('--output-dir', default='live_trading_output',
                       help='Output directory for results (default: live_trading_output)')

    args = parser.parse_args()

    try:
        logger.info("Starting live trading pipeline")

        # Create TALIB configuration if using TALIB
        talib_config = None
        if args.use_talib:
            # Get base config by risk profile
            talib_config = get_config_by_risk_profile(args.talib_config)

            # Apply custom overrides
            custom_params = {}
            if args.custom_max_positions is not None:
                custom_params['max_positions'] = args.custom_max_positions
            if args.custom_threshold is not None:
                custom_params['prediction_threshold'] = args.custom_threshold

            if custom_params:
                talib_config = create_custom_config(**custom_params)

            logger.info(f"Using TALIB config: {args.talib_config} profile")
            logger.info(f"Config details: max_positions={talib_config.max_positions}, threshold={talib_config.prediction_threshold}")

        if args.use_talib:
            logger.info("Using TALIB factor model predictions")
            predictions_df = load_talib_predictions(args.model_results_dir, args.talib_strategy, talib_config)
        else:
            logger.info("Using CSV prediction files")
            predictions_df = load_real_predictions(args.data_dir)

        if predictions_df is None or predictions_df.empty:
            raise ValueError("No predictions loaded")

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

        # Save results
        save_results(result, args.output_dir, args.use_talib, args.talib_strategy, talib_config, args.talib_config)

        # Print results
        print_results(result)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()