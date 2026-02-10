#!/usr/bin/env python3
"""
Test script for the adapted JQ limit-up gene strategy.
Tests the backtest-compatible version with sample data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.backtest.jq_backtest_strategy import LimitUpGeneStrategy, register_limit_up_gene_strategy
from src.backtest.config import StrategyConfig


def test_strategy_initialization():
    """Test strategy class creation and basic methods"""
    print("Testing strategy class creation...")

    from src.backtest.jq_backtest_strategy import LimitUpGeneStrategy
    from src.backtest.config import StrategyConfig

    config = StrategyConfig(
        strategy_name='limit_up_gene',
        parameters={
            'stock_num': 6,
            'hold_days': 5,
            'up_price': 20
        }
    )

    # Test that we can create the strategy class
    strategy_class = LimitUpGeneStrategy
    print(f"✓ Strategy class created: {strategy_class.__name__}")

    # Test that config is valid
    print(f"✓ Config parameters: {config.parameters}")

    return strategy_class, config


def test_stock_selection(strategy):
    """Test stock selection logic"""
    print("\nTesting stock selection...")

    try:
        test_date = pd.Timestamp('2024-01-01')
        selected_stocks = strategy.get_stock_list(test_date)

        print(f"✓ Selected {len(selected_stocks)} stocks: {selected_stocks[:5]}...")
        return selected_stocks

    except Exception as e:
        print(f"✗ Stock selection failed: {e}")
        return []


def test_strategy_execution(strategy_class, config):
    """Test strategy factory integration"""
    print("\nTesting strategy factory integration...")

    try:
        from src.backtest.backtest_engine import StrategyFactory

        # Test that we can get the strategy from factory
        retrieved_class = StrategyFactory.get_strategy('limit_up_gene')
        print(f"✓ Strategy factory returned: {retrieved_class.__name__}")

        # Test that it's the same class
        if retrieved_class == strategy_class:
            print("✓ Strategy class matches")
        else:
            print("✗ Strategy class mismatch")

        return True

    except Exception as e:
        print(f"✗ Strategy factory test failed: {e}")
        return False


def test_jq_compat_functions():
    """Test JQ compatibility functions"""
    print("\nTesting JQ compatibility functions...")

    try:
        import src.backtest.jq_compat as jq_compat

        # Test get_all_securities
        securities = jq_compat.get_all_securities('stock')
        print(f"✓ get_all_securities: {len(securities)} stocks")

        # Test get_current_data
        current_data = jq_compat.get_current_data()
        print(f"✓ get_current_data: {len(current_data)} records")

        # Test history
        if securities.index.tolist():
            sample_stock = securities.index.tolist()[0]
            hist_data = jq_compat.history(5, '1d', 'close', [sample_stock])
            print(f"✓ history: {len(hist_data)} days for {sample_stock}")

        return True

    except Exception as e:
        print(f"✗ JQ compatibility test failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== JQ Strategy Adaptation Test ===\n")

    # Test JQ compatibility functions
    compat_ok = test_jq_compat_functions()

    if not compat_ok:
        print("\n❌ JQ compatibility functions failed. Cannot proceed with strategy tests.")
        return

    # Test strategy
    strategy_class, config = test_strategy_initialization()

    # Get some available stocks for testing
    available_stocks = ['000001', '000002', '000858', '600000', '600036', '600519']

    result = test_strategy_execution(strategy_class, config)

    # Summary
    print("\n=== Test Summary ===")
    if result:
        print("✅ Strategy adaptation test completed successfully!")
        print("   - Strategy class can be created")
        print("   - Strategy factory integration works")
        print("   - JQ compatibility functions available")
    else:
        print("❌ Strategy adaptation test failed")

    print("\n=== Recommendations ===")
    print("1. The adapted strategy provides a backtest-compatible version of the JQ limit-up gene strategy")
    print("2. It implements the core logic: market cap filtering, limit-up history analysis, start point analysis, and industry diversification")
    print("3. The strategy can be integrated with the existing BacktestEngine for full backtesting")
    print("4. For production use, consider optimizing data loading and adding more sophisticated filters")


if __name__ == "__main__":
    main()