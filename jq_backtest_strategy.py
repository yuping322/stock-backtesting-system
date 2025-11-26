"""
Backtest-compatible version of the JQ limit-up gene stock pool rotation strategy.
Adapted from strategies/jq.py for use with the current stock backtesting framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import framework components
from backtest_engine import BacktestEngine, BaseStrategy, StrategyFactory
from config import SystemConfig, StrategyConfig
import data
import jq_compat  # Import our compatibility layer


class LimitUpGeneStrategy(BaseStrategy):
    """
    Backtest-compatible version of the limit-up gene stock rotation strategy.
    Implements multi-stage stock selection based on historical limit-up patterns.
    """

    def __init__(self):
        super().__init__()
        # Strategy parameters from config
        self.stock_num = getattr(self.config, 'stock_num', 6)
        self.up_price = getattr(self.config, 'up_price', 20)
        self.limit_days_window = getattr(self.config, 'limit_days_window', 3 * 250)
        self.init_stock_count = getattr(self.config, 'init_stock_count', 1000)
        self.hold_days = getattr(self.config, 'hold_days', 5)  # Rebalance every N days

        # State variables
        self.hold_list = []
        self.target_list = []
        self.not_buy_again = []
        self.last_rebalance_date = None

    def initialize(self):
        """Initialize strategy parameters"""
        print("Initializing Limit-Up Gene Strategy")
        print(f"Stock num: {self.stock_num}, Hold days: {self.hold_days}")

    def prepare_stock_list(self, current_date: pd.Timestamp) -> List[str]:
        """
        Prepare the initial stock pool based on market cap filtering.
        Equivalent to prepare_stock_list in original JQ strategy.
        """
        try:
            # Get all securities
            all_securities = jq_compat.get_all_securities('stock', current_date)
            if all_securities.empty:
                print("No securities data available")
                return []

            initial_list = all_securities.index.tolist()[:self.init_stock_count]
            print(f"Initial stock pool: {len(initial_list)} stocks")

            # Filter out new stocks, ST stocks, paused stocks
            initial_list = self._filter_basic_stocks(initial_list, current_date)

            # Get valuation data for market cap sorting
            market_cap_data = {}
            for code in initial_list[:100]:  # Limit to avoid too many API calls
                try:
                    val_df = data.get_valuation(code, date=current_date)
                    if not val_df.empty and 'circulating_cap' in val_df.columns:
                        market_cap = val_df['circulating_cap'].iloc[0]
                        if market_cap and market_cap > 0:
                            market_cap_data[code] = market_cap
                except Exception as e:
                    continue

            # Sort by market cap ascending and take top init_stock_count
            sorted_by_cap = sorted(market_cap_data.items(), key=lambda x: x[1])
            initial_list = [code for code, _ in sorted_by_cap[:self.init_stock_count]]

            print(f"After market cap filtering: {len(initial_list)} stocks")
            return initial_list

        except Exception as e:
            print(f"Error in prepare_stock_list: {e}")
            return []

    def get_stock_list(self, current_date: pd.Timestamp) -> List[str]:
        """
        Main stock selection logic - equivalent to get_stock_list in JQ strategy.
        """
        try:
            # Get initial stock pool
            initial_list = self.prepare_stock_list(current_date)

            if not initial_list:
                return []

            # Filter limit-up and limit-down stocks
            initial_list = self._filter_limit_stocks(initial_list, current_date)

            # Get historical limit-up frequency
            limit_up_stocks = self._get_history_highlimit(initial_list, current_date)

            # Get start point analysis (price bias from historical lows)
            start_point_stocks = self._get_start_point_analysis(limit_up_stocks, current_date)

            # Get industry diversification
            final_list = self._get_industry_diversified_stocks(start_point_stocks)

            print(f"Selected {len(final_list)} stocks for date {current_date.date()}")
            return final_list[:self.stock_num * 2]  # Return more for rebalancing flexibility

        except Exception as e:
            print(f"Error in get_stock_list: {e}")
            return []

    def execute_strategy(self):
        """
        Main strategy execution logic - called by Backtrader's next() method.
        This method must be implemented by all BaseStrategy subclasses.
        """
        try:
            current_date = pd.Timestamp(self.current_dt.date())

            # Check if we need to rebalance
            if self._should_rebalance(current_date):
                self.target_list = self.get_stock_list(current_date)
                self._execute_rebalance(current_date)
                self.last_rebalance_date = current_date

            # Execute trades based on target holdings
            self._execute_trades(current_date)

        except Exception as e:
            print(f"Error in execute_strategy: {e}")

    def _should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if we should rebalance based on hold_days parameter"""
        if self.last_rebalance_date is None:
            return True

        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.hold_days

    def _execute_trades(self, current_date: pd.Timestamp):
        """Execute actual trades based on target holdings"""
        if not self.hold_list:
            return

        total_value = self.broker.getvalue()
        if total_value == 0:
            return

        target_value_per_stock = total_value / len(self.hold_list)

        for stock in self.hold_list:
            data_feed = self._find_data(stock)
            if data_feed is None:
                continue

            position = self.broker.getposition(data_feed)
            current_value = position.size * data_feed.close[0] if position.size > 0 else 0

            if abs(target_value_per_stock - current_value) / total_value > 0.01:  # 1% threshold
                size_change = int((target_value_per_stock - current_value) / data_feed.close[0])
                if abs(size_change) > 0:
                    action = "BUY" if size_change > 0 else "SELL"
                    self.order_target_value(data_feed, target_value_per_stock)
                    traded_value = abs(size_change) * data_feed.close[0]
                    self._record_trade(stock, action, size_change, data_feed.close[0], traded_value)
                    stock_name = getattr(data_feed, '_name', stock)
                    self.log(f"{action} {stock} {stock_name} target_value={target_value_per_stock:.2f}")

    def _execute_rebalance(self, current_date: pd.Timestamp):
        """Execute rebalancing trades"""
        try:
            # Update target holdings
            self.hold_list = self.target_list[:self.stock_num] if self.target_list else []

            # Record holdings for this date
            holdings = {}
            for stock in self.hold_list:
                holdings[stock] = 1.0 / len(self.hold_list) if self.hold_list else 0

            self._record_holdings(current_date, holdings)

            print(f"Rebalanced to {len(self.hold_list)} stocks: {self.hold_list}")

        except Exception as e:
            print(f"Error in _execute_rebalance: {e}")

    def _filter_basic_stocks(self, stock_list: List[str], current_date: pd.Timestamp) -> List[str]:
        """Filter out new stocks, ST stocks, and paused stocks"""
        filtered = []

        for stock in stock_list:
            try:
                # Check if it's a new stock (listed less than 375 days ago)
                security_info = jq_compat.get_security_info(stock)
                if security_info and security_info.start_date:
                    days_since_listing = (current_date.date() - security_info.start_date).days
                    if days_since_listing < 375:
                        continue

                # Check ST status - skip if network access fails
                try:
                    extras_data = jq_compat.get_extras('is_st', [stock], df=True)
                    if not extras_data.empty and extras_data.iloc[0].get('is_st', False):
                        continue
                except Exception as e:
                    print(f"Warning: Skipping ST check for {stock} due to network error: {e}")
                    # Continue without ST filtering

                # Check if paused - skip if network access fails
                try:
                    current_data = jq_compat.get_current_data()
                    if stock in current_data and current_data[stock].paused:
                        continue
                except Exception as e:
                    print(f"Warning: Skipping pause check for {stock} due to network error: {e}")
                    # Continue without pause filtering

                # Check price filter
                try:
                    price_data = jq_compat.history(1, unit='1d', field='close', security_list=[stock], df=True)
                    if stock in price_data and not price_data[stock].empty:
                        last_price = price_data[stock].iloc[-1]
                        if last_price <= self.up_price:
                            filtered.append(stock)
                    else:
                        # If no price data, include the stock anyway
                        filtered.append(stock)
                except Exception as e:
                    print(f"Warning: Skipping price check for {stock} due to network error: {e}")
                    # Include stock if price check fails
                    filtered.append(stock)

            except Exception as e:
                print(f"Warning: Error processing {stock}: {e}")
                continue

        return filtered

    def _filter_limit_stocks(self, stock_list: List[str], current_date: pd.Timestamp) -> List[str]:
        """Filter out stocks that hit limit up or limit down recently"""
        filtered = []

        for stock in stock_list:
            try:
                # Get recent price data - skip if network fails
                try:
                    price_data = jq_compat.history(2, unit='1d', field='close', security_list=[stock], df=True)
                    if stock not in price_data or price_data[stock].empty:
                        # Include stock if no price data
                        filtered.append(stock)
                        continue
                except Exception as e:
                    print(f"Warning: Skipping limit check for {stock} due to network error: {e}")
                    # Include stock if price check fails
                    filtered.append(stock)
                    continue

                # Get current data - skip if network fails
                try:
                    current_data = jq_compat.get_current_data()
                    if stock not in current_data:
                        # Include stock if no current data
                        filtered.append(stock)
                        continue
                except Exception as e:
                    print(f"Warning: Skipping current data check for {stock} due to network error: {e}")
                    # Include stock if current data check fails
                    filtered.append(stock)
                    continue

                last_price = price_data[stock].iloc[-1]
                high_limit = current_data[stock].high_limit
                low_limit = current_data[stock].low_limit

                # Skip if currently at limit up
                if last_price >= high_limit:
                    continue

                # Skip if at limit down
                if last_price <= low_limit:
                    continue

                filtered.append(stock)

            except Exception as e:
                print(f"Warning: Error in limit filter for {stock}: {e}")
                # Include stock on error
                filtered.append(stock)

        return filtered

    def _get_history_highlimit(self, stock_list: List[str], current_date: pd.Timestamp) -> List[str]:
        """Get stocks with high historical limit-up frequency"""
        try:
            limit_counts = {}

            # Calculate lookback period
            start_date = current_date - pd.Timedelta(days=self.limit_days_window)

            for stock in stock_list[:50]:  # Limit to avoid too many API calls
                try:
                    # Get historical price data
                    price_data = data.load_oss_stocks(
                        codes=[stock],
                        start=start_date.date(),
                        end=current_date.date()
                    )

                    if price_data.empty:
                        continue

                    # Count limit-up days (simplified: close == high * 1.1)
                    close_prices = price_data['close']
                    high_prices = price_data['high']

                    # Simple approximation: close very close to high
                    limit_up_days = ((close_prices - high_prices) / high_prices < 0.001).sum()

                    if limit_up_days > 0:
                        limit_counts[stock] = limit_up_days

                except Exception as e:
                    continue

            # Sort by limit-up frequency and take top 10%
            sorted_stocks = sorted(limit_counts.items(), key=lambda x: x[1], reverse=True)
            top_10_percent = max(1, int(len(sorted_stocks) * 0.1))

            return [stock for stock, _ in sorted_stocks[:top_10_percent]]

        except Exception as e:
            print(f"Error in _get_history_highlimit: {e}")
            return stock_list

    def _get_start_point_analysis(self, stock_list: List[str], current_date: pd.Timestamp) -> List[str]:
        """Analyze price bias from historical start points (simplified version)"""
        try:
            price_bias = {}

            for stock in stock_list[:30]:  # Limit API calls
                try:
                    # Get historical data for analysis
                    hist_data = data.load_oss_stocks(
                        codes=[stock],
                        start=(current_date - pd.Timedelta(days=250)).date(),
                        end=current_date.date()
                    )

                    if hist_data.empty:
                        continue

                    # Find recent low points (simplified start point analysis)
                    recent_low = hist_data['low'].min()
                    current_price = hist_data['close'].iloc[-1]

                    if recent_low > 0:
                        bias = current_price / recent_low
                        price_bias[stock] = bias

                except Exception as e:
                    continue

            # Sort by price bias (prefer stocks closer to historical lows)
            sorted_stocks = sorted(price_bias.items(), key=lambda x: x[1])

            return [stock for stock, _ in sorted_stocks]

        except Exception as e:
            print(f"Error in _get_start_point_analysis: {e}")
            return stock_list

    def _get_industry_diversified_stocks(self, stock_list: List[str]) -> List[str]:
        """Apply industry diversification (simplified to select different industries)"""
        try:
            industry_groups = {}

            for stock in stock_list:
                try:
                    industry_data = jq_compat.get_industry([stock])
                    if stock in industry_data:
                        industry = industry_data[stock].get('sw_l1', '其他')
                        if industry not in industry_groups:
                            industry_groups[industry] = []
                        industry_groups[industry].append(stock)
                    else:
                        # If no industry data, put in '其他' category
                        if '其他' not in industry_groups:
                            industry_groups['其他'] = []
                        industry_groups['其他'].append(stock)
                except Exception as e:
                    print(f"Warning: Failed to get industry for {stock}: {e}")
                    # Put in '其他' category on error
                    if '其他' not in industry_groups:
                        industry_groups['其他'] = []
                    industry_groups['其他'].append(stock)

            # Select top stock from each industry
            diversified_stocks = []
            for industry, stocks in industry_groups.items():
                if stocks:
                    diversified_stocks.append(stocks[0])  # Take first stock from each industry

            return diversified_stocks[:10]  # Limit to 10 different industries

        except Exception as e:
            print(f"Error in _get_industry_diversified_stocks: {e}")
            # Return original list if industry diversification fails
            return stock_list[:10]


# Register the strategy
def register_limit_up_gene_strategy():
    """Register the limit-up gene strategy with the factory"""
    from backtest_engine import StrategyFactory
    StrategyFactory._strategies['limit_up_gene'] = LimitUpGeneStrategy
    print("Limit-up gene strategy registered successfully")


if __name__ == "__main__":
    # Test the strategy
    register_limit_up_gene_strategy()

    # Create a simple test
    config = StrategyConfig(
        strategy_name='limit_up_gene',
        parameters={
            'stock_num': 6,
            'hold_days': 5,
            'up_price': 20
        }
    )

    # Note: In actual backtesting, strategy is instantiated by Backtrader with config passed via params
    # This test is simplified and won't work with the current BaseStrategy structure
    print("Strategy class defined successfully")
    print("For actual testing, use the backtesting framework with main.py")