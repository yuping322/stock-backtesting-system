import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from factor_old.factor import FactorTester, CFG
from factor_old.factor_calculator import FileFactorCalculator

class AdaptableFileFactorCalculator(FileFactorCalculator):
    """
    A subclass of FileFactorCalculator that handles 'stock' column name by renaming it to 'code'.
    """
    def _load_file(self) -> pd.DataFrame:
        """Load factor file with column adaptation"""
        try:
            # Read CSV without specifying dtype to let pandas infer
            # We read it here to rename columns before the parent class validation logic would run
            # But since we are overriding the whole method, we just reimplement the loading logic
            # copying the core logic from FileFactorCalculator._load_file but adding the rename
            
            df = pd.read_csv(self.file_path)
            
            # Rename 'stock' to 'code' if present
            if 'stock' in df.columns and 'code' not in df.columns:
                df.rename(columns={'stock': 'code'}, inplace=True)
            
            # Check necessary columns
            if 'date' not in df.columns or 'code' not in df.columns:
                raise ValueError(f"File must contain 'date' and 'code' columns (or 'stock')")
            
            if self.factor_name not in df.columns:
                raise ValueError(f"File must contain factor column '{self.factor_name}'")
            
            # Date conversion
            df['date'] = pd.to_datetime(df['date'])
            
            # Normalize code
            df['code'] = df['code'].astype(str).str.strip()
            df['code'] = df['code'].apply(self._normalize_code)
            df['code'] = df['code'].str.zfill(6)
            
            # Set MultiIndex
            df = df.set_index(['date', 'code']).sort_index()
            
            # Record metadata
            if not df.empty:
                dates = df.index.get_level_values('date')
                self._file_date_range = (dates.min(), dates.max())
                self._file_stocks = sorted(df.index.get_level_values('code').unique().tolist())
            
            return df
            
        except Exception as e:
            print(f"Failed to load factor file {self.file_path}: {e}")
            return pd.DataFrame()

def main():
    # 1. Configuration
    # Factors from the notebook
    FACTOR_GROUPS = [
        (['sales_to_price_ratio', 'share_turnover_monthly', 'natural_log_of_market_cap'], [0.0002666821, -0.0020518674, -0.0023101097], 0.0507803593),
        (['size', 'roe_ttm', 'current_asset_turnover_rate'], [-0.0008979094, -0.0000039691, 0.0002272270], -0.0003337466),
        (['VOL10', 'single_day_VPT_12'], [-0.0006370810, -0.0000001720], 0.0027864796),
        (['adjusted_profit_to_total_profit'], [-0.0000013402], 0.0013302010),
        (['super_quick_ratio', 'cube_of_size', 'cfo_to_ev'], [0.0000357266, -0.0003667557, 0.0130488065], 0.0002890622),
        (['cash_to_current_liability', 'operating_tax_to_operating_revenue_ratio_ttm', 'Price3M'], [-0.0003459985, 0.0010498108, -0.0233277951], 0.0013685457),
        (['liquidity', 'roa_ttm'], [-0.0027426855, -0.0027239563], 0.0013502358),
        (['VSTD10', 'ROC60'], [-0.0000000004, -0.0000823648], 0.0013880176),
    ]
    
    # Flatten factor list
    all_factors = set()
    for factors, _, _ in FACTOR_GROUPS:
        all_factors.update(factors)
    all_factors = list(all_factors)
    
    # Data file path
    data_path = os.path.join(project_root, 'data', 'model_tasks', 'formatted_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    # 2. Setup FactorTester
    class Args:
        start = '2020-01-01' 
        end = datetime.now().strftime('%Y-%m-%d')
        stock_pool = 'stock'
        factors = all_factors
        quantiles = 5
        periods = [5]
        fillna = 0
        winsorize = 0
        neutralize = 0
        standardize = 0
        roll_win = 20
        monitor_csv = 'monitor.csv'
        last_only = False
        factor_dir = None
        max_stocks = None

    cfg = CFG(Args())
    
    # Create AdaptableFileFactorCalculators for each factor
    custom_factors = {}
    for f in all_factors:
        custom_factors[f] = AdaptableFileFactorCalculator(data_path, f)
            
    tester = FactorTester(cfg, custom_factors)
    
    # 3. Get Factors
    factor_data = tester.get_factors()
    
    if not factor_data:
        print("No factor data retrieved.")
        return

    # 4. Combine and Score
    print("Combining factors and scoring...")
    
    df_factors = pd.DataFrame(factor_data)
    
    for f in all_factors:
        if f not in df_factors.columns:
            print(f"Warning: Factor {f} missing from retrieved data. Filling with 0.")
            df_factors[f] = 0.0
            
    df_factors['total_score'] = 0.0
    
    for factors, coefs, intercept in FACTOR_GROUPS:
        group_score = intercept
        for f, c in zip(factors, coefs):
            if f in df_factors.columns:
                group_score += df_factors[f].fillna(0) * c
        df_factors['total_score'] += group_score
        
    # 5. Select Stocks
    print("Selecting stocks...")
    results = []
    
    df_scored = df_factors.reset_index()
    if 'asset' in df_scored.columns:
        df_scored.rename(columns={'asset': 'code'}, inplace=True)
        
    top_n = 30
    
    for date, group in df_scored.groupby('date'):
        sorted_group = group.sort_values('total_score', ascending=False)
        top_picks = sorted_group.head(top_n)
        weight = 1.0 / len(top_picks)
        
        for _, row in top_picks.iterrows():
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'code': row['code'],
                'weight': weight,
                'score': row['total_score']
            })
            
    # 6. Save
    output_file = 'jq_migration_predictions.csv'
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    main()
