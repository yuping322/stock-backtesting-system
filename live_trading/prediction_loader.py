"""Utilities to aggregate latest prediction files into a unified DataFrame.

Assumptions:
- Each prediction file has columns: date, code, weight (extra columns allowed, ignored by default)
- Files reside under `data_dir` following a naming convention, but we accept any CSV matching pattern.
- Codes are 6-digit raw; normalization to exchange form delegated to existing data utilities if needed.

This module avoids importing heavy backtest modules to stay lightweight for live scheduling.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .live_config import DataIngestionConfig


class PredictionLoader:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def list_files(self) -> List[str]:
        pattern = os.path.join(self.config.data_dir, self.config.file_pattern)
        return sorted(glob.glob(pattern))

    def load_latest(self) -> pd.DataFrame:
        files = self.list_files()
        if not files:
            return pd.DataFrame(columns=self.config.required_columns)

        # load all, filter by latest N distinct dates
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if not set(self.config.required_columns).issubset(df.columns):
                continue
            dfs.append(df[self.config.required_columns].copy())
        if not dfs:
            return pd.DataFrame(columns=self.config.required_columns)

        all_df = pd.concat(dfs, ignore_index=True)
        all_df['date'] = pd.to_datetime(all_df['date']).dt.normalize()
        # choose latest distinct dates
        unique_dates = sorted(all_df['date'].unique())
        latest_dates = unique_dates[-self.config.latest_days:]
        live_df = all_df[all_df['date'].isin(latest_dates)].copy()

        if self.config.allow_missing_weight and 'weight' in live_df.columns:
            live_df['weight'] = live_df['weight'].fillna(1.0)
        if self.config.normalize_codes:
            live_df['code'] = live_df['code'].astype(str).str.zfill(6)

        # aggregate duplicates (if multiple models produce same code/date) by mean
        live_df = live_df.groupby(['date', 'code'], as_index=False)['weight'].mean()

        return live_df.sort_values(['date', 'weight'], ascending=[True, False]).reset_index(drop=True)

    def load_ensemble(self, date: str, models: List[str], top_n: int = 50) -> pd.DataFrame:
        """Load ensemble predictions from multiple models for a specific date.
        
        Args:
            date: Date string in YYYYMMDD format (e.g., '20250603')
            models: List of model directory names (e.g., ['model_a', 'model_b'])
            top_n: Number of top stocks to select
            
        Returns:
            DataFrame with columns: code, weight (normalized weights, sum to 1)
        """
        dfs = []
        for model in models:
            model_path = os.path.join(self.config.data_dir, model, f'{date}.csv')
            if not os.path.exists(model_path):
                continue
            try:
                df = pd.read_csv(model_path, dtype={'code': str})
                if 'code' not in df.columns or 'score' not in df.columns:
                    continue
                df['code'] = df['code'].astype(str).str.zfill(6)
                df['model'] = model
                dfs.append(df[['code', 'score', 'model']])
            except Exception as e:
                continue
        
        if not dfs:
            return pd.DataFrame(columns=['code', 'weight'])
        
        # Concatenate all model predictions
        df = pd.concat(dfs, ignore_index=True)
        
        # Step 1: Score -> Weight: rank(pct=True) - 0.5 (neutralized)
        df['weight'] = df.groupby('model')['score'].transform(
            lambda x: x.rank(pct=True) - 0.5
        )
        
        # Step 2: Equal-weight ensemble across models
        out = df.groupby('code')['weight'].mean().reset_index()
        
        # Step 3: Select Top-N with weight >= 0
        out = out.nlargest(top_n, 'weight')
        out = out[out['weight'] >= 0].copy()
        
        if out.empty:
            return pd.DataFrame(columns=['code', 'weight'])
        
        # Step 4: Normalize to sum to 1
        total_weight = out['weight'].sum()
        if total_weight > 0:
            out['weight'] = out['weight'] / total_weight
        else:
            out['weight'] = 1.0 / len(out)
        
        return out[['code', 'weight']].sort_values('weight', ascending=False).reset_index(drop=True)
