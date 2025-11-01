"""Portfolio construction logic for live trading.

Takes aggregated prediction DataFrame (date, code, weight) and produces a target
portfolio for the latest date with constraints: top-N, per-stock cap, industry cap,
minimum weight threshold, normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

from .live_config import PortfolioConfig


@dataclass
class PortfolioResult:
    date: pd.Timestamp
    target_weights: pd.DataFrame  # columns: code, weight
    meta: Dict[str, float]


class PortfolioBuilder:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self._industry_map = self._load_industry_map(config.industry_map_file) if config.industry_map_file else {}

    def _load_industry_map(self, path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        try:
            df = pd.read_csv(path)
        except Exception:
            return {}
        # attempt to detect columns
        code_col = 'code' if 'code' in df.columns else df.columns[0]
        ind_col = 'industry' if 'industry' in df.columns else df.columns[-1]
        mapping = dict(zip(df[code_col].astype(str).str.zfill(6), df[ind_col].astype(str)))
        return mapping

    def build(self, pred_df: pd.DataFrame) -> Optional[PortfolioResult]:
        if pred_df.empty:
            return None
        # ensure date column is datetime for consistency
        if not pd.api.types.is_datetime64_any_dtype(pred_df['date']):
            try:
                pred_df['date'] = pd.to_datetime(pred_df['date']).dt.normalize()
            except Exception:
                # fallback: leave as-is
                pass
        latest_date = pred_df['date'].max()
        today_df = pred_df[pred_df['date'] == latest_date].copy()
        if today_df.empty:
            return None

        # rank & select top N
        top_n = self.config.top_n
        today_df = today_df.sort_values('weight', ascending=False).head(top_n).copy()

        # apply per-stock cap
        cap = self.config.max_stock_weight
        # initial normalization
        today_df['weight'] = today_df['weight'] / today_df['weight'].sum()
        today_df['weight'] = today_df['weight'].clip(upper=cap)

        # re-normalize after clipping
        today_df['weight'] = today_df['weight'] / today_df['weight'].sum()

        # attach & enforce industry constraints only if we actually have a mapping
        if self._industry_map:
            today_df['industry'] = today_df['code'].astype(str).str.zfill(6).map(self._industry_map).fillna('UNKNOWN')
            # if only one industry category present, skip scaling (acts like no industry constraint)
            if today_df['industry'].nunique() > 1:
                ind_cap = self.config.max_industry_weight
                adjusted_rows = []
                for ind, sub in today_df.groupby('industry'):
                    total_ind_weight = sub['weight'].sum()
                    if total_ind_weight > ind_cap:
                        scale = ind_cap / total_ind_weight
                        sub['weight'] = sub['weight'] * scale
                    adjusted_rows.append(sub)
                adjusted_df = pd.concat(adjusted_rows, ignore_index=True)
            else:
                adjusted_df = today_df
        else:
            # No industry constraints -> fully utilize capital by normalizing to 1
            adjusted_df = today_df
            adjusted_df['weight'] = adjusted_df['weight'] / adjusted_df['weight'].sum()

        # After industry scaling the total weight may be < 1 (cash remainder) or > 1 (rare if caps ignored).
        total_w = adjusted_df['weight'].sum()
        if total_w > 1.0:
            # scale down uniformly to sum=1 while preserving industry caps (since we only shrink)
            adjusted_df['weight'] = adjusted_df['weight'] / total_w
        elif not self._industry_map and total_w != 1.0:
            # ensure sum==1 for basic case without industry mapping
            adjusted_df['weight'] = adjusted_df['weight'] / total_w

        # drop very small weights (treat as zero / cash)
        thr = self.config.min_weight_threshold
        adjusted_df = adjusted_df[adjusted_df['weight'] >= thr].copy()

        # IMPORTANT: do NOT renormalize upward if sum < 1, to avoid re-violating industry caps; leave residual as cash.

        # compute concentration (HHI)
        hhi = (adjusted_df['weight'] ** 2).sum()
        # robust timestamp extraction
        try:
            date_ts_val = float(pd.Timestamp(latest_date).value)
        except Exception:
            date_ts_val = 0.0
        meta = {
            'hhi': float(hhi),
            'stock_count': float(len(adjusted_df)),
            'date_timestamp': date_ts_val,
        }

        return PortfolioResult(date=latest_date, target_weights=adjusted_df[['code', 'weight']], meta=meta)
