"""Data acquisition and preparation module (NO LOCAL CACHE VERSION).

This variant removes all local disk caching: every call attempts network fetch and
falls back gracefully to empty DataFrames or offline sample lists when upstream
endpoints fail. Suitable when:
    - Running inside ephemeral environments (containers, CI) where persistence is unnecessary.
    - Ensuring latest data without relying on potentially stale cached files.

Provided datasets (columns):
    industry:     [code, industry, sector]
    basic_panel:  [code, name, last_price, market_cap, float_market_cap, volume, amount, turnover_rate]
    suspension:   [code, is_suspended, date]
    blacklist:    [code, reason, date]
    daily_prices: [code, trade_date, open, high, low, close, volume, amount]

Resilience strategy:
    - Multi-step universe fallback → full → HS300 → fixed sample.
    - Each fetch wrapped in retry; on failure returns empty frame (except universe which may fallback to sample codes).
    - No read/write side effects (pure network operations + in-memory transformations).

Usage Example:
        from live_trading.data_provider import DataProvider, ProviderConfig
        cfg = ProviderConfig(universe_mode='all')
        dp = DataProvider(cfg)
        codes = dp.load_universe()
        panel = dp.fetch_basic_panel()
        industry = dp.fetch_industry()
        blacklist = dp.build_blacklist(panel)
"""
from __future__ import annotations
import os  # retained only for potential future path logic (unused now)
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

try:
    import akshare as ak  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("akshare is required for DataProvider. Please install via `pip install akshare`." ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    universe_mode: str = "all"  # or 'hs300', 'zz1000', 'custom'
    custom_universe: List[str] = field(default_factory=list)
    retry: int = 3
    retry_sleep: float = 1.5
    # Minimal liquidity thresholds (used in blacklist building)
    min_price: float = 2.0
    min_turnover_amount: float = 5_000_000  # RMB
    st_name_patterns: Sequence[str] = ("ST", "*ST")
    allow_offline_fallback: bool = True  # return sample universe if all remote calls fail

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Cache utilities removed (no local persistence in this version).


def _retry_call(func, *args, retry: int = 3, sleep: float = 1.5, **kwargs):
    last_err = None
    for i in range(retry):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # pragma: no cover (network variance)
            last_err = e
            logger.warning("Call failed (%s/%s): %s", i+1, retry, e)
            time.sleep(sleep)
    raise last_err

# ---------------------------------------------------------------------------
# DataProvider implementation
# ---------------------------------------------------------------------------
class DataProvider:
    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    # ---------------------- Universe ----------------------
    def load_universe(self) -> List[str]:
        """Load stock universe based on config.

        Modes:
        - all: All A-share codes via ak.stock_a_all()
        - hs300: CSI 300 constituents
        - zz1000: CSI 1000 constituents
        - custom: Use cfg.custom_universe
        """
        mode = self.cfg.universe_mode
        if mode == "custom":
            return list(self.cfg.custom_universe)
        if mode == "hs300":
            try:
                df = _retry_call(ak.index_stock_cons, symbol="000300")
                return df["代码"].astype(str).str.replace(".SH", "").str.replace(".SZ", "").tolist()
            except Exception as e:
                logger.warning("HS300 fetch failed: %s", e)
                if self.cfg.allow_offline_fallback:
                    return self._offline_sample_universe()
                raise
        if mode == "zz1000":
            try:
                df = _retry_call(ak.index_stock_cons, symbol="000852")
                return df["代码"].astype(str).str.replace(".SH", "").str.replace(".SZ", "").tolist()
            except Exception as e:
                logger.warning("ZZ1000 fetch failed: %s", e)
                if self.cfg.allow_offline_fallback:
                    return self._offline_sample_universe()
                raise
        # default all (robust fallback chain)
        try:
            df = _retry_call(ak.stock_info_a_code_name)
            source = "stock_info_a_code_name"
        except Exception as e:
            logger.warning("Full universe fetch failed (%s). Trying HS300 fallback.", e)
            try:
                df = _retry_call(ak.index_stock_cons, symbol="000300")
                source = "hs300_fallback"
            except Exception as e2:
                logger.warning("HS300 fallback also failed (%s).", e2)
                if self.cfg.allow_offline_fallback:
                    logger.warning("Returning offline sample universe.")
                    return self._offline_sample_universe()
                raise
        candidate_cols = ["code", "证券代码", "代码", "品种代码", "symbol"]
        code_col = next((c for c in candidate_cols if c in df.columns), df.columns[0])
        codes = (df[code_col].astype(str)
                 .str.replace(".SH", "")
                 .str.replace(".SZ", "")
                 .tolist())
        logger.info("Universe size (%s): %d", source, len(codes))
        return codes

    @staticmethod
    def _offline_sample_universe() -> List[str]:
        """Return a fixed small list of representative liquid tickers for offline/demo mode."""
        return ["600000", "000001", "600519", "000858", "300750", "601318", "002594", "000333", "600036", "000651"]

    # ---------------------- Industry Classification ----------------------
    def fetch_industry(self) -> pd.DataFrame:
        """Fetch industry classification.

        Uses Tonghuashun (ths) concept & industry boards as proxy. Merges concept + industry into sector fallback.
        Returns columns: [code, industry, sector]
        """
        # Industry board constituents
        # Note: akshare provides functions like stock_board_industry_name_ths & stock_board_industry_cons_ths
        try:
            industry_list = _retry_call(ak.stock_board_industry_name_ths)
            frames = []
            for _, row in industry_list.iterrows():
                board = row["名称"]
                cons = _retry_call(ak.stock_board_industry_cons_ths, symbol=board)
                # Expect columns: '代码','名称'
                if "代码" not in cons.columns:
                    continue
                tmp = cons[["代码"]].copy()
                tmp["industry"] = board
                frames.append(tmp)
            industry_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["代码","industry"])
            industry_df.rename(columns={"代码": "code"}, inplace=True)
        except Exception as e:  # pragma: no cover
            logger.warning("Industry fetch failed: %s", e)
            industry_df = pd.DataFrame(columns=["code", "industry"])  # empty fallback
        industry_df["sector"] = industry_df["industry"]  # simple mapping for now
        return industry_df

    # ---------------------- Basic Panel ----------------------
    def fetch_basic_panel(self) -> pd.DataFrame:
        """Fetch spot snapshot for basic metrics.

        Data source: ak.stock_zh_a_spot_em() / fallback to ak.stock_zh_a_spot().
        Returns columns standardized.
        """
        try:
            try:
                df = _retry_call(ak.stock_zh_a_spot_em)
            except Exception:
                df = _retry_call(ak.stock_zh_a_spot)
        except Exception as e:  # pragma: no cover
            logger.warning("Basic panel fetch failed: %s; returning empty frame", e)
            empty_cols = ["code","name","last_price","market_cap","float_market_cap","volume","amount","turnover_rate"]
            empty_df = pd.DataFrame(columns=empty_cols)
            return empty_df
        # Normalize
        # Common columns: "代码","名称","最新价","总市值","流通市值","成交量","成交额","换手率"
        col_map = {
            "代码": "code",
            "名称": "name",
            "最新价": "last_price",
            "总市值": "market_cap",
            "流通市值": "float_market_cap",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        panel_cols = ["code","name","last_price","market_cap","float_market_cap","volume","amount","turnover_rate"]
        panel = df.copy()
        # Ensure all expected columns exist even if upstream missing
        for c in panel_cols:
            if c not in panel.columns:
                panel[c] = pd.NA
        panel = panel[panel_cols]
        return panel

    # ---------------------- Suspension / Delisting ----------------------
    def fetch_suspension(self, codes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Fetch suspension status (simplified).

        Akshare doesn't expose a single bulk endpoint for all suspended stocks in spot form; we approximate:
        - Use basic panel: if volume == 0 and last_price unchanged (not available here), treat as potential suspended.
        - For more accurate results, integrate exchange bulletins (TODO).
        Returns: [code, is_suspended, date]
        """
        panel = self.fetch_basic_panel()
        today = pd.Timestamp.today().strftime("%Y%m%d")
        if codes:
            panel = panel[panel.code.isin(codes)]
        # Heuristic: volume == 0 -> maybe suspended (rough approximation)
        susp = panel[["code","volume"]].copy()
        susp["is_suspended"] = susp["volume"].fillna(0) == 0
        susp = susp[["code","is_suspended"]]
        susp["date"] = today
        return susp

    # ---------------------- Blacklist ----------------------
    def build_blacklist(self, panel: Optional[pd.DataFrame] = None, suspension: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct blacklist based on:
        - ST naming patterns
        - Low price (< cfg.min_price)
        - Extremely low turnover amount (< cfg.min_turnover_amount)
        - Suspended status
        Returns columns: [code, reason, date]
        """
        if panel is None:
            panel = self.fetch_basic_panel()
        if suspension is None:
            suspension = self.fetch_suspension(panel.code.tolist())
        today = pd.Timestamp.today().strftime("%Y%m%d")
        rows = []
        # ST pattern
        for _, r in panel.iterrows():
            name = str(r.get("name", ""))
            code = str(r.get("code"))
            reasons = []
            if any(p in name for p in self.cfg.st_name_patterns):
                reasons.append("ST")
            if pd.notnull(r.get("last_price")) and r.get("last_price") < self.cfg.min_price:
                reasons.append("LOW_PRICE")
            if pd.notnull(r.get("amount")) and r.get("amount") < self.cfg.min_turnover_amount:
                reasons.append("LOW_LIQUIDITY")
            susp_row = suspension[suspension.code == code]
            if not susp_row.empty and bool(susp_row.iloc[0].is_suspended):
                reasons.append("SUSPENDED")
            if reasons:
                rows.append({"code": code, "reason": ";".join(sorted(set(reasons))), "date": today})
        blacklist = pd.DataFrame(rows, columns=["code","reason","date"]).sort_values("code")
        return blacklist

    # ---------------------- Daily Prices ----------------------
    def fetch_daily_prices(self, trade_date: str, codes: Sequence[str]) -> pd.DataFrame:
        """Fetch daily historical prices for given codes on or around trade_date.

        Akshare historical API requires start/end range; we pull single day window.
        Returns columns standardized: [code, trade_date, open, high, low, close, volume, amount]
        NOTE: This iterates codes; consider optimizing via parallel or local mirror if universe large.
        """
        start = trade_date
        end = trade_date
        records = []
        for code in codes:
            # Akshare symbol formatting: 600000 -> SH600000; 000001 -> SZ000001 for some endpoints.
            symbol = self._format_symbol(code)
            try:
                hist = _retry_call(ak.stock_zh_a_hist, symbol=symbol, start_date=start, end_date=end, adjust="")
            except Exception as e:  # pragma: no cover
                logger.warning("Price fetch failed for %s: %s", code, e)
                continue
            if hist.empty:
                continue
            # Standard columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
            row = hist.iloc[0]
            rec = {
                "code": code,
                "trade_date": row.get("日期"),
                "open": row.get("开盘"),
                "close": row.get("收盘"),
                "high": row.get("最高"),
                "low": row.get("最低"),
                "volume": row.get("成交量"),
                "amount": row.get("成交额"),
            }
            records.append(rec)
        df = pd.DataFrame(records)
        if not df.empty:
            df = df[["code","trade_date","open","high","low","close","volume","amount"]]
        return df

    # ---------------------- Real-time Index (Lightweight) ----------------------
    def fetch_index_snapshot(self, indices: Sequence[str]) -> pd.DataFrame:
        """Fetch lightweight index change data.

        Uses ak.index_zh_a_hist for last bar (simplified) or spot endpoints if available.
        Returns: [index_code, pct_change, timestamp]
        """
        rows = []
        for idx in indices:
            try:
                data = _retry_call(ak.stock_zh_index_daily, symbol=idx)
            except Exception as e:  # pragma: no cover
                logger.warning("Index fetch failed %s: %s", idx, e)
                continue
            if data.empty:
                continue
            last = data.iloc[-1]
            # Columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
            pct = None
            if pd.notnull(last.get("开盘")) and pd.notnull(last.get("收盘")) and last.get("开盘") != 0:
                pct = (last.get("收盘") - last.get("开盘")) / last.get("开盘")
            rows.append({
                "index_code": idx,
                "pct_change": pct,
                "timestamp": last.get("日期")
            })
        return pd.DataFrame(rows, columns=["index_code","pct_change","timestamp"])

    # ---------------------- Helper ----------------------
    @staticmethod
    def _format_symbol(code: str) -> str:
        """Format raw numeric code to akshare symbol.
        For history API ak.stock_zh_a_hist uses like 'sh600000', 'sz000001' (lowercase variant accepted).
        """
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"  # assume SZ for others (000/300 etc.)

# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_default_provider() -> DataProvider:
    """Create a DataProvider with defaults (no caching)."""
    cfg = ProviderConfig(universe_mode="all")
    return DataProvider(cfg)

# ---------------------------------------------------------------------------
# CLI Entry (Optional quick test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    provider = create_default_provider()
    codes = provider.load_universe()[:20]  # limit for demo
    industry = provider.fetch_industry()
    panel = provider.fetch_basic_panel()
    susp = provider.fetch_suspension(codes)
    blacklist = provider.build_blacklist(panel, susp)
    today = pd.Timestamp.today().strftime("%Y%m%d")
    prices = provider.fetch_daily_prices(trade_date=today, codes=codes[:5])

    summary = {
        "universe_size": len(codes),
        "industry_rows": len(industry),
        "panel_rows": len(panel),
        "blacklist_size": len(blacklist),
        "prices_rows": len(prices),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
