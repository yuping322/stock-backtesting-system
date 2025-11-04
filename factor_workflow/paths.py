"""Centralized path definitions for the factor workflow pipeline."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "exported_data_all"
FEATURES_FILE = DATA_ROOT / "features_panel.pkl"
LABEL_FILE = DATA_ROOT / "label_panel.pkl"
META_FILE = DATA_ROOT / "meta_series.pkl"
IC_FILE = DATA_ROOT / "factor_ic_daily.pkl"
CLEAN_PRICES_FILE = DATA_ROOT / "prices_cleaned.csv"
RAW_PRICES_FILE = DATA_ROOT / "prices_raw.csv"

RESULTS_ROOT = PROJECT_ROOT / "results" / "factor_workflow"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

BACKTEST_OUTPUT_DIR = RESULTS_ROOT / "backtest"
BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_OUTPUT_FILE = RESULTS_ROOT / "scores.csv"

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "FEATURES_FILE",
    "LABEL_FILE",
    "META_FILE",
    "IC_FILE",
    "CLEAN_PRICES_FILE",
    "RAW_PRICES_FILE",
    "RESULTS_ROOT",
    "BACKTEST_OUTPUT_DIR",
    "SCORES_OUTPUT_FILE",
]
