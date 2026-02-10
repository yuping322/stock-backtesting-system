"""Centralized path definitions for the factor workflow pipeline."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # Go up two levels to reach the repo root

DATA_ROOT = PROJECT_ROOT / "data" / "model_tasks"
LATEST_LINK = DATA_ROOT / "latest"
LATEST_MARKER = DATA_ROOT / "latest_run.txt"


def _resolve_active_root() -> Path:
    if LATEST_LINK.exists():
        try:
            return LATEST_LINK.resolve(strict=True)
        except FileNotFoundError:
            pass
        except OSError:
            return LATEST_LINK
    if LATEST_MARKER.exists():
        try:
            candidate = Path(LATEST_MARKER.read_text().strip())
            if candidate.exists():
                return candidate
        except Exception:
            pass
    return DATA_ROOT


ACTIVE_DATA_ROOT = _resolve_active_root()
FEATURES_FILE = ACTIVE_DATA_ROOT / "features_panel.pkl"
LABEL_FILE = ACTIVE_DATA_ROOT / "label_panel.pkl"
META_FILE = ACTIVE_DATA_ROOT / "meta_series.pkl"
IC_FILE = ACTIVE_DATA_ROOT / "factor_ic_daily.pkl"
CLEAN_PRICES_FILE = ACTIVE_DATA_ROOT / "prices_cleaned.csv"
RAW_PRICES_FILE = ACTIVE_DATA_ROOT / "prices_raw.csv"

RESULTS_ROOT = PROJECT_ROOT / "results" / "factor_workflow"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

BACKTEST_OUTPUT_DIR = RESULTS_ROOT / "backtest"
BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_OUTPUT_FILE = RESULTS_ROOT / "scores.csv"

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "ACTIVE_DATA_ROOT",
    "LATEST_LINK",
    "LATEST_MARKER",
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
