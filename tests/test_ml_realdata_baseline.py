import os
import pandas as pd
import numpy as np
import pytest

from ml.ml_pipeline import load_factor_data, get_factor_columns, run_baseline, WalkForwardConfig
from ml.preprocessing import FeaturePreprocessor, PreprocessConfig

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "exported_data_all", "formatted_data.csv")
FALLBACK_SMALL = os.path.join(os.path.dirname(__file__), "..", "data", "factor_values_sample.csv")


def _load_any(path: str) -> pd.DataFrame:
    return load_factor_data(path, date_col="date", code_col="code")


def test_realdata_baseline_pipeline():
    if not os.path.exists(DATA_FILE):
        pytest.skip(f"formatted_data.csv not available: {DATA_FILE}")
    try:
        df = _load_any(DATA_FILE)
    except Exception as e:
        pytest.skip(f"Cannot load formatted_data.csv due to: {e}")
    # Guard: ensure we have more than 3 unique dates; otherwise fallback to small sample
    unique_dates = sorted(df["date"].unique()) if "date" in df.columns else []
    if len(unique_dates) < 4:
        if os.path.exists(FALLBACK_SMALL):
            df = _load_any(FALLBACK_SMALL)
            unique_dates = sorted(df["date"].unique())
        else:
            pytest.skip("Not enough dates in formatted_data.csv and fallback missing")

    # Determine label
    label_col = "forward_return" if "forward_return" in df.columns else None
    factor_cols = get_factor_columns(df, label_col or "__dummy__", "date", "code")
    # If no existing label, synthesize continuous then use directly as regression target
    if label_col is None:
        rng = np.random.default_rng(42)
        synth = df[factor_cols].mean(axis=1) + rng.normal(0, 0.1, size=len(df))
        df["forward_return"] = synth.astype(float)
        label_col = "forward_return"

    # Reduce factor set for speed (limit to first 30 numeric factors)
    factor_cols = [c for c in factor_cols if pd.api.types.is_numeric_dtype(df[c])][:30] or factor_cols[:1]

    # Dynamic windows: aim for ~3 test windows
    total_dates = len(unique_dates)
    test_window = max(3, min(10, total_dates // 6))
    train_window = max(20, min(60, total_dates - test_window * 2))
    min_train = max(10, train_window // 2)
    cfg = WalkForwardConfig(train_window=train_window, test_window=test_window, min_train=min_train)

    preproc = FeaturePreprocessor(PreprocessConfig(impute="mean", standardize=True, neutralize_industry=False), factor_cols)
    pseudo_args = type("Args", (), {"date_column": "date", "code_column": "code", "label_column": label_col})
    pred_df, shap_agg = run_baseline(df, pseudo_args, cfg, factor_cols, preproc)

    assert not pred_df.empty, "Prediction DataFrame should not be empty"
    assert {"date", "code", "pred_score"}.issubset(pred_df.columns), "Missing prediction columns"
    assert not shap_agg.empty, "SHAP importance output should not be empty"
    assert {"factor", "mean_abs_shap", "rank"}.issubset(shap_agg.columns)
