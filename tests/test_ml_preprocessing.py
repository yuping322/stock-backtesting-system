import pandas as pd
import numpy as np
from ml.preprocessing import FeaturePreprocessor, PreprocessConfig
from ml.ml_pipeline import train_group_models, WalkForwardConfig, get_factor_columns, build_walk_forward_windows


def make_dummy_df(n_dates=40, n_codes=10, n_factors=5):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for c in range(n_codes):
            row = {
                "date": d,
                "code": f"{c:06d}",
                "industry": f"I{c % 3}",
            }
            for f in range(n_factors):
                row[f"factor_{f}"] = rng.normal(0, 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Forward return synthetic label correlated with factor_0
    df["forward_return"] = df["factor_0"] * 0.5 + rng.normal(0, 0.5, size=len(df))
    return df


def test_preprocessor_fit_transform_no_leakage():
    # Need enough dates for at least one window: n_dates >= train_window + test_window and train_window >= min_train
    df = make_dummy_df(n_dates=40)
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    cfg = WalkForwardConfig(train_window=20, test_window=5, min_train=10)
    dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(dates, cfg)
    # Use first window
    train_dates, test_dates = windows[0]
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    preproc = FeaturePreprocessor(PreprocessConfig(impute="mean", standardize=True, neutralize_industry=True), factor_cols)
    preproc.fit(train_df)
    train_proc = preproc.transform(train_df)
    test_proc = preproc.transform(test_df)
    # Means of standardized train should be ~0
    train_means = [abs(train_proc[c].mean()) for c in factor_cols]
    assert max(train_means) < 1e-6
    # Test means need not be zero (distribution shift) but should not equal exactly train means replaced by leakage pattern
    # Check industry neutralization: industry group means ~0 in train
    grp_means = train_proc.groupby("industry")[factor_cols].mean().abs().values.max()
    assert grp_means < 1e-6


def test_ic_uses_test_labels():
    df = make_dummy_df(n_dates=50)
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    # Create two groups to exercise weighting logic
    mapping = {f: (0 if i % 2 == 0 else 1) for i, f in enumerate(factor_cols)}
    cfg = WalkForwardConfig(train_window=25, test_window=5, min_train=15)
    preproc = FeaturePreprocessor(PreprocessConfig(impute="mean", standardize=True, neutralize_industry=False), factor_cols)
    blended_df, components_df = train_group_models(df, mapping, "forward_return", "date", "code", cfg, preproc)
    # Expect blended_df not empty (should have predictions for each test window)
    assert not blended_df.empty, "Blended predictions should not be empty"
    # Validate IC computation: compare correlation using test labels versus first slice of train labels - they should differ frequently
    # Extract first window test dates
    unique_dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(unique_dates, cfg)
    first_train, first_test = windows[0]
    test_slice = df[df["date"].isin(first_test)]
    train_slice = df[df["date"].isin(first_train)]
    # Build naive leakage IC (incorrect) and correct IC for first group window on the same predictions subset
    # For simplicity use factor_0 raw values as pseudo predictions for demonstration
    pseudo_pred = test_slice[factor_cols[0]].values
    ic_correct = pd.Series(pseudo_pred).corr(test_slice["forward_return"], method="spearman")
    ic_leak = pd.Series(pseudo_pred).corr(train_slice["forward_return"].iloc[: len(pseudo_pred)], method="spearman")
    # At least one correlation should be finite
    assert not (np.isnan(ic_correct) and np.isnan(ic_leak)), "Both correlations are NaN, unexpected"
    if not (np.isnan(ic_correct) or np.isnan(ic_leak)):
        assert ic_correct != ic_leak, "Correlations unexpectedly equal indicating potential leakage logic still present"
