# Factor Workflow Example

An end-to-end reference that trains two light-weight model suites (ridge & HistGradientBoosting), fuses them with IC-driven weights, neutralizes style exposures, and exports daily normalized portfolio weights. The pipeline is numerically robust: processors copy data to avoid `SettingWithCopy` warnings, ridge solvers fall back gracefully, and export weights always sum to 1 per trading day.

---

## Integration Blueprint

| Layer | File(s) | Responsibility |
| --- | --- | --- |
| Data access | `dataset_config.py`, `paths.py`, `exported_data_all/` | Auto-detect calendar/instruments, configure `DatasetH` with safe processors. |
| Model specs | `models_config.py`, `local_models.py` | Declare ridge & HistGB suites and provide sklearn-based implementations with SVD-stabilized ridge solver. |
| Training + fusion | `model_pipeline.py`, `workflow_main.py` | Fit models, compute rolling IC, update EMA weights, persist intermediate predictions. |
| Signal post-processing | `combine_signal.py` | Neutralize industry/size exposures via ridge regression and clip extremes. |
| Evaluation & export | `backtest_evaluation.py`, `export_scores.py` | Build tradable signal, run sample backtest, emit normalized `weights.csv`. |
| Factor maintenance | `config_factors.py`, `update_factors.py` | Manage core/candidate pools using historical IC statistics. |
| Quality gate | `tests/examples/test_factor_workflow.py` | Slow pytest smoke test covering train → backtest → export.

When integrating into another repository, copy the entire folder or cherry-pick the layers you need; each module only depends on qlib core, numpy/pandas, and scikit-learn.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r examples/factor_workflow/requirements.txt

# 2. (Optional) Update factor lists if you track IC statistics
python -m factor_workflow.update_factors
# or run locally without installation
python factor_workflow/update_factors.py

# 3. Train model suites and persist predictions
python -m factor_workflow.workflow_main  # package run
# or run locally without installation
python factor_workflow/workflow_main.py

# 4. Run evaluation (produces signal + backtest artifacts)
python -m factor_workflow.backtest_evaluation
# or run locally without installation
python factor_workflow/backtest_evaluation.py

# 5. Export normalized weights to CSV
python -m factor_workflow.export_scores
# or run locally without installation
python factor_workflow/export_scores.py
```

Outputs land in `results/factor_workflow/` (`backtest/`, `scores.csv`, etc.) and under `workflow/` (qlib experiment logs).

---

## Data Assumptions

QLib-ready artifacts are expected under `exported_data_all/`:

- `features_panel.pkl`: MultiIndex `(datetime, instrument)` DataFrame of factor values.
- `label_panel.pkl`: Matching Series/DataFrame with forward returns.
- `meta_series.pkl`: Dictionary with auxiliary series such as industry codes and market cap (used for neutralization & instrument discovery).

`dataset_config.py` detects actual calendars/instruments from the feature panel and falls back to defaults when files are missing, so you can drop in your own pickles without rewriting code.

---

## Why It’s Robust

- **Safe processors** (`processors.py`): All handler processors operate on copies to silence chained-assignment warnings.
- **Numeric guards**: `local_models.py` and `combine_signal.py` sanitize NaN/inf values and use SVD-based ridge solving with lstsq fallback.
- **Consistent exports**: `export_scores.py` renames symbols to `code`, normalizes weights per day, and is covered by an automated regression test.
- **Logging & fallbacks**: `backtest_evaluation.generate_final_signal` logs degraded paths and still returns baseline signals if neutralization fails.

---

## Customisation Pointers

- Adjust training/test windows in `dataset_config.py` (defaults adapt to data length, respecting a 60%/40% split with minimum evaluation days).
- Tweak ridge penalty or HistGB hyper-parameters via `models_config.py`.
- Inject additional style factors into `combine_signal.combine_predictions` by passing `extra_styles`.
- Replace or extend model specs (e.g., ElasticNet) by editing `models_config.py` and adding implementations to `local_models.py`.
- For long-short portfolios, modify `export_scores.py` to allow negative weights and update the smoke test accordingly.

---

## Testing

This repo ships with a regression smoke test to keep the workflow honest:

```bash
pytest -m slow tests/examples/test_factor_workflow.py
```

The test retrains on the sample data, exports weights, and asserts that every day’s weights sum to 1 within tolerance.

---

## Notes & Next Steps

- The example ships with toy data; replace `sample_data/` with your production pickles.
- For live deployment, connect `export_scores.export_scores` to your orchestration layer (Airflow, cron, etc.) and surface logs from `workflow_main`.
- Consider adding richer analytics (rolling IC plots, turnover stats) in `backtest_evaluation.py` once integrated.

