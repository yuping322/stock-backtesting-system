"""ML package exposing factor modeling pipeline.

Exports:
  - main()                   CLI entry (same as legacy script)
  - run_baseline             Walk-forward baseline XGBoost + SHAP aggregation
  - build_factor_groups      Correlation + SHAP guided clustering
  - train_group_models       Per-group submodel training & IC-weighted blending
  - build_prediction_weights Top-N equal-weight selection helper

Usage examples:
  python -m ml.ml_pipeline --factor-file data/factors.csv --mode baseline
  python scripts/ml_pipeline.py --factor-file data/factors.csv --mode all
"""
from .ml_pipeline import (
    main,
    run_baseline,
    build_factor_groups,
    train_group_models,
    build_prediction_weights,
)

__all__ = [
    "main",
    "run_baseline",
    "build_factor_groups",
    "train_group_models",
    "build_prediction_weights",
]
