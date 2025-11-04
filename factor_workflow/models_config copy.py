"""Model configuration registry for the factor workflow multi-model pipeline.

All models are backed by light-weight scikit-learn estimators implemented in
``local_models.py`` to avoid heavy optional dependencies such as LightGBM.

Each ``*_model_specs`` entry is a dictionary containing:
    - ``name``: identifier used in logs/metrics.
    - ``config``: qlib-style init config for the estimator.
    - ``weighting`` (optional): overrides for fusion stage per model (e.g.
      custom ema span).
"""

from pathlib import Path

LOCAL_MODEL_PATH = str((Path(__file__).resolve().parent / "local_models.py"))


def _linear(estimator: str, alpha: float, include_valid: bool = True, **kwargs):
    return {
        "class": "SklearnLinear",
        "module_path": LOCAL_MODEL_PATH,
        "kwargs": {
            "estimator": estimator,
            "alpha": alpha,
            "fit_intercept": True,
            "include_valid": include_valid,
            **kwargs,
        },
    }


def _hist_gb(learning_rate: float, max_iter: int, max_leaf_nodes: int | None = 31, **kwargs):
    return {
        "class": "SklearnHistGB",
        "module_path": LOCAL_MODEL_PATH,
        "kwargs": {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            **kwargs,
        },
    }


long_model_specs = [
    {
        "name": "ridge_core",
        "config": _linear("ridge", alpha=2.5, include_valid=True),
    },
    {
        "name": "lasso_sparse",
        "config": _linear("lasso", alpha=0.8, include_valid=True),
    },
    {
        "name": "histgb_core",
        "config": _hist_gb(learning_rate=0.06, max_iter=400, max_leaf_nodes=45, include_valid=True),
        "weighting": {"ema_span": 90},
    },
]


short_model_specs = [
    {
        "name": "lasso_fast",
        "config": _linear("lasso", alpha=0.6, include_valid=True),
        "weighting": {"ema_span": 30},
    },
    {
        "name": "ridge_fast",
        "config": _linear("ridge", alpha=1.2, include_valid=True),
        "weighting": {"ema_span": 40},
    },
    {
        "name": "histgb_fast",
        "config": _hist_gb(learning_rate=0.08, max_iter=300, max_leaf_nodes=25, include_valid=True, max_depth=6),
        "weighting": {"ema_span": 25},
    },
]


fusion_config = {
    "long": {
        "ema_span": 75,
        "min_periods": 30,
        "min_weight": 0.05,
        "ic_floor": 0.0,
    },
    "short": {
        "ema_span": 35,
        "min_periods": 20,
        "min_weight": 0.05,
        "ic_floor": -0.02,
    },
}


__all__ = ["long_model_specs", "short_model_specs", "fusion_config"]
