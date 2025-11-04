"""Model configuration registry for the simplified factor workflow pipeline.

We intentionally固定两类模型：
    - Ridge (线性基线，稳定可解释)
    - Histogram Gradient Boosting (HistGB，适度非线性增强)

所有模型均封装在 ``local_models.py``，便于保持依赖精简和参数统一。
每个 spec 定义:
    - ``name``: 日志中使用的名称
    - ``config``: qlib 初始化参数
    - ``weighting``: 可选，覆盖融合阶段的 EMA/阈值配置
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


_RIDGE_SPEC = {
    "name": "ridge_core",
    "config": _linear("ridge", alpha=2.0, include_valid=True),
}

_HISTGB_SPEC = {
    "name": "histgb_core",
    "config": _hist_gb(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=45,
        include_valid=True,
    ),
    "weighting": {"ema_span": 90, "ic_floor": 0.0},
}


long_model_specs = [_RIDGE_SPEC, _HISTGB_SPEC]

# 短期套件沿用相同模型，但允许更快的权重响应
short_model_specs = [
    {
        **_RIDGE_SPEC,
        "name": "ridge_short",
        "weighting": {"ema_span": 40, "ic_floor": -0.01},
    },
    {
        **_HISTGB_SPEC,
        "name": "histgb_short",
        "weighting": {"ema_span": 30, "ic_floor": -0.01},
    },
]


fusion_config = {
    "long": {
        "ema_span": 75,
        "min_periods": 25,
        "min_weight": 0.05,
        "ic_floor": 0.0,
    },
    "short": {
        "ema_span": 35,
        "min_periods": 20,
        "min_weight": 0.05,
        "ic_floor": -0.01,
    },
}


__all__ = ["long_model_specs", "short_model_specs", "fusion_config"]
