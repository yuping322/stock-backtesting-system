"""Workflow script: train fixed ridge + HistGB suites and persist fused predictions.

Run: ``python workflow_main.py``
Make sure ``qlib.init`` is pointed to your data directory.
"""
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R

try:  # Prefer package-relative imports when available
    from .dataset_config import long_dataset_config, short_dataset_config
    from .model_pipeline import train_model_suite
    from .models_config import fusion_config, long_model_specs, short_model_specs
except ImportError:  # Fallback for ``python factor_workflow/workflow_main.py``
    from pathlib import Path
    import sys

    _CURRENT_DIR = Path(__file__).resolve().parent
    _PACKAGE_ROOT = _CURRENT_DIR.parent
    _PACKAGE_PATH = str(_PACKAGE_ROOT)
    if _PACKAGE_PATH not in sys.path:
        sys.path.insert(0, _PACKAGE_PATH)

    from factor_workflow.dataset_config import long_dataset_config, short_dataset_config
    from factor_workflow.model_pipeline import train_model_suite
    from factor_workflow.models_config import fusion_config, long_model_specs, short_model_specs


def _save_suite_outputs(tag: str, suite_result):
    metrics = suite_result.metrics_table()
    weight_series = (pd.Series(suite_result.weights, name="weight") if suite_result.weights else pd.Series())
    obj_to_save = {
        f"predict_{tag}": suite_result.fused_prediction,
        f"metrics_{tag}": metrics,
        f"weights_{tag}": weight_series,
    }
    for res in suite_result.model_results:
        obj_to_save[f"predict_{tag}_{res.name}"] = res.prediction
        obj_to_save[f"daily_ic_{tag}_{res.name}"] = res.daily_ic
    R.save_objects(**obj_to_save)


def train_long_short(provider_uri="~/.qlib/qlib_data", region="cn"):
    qlib.init(provider_uri=provider_uri, region=region)

    long_dataset = init_instance_by_config(long_dataset_config)
    short_dataset = init_instance_by_config(short_dataset_config)

    with R.start(experiment_name="exp_long_suite"):
        long_suite = train_model_suite("long", long_dataset, long_model_specs, fusion_config["long"])
        _save_suite_outputs("long", long_suite)

    with R.start(experiment_name="exp_short_suite"):
        short_suite = train_model_suite("short", short_dataset, short_model_specs, fusion_config["short"])
        _save_suite_outputs("short", short_suite)

    pred_long = long_suite.fused_prediction
    pred_short = short_suite.fused_prediction

    return pred_long, pred_short, long_dataset, short_dataset


if __name__ == "__main__":
    train_long_short()
