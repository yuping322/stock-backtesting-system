"""Dataset configuration for long-term and short-term models.

Instead of hard-coding calendar windows we infer the available date range from
``features_panel.pkl`` so the workflow adapts automatically to new data dumps.
"""
from pathlib import Path
import sys
import pickle
import pandas as pd

from .paths import FEATURES_FILE, LABEL_FILE, META_FILE

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
REPO_PATH_STR = str(REPO_ROOT)
if REPO_PATH_STR not in sys.path:
    sys.path.insert(0, REPO_PATH_STR)

FEATURE_PATH = str(FEATURES_FILE.resolve())
LABEL_PATH = str(LABEL_FILE.resolve())
META_PATH = META_FILE

DEFAULT_START_DATE = "2025-07-28"
DEFAULT_END_DATE = "2025-10-24"
DEFAULT_SPLIT_DATE = "2025-09-30"
TRAIN_RATIO = 0.6  # fraction of trading days used for training when enough history exists
MIN_TEST_DAYS = 30  # ensure we keep at least this many days for evaluation when possible


def _format_date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _infer_calendar():
    try:
        feature_df = pd.read_pickle(FEATURE_PATH)
        dates = feature_df.index.get_level_values("datetime")
        dates = pd.Index(pd.to_datetime(dates.unique())).sort_values()
        if dates.empty:
            raise ValueError("empty date index")
        start_dt = dates[0]
        end_dt = dates[-1]
        if len(dates) < 3:
            train_end_dt = dates[-1]
            test_start_dt = dates[-1]
        else:
            candidate_idx = max(int(len(dates) * TRAIN_RATIO), 1)
            max_train_idx = len(dates) - MIN_TEST_DAYS
            if max_train_idx < 1:
                split_idx = min(candidate_idx, len(dates) - 1)
            else:
                split_idx = min(candidate_idx, max_train_idx)
                split_idx = max(split_idx, 1)
            if split_idx >= len(dates):
                split_idx = len(dates) - 1
            train_end_dt = dates[split_idx - 1]
            test_start_dt = dates[split_idx]
        return (
            _format_date(start_dt),
            _format_date(end_dt),
            _format_date(train_end_dt),
            _format_date(test_start_dt),
        )
    except Exception:
        return DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SPLIT_DATE, DEFAULT_SPLIT_DATE


def _infer_instruments():
    try:
        with META_PATH.open("rb") as f:
            _meta = pickle.load(f)
        return sorted(_meta["industry"].index.get_level_values("instrument").unique())
    except Exception:
        return ["STK_A", "STK_B", "STK_C"]


START_DATE, END_DATE, TRAIN_END_DATE, TEST_START_DATE = _infer_calendar()
SAMPLE_INSTRUMENTS = _infer_instruments()
FIT_START_DATE = START_DATE
FIT_END_DATE = TRAIN_END_DATE

# Long-term handler config
long_handler_config = {
    "class": "qlib.data.dataset.handler.DataHandlerLP",
    "kwargs": {
        "start_time": START_DATE,
        "end_time": END_DATE,
        "instruments": SAMPLE_INSTRUMENTS,
        "data_loader": {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": FEATURE_PATH,
                    "label": LABEL_PATH,
                },
            },
        },
        "infer_processors": [
            {
                "class": "SafeRobustZScoreNorm",
                "module_path": "factor_workflow.processors",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": FIT_START_DATE,
                    "fit_end_time": FIT_END_DATE,
                },
            },
            {
                "class": "SafeFillna",
                "module_path": "factor_workflow.processors",
                "kwargs": {"fields_group": "feature"},
            },
        ],
        "learn_processors": [
            # {"class": "DropnaLabel", "kwargs": {}},  # Skip dropna since we have no valid labels
            {
                "class": "SafeCSRankNorm",
                "module_path": "factor_workflow.processors",
                "kwargs": {"fields_group": "feature"},
            },
        ],
    },
}

short_handler_config = {
    "class": "qlib.data.dataset.handler.DataHandlerLP",
    "kwargs": {
        "start_time": START_DATE,
        "end_time": END_DATE,
        "instruments": SAMPLE_INSTRUMENTS,
        "data_loader": {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": FEATURE_PATH,
                    "label": LABEL_PATH,
                },
            },
        },
        "infer_processors": [
            {
                "class": "SafeRobustZScoreNorm",
                "module_path": "factor_workflow.processors",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": FIT_START_DATE,
                    "fit_end_time": FIT_END_DATE,
                },
            },
            {
                "class": "SafeFillna",
                "module_path": "factor_workflow.processors",
                "kwargs": {"fields_group": "feature"},
            },
        ],
        "learn_processors": [
            # {"class": "DropnaLabel", "kwargs": {}},  # Skip dropna since we have no valid labels
            {
                "class": "SafeCSRankNorm",
                "module_path": "factor_workflow.processors",
                "kwargs": {"fields_group": "feature"},
            },
        ],
    },
}

long_dataset_config = {
    "class": "qlib.data.dataset.DatasetH",
    "kwargs": {
        "handler": long_handler_config,
        "segments": {
            "train": (START_DATE, TRAIN_END_DATE),
            "test": (TEST_START_DATE, END_DATE),
        },
    },
}

short_dataset_config = {
    "class": "qlib.data.dataset.DatasetH",
    "kwargs": {
        "handler": short_handler_config,
        "segments": {
            "train": (START_DATE, TRAIN_END_DATE),
            "test": (TEST_START_DATE, END_DATE),
        },
    },
}

__all__ = [
    "long_dataset_config", "short_dataset_config", "long_handler_config", "short_handler_config"
]
