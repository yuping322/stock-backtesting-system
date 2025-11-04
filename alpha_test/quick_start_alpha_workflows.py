"""Quick start script: Run LightGBM on Alpha158 and Alpha360 datasets.

Usage:
    python examples/quick_start_alpha_workflows.py --data ~/.qlib/qlib_data/cn_data

It will:
1. Init qlib
2. Define two tasks (Alpha158 & Alpha360) using simplified configs
3. Train LightGBM model for each (single run) and record signals & basic analysis
4. Print IC mean/std of validation & test and portfolio analysis summary files location.

Requirements:
    pip install qlib lightgbm

Optional:
    --seed to set random seed

Note: This script is a condensed version of benchmark workflows, focusing on getting a first successful run quickly.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.contrib.data.loader import Alpha158DL

# Common model config (can override learning_rate between datasets)
MODEL_BASE = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 4,  # lower threads for laptop
    },
}

DATA_HANDLER_BASE = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}
SEGMENTS = {
    "train": ["2008-01-01", "2014-12-31"],
    "valid": ["2015-01-01", "2016-12-31"],
    "test": ["2017-01-01", "2020-08-01"],
}

def build_task(dataset_class: str, handler_class: str, learning_rate: float):
    model_conf = MODEL_BASE.copy()
    model_conf["kwargs"] = dict(model_conf["kwargs"], learning_rate=learning_rate)
    task = {
        "model": model_conf,
        "dataset": {
            "class": dataset_class,
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": handler_class,
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": DATA_HANDLER_BASE,
                },
                "segments": SEGMENTS,
            },
        },
        "record": [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            },
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            },
        ],
    }
    return task

def get_alpha158_features(full: bool = False):
    """Return (fields, names) for Alpha158.
    full=True will use an extended config approximating the complete 158-factor set (price+volume multiple windows + rolling operators).
    """
    if full:
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4],
            },
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": None,  # None => use default operators
                "exclude": [],
            },
        }
    else:
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
    return Alpha158DL.get_feature_config(conf)

def _print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def ensure_data(path: str):
    exp_path = os.path.expanduser(path)
    # Simple heuristic: expect instruments directory
    expected = os.path.join(exp_path, "instruments")
    if os.path.exists(expected):
        print(f"Data found at {exp_path}")
        return
    _print_header("Data not found; auto downloading simplified CN daily dataset (qlib_data_simple)...")
    from qlib.tests.data import GetData
    try:
        GetData().qlib_data(name="qlib_data_simple", target_dir=exp_path, interval="1d", region="cn", exists_skip=False, delete_old=False)
        print("Download finished.")
    except Exception as e:
        print(f"Auto download failed: {e}\nPlease manually run: python scripts/get_data.py qlib_data --target_dir {exp_path} --region cn")

def run_task(name: str, task_conf: dict, with_port: bool = False):
    print(f"===== Running task: {name} =====")
    # Start experiment context
    with R.start(experiment_name=name):
        # Build model & dataset
        dataset = init_instance_by_config(task_conf["dataset"])
        model = init_instance_by_config(task_conf["model"])
        train_ds = dataset.get_segment("train")
        model.fit(train_ds)
        # Predictions (trigger any internal caching)
        _ = model.predict(dataset.get_segment("valid"))
        _ = model.predict(dataset.get_segment("test"))
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
        sig_rec = SignalRecord(model=model, dataset=dataset)
        sig_rec.generate()
        ana_rec = SigAnaRecord(ann_scaler=252, ana_long_short=False)
        ana_rec.generate()
        if with_port:
            try:
                from qlib.workflow.record_temp import PortAnaRecord
                print('[PORT] Generating portfolio analysis record...')
                port_conf = {
                    'strategy': {
                        'class': 'TopkDropoutStrategy',
                        'module_path': 'qlib.contrib.strategy.strategy',
                        'kwargs': {
                            'signal': '<SignalRecord>',
                            'topk': 5,
                            'n_drop': 0,
                        },
                    },
                    'backtest': {
                        'exchange': {
                            'class': 'SIMExchange',
                            'module_path': 'qlib.backtest.exchange',
                            'kwargs': {
                                'freq': 'day',
                                'limit_threshold': 0.095,
                                'deal_price': 'close',
                                'open_cost': 0.0005,
                                'close_cost': 0.0005,
                                'min_cost': 5,
                            },
                        },
                        'benchmark': 'SH000300',
                        'account': 100000000,
                        'pos_type': 'Position',
                        'verbose': False,
                        'risk_model': {
                            'class': 'RiskModelDiscrete',
                            'module_path': 'qlib.contrib.backtest.risk_model',
                        },
                    },
                }
                PortAnaRecord(config=port_conf).generate()
            except Exception as e:
                print(f'[PORT][WARN] Portfolio analysis failed: {e}')
        rec = R.get_recorder()
        try:
            ic_series = rec.load_object("ic")
            if ic_series is not None and hasattr(ic_series, "mean"):
                print(f"IC summary ({name}): mean={ic_series.mean():.4f} std={ic_series.std():.4f} count={len(ic_series)}")
        except Exception as e:
            print(f"No IC object for {name}: {e}")
        try:
            objs = rec.list_objects()
            print(f'[RECORDER] {name} objects: {objs}')
        except Exception as e:
            print(f'[RECORDER][WARN] list_objects failed: {e}')
        print(f"Artifacts path: {rec.get_artifact_uri()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-alpha158", action="store_true", help="Print Alpha158 factor names before running")
    parser.add_argument("--full-alpha158", action="store_true", help="Use extended Alpha158 config when printing list (does not change training handler).")
    parser.add_argument("--with-port", action="store_true", help="Generate portfolio analysis record.")
    args = parser.parse_args()

    provider_uri = os.path.expanduser(args.data)
    print(f"[INFO] Using data directory: {provider_uri}")
    ensure_data(provider_uri)
    qlib.init(mount_path=provider_uri, region=REG_CN)

    if args.show_alpha158:
        fields, names = get_alpha158_features(full=args.full_alpha158)
        print(f"[INFO] Alpha158 feature count: {len(names)}")
        # Show a preview
        preview = 20
        print("[INFO] First %d names: %s" % (min(preview, len(names)), ", ".join(names[:preview])))
        if len(names) > preview:
            print("[INFO] Last name example: %s" % names[-1])

    # Alpha158
    task_alpha158 = build_task("DatasetH", "Alpha158", learning_rate=0.2)
    try:
        run_task("quick_lightgbm_alpha158", task_alpha158, with_port=args.with_port)
    except Exception as e:
        print(f"[ERROR] Alpha158 task failed: {e}")

    # Alpha360 (different lr)
    task_alpha360 = build_task("DatasetH", "Alpha360", learning_rate=0.0421)
    try:
        run_task("quick_lightgbm_alpha360", task_alpha360, with_port=args.with_port)
    except Exception as e:
        print(f"[ERROR] Alpha360 task failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
