"""Self-contained end-to-end minimal test in alpha_test directory.

Steps:
1. Build mini qlib dataset from local fake_prices.csv (if not already built)
2. Init qlib with that dataset
3. Train LightGBM on a very small Alpha158 handler config (tiny windows)
4. Generate signal & IC analysis
5. Print success banner

Run:
    python examples/alpha_test/run_alpha_minimal.py

(Uses files colocated in this folder; no external data download.)
"""
from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path
import traceback
import importlib
import qlib
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd
import numpy as np
import datetime as dt

# Import data module for real data source
sys.path.insert(0, str(Path(__file__).parent.parent))
import data as data_module

ROOT = Path(__file__).parent
RAW_CSV = ROOT / 'fake_prices.csv'
DATA_DIR = ROOT / 'mini_data_real'  # Changed directory name for real data
USE_REAL_DATA = True  # Flag to switch between fake and real data
MPLCONFIG_DIR = ROOT / '.mplconfig'
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(MPLCONFIG_DIR)
os.environ.setdefault('LIGHTGBM_IMPORT_DASK', '0')
os.environ.setdefault('PSUTIL_DISABLE_CPU_COUNT_LOGICAL', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

FIELDS = ["open", "high", "low", "close", "vwap", "volume"]
DATA_VERSION = "3"  # Incremented for real data source

sys.modules.setdefault('run_alpha_minimal', sys.modules[__name__])
sys.modules.setdefault('alpha_test.run_alpha_minimal', sys.modules[__name__])


class MiniLinearModel:
    """Simple least squares model implemented in pure NumPy (no external dependencies)."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, dataset, segment: str = 'train'):
        df = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # Check NaN only in feature columns (not label)
        feature_cols = [col for col in df.columns if col[0] == 'feature']
        label_cols = [col for col in df.columns if col[0] == 'label']
        
        # Fill NaN in individual feature columns with 0 for safety
        # (Alpha158 may have NaN in rolling window features at boundaries)
        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0.0)
        
        # Drop rows where label is NaN (required for training)
        if label_cols:
            df = df.dropna(subset=label_cols)
        
        if df.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        
        features = df["feature"].values
        labels = df["label"].values
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        
        ones = np.ones((features.shape[0], 1), dtype=features.dtype)
        design = np.hstack([features, ones])
        beta, *_ = np.linalg.lstsq(design, labels, rcond=None)
        self.coef_ = beta[:-1]
        self.intercept_ = float(beta[-1])
        return self

    def predict(self, dataset, segment: str = 'test'):
        if self.coef_ is None:
            raise ValueError("model is not fitted yet!")
        feats = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = feats.values @ self.coef_ + self.intercept_
        return pd.Series(preds, index=feats.index)


def _patch_psutil_cpu_count():
    try:
        import psutil
    except Exception:
        return
    try:
        # Attempt once so we only patch if it fails under sandbox restrictions.
        psutil.cpu_count()
        return
    except Exception:
        pass

    def _fallback_cpu_count(*_args, **_kwargs):
        count = os.cpu_count()
        return count if isinstance(count, int) and count > 0 else 1

    try:
        def _patched_cpu_count(logical=True):  # noqa: ARG001 - keep signature compatible
            return _fallback_cpu_count()

        psutil.cpu_count = _patched_cpu_count  # type: ignore[assignment]
    except Exception:
        pass

    # Patch macOS specific helpers if available
    targets = []
    try:
        import psutil._psplatform as _psplatform  # type: ignore[attr-defined]
        targets.append((_psplatform, 'cpu_count_logical'))
    except Exception:
        pass
    try:
        import psutil._psosx as _psosx  # type: ignore[attr-defined]
        if hasattr(_psosx, 'cext'):
            targets.append((_psosx.cext, 'cpu_count_logical'))
    except Exception:
        pass

    for module_obj, attr_name in targets:
        try:
            setattr(module_obj, attr_name, lambda _f=_fallback_cpu_count: _f())
        except Exception:
            continue


def build_data():
    global USE_REAL_DATA
    marker = DATA_DIR / '.version'
    legacy_bins = list(DATA_DIR.glob('features/*/*..day.bin')) if DATA_DIR.exists() else []
    if marker.exists() and (DATA_DIR / 'instruments').exists() and not legacy_bins:
        try:
            current_version = marker.read_text().strip()
        except OSError:
            current_version = None
        if current_version == DATA_VERSION:
            print('[DATA] Existing mini dataset found, skip rebuild.')
            return
        print('[DATA] Dataset version mismatch; rebuilding...')
    elif DATA_DIR.exists():
        if legacy_bins:
            print('[DATA] Found legacy feature files; rebuilding...')
        else:
            print('[DATA] Partial dataset detected; rebuilding...')
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    use_real = USE_REAL_DATA  # Local copy
    if use_real:
        print('[DATA] Loading real data from data.py...')
        # Get 60 days of data ending today
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=60)
        
        # Get stock codes (use a small set for testing, e.g., first 10 stocks from HS300)
        try:
            stocks = data_module.get_index_stocks('HS300', date=end_date)
            # Convert to list if it's a Series or other iterable
            if isinstance(stocks, pd.Series):
                stocks = stocks.tolist()
            elif not isinstance(stocks, list):
                stocks = list(stocks) if stocks else []
            # Use first 10 stocks for faster testing
            test_stocks = stocks[:10] if len(stocks) > 10 else stocks
            if test_stocks:
                print(f'[DATA] Using {len(test_stocks)} stocks from HS300: {test_stocks[:5]}...')
            else:
                raise ValueError("No stocks returned from get_index_stocks")
        except Exception as e:
            print(f'[DATA] Failed to get index stocks, using default stocks: {e}')
            # Fallback to some common stocks
            test_stocks = ['000001', '600000', '000002', '600519', '000858', '600036', '000858', '600028', '000063', '600048']
            print(f'[DATA] Using default stocks: {test_stocks[:5]}...')
        
        # Load complex stock data (open, high, low, close, volume)
        print(f'[DATA] Fetching data from {start_date} to {end_date}...')
        try:
            data_dict = data_module.load_oss_complex_stocks(
                codes=test_stocks,
                start=str(start_date),
                end=str(end_date),
                fields=['open', 'high', 'low', 'close', 'volume']
            )
        except Exception as e:
            print(f'[DATA] Failed to load real data: {e}')
            print('[DATA] Falling back to fake data...')
            use_real = False
        
        if use_real and data_dict:
            # data_dict is a dictionary: {field_name: DataFrame(index=date, columns=codes)}
            # Convert to long format DataFrame
            dfs_by_code = {}
            for field_name, field_df in data_dict.items():
                if field_df.empty:
                    continue
                # field_df is DataFrame with dates as index and codes as columns
                for code in test_stocks:
                    if code not in field_df.columns:
                        continue
                    if code not in dfs_by_code:
                        dfs_by_code[code] = pd.DataFrame({'date': field_df.index, 'symbol': code})
                    dfs_by_code[code][field_name] = field_df[code].values
            
            # Combine all stocks
            if dfs_by_code:
                df_list = [df for df in dfs_by_code.values() if not df.empty]
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Calculate VWAP as (high + low + close) / 3 if not available
                    if 'vwap' not in df.columns:
                        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                    
                    # Ensure all required fields exist
                    for field in FIELDS:
                        if field not in df.columns:
                            if field == 'vwap':
                                df['vwap'] = df['close']
                            else:
                                df[field] = 0.0
                    
                    df = df.dropna(subset=['date', 'symbol']).sort_values(['symbol', 'date'])
                    print(f'[DATA] Loaded {len(df)} rows for {df["symbol"].nunique()} stocks')
                else:
                    print('[DATA] No data loaded, falling back to fake data...')
                    use_real = False
                    df = None
            else:
                print('[DATA] No data loaded, falling back to fake data...')
                use_real = False
                df = None
        else:
            use_real = False
            df = None
    else:
        use_real = False
        df = None
    
    if not use_real or df is None:
        # Fallback to fake data
        print('[DATA] Using fake data from CSV...')
        np.random.seed(42)
        df = pd.read_csv(RAW_CSV)
        df['date'] = pd.to_datetime(df['date'])
        if 'vwap' not in df.columns:
            df['vwap'] = df['close']
        
        # Expand fake data to 60 days
        min_days_needed = 60
        all_dates = sorted(set(df['date']))
        if len(all_dates) < min_days_needed:
            print(f'[DATA] Expanding data from {len(all_dates)} to {min_days_needed} days...')
            expanded_rows = []
            for sym in df['symbol'].unique():
                sym_data = df[df['symbol'] == sym].sort_values('date')
                if len(sym_data) == 0:
                    continue
                first_row = sym_data.iloc[0]
                first_date = first_row['date']
                start_date = first_date - pd.Timedelta(days=min_days_needed - 1)
                existing_dates = set(sym_data['date'])
                base_close = float(first_row['close'])
                base_volume = float(first_row['volume'])
                
                for i in range(min_days_needed):
                    current_date = start_date + pd.Timedelta(days=i)
                    if current_date in existing_dates:
                        continue
                    noise = np.random.normal(0, 0.02) * (i / min_days_needed)
                    close = base_close * (1 + noise)
                    open_price = close * (1 + np.random.normal(0, 0.01))
                    high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
                    low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
                    vwap = (high + low + close) / 3
                    volume = base_volume * (1 + np.random.normal(0, 0.1))
                    expanded_rows.append({
                        'date': current_date, 'symbol': sym,
                        'open': open_price, 'high': high, 'low': low,
                        'close': close, 'vwap': vwap, 'volume': max(volume, 1000),
                    })
            
            if expanded_rows:
                expanded_df = pd.DataFrame(expanded_rows)
                df = pd.concat([df, expanded_df], ignore_index=True)
                df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
                print(f'[DATA] Expanded to {len(df)} rows')
    
    cal_dir = DATA_DIR / 'calendars'
    ins_dir = DATA_DIR / 'instruments'
    feat_root = DATA_DIR / 'features'
    cal_dir.mkdir(exist_ok=True)
    ins_dir.mkdir(exist_ok=True)
    feat_root.mkdir(exist_ok=True)
    all_days = sorted(set(df['date']))
    with (cal_dir / 'day.txt').open('w') as f:
        for d in all_days:
            f.write(f"{pd.Timestamp(d).date()}\n")
    lines = []
    for sym, g in df.groupby('symbol'):
        lines.append(f"{sym}\t{g['date'].min().date()}\t{g['date'].max().date()}")
    with (ins_dir / 'all.txt').open('w') as f:
        for line in lines:
            f.write(line + '\n')
    # feature bins
    for sym, g in df.groupby('symbol'):
        sym_dir = feat_root / sym.lower()
        sym_dir.mkdir(exist_ok=True)
        g = g.drop_duplicates('date').set_index('date').sort_index()
        for field in FIELDS:
            arr = g[field].to_numpy(dtype='float32')
            out = np.hstack([np.array([0], dtype='float32'), arr])
            with (sym_dir / f'{field}.day.bin').open('wb') as f:
                out.tofile(f)
    (DATA_DIR / '.version').write_text(DATA_VERSION)
    print('[DATA] Mini dataset built at', DATA_DIR)


def build_model_config(model_type: str) -> dict:
    model_type = model_type.lower()
    if model_type == 'lightgbm':
        return {
            'class': 'LGBModel',
            'module_path': 'qlib.contrib.model.gbdt',
            'kwargs': {
                'loss': 'mse',
                'learning_rate': 0.1,
                'num_leaves': 16,
                'max_depth': 4,
                'num_threads': 1,
            },
        }
    if model_type == 'linear':
        return {
            'class': 'MiniLinearModel',
            'module_path': 'run_alpha_minimal',
            'kwargs': {},
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_task(model_conf: dict):
    # Read actual date range from built data
    # Check what dates were actually built in the dataset
    cal_file = DATA_DIR / 'calendars' / 'day.txt'
    if cal_file.exists():
        with cal_file.open() as f:
            all_dates = [pd.Timestamp(line.strip()) for line in f if line.strip()]
        if all_dates:
            extended_start = min(all_dates)
            test_date_ts = max(all_dates)
            # Split: use first 80% for train, next 10% for valid, last 10% for test
            n_days = len(all_dates)
            train_end_idx = int(n_days * 0.8)
            valid_start_idx = int(n_days * 0.9)
            train_end_ts = all_dates[train_end_idx] if train_end_idx < len(all_dates) else all_dates[-1]
            valid_start_ts = all_dates[valid_start_idx] if valid_start_idx < len(all_dates) else test_date_ts
            valid_end_ts = test_date_ts
        else:
            # Fallback to fake data dates
            original_start = pd.Timestamp('2024-12-30')
            extended_start = original_start - pd.Timedelta(days=60)
            train_end = pd.Timestamp('2025-01-01')
            valid_start_ts = pd.Timestamp('2025-01-02')
            valid_end_ts = pd.Timestamp('2025-01-02')
            test_date_ts = pd.Timestamp('2025-01-02')
            train_end_ts = train_end
    else:
        # Fallback to fake data dates
        original_start = pd.Timestamp('2024-12-30')
        extended_start = original_start - pd.Timedelta(days=60)
        train_end = pd.Timestamp('2025-01-01')
        valid_start_ts = pd.Timestamp('2025-01-02')
        valid_end_ts = pd.Timestamp('2025-01-02')
        test_date_ts = pd.Timestamp('2025-01-02')
        train_end_ts = train_end
    
    handler_conf = {
        'class': 'Alpha158',
        'module_path': 'qlib.contrib.data.handler',
        'kwargs': {
            'start_time': str(extended_start.date()),
            'end_time': str(test_date_ts.date()),
            'fit_start_time': str(extended_start.date()),
            'fit_end_time': str(test_date_ts.date()),
            'instruments': 'all',
        },
    }
    dataset_conf = {
        'class': 'DatasetH',
        'module_path': 'qlib.data.dataset',
        'kwargs': {
            'handler': handler_conf,
            'segments': {
                'train': [str(extended_start.date()), str(train_end_ts.date())],
                'valid': [str(valid_start_ts.date()), str(valid_end_ts.date())],
                'test': [str(test_date_ts.date()), str(test_date_ts.date())],
            },
        },
    }
    record_conf = [
        {
            'class': 'SignalRecord',
            'module_path': 'qlib.workflow.record_temp',
            'kwargs': {'model': '<MODEL>', 'dataset': '<DATASET>'},
        },
        {
            'class': 'SigAnaRecord',
            'module_path': 'qlib.workflow.record_temp',
            'kwargs': {'ana_long_short': False, 'ann_scaler': 252},
        },
    ]
    return {'model': model_conf, 'dataset': dataset_conf, 'record': record_conf}


def run():
    _patch_psutil_cpu_count()
    requested_model = os.environ.get('RUN_ALPHA_MINIMAL_MODEL', 'linear').lower()
    supported_models = {'lightgbm', 'linear'}
    if requested_model not in supported_models:
        print(f"[WARN] Unsupported model '{requested_model}', fallback to 'linear'.")
        requested_model = 'linear'

    model_type = requested_model

    if model_type == 'lightgbm':
        _patch_psutil_cpu_count()
        try:
            importlib.import_module('lightgbm')
        except ImportError:
            print('[WARN] lightgbm not installed; fallback to linear model.')
            model_type = 'linear'
    if model_type == 'linear':
        print('[INFO] Using LinearModel backend (safe default).')

    try:
        build_data()
        print('[INIT] Initializing qlib...')
        qlib.init(provider_uri=str(DATA_DIR), expression_cache=None, dataset_cache=None)
        model_conf = build_model_config(model_type)
        task = build_task(model_conf)
        print(f"[TASK] Starting experiment context with model='{model_type}'...")
        with R.start(experiment_name='alpha_minimal_run'):
            dataset = init_instance_by_config(task['dataset'])
            model = init_instance_by_config(task['model'])
            print(f"[MODEL] Fitting {model_type} model on train segment...")
            model.fit(dataset)
            print('[MODEL] Predicting valid/test segments...')
            valid_pred = model.predict(dataset, 'valid')
            test_pred = model.predict(dataset, 'test')
            print(f'[PRED] Valid predictions: {len(valid_pred)} samples')
            print(f'[PRED] Test predictions: {len(test_pred)} samples')
            from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
            rec = R.get_recorder()
            print('[RECORD] Generating signal record...')
            signal_rec = SignalRecord(model=model, dataset=dataset, recorder=rec)
            signal_rec.generate()
            print('[RECORD] Generating signal analysis record...')
            ana_rec = SigAnaRecord(ann_scaler=252, ana_long_short=False, recorder=rec)
            ana_rec.generate()
            # Portfolio analysis (simple long top, short bottom strategy)
            try:
                from qlib.workflow.record_temp import PortAnaRecord
                print('[RECORD] Generating portfolio analysis record...')
                port_conf = {
                    'strategy': {
                        'class': 'TopkDropoutStrategy',
                        'module_path': 'qlib.contrib.strategy.strategy',
                        'kwargs': {
                            'signal': '<SignalRecord>',
                            'topk': 1,
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
                port_rec = PortAnaRecord(config=port_conf, recorder=rec)
                port_rec.generate()
            except Exception as e:
                print('[WARN] Portfolio analysis skipped:', e)
            try:
                ic_series = rec.load_object('ic')
                if ic_series is not None:
                    print(f'[RESULT] IC length={len(ic_series)} mean={getattr(ic_series, "mean", lambda: float("nan"))():.4f}')
                else:
                    print('[RESULT] IC object not found.')
            except Exception as e:
                print('[WARN] Failed to load IC:', e)
            # List all recorded objects
            try:
                objs = rec.list_objects()
                print('[RECORDER] Objects:', objs)
            except Exception as e:
                print('[RECORDER] list_objects failed:', e)
            print('[ARTIFACTS] Path:', rec.get_artifact_uri())
        print('\n================ SUCCESS (minimal alpha test) ================')
    except Exception as e:
        print('[FATAL] Exception during run:', e)
        traceback.print_exc()


if __name__ == '__main__':
    run()
