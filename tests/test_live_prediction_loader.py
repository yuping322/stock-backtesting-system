import os
import pandas as pd
from live_trading.prediction_loader import PredictionLoader
from live_trading.live_config import DataIngestionConfig


def test_prediction_loader_aggregate(tmp_path):
    # create sample prediction files
    f1 = tmp_path / 'pred_a.csv'
    f2 = tmp_path / 'pred_b.csv'
    df1 = pd.DataFrame({
        'date': ['2025-10-23', '2025-10-23', '2025-10-24'],
        'code': ['000001', '000002', '000001'],
        'weight': [0.1, 0.2, 0.3],
    })
    df2 = pd.DataFrame({
        'date': ['2025-10-23', '2025-10-24'],
        'code': ['000001', '000002'],
        'weight': [0.4, 0.5],
    })
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)

    cfg = DataIngestionConfig(data_dir=str(tmp_path), file_pattern='*.csv', latest_days=2)
    loader = PredictionLoader(cfg)
    result = loader.load_latest()

    # check columns
    assert set(['date', 'code', 'weight']).issubset(result.columns)
    # dates normalized
    assert pd.api.types.is_datetime64_any_dtype(result['date'])
    # aggregation: weight for 2025-10-23 code 000001 should be mean of (0.1,0.4) = 0.25
    w_23 = result[(result['date'] == pd.Timestamp('2025-10-23')) & (result['code'] == '000001')]['weight'].iloc[0]
    assert abs(w_23 - 0.25) < 1e-6

    # latest day retained
    assert '2025-10-24' in {d.strftime('%Y-%m-%d') for d in result['date'].unique()}


def test_prediction_loader_missing_files(tmp_path):
    cfg = DataIngestionConfig(data_dir=str(tmp_path), file_pattern='*.csv', latest_days=1)
    loader = PredictionLoader(cfg)
    result = loader.load_latest()
    assert result.empty
