import os
import pandas as pd
from live_trading.run_live import run_live
from live_trading.live_config import LiveConfig, DataIngestionConfig, PortfolioConfig, PersistenceConfig


def test_run_live_end_to_end(tmp_path):
    # prepare prediction file
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    pred_file = data_dir / 'pred.csv'
    df = pd.DataFrame({
        'date': ['2025-10-24'] * 4,
        'code': ['000001', '000002', '000003', '000004'],
        'weight': [0.4, 0.3, 0.2, 0.1]
    })
    df.to_csv(pred_file, index=False)

    persistence_dir = tmp_path / 'state'
    cfg = LiveConfig(
        data=DataIngestionConfig(data_dir=str(data_dir), file_pattern='*.csv', latest_days=1),
        portfolio=PortfolioConfig(top_n=3),
        persistence=PersistenceConfig(state_dir=str(persistence_dir)),
    )

    run_live(cfg, total_equity=1_000_000)

    # check outputs
    assert (persistence_dir / 'positions.csv').exists()
    assert (persistence_dir / 'nav.csv').exists()
    assert (persistence_dir / 'audit.log').exists()
