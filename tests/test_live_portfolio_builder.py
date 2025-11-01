import pandas as pd
from live_trading.portfolio_builder import PortfolioBuilder
from live_trading.live_config import PortfolioConfig


def test_portfolio_builder_basic():
    cfg = PortfolioConfig(top_n=3, max_stock_weight=0.5, max_industry_weight=0.6, min_weight_threshold=0.01)
    builder = PortfolioBuilder(cfg)
    data = pd.DataFrame({
        'date': ['2025-10-24'] * 5,
        'code': ['000001', '000002', '000003', '000004', '000005'],
        'weight': [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    res = builder.build(data)
    assert res is not None
    tw = res.target_weights
    # top_n=3 selects 3 codes
    assert len(tw) == 3
    assert abs(tw['weight'].sum() - 1.0) < 1e-9
    # HHI within expected bounds
    hhi = (tw['weight'] ** 2).sum()
    assert hhi > 0


def test_portfolio_builder_industry_cap(tmp_path):
    # create industry map file
    ind_file = tmp_path / 'industry.csv'
    ind_df = pd.DataFrame({
        'code': ['000001', '000002', '000003', '000004'],
        'industry': ['A', 'A', 'A', 'B']
    })
    ind_df.to_csv(ind_file, index=False)
    cfg = PortfolioConfig(top_n=4, max_stock_weight=0.7, max_industry_weight=0.5, industry_map_file=str(ind_file))
    builder = PortfolioBuilder(cfg)
    data = pd.DataFrame({
        'date': ['2025-10-24'] * 4,
        'code': ['000001', '000002', '000003', '000004'],
        'weight': [0.9, 0.8, 0.7, 0.6],
    })
    res = builder.build(data)
    assert res is not None
    tw = res.target_weights
    a_weight = tw[tw['code'].isin(['000001', '000002', '000003'])]['weight'].sum()
    # industry cap enforced
    assert a_weight <= cfg.max_industry_weight + 1e-6
