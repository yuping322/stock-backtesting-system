import pandas as pd
from src.live_trading.risk_manager import CompositeRiskEngine, BlacklistRule, MarketCapRule


def test_risk_manager_blacklist_rule():
    """Test that blacklist rule filters out blacklisted stocks"""
    rules = [BlacklistRule()]
    risk_engine = CompositeRiskEngine(rules)
    
    weights = pd.DataFrame({
        'code': ['000001', '000002', '000003'],
        'target_weight': [0.4, 0.3, 0.3]
    })
    
    blacklist = pd.DataFrame({
        'code': ['000002'],
        'reason': ['suspended']
    })
    
    ctx = {
        'weights': weights,
        'blacklist': blacklist,
        'panel': pd.DataFrame(),
        'industry': pd.DataFrame()
    }
    
    result = risk_engine.run(ctx)
    
    # Should remove blacklisted stock
    assert len(result['weights']) == 2
    assert '000002' not in result['weights']['code'].values
    assert '000001' in result['weights']['code'].values
    assert '000003' in result['weights']['code'].values


def test_risk_manager_market_cap_rule():
    """Test that market cap rule filters out low market cap stocks"""
    rules = [MarketCapRule(min_market_cap=1e9)]  # 1 billion
    risk_engine = CompositeRiskEngine(rules)
    
    weights = pd.DataFrame({
        'code': ['000001', '000002'],
        'target_weight': [0.5, 0.5]
    })
    
    panel = pd.DataFrame({
        'code': ['000001', '000002'],
        'market_cap': [2e9, 5e8]  # 000002 below threshold
    })
    
    ctx = {
        'weights': weights,
        'blacklist': pd.DataFrame(),
        'panel': panel,
        'industry': pd.DataFrame()
    }
    
    result = risk_engine.run(ctx)
    
    # Should remove low market cap stock
    assert len(result['weights']) == 1
    assert '000001' in result['weights']['code'].values
    assert '000002' not in result['weights']['code'].values
