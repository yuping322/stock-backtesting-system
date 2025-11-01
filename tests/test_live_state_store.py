import pandas as pd
from datetime import datetime
from live_trading.state_store import StateStore
from live_trading.live_config import PersistenceConfig


def test_state_store_persistence(tmp_path):
    cfg = PersistenceConfig(state_dir=str(tmp_path))
    store = StateStore(cfg)

    # initial load empty
    state = store.load_state()
    assert state.positions.empty
    assert state.nav_history.empty

    # save positions
    pos = pd.DataFrame({'code': ['000001', '000002'], 'weight': [0.6, 0.4], 'avg_price': [10.5, 12.3]})
    store.save_positions(pos)
    state2 = store.load_state()
    assert len(state2.positions) == 2

    # append nav
    store.append_nav(datetime(2025, 10, 24), 1_000_000)
    store.append_nav(datetime(2025, 10, 25), 1_010_000)
    state3 = store.load_state()
    assert len(state3.nav_history) == 2
    assert state3.nav_history.iloc[1]['nav'] == 1_010_000

    # audit log
    store.audit("test_event", foo=123)
    audit_path = tmp_path / cfg.audit_file
    assert audit_path.exists()
    content = audit_path.read_text(encoding='utf-8')
    assert "test_event" in content
