import pandas as pd
from live_trading.drift_detector import DriftDetector
from live_trading.live_config import RiskConfig


def test_drift_detector_ic_and_trigger():
    cfg = RiskConfig(ic_rolling_window=3, min_ic_threshold=0.05)
    detector = DriftDetector(cfg)

    # simulate three days of predictions + realized returns with descending correlation
    for i, ic_pattern in enumerate([0.5, 0.02, -0.1]):
        date = f"2025-10-2{i+1}"
        # construct weights/returns with controllable correlation
        # For simplicity, weight = [0,1,2,3,4]; returns = weight * ic_pattern + noise
        weights = list(range(5))
        returns = [w * ic_pattern for w in weights]
        pred_df = pd.DataFrame({'date': [date]*5, 'code': [f"00000{j}" for j in range(5)], 'weight': weights})
        ret_df = pd.DataFrame({'date': [date]*5, 'code': [f"00000{j}" for j in range(5)], 'return': returns})
        detector.update(pred_df, ret_df)

    status = detector.evaluate()
    assert status is not None
    # last ic should reflect last pattern sign (negative or near)
    assert status.latest_ic <= 0.01
    # rolling ic below threshold triggers retrain
    assert status.trigger_retrain is True
