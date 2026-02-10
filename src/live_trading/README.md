Live Trading skeleton

This package provides minimal scaffolding for the live trading components described in `docs/trading_system_design.md`.

Modules:
- prediction_loader.py: load and aggregate model outputs
- portfolio_builder.py: simple weight normalization
- risk_manager.py: rule chain skeleton
- execution_engine.py: diff -> orders + broker calls
- state_store.py: in-memory state snapshot
- drift_detector.py: IC/PSI check and retrain flag
- broker_adapter.py: BrokerAdapter interface + Xueqiu placeholder

Run quick import test:

```bash
python -c "import live_trading; print('ok')"
```
