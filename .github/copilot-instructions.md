# Copilot Instructions for Stock Backtesting System
- `integrated_backtesting_system.py`: Legacy all-in-one Streamlit app; deprecated, slated for deletion—use `app.py`.
- `backtrader_base_strategy.py`: Historical base; migrate needed logic to `backtest_engine.py`, then remove.
- `start.sh`: Points to deprecated file; update to `streamlit run app.py`.
- Refactor: Ensure `app.py` and `main.py` import from `backtest_engine.py`; avoid stale references.

## Architecture snapshot
- `app.py` is the Streamlit UI: builds sidebar configs, scans `data/` for prediction CSVs, delegates to `BacktestEngine` for execution and displays results via tabs (NAV charts, holdings, trades, metrics).
- `main.py` is a CLI wrapper: parses args (e.g., --data-file, --strategy, --benchmark), instantiates `BacktestEngine`, runs backtest, and saves results to `results/` or logs.
- `backtest_engine.py` owns the core `BacktestEngine` class, `StrategyFactory` for strategy registration, `DataLoader` for prediction data, and `AnalysisBuilder` for metrics; consolidates logic from legacy `backtrader_base_strategy.py`.
- `data.py` centralizes data fetchers: OSS buckets for snapshots, ModelScope for CSVs, AkShare for real-time/fallbacks; includes Alphalens integration for factor analysis and caching in `stock_cache/`.
- `config.py` defines `SystemConfig` (fees, dates), `StrategyConfig` (params like top_n, hold_days), and constants like `BENCHMARK_INDICES`.

Data flows: Prediction CSVs (date, code, weight) → `DataLoader.load_prediction_data` → normalize codes/dates → `BacktestEngine.run_single_backtest` → load stock feeds via `data.load_bt_stocks` → execute strategy → align NAVs against benchmark from `data.get_index_daily` → return `BacktestResult` for analysis.

Structural decisions: Separation of UI/CLI from engine enables headless runs; modular data sources allow graceful degradation (OSS → AkShare); Backtrader integration provides robust simulation with custom strategies subclassing `BaseStrategy`.

## Data sources & environment
- Real-time/history data: OSS buckets (via `OSS_ACCESS_KEY_ID|SECRET|ENDPOINT|BUCKET`), ModelScope endpoints, AkShare HTTP calls; credentials optional—missing ones disable features without crashes.
- Prediction CSVs in `data/`: Must have `date`, `code` (normalized to 6-digit), optional `weight` (defaults to 1.0); `DataLoader.load_prediction_data` handles normalization.
- Stock codes: Always normalize via `data._normalize_code_arg` or `data._ensure_exchange_prefix` (e.g., '000001' → '000001.XSHE'); avoid raw symbols.
- Caching: Stock data cached in `stock_cache/` as `.pkl`; delete outdated files to force refresh. Factor results saved to OSS `daily_metrics/` as multi-indexed CSVs.

## Running backtests
- UI: `streamlit run app.py` (scans `data/` for CSVs, configures via sidebar, runs via `BacktestEngine`).
- CLI: `python main.py --data-file data/test_sample_predictions.csv --strategy weighted_top_n --benchmark sh000300 --output-dir results/sample_run` (overrides defaults like --hold-days, --top-n).
- Engine: `BacktestEngine.run_single_backtest` filters predictions to stocks with feeds, reindexes NAVs; mismatches (empty feeds) logged but don't crash—check `BacktestResult.valid_stocks`.

## Strategy development
- Add strategies: Subclass `BaseStrategy` in `backtest_engine.py`, implement `execute_strategy` (e.g., select top_n by weight, rebalance every hold_days), register in `StrategyFactory._strategies`.
- Configs: Expose params in `config.STRATEGY_PARAMS` (e.g., `top_n`, `hold_days`) for UI/CLI knobs; `StrategyConfig.parameters` passes them.
- Recording: Use `_record_trade`, `_record_holdings` in `BaseStrategy`; NAVs auto-tracked in `self._daily_navs`.

## Working with data utilities
- Feeds: `data.load_bt_stocks` prefers OSS snapshots, falls back to AkShare via `_load_bt_stocks_fallback`, synthesizes if needed.
- Factor analysis: `data.factor_for_al` expects (date, asset) MultiIndex series; calls Alphalens `mean_information_coefficient` etc.; persist via `data.save_result` to OSS.
- Dates: Always `pd.to_datetime(...).dt.normalize()` (e.g., in `DataLoader`); mismatched times break Backtrader alignments.
- Empty handling: Loaders return `pd.DataFrame(dtype=float)` if no data—check `.empty` before operations.

## Testing & verification
- Tests in `tests/`: Mock `akshare`, `alphalens`, `oss2` (e.g., via `sys.modules` stubs); run with `pytest tests/test_data.py`.
- Validation: After changes, run CLI backtest (`python main.py ...`) to verify feeds, NAV alignment, metrics; check logs for errors.

## Conventions & pitfalls
- Code normalization: Use `_normalize_code_arg` for consistency; raw codes cause mismatches.
- Date handling: Normalize to date-only; time components misalign NAVs.
- Empty DataFrames: Guard with `.empty` checks; common in degraded modes.
- OSS guards: Check `bucket is not None` before operations; local/dev runs often lack creds.
- Chinese headers: `_wide_to_ohlcv` relies on exact strings like `"今开"`, `"成交量"`.
- Imports: Bilingual-friendly; avoid assuming English-only.

## Deprecations & migration notes
- `integrated_backtesting_system.py` is a legacy “all-in-one” Streamlit runner kept only for reference and queued for deletion; avoid wiring new features there.
- `backtrader_base_strategy.py` and its cache helpers (`stock_cache/`, `etf_cache/`) represent the old layering. New work should live in `backtest_engine.py`; plan to migrate any still-needed logic before dropping the legacy file.
- When pruning historical modules, double-check `app.py` and `main.py` imports point to the consolidated engine to prevent stale references.
