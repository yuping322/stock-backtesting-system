# 文档索引 — docs/

本文件为 `docs/` 目录的索引与整理，快速指引仓库中各份文档的用途、代码映射与推荐的后续动作。

目录（按文件名）

- `PROJECT_ANALYSIS.md`
  - 概要：项目的架构与功能整体快照，包含技术栈、主要模块与核心能力摘要。
  - 代码映射：总体对应 `app.py`, `backtrader_base_strategy.py`, `data.py`, `backtest_engine.py`。
  - 用途：项目简介、对外汇报和新成员入门。

- `analysis_layer_design.md`
  - 概要：分析/展示层重构设计，定义 `BacktestEngine` 输出结构与 `AnalysisBuilder` 的拆分接口。
  - 代码映射：`backtest_engine.py::AnalysisBuilder`、`app.py` 中的展示调用。
  - 用途：为将来将分析逻辑从 UI 中抽离成可复用模块提供设计文档。

- `data_module.md`
  - 概要：`data.py` 的功能说明，列出与 OSS、ModelScope、日线/快照相关的主要接口与注意事项。
  - 代码映射：`data.py`（`load_bt_stocks`, `load_bt_oss_stocks`, `_wide_to_ohlcv`, `get_index_daily` 等）。
  - 用途：数据工程、调试数据源、确认 CSV/OSS 格式与字段约定。

- `backtest_improvements.md`
  - 概要：回测/评价层面的改进建议（交易成本、数据质量、风险指标、验证框架等）。
  - 代码映射：`backtest_engine.py`（成本、性能指标）、`app.py`（展示）与 `data.py`（数据质量）。
  - 用途：用于 roadmap、技术负债清单与优先级评估。

- `strategy_selection.md`  (新整理)
  - 概要：针对 `date,code,weight` 类型预测文件，逐条说明 `direct_execution`, `weighted_top_n`, `equal_weight`, `momentum` 的行为差异、参数与测试建议。
  - 代码映射：`backtest_engine.py` 中的相应类与 `config.py` 的参数定义。
  - 用途：帮助开发者/产品人员选择合适策略并解释潜在陷阱（例如权重归一化、缺席是否卖出等）。

快速导航

- 想看数据接口（字段/OSS 路径/格式）→ 打开 `docs/data_module.md`。
- 想理解分析层输出结构与复用接口 → 打开 `docs/analysis_layer_design.md`。
- 想选择策略并知道行为差异 → 打开 `docs/strategy_selection.md`。
- 想查看项目整体与 roadmap → 打开 `docs/PROJECT_ANALYSIS.md` 和 `docs/backtest_improvements.md`。

建议的整理动作（短期：可在一两天内完成）

1. 在每个 docs 文件顶部加上 `Last updated: YYYY-MM-DD`（便于追踪文档新旧）。我可以替你批量添加。
2. 在 `docs/` 下建立 `examples/` 或 `notebooks/` 文件夹，用于放入 smoke test 的最小示例 CSV、以及演示脚本（例如 `examples/run_direct_execution_smoke.py`）。
3. 将 `docs/strategy_selection.md` 的建议小改进（如 direct_execution 的缺席清仓）变成 GitHub issue 或 TODO，并优先排序。

如何我可以帮助下一步

- 直接实现 `DirectExecutionStrategy` 的“缺席即清仓”代码并运行 smoke test（会修改 `backtest_engine.py` 并执行 `main.py` 的一轮短回测）。
- 批量给每个 docs 文件加上 `Last updated` 时间戳，并把 `docs/` 目录下文件名规范化（例如 `data_module.md` → `data-module.md`，如果你同意）。

若你同意其中某项，我就开始做并在完成后提交补丁与回测结果摘要。
