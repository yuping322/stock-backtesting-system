# 分析层重构设计说明

## 1. 背景

当前 `app.py` 中的 `display_detailed_analysis` 同时承担了数据准备、指标计算和 Streamlit 展示，逻辑集中在一个函数里，维护成本较高。一旦页面结构扩展或需要在其他入口复用分析逻辑，就必须拆分出独立的分析层。本设计旨在明确：

1. 如何独立运行 `BacktestEngine` 并产出统一的数据包；
2. 如何围绕该数据包划分分析大类，形成可复用的准备函数；
3. 如何在 UI 层将这些分析结果映射到七个 Tab，并保持现有体验。

## 2. 设计目标

- **解耦**：将数据计算/加工与 Streamlit 展示逻辑分离。
- **复用**：分析函数可被不同界面或导出工具调用。
- **易测**：便于针对单个分析模块做单元测试或截图测试。
- **可扩展**：后续增加新的分析维度时尽量减少对现有模块的影响。

## 3. 数据产出入口

### 3.1 调用方式

```python
from backtest_engine import BacktestEngine, DataLoader
from config import SystemConfig, StrategyConfig

system_config = SystemConfig(
    data_dir='data',
    initial_cash=1_000_000,
    commission_rate=0.0002,
    slippage_rate=0.0001,
    show_plots=False,
    save_results=False,
)
system_config.benchmark_index = 'sh000300'

strategy_config = StrategyConfig(
    strategy_name='weighted_top_n',
    parameters={'top_n_stocks': 5, 'hold_days': 2}
)

engine = BacktestEngine(system_config, strategy_config)

pred_df = DataLoader.load_prediction_data('data/sample.csv')
result = engine.run_single_backtest(pred_df, 'weighted_top_n', 'sample.csv')
```

### 3.2 `result` 字典结构

| Key | 描述 |
| --- | --- |
| `strategy_nav` | `pd.Series`，策略净值序列（DatetimeIndex） |
| `benchmark_nav` | `pd.Series`，基准净值序列，与策略对齐 |
| `performance` | `pd.DataFrame`，指标表（index 为指标 key，列包含 `value` 等） |
| `detailed_metrics` | dict，包含 `drawdown_series`、`running_max` 等辅助序列 |
| `monthly_stats` / `yearly_stats` | `pd.DataFrame`，期间收益对比（策略/基准/超额/胜率） |
| `trade_history` | list[dict]，交易记录（日期、操作、数量、价格、价值、组合价值） |
| `daily_holdings` | list[dict]，每日持仓快照（日期、持仓列表、总资产、现金） |
| `final_value` | float，最终组合市值 |
| `valid_stocks` | int，回测中实际加载的股票数量 |
| `strategy_name` / `file_name` | 元信息 |

> 约定：后续分析函数只依赖该字典及少量外部配置（如 `SystemConfig`、选定指标列表），确保 UI 层以外的调用也能复用。

## 4. 分析大类拆分

围绕 `result` 将分析逻辑划分为七个大模块，对应 Streamlit 的七个标签页。自 2025-10 起，`backtest_engine.py` 中新增的 `AnalysisBuilder` 已经实现了所有 `prepare_*` 接口，可直接复用。每个模块仍建议保留两个接口：

1. **`prepare_*`**：读取 `BacktestResult` 并做数据加工/统计，返回结构化结果（当前由 `AnalysisBuilder` 提供）。
2. **`render_*`**（UI 层）：在特定 Tab 内调用对应的准备结果，完成可视化。

`AnalysisBuilder` 已实现的七大模型：

- `prepare_overview(result, system_config, selected_metrics=None)`
- `prepare_net_value(result)`
- `prepare_returns(result)`
- `prepare_risk(result)`
- `prepare_period_stats(result)`
- `prepare_holdings(result)`
- `prepare_trades(result)`

### 4.1 概览（Overview）

- 输入：`result`、`system_config`、`selected_metrics`。
- 输出：
  - 概览指标字典（初始资金、最终资金、收益率、有效股票数、基准名称等）。
  - 关键指标列表（根据用户勾选筛选 `performance` 表）。
- 作用：支持展示指标卡、Hint 气泡等。

### 4.2 净值与相对收益（Net Value）

- 输入：`strategy_nav`、`benchmark_nav`、`performance`。
- 输出：
  - 归一化策略净值、基准净值、相对收益序列。
  - 相对收益终值，用于配色。
  - 关键曲线指标（总收益、年化、波动率等）。
- 作用：用于绘制上下两个子图（对比曲线/相对收益）和右侧指标列。

### 4.3 收益分析（Returns）

- 输入：策略日收益、基准日收益。
- 输出：
  - 日收益统计（均值、标准差、正负收益天数、胜率）。
  - 日收益直方图、累计收益曲线数据。
  - 月度收益表（策略、基准、超额），默认取最近 6 期。
- 作用：支撑收益分布图、累计收益图和表格。

### 4.4 风险分析（Risk）

- 输入：`strategy_nav`、`detailed_metrics`、`performance`。
- 输出：
  - 回撤序列、历史高点序列。
  - 风险指标表（最大回撤、波动率、VaR、CVaR、信息比率、Calmar 等）。
- 作用：用于绘制回撤面积图、风险指标表，可通过开关隐藏。

### 4.5 期间统计（Period Stats）

- 输入：`monthly_stats`、`yearly_stats`。
- 输出：
  - 结构化的月度/年度 DataFrame（含策略收益、基准收益、超额收益、胜率）。
  - 可选：对缺失数据的提示信息。
- 作用：展示两个可滚动表格，可按需扩展到季度等。

### 4.6 持仓分析（Holdings）

- 输入：`daily_holdings`、`code2name`。
- 输出：
  - 最近一期持仓明细（兼容 dict/list 结构）。
  - 资产曲线（日期、总资产、现金）。
- 作用：渲染持仓快照表和资产折线图。

### 4.7 交易记录（Trades）

- 输入：`trade_history`。
- 输出：
  - DataFrame 化的交易记录，含格式化的日期、数量、价格、金额、组合价值。
  - 针对空数据的反馈信息。
- 作用：用于可滚动表格，可扩展排序/过滤/导出。

> 注：`AnalysisBuilder` 中的实现还包含 VaR、CVaR、跟踪误差、信息比率、Calmar Ratio 等辅助指标，UI 层可按需挑选展示。

## 5. UI 集成流程

1. **入口函数**（例如 `display_results` 或新建 `render_backtest_results`）：
   - 调用 `prepare_*` 函数缓存各 Tab 所需数据。
   - 创建 Streamlit Tabs：`overview_tab, nav_tab, ... = st.tabs([...])`。
   - 在每个 `with tab:` 中调用对应的 `render_*`，传入预处理结果。
2. **去重数据准备**：一些共用数据（例如 `daily_returns`）在入口函数中统一计算，以避免重复操作。
3. **参数传递**：
   - `SystemConfig`、用户勾选的指标或开关通过入口函数统一传给 `prepare_*`，避免函数深层套娃。
   - 对需国际化的标题/描述可集中维护字典。

## 6. 测试与验证

- **函数级单测**：针对 `prepare_*` 输出结构做断言（列名、长度、缺失值处理、异常输入）。
- **冒烟测试**：调用引擎生成 `result`，判断各 Tab 生成的数据结构非空、关键字段存在。
- **UI 冒烟**：在 Streamlit 中人工检查图表/表格是否渲染成功；后续可结合截图测试。
- **OSS 覆盖**：确保 `load_bt_stocks` 获取到行情，否则提前抛错，而不是悄悄 fallback 到 AkShare（当前 fallback 已计划移除）。

## 7. 后续扩展方向

- **导出能力**：基于 `prepare_*` 的结果输出 CSV/Markdown/图片。
- **可配置指标集**：对于不同用户群（机构/个人）预设指标模板。
- **多策略对比**：复用分析函数，在 Tab 级别扩展“策略对比”页。
- **缓存策略**：对 `result` 或 `prepare_*` 结果做缓存，加速重新渲染。

---

通过上述设计，回测引擎与 UI 分层更加清晰：`BacktestEngine` 负责生产统一的 `result` 数据包；`prepare_*` 函数组成 **分析层**，负责加工数据；`render_*` 是最顶层的 **展示层**。这样既能保证 Streamlit 页面的交互体验，又方便未来在 CLI、报告工具等环境中复用同一套分析逻辑。