# 技术设计：因子服务 (Factor MCP + Skills)

因子服务负责因子的**定义、计算、存储与评测**。它构建在 `src/factor/generator_v2` 这一现代化的因子生成框架之上，通过 DuckDB 进行数据管理，并整合 Alphalens 进行专业评测。

## 1. 核心架构

因子服务是连接“基础数据”与“策略信号”的桥梁。

- **输入 (Data Source)**: 
    - 依赖 **Data Service** 的 `market.db` (OHLCV) 和 `finance.db` (财务数据)。
    - `generator_v2.DataLoader` 将被改造为直接从 DuckDB 读取数据，而非调用 OSS。
- **计算 (Engine)**: 
    - 使用 `generator_v2` 的 `FactorCalculator` 体系（支持内置、TA-Lib、自定义函数）。
- **存储 (Storage)**: 
    - **`factors.db`**: 因子数据的专用 DuckDB 仓库。
    - **Schema**: `(date, code, factor_name, value)`。长表存储，方便扩展任意数量的因子。
- **评测 (Evaluation)**: 
    - 内置 Alphalens 分析流程，生成 PDF/Markdown 报告。

---

## 2. MCP 接口 (Tools)

### 2.1 因子计算与存储 (Calculation & Storage)

#### `list_factors()`
- **功能**: 列出系统支持的因子库。
- **返回**: 
    - `builtin`: ["VOL10", "MA_20", "RSI_14", ...] (开箱即用)
    - `talib_patterns`: ["TALIB_{INDICATOR}_{PARAMS}"] (动态支持所有 TA-Lib 指标)
    - `stored`: ["MOMENTUM_1M", "VALUE_PE"] (已存储在 factors.db 中的自定义因子)

#### `calculate_factor(factor_name, start_date, end_date, stock_pool='all', save=True)`
- **功能**: 计算指定因子并（可选）存入数据库。
- **参数**:
    - `factor_name`: 支持动态语法 (如 `TALIB_RSI_14`) 或已注册的因子名。
    - `save`: 是否写入 `factors.db`。如果不存，仅返回临时文件路径（用于调试）。
- **实现**: 
    1. 从 `market.db` 读取 OHLCV。
    2. 调用 `generator_v2` 计算。
    3. `UPSERT` 到 `factors.db`。

#### `register_new_factor(name, description, definition_file_path)`
- **功能**: 注册一个新的 Python 脚本因子 (Custom Factor)。
- **逻辑**: 将脚本复制到 `src/factor/custom/` 目录，并注册到系统配置中，使其可通过 `calculate_factor` 调用。

### 2.2 质量与管理 (Quality & Management)

#### `verify_factor_data(factor_name)`
- **功能**: 对 `factors.db` 中的指定因子进行质量检查。
- **检查项**:
    - 覆盖率 (Coverage): 是否覆盖了 stock_pool 中的大部分股票？
    - 极端值 (Outliers): 是否有 infinite 或超过 5倍标准差的值？
    - 连续性: 历史数据是否有断层？
- **输出**: 质量诊断报告。

#### `clean_factor_data(factor_name, before_date)`
- **功能**: 清理旧数据，释放 `factors.db` 空间。

### 2.3 评测分析 (Analysis)

#### `analyze_single_factor(factor_name, start, end, periods=[1, 5, 10])`
- **功能**: 单因子有效性分析 (Alphalens Workflow)。
- **流程**:
    1. 从 `factors.db` 读取因子值。
    2. 从 `market.db` 读取未来收益率 (Forward Returns)。
    3. 计算 IC序列, IC_IR, 分层收益率 (Quantile Returns)。
    4. 生成图表和报告。
- **输出**: 报告 URL/路径。

---

## 3. Skills (组合能力)

### `factor_mining_workflow` (因子挖掘流水线)
**场景**: 用户想验证 "RSI + 波动率" 的组合效果。
**步骤**:
1. **生成**: 调用 `calculate_factor` 生成 `TALIB_RSI_14` 和 `TALIB_ATR_14`。
2. **合成**: (可选) 将两个因子合成新因子 (如 `RSI / ATR`)。
3. **评测**: 对新因子运行 `analyze_single_factor`。
4. **报告**: 输出最终的 IC/IR 评分，判断是否可用。

### `portfolio_risk_check` (组合风险检查)
**场景**: 检查当前持仓是否在某个因子上暴露过高。
**步骤**:
1. 获取用户持仓列表。
2. 从 `factors.db` 获取最新的风格因子值（如 Size, Volatility）。
3. 计算持仓的因子加权暴露度。
4. 警告如果暴露度超过阈值 (如 Z-Score > 2.0)。

---

## 4. 数据库设计 (DuckDB)

### `factors.db`

采用 **EAV (Entity-Attribute-Value)** 变体长表结构，以适应因子数量的无限扩展：

| Field | Type | Description |
|-------|------|-------------|
| date | DATE | 交易日 (Index) |
| code | VARCHAR | 股票代码 (Index) |
| factor | VARCHAR | 因子名称 (如 'RSI_14') |
| value | DOUBLE | 因子值 |

*注：对于极高频访问的基础因子（如市值），可以单独建宽表优化。*

---

## 5. 目录结构 (src/factor 重构)

```text
src/factor/
├── __init__.py
├── api.py                # MCP 暴露的接口 (list, calculate, analyze)
├── db.py                 # DuckDB 交互 (factors.db)
├── generator_v2/         # [保持不变] 核心计算引擎
│   └── ...
├── analysis/             # 分析模块
│   └── alphalens_wrapper.py # Alphalens 封装
└── custom/               # 用户自定义因子脚本存放处
    └── my_alpha_001.py
```
