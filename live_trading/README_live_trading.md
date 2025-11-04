# Live Trading 模块说明

该目录提供一个与回测逻辑**解耦**的线上实时（或准实时）交易工作流原型，覆盖：预测聚合、组合构建、风险管理、执行、状态持久化、漂移检测与审计。

> 目的：从模型每日在 `data/` 目录生成的多个预测文件出发，构建一个可迭代的实际交易层框架，方便逐步替换模拟组件为真实接口。

## 目录结构
```
live_trading/
  live_config.py          # 配置数据类集合
  prediction_loader.py    # 聚合多个预测CSV（支持多模型集成）
  portfolio_builder.py    # 组合构建 + 行业/权重约束
  risk_manager.py         # 回撤/波动/集中度 + 熔断判断
  execution_engine.py     # 订单生成 & 模拟成交（待接Broker）
  state_store.py          # 持仓、NAV、审计、漂移指标持久化
  drift_detector.py       # 滚动IC & 漂移触发（支持每模型IC）
  run_live.py             # 主流程编排脚本（原有）
  run_premarket.py        # 盘前流程：集成多模型预测 → 生成订单
  run_trade.py            # 盘中执行：读取 orders.csv → 下单
  run_settle.py           # 收盘结算：更新NAV → 计算IC → 触发重训
  README_live_trading.md  # 文档
```

## 核心流程阶段
| 阶段 | 脚本职责 | 主要输入 | 主要输出 |
|------|----------|----------|----------|
| 盘前 (Pre-market) | 聚合预测、构建目标组合 | `data/` 中最新CSV | 目标权重表 `code, weight` |
| 开盘前 | 风险评估与可能降仓 | 历史NAV、组合权重 | 调整后权重 / 审计记录 |
| 盘中 (模拟) | 生成&执行订单 | 当前持仓 + 目标权重 | 执行订单、滑点统计 |
| 收盘 | 更新NAV、漂移检测 | 当日持仓、模拟收益 | `nav.csv`、漂移指标、审计日志 |

## 配置说明
在 `live_config.py` 中：
- `DataIngestionConfig`: 预测数据加载行为（文件模式、最近天数、**模型列表**）。
  - `models`: 模型目录名列表（如 `['model_a', 'model_b']`）
- `PortfolioConfig`: Top-N、单股/行业权重上限、最小权重阈值、再平衡节奏。
  - `top_n`: 只做多前N只股票（默认50）
  - `max_stock_weight`: 单票上限（默认0.10）
  - `max_industry_weight`: 行业上限（默认0.35）
- `RiskConfig`: 回撤、熔断、波动率目标、HHI 集中度限制、IC 漂移窗口阈值。
  - `min_ic_threshold`: IC阈值（低于此值触发重训，默认0.02）
- `ExecutionConfig`: 是否模拟、滑点BP上限、并发订单数量、价格源占位。
- `PersistenceConfig`: 状态文件路径（持仓、NAV、审计、漂移结果、**目标权重、订单、每模型IC、重训标记**）。
- `MonitoringConfig`: 指标与告警阈值（预留）。

## 天级多模型策略流程（MVP）

### 数据格式
每天 **07:00 前** 多个模型已落盘 `csv`：
```
data/
  model_a/20250603.csv
  model_b/20250603.csv
```
每个 CSV 仅两列：`code,score`（score 越高越看涨）。

### 流程拆分（按时间段执行）

#### 1. 盘前流程（07:00）- `run_premarket.py`
- 加载多模型预测并集成（score → rank → 等权合并）
- 应用风险检查（单票≤10%，行业≤30%）
- 生成目标权重 `target_w.csv`
- 生成订单清单 `orders.csv`（code,side,shares）

```bash
python -m live_trading.run_premarket 20250603
```

#### 2. 盘中执行（09:30）- `run_trade.py`
- 读取 `orders.csv`
- 执行订单（模拟或真实Broker）
- 更新持仓 `positions.csv`

```bash
python -m live_trading.run_trade 20250603
```

#### 3. 收盘结算（15:15）- `run_settle.py`
- 拉收盘价 → 计算 NAV → 追加 `nav.csv`
- 计算各模型 IC → 写入 `model_ic.csv`
- 若连续 5 天平均 IC < 0.02 → 生成 `retrain.flag`

```bash
python -m live_trading.run_settle 20250603
```

### 输出文件清单
在 `live_state/` 目录生成：
- `target_w.csv`：目标权重（code, weight）
- `orders.csv`：订单清单（code, side, shares）
- `positions.csv`：最新持仓（权重+模拟均价）
- `nav.csv`：净值历史
- `model_ic.csv`：每个模型的IC追踪（date, model, ic）
- `retrain.flag`：重训信号标记文件
- `audit.log`：关键事件记录

### Cron 调度示例
```bash
# 盘前 07:00
00 07 * * 1-5  cd /path/to/project && python -m live_trading.run_premarket $(date +\%Y\%m\%d)

# 盘中 09:30
30 09 * * 1-5  cd /path/to/project && python -m live_trading.run_trade $(date +\%Y\%m\%d)

# 收盘 15:15
15 15 * * 1-5  cd /path/to/project && python -m live_trading.run_settle $(date +\%Y\%m\%d)
```

## 运行 (原有单脚本模式)
```bash
python -m live_trading.run_live
```
运行后，会在 `live_state/` 目录生成：
- `positions.csv`：最新持仓（权重+模拟均价）
- `nav.csv`：净值历史（当前示例使用恒定初始净值占位）
- `audit.log`：关键事件记录（订单执行、风险决策、漂移结果等）
- `drift_metrics.csv`：IC 滚动与重训触发标记

## 后续可扩展点
1. **接入真实价格/行情**：替换 `ExecutionEngine._mock_price` 为实时报价接口，并在收盘重新计算NAV。  
2. **订单路由**：实现 `BrokerAdapter`（下单、撤单、成交回报回调）。  
3. **风控深化**：引入动态波动缩放、仓位分层（核心/卫星仓）。  
4. **高级优化**：使用二次规划（CVXPy）构建目标组合，权重受风险模型约束。  
5. **漂移检测增强**：IC 分解（行业中性、风格中性），预测收益分布偏度 & 置信区间。  
6. **指标上报**：接入 Prometheus / InfluxDB， Grafana Dashboard。  
7. **调度拆分**：将 `run_live.py` 中四阶段拆到单独脚本按具体时间段执行（如 cron / Airflow DAG）。  
8. **日志结构化**：使用 JSON Lines + 日志采集管道（ELK）。  
9. **回滚策略**：加入“安全模式”自动切换等权或全现金。  
10. **与回测联动**：定期对比线上 vs 回测指标漂移曲线，自动生成差异报告。

## 与原回测系统的关系
- 不修改原有 `backtest_engine.py` / `main.py`；此模块作为独立层。  
- 可以在需要时引用原数据处理函数（如代码规范化）；当前保持轻量避免耦合。  
- 回测阶段选出的策略/参数可直接映射到 `live_config.py` 的 `PortfolioConfig` 和 `RiskConfig`。  

## 数据要求

### 多模型集成模式（MVP）
预测CSV格式：`code,score`
- 文件路径：`data/{model_name}/{YYYYMMDD}.csv`
- `code`：六位股票代码（如 '000001'）
- `score`：模型打分（越高越看涨）

集成逻辑：
1. 每个模型内：`score → rank(pct=True) - 0.5`（中性化）
2. 模型间：等权平均
3. Top-N 筛选（权重 ≥ 0）
4. 归一化到权重和=1

### 原有模式（兼容）
预测CSV最少列：`date, code, weight`。若 `weight` 缺失且 `allow_missing_weight=True`，自动补 1.0。多模型重复同一 `code,date` 时取均值。日期需可被 `pd.to_datetime` 解析并标准化为日。股票代码需为六位数字串。

## 审计规范建议
- 每次重大决策（熔断、降仓、漂移触发）在 `audit.log` 写入：时间戳 + 事件类型 + 关键指标。  
- 定期归档 audit 文件并对比版本差异，防止未经批准的参数变更。

## 风险提示
当前实现仍是原型：
- NAV 计算简化为常数，需要替换为真实估值。  
- 漂移检测使用模拟收益，需接入实际的次日或当日真实回报。  
- 行业分类加载逻辑简单，需校验数据质量。  
- 无真实持仓数量/股数换算与现金余额管理。  

## 快速验证
确保 `data/` 下存在至少一个预测CSV（如 `factor_values_sample.csv`），然后运行脚本并检查 `live_state/` 中文件生成是否正确。

---
欢迎继续提出：是否需要接入真实价格、增加订单分批算法、或把风险/漂移指标推送到监控系统。