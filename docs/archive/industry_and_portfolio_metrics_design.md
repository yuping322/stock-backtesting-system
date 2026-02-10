# 行业与组合结构指标设计文档

## 1. 背景与动机
当前回测系统已经具备基础收益与风险指标 (总收益、年化收益、最大回撤、Sharpe 等)，以及占位的行业集中度指标。为了在参数调优、策略对比、线上风控中做到更全面的评估，需要补充组合结构、分散度、交易效率、下行风险、路径风险、容量与流动性等维度的量化指标，形成可迭代的指标体系，以支撑：
- 更精准的策略选择 (收益质量 vs 集中度 vs 换手成本)。
- 风险敞口及时监控 (行业挤压、集中度过高、路径风险恶化)。
- 线上风控联动：当指标越界自动降风险。
- 数据驱动的策略迭代 (多维指标雷达图 + 历史滚动趋势)。

## 2. 目标与范围
本设计覆盖：
- 指标分阶段落地规划 (Phase 1 ~ Phase 3+)。
- 指标计算的数据来源与依赖结构。
- 模块化实现架构：分析层新增 PortfolioMetrics 组件。
- 与现有 `BacktestEngine` / `AnalysisBuilder` 的集成方式。
- 测试策略与可观测性。

不包括：
- 真实交易执行微观市场冲击建模。
- 因子暴露的具体因子库构建与获取细节 (后续扩展)。
- 资金容量评估的外部成交量质量修正 (暂采用基础成交量)。

## 3. 指标分类与分阶段计划
### Phase 1（快速收益 + 分散度增强）
| 类别 | 指标 | 说明 |
| ---- | ---- | ---- |
| 分散度 | Effective Number of Positions (ENP) | 1 / ∑w_i^2，越大越分散 |
| 分散度 | Weight Entropy (归一化) | -∑w_i log w_i / log N |
| 分散度 | Gini (权重) | 权重集中度；0~1 越高越集中 |
| 行业 | Industry HHI (真实) | ∑(行业权重^2) |
| 行业 | Top Industry Weight | 最大行业权重 |
| 行业 | Industry Count | 持仓涉及行业数量 |
| 交易 | Turnover (日/总) | 总成交金额 / 平均资产规模 |
| 交易 | Avg Holding Period | 持仓从买入到卖出平均天数 |
| 收益风险 | Sortino Ratio | 年化超额收益 / 下行波动 |
| 收益风险 | Downside Deviation | 负收益标准差年化 |
| 收益风险 | Tail Ratio | 上尾分位数回报 / 下尾绝对值 |
| 分布 | Skewness / Kurtosis | 收益分布偏度/峰度 |
| 路径 | Ulcer Index | 回撤平方均值的平方根 |

### Phase 2（行业轮动 + 交易质量 + 风险扩展）
| 类别 | 指标 | 说明 |
| ---- | ---- | ---- |
| 行业 | Industry Rotation Rate | 行业权重向量相邻日余弦距离 |
| 行业 | Industry Stability (Std) | 行业权重变化的标准差 |
| 交易 | Commission Ratio | 手续费 / 成交金额 |
| 交易 | Implementation Shortfall | 理论价 vs 实际成交价差收益影响 |
| 交易 | Trade Win Rate | 闭合交易盈利占比 |
| 收益风险 | Rolling Sharpe / Vol | 20/60/120 日滚动序列 |
| 路径 | Max Consecutive Loss/Win Days | 连续亏损或盈利天数峰值 |
| 路径 | Recovery Time | 最大回撤到恢复所需天数 |
| 收益质量 | Payoff Ratio / Expectancy | 平均盈利 / 平均亏损，单笔期望 |

### Phase 3（风险分解 + 容量 + 因子归因）
| 类别 | 指标 | 说明 |
| ---- | ---- | ---- |
| 组合风险 | Component Volatility | 风险贡献：w_i (Σ w)_i |
| 组合风险 | Diversification Ratio | ∑ w_i σ_i / σ_p |
| 组合风险 | Marginal VaR / CVaR | 个股边际风险贡献 |
| 因子 | Factor Exposure / Brinson Attribution | 收益拆分 (Allocation/Selection) |
| 流动性 | ADV Coverage | 持仓价值 / 日均成交额 |
| 流动性 | Days-To-Liquidate | 以 X% ADV 逐日退出天数 |
| 容量 | Capacity Estimate | 根据 ADV 占用反算最大资金规模 |
| 信号 | IC Stability / IC IR | 滚动信息系数均值/标准差 |
| 稳健 | Stress Test P&L | 假设市场冲击情景收益模拟 |

## 4. 数据来源与依赖
| 数据 | 来源/结构 | 用途 |
| ---- | -------- | ---- |
| strategy_nav | 回测结果 Series | 所有收益/回撤/路径类指标基础 |
| daily_returns | strategy_nav.pct_change | Sortino/Downside/Skew/Kurtosis/Ulcer |
| benchmark_nav | 基准 Series | Upside/Downside Capture、Active、TrackingError |
| trade_history | 列表[{date, code, action, size, price, value}] | Turnover、Commission、持仓周期、盈亏闭合 |
| daily_holdings | 列表[{date, holdings[{code, weight, value, buy_date}], total_value}] | 分散度、行业权重、轮动、最大权重 |
| pricing (optional) | load_bt_stocks/合成行情 | 波动率、因子回归、容量估算 |
| industry_map | get_industry_category | 行业权重聚合 |
| concept_map (扩展) | get_concept_categories | 概念集中度 |
| 手续费率 | system_config.commission_rate | Commission Ratio |

## 5. 架构设计概览
新增模块：`metrics/portfolio_structure.py` 与 `metrics/trade_metrics.py`
- portfolio_structure：分散度、行业轮动、行业聚合、权重统计。
- trade_metrics：换手、闭合交易重建、盈亏/期望、commission 统计。
- risk_extension：Sortino/Tail/Ulcer/Skew/Kurtosis/Rolling windows。

`AnalysisBuilder` 新增 orchestrator：`prepare_extended_metrics(result: BacktestResult)`
返回：
```python
{
  'structure': {...},
  'trading': {...},
  'risk_ext': {...},
  'time_series': {
     'rolling_sharpe': pd.Series,
     'rolling_vol_60': pd.Series,
  }
}
```
与现有 `prepare_risk/prepare_holdings` 并行，不破坏已有接口。

## 6. 模块 API 设计 (示例)
```python
# metrics/portfolio_structure.py
class PortfolioStructureMetrics:
    @staticmethod
    def industry_breakdown(latest_holdings: list) -> dict:  # {industry: weight}
    @staticmethod
    def diversification_basic(weights: pd.Series) -> dict:  # ENP, entropy, gini
    @staticmethod
    def industry_concentration(industry_weights: dict) -> dict:  # hhi, top_weight, count
    @staticmethod
    def rotation(prev_industry: dict, curr_industry: dict) -> float:  # cosine distance

# metrics/trade_metrics.py
class TradeMetrics:
    @staticmethod
    def turnover(trades: list, nav_series: pd.Series) -> dict:  # daily_turnover, total_turnover
    @staticmethod
    def rebuild_round_trips(trades: list) -> list:  # [{code, entry_date, exit_date, pnl, holding_days}]
    @staticmethod
    def holding_period(round_trips: list) -> float
    @staticmethod
    def payoff(round_trips: list) -> dict  # win_rate, payoff_ratio, expectancy

# metrics/risk_extension.py
class ExtendedRiskMetrics:
    @staticmethod
    def sortino(daily_returns: pd.Series, rf: float=0.0) -> float
    @staticmethod
    def downside_deviation(daily_returns: pd.Series, rf: float=0.0) -> float
    @staticmethod
    def tail_ratio(daily_returns: pd.Series) -> float
    @staticmethod
    def ulcer_index(nav: pd.Series) -> float
    @staticmethod
    def distribution_stats(daily_returns: pd.Series) -> dict  # skew, kurtosis
    @staticmethod
    def rolling_sharpe(daily_returns: pd.Series, window: int=60) -> pd.Series
```

## 7. 关键算法与实现细节
- ENP：`1.0 / (weights.pow(2).sum())`，权重需归一化 sum=1。
- Entropy：`-(weights * np.log(weights + 1e-12)).sum() / np.log(len(weights))`。
- Gini：对权重升序 w，`G = (2*∑(i*w_i)/(N*∑w) - (N+1)/N)`。
- Industry Rotation：`1 - (w_t · w_{t-1}) / (||w_t|| * ||w_{t-1}||)`，行业缺失补 0。
- Turnover：日度 = `sum(abs(delta_position_value)) / portfolio_value_prev`；简化用 trade_history 合计成交额 / 平均资产。
- Holding Period：从 round trip 重建：利用栈或 FIFO 匹配 BUY/SELL；多次增减仓处理：按份额分段平均。初期简化为首次买入到完全清仓。
- Implementation Shortfall (后续)：理想基准价选择前收或开盘价，需要历史价。
- Sortino：`(annual_return - rf) / (downside_std * sqrt(252))`；或先直接采用日度收益：`mean(excess)/std(negative)` * √252。
- Tail Ratio：`np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))`。
- Ulcer Index：对归一化净值回撤序列 `drawdown = nav/nav.cummax() - 1`，`sqrt((drawdown.clip(upper=0)**2).mean())`。

## 8. 数据处理与鲁棒性
- 权重归一化：若持仓价值和总资产不匹配，使用记录的 weight 字段；若缺失则由 value/total_value 重新计算。
- 缺失行业：赋值为 "UNKNOWN" 并参与集中度（可选是否剔除）。
- 交易配对不完整：round trip 重建失败的残余持仓不计入盈亏比 (需在文档中注明)。
- 窗口不足：滚动指标窗口 < size 时用 NaN 或跳过。
- 全部权重为 0：分散度指标返回 None 或 0 并打日志。

## 9. 流程整合
1. 回测结束获得 `BacktestResult`。
2. `AnalysisBuilder.prepare_extended_metrics(result)`：
   - 提取 `daily_holdings[-2]` 与 `[-1]` 做行业轮动 (若长度>=2)。
   - 最新持仓计算权重分散度与行业集中度。
   - 从 `trade_history` 重建 round trips → 交易质量指标。
   - 从 `strategy_nav` 计算扩展风险指标与滚动序列。
3. 将结构化字典返回；UI 层可以：
   - 显示当日结构表。
   - 绘制滚动 Sharpe/Vol 曲线。
   - 雷达图对比多策略 (归一化 0~1)。

## 10. 测试策略
类别：
- 单元测试：每个静态方法给定伪造数据断言数值范围（例如 ENP 应该 >= 实际持仓数的下限 1）。
- 集成测试：构造一个含多日持仓 & 多笔交易的 `BacktestResult` 验证整体输出结构与关键字段存在。
- 边界测试：空持仓、单一持仓、全部同一行业、权重极度集中、无交易、只买未卖。
- 性能测试（后续）：对 1 年 250 日、1000 笔交易的处理不超过阈值 (如 < 50ms)。

## 11. 迭代路线图
- Week 1：实现 Phase 1 全部指标 + 测试 + UI 基础展示。
- Week 2：交易闭合与轮动、委托成本指标；滚动风险序列。
- Week 3：容量与流动性、行业稳定、Stress Test 简单场景。
- Week 4+：因子归因、风险贡献、策略间对比仪表盘。

## 12. 风险与缓解
| 风险 | 描述 | 缓解 |
| ---- | ---- | ---- |
| 持仓权重缺失 | 历史数据中未记录 weight | 用 value/total_value 回推 |
| 回测数据过短 | 指标不稳定 | 对滚动指标加最小样本阈值 & 标注“低样本” |
| 交易闭合复杂 | 多次分批买卖组合难度 | 初期用简单 FIFO, 后续精细颗粒度 |
| 性能退化 | 多循环统计 | 向量化 + 局部缓存行业映射 |
| 指标滥用 | 过多指标影响决策清晰度 | 分层分组 + 推荐核心 KPI 集合 |

## 13. 验收标准
- Phase 1 指标函数均有测试覆盖率 >90%。
- 集成方法返回字典包含 `structure`, `trading`, `risk_ext` 等关键分组。
- UI 或 CLI 能列出新增指标并格式化展示。
- 日志中无未处理异常或大量警告。
- 运行时间：单次回测后指标扩展计算 < 100ms（一般数据规模）。

## 14. 后续扩展挂钩点
- 与实时交易模块：实时更新 ENP、行业 HHI，超过阈值触发降仓逻辑。
- 与参数扫描脚本：将分散度/换手/Sortino 加入筛选条件与输出 CSV。
- 因子系统：利用因子暴露与结构指标联动优化持仓构建（风险预算）。

## 15. 实现优先级建议
立即落地：ENP、真实行业集中度、Entropy、Gini、Turnover、Avg Holding Period、Sortino、Ulcer、Tail Ratio、Skew/Kurtosis。
随后：Industry Rotation、Rolling Sharpe、Commission Ratio、Round Trip 盈亏结构。
再后：容量、因子归因、Stress Test。

---
**附注**：目前 `BacktestEngine.calculate_detailed_metrics` 嵌入的行业指标为占位；真实实现将迁移至新模块，避免该函数膨胀并保持职责单一。

如需，我可以下一步直接生成 `metrics/portfolio_structure.py` 和相关测试骨架。
