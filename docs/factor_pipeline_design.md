# 多因子筛选与存活判定设计文档

**版本**: v1.0  
**日期**: 2025-10-26  
**作者**: 规划生成（后续可补负责人）

---
## 1. 背景与目标
近期测试窗口 (2025-07-26 至 2025-10-24) 对 252 个因子在 3 个调仓周期（5/10/15天）上生成了 756 组合的评估结果。由于样本期极短（≈60 交易日），统计显著性不足、部分半衰期为 `NaN`。需要一套**统一、可扩展、可程序化**的因子状态判定与后续多因子构建方法：
1. 标准化 alive / warning / dead 逻辑，减少“优秀但 dead”语义冲突。
2. 生成候选因子白名单（可直接用于二次回测、组合构建）。
3. 处理冗余（相关性聚类），挑选代表因子或构建主题因子集。
4. 构造第一版多因子综合打分与回测方案（长/短调仓对比）。
5. 对稳定负 IC 的技术类因子进行反向测试验证潜在 Alpha。
6. 提供实施阶段路线 + 输出格式规范，便于快速集成到现有 `factor_calculator` / 回测引擎。

---
## 2. 数据范围与限制
| 方面 | 当前情况 | 影响 |
|------|----------|------|
| 时间跨度 | ~3 个月 | t / IR 不稳定、半衰期拟合失败多 |
| 市场状态 | 单一行情阶段 | Regime 敏感因子可能被错误丢弃 |
| 费用模型 | 固定费率 + 简单滑点 | 高换手策略真实净 IR 可能更差 |
| 行业 / 风险中性 | 未执行 | 部分规模/盈利类因子或仅捕捉风险溢价 |
| 质量控制 | 缺缺失率、异常值统计 | 部分因子“噪声驱动”表现难识别 |

后续不再简单追求“≥1 年”全量拼接：应构建**多风格 / 多市场状态分段样本**。做法：
1) 基础滚动窗口：采用 3M / 6M 滚动切片分别统计因子 IC 与稳定性；不强制合并为整年，避免风格漂移被平均。
2) Regime 标注：根据指数趋势 (30~60 日收益斜率)、波动分位数 (Realized Volatility 处于历史分位)、市场广度 (上涨家数占比)、资金偏好（成长 vs 价值相对强度）打标签（Bull / Bear / Sideways / Rotation）。
3) 分段统计：对每个 regime 独立计算 IC_mean / IR / Monotonicity；输出 `regime_ic_matrix` 评价因子是否“跨风格稳定”还是“特定风格 Alpha”。
4) 加权融合：综合评分使用 `OverallScore = Σ RegimeWeight_r * Score_r`，RegimeWeight 可按最近出现频率或风险平衡（例如对冲下行区间给予更高权重）。
5) 结构性变化检测：当相邻窗口因子 IC 分布通过 Pettitt / Chow Test 显著变化时，重置历史统计（防止旧 regime 拉低当前决策）。
6) 输出扩展字段：`regime_stability_ratio`（在所有出现的 regime 中保持 alive 的比例）、`regime_specific_tag`（只在某类 regime 有效）。
这样可以解决简单延长时间导致“牛熊抵消”与错误剔除问题，并为后续多因子权重的情境自适应打基础。

---
## 3. 指标定义
| 指标 | 说明 | 备注 |
|------|------|------|
| IC_mean | 截面信息系数均值 | Pearson/Spearman，需统一选择（推荐 Spearman） |
| IC_std | IC 序列标准差 | 用于 IR 及 t 转换 |
| IR | IC_mean / IC_std | 短窗口下易偏态 |
| IC_t | t = IC_mean / (IC_std / sqrt(N)) | N = 有效期数 |
| Monotonicity | 分位组合收益单调性度量（秩相关或线性趋势 R） | [-1,1] |
| Net_IR_after_cost | 多空或分位策略扣费后 IR | 成本模型需透明化 |
| Top_Turnover | Top 组合换手率 | 调仓日变化比例 |
| Capacity | 资金承载能力估算（亿元） | 基于成交额/流动性模型 |
| NegReturnRatio | Top 组合负收益日占比 | 波动质量指标 |
| HalfLife | IC 半衰期（指数衰减拟合） | 最少点数门槛 |
| Q5-Q1_Sharpe | 最高 vs 最低分组年化夏普 | 可作为辅助筛选 |
| Consistency | 多周期 alive 核心项满足比例 | 后处理聚合 |
| ReverseFlag | 满足“稳定负 IC”反向潜力 | 需二次回测验证 |

核心项 (Core Items)：`IC_mean`, `IC_t/IR`, `Monotonicity`, `Net_IR_after_cost`。

---
## 4. 存活状态判定逻辑
### 4.1 单周期阈值
| 指标 | alive | warning | dead |
|------|-------|---------|------|
| IC_mean | > 0.025 (或 < -0.025 记录可反向) | 0.010 ~ 0.025 | < 0.010 且 |IC_mean| < 反向阈值 |
| IC_t (或 IR) | t > 2.0 或 IR > 0.5 | 1.2 ~ 2.0 / 0.3 ~ 0.5 | t ≤ 1.2 / IR ≤ 0.3 |
| Monotonicity | > 0.60 | 0.40 ~ 0.60 | < 0.40 |
| Net_IR_after_cost | > 0 | -0.3 ~ 0 | < -0.3 |
| Top_Turnover | < 0.35 | 0.35 ~ 0.60 | > 0.60 |
| Capacity | > 500 | 100 ~ 500 | < 100 |
| NegReturnRatio | < 45% | 45% ~ 55% | > 55% |
| HalfLife | 20 ~ 240 | 240 ~ 400 或 NaN(暂缓) | NaN 且长样本后仍失败 |

说明：
- 反向资格：`IC_mean < -0.025` 且 `t < -2.0` 且 `|Monotonicity| > 0.6`。
- 高频价量类（动量/成交量波动）可放宽 Turnover 上限到 0.60（需附因子标签）。

### 4.2 多周期聚合
设周期集合 P = {5d,10d,15d}；对每个因子：
- `AliveCoreCount` = 满足 4 个核心项 alive 的周期数。
- `WarningCoreCount` = 核心项落入 warning 的次数。
- `DeadCoreCount` = 核心项进入 dead 的次数。

最终分类：
1. `overall = alive` 若 ≥ 2/3 周期核心项均 alive（允许 ≤25% 核心项为 warning）。
2. `overall = warning` 若未达 alive，但 dead 核心项比例 < 50%，且至少 1/3 周期有 alive 或 warning。
3. 其余 → `overall = dead`。
4. 若反向资格在 ≥ 1/2 周期成立 → 标记 `reverse_candidate`，单独通道再测试。

### 4.3 输出字段
```
因子, 周期, IC_mean, IC_t(or IR), Monotonicity, Net_IR_after_cost, Top_Turnover, Capacity, NegReturnRatio, HalfLife, PeriodStatus, ReverseFlag
```
汇总：`OverallStatus`, `ConsistencyAliveRatio = AliveCoreCount / (4 * |P|)`, `CandidateTag (whitelist / reverse_candidate / drop)`。

### 4.4 现有实现差异梳理 (factor/factor.py)
当前程序的 `quick_score()` 使用 period→阈值字典（IC、IR、年化收益、单调性、换手、半衰期、夏普、净IR、容量），与本设计的静态分段阈值存在冲突；滚动状态采用 `roll_ir`、近窗口胜率、`net_ir`、最近 5 个 IC 符号及 `q5q1_shrp` 组合判定。尚未集成：FDR、多周期一致性、显著性 p 值、负收益日占比、真实容量估算与半衰期回退。

### 4.5 优化后的三层判定框架
1) 信号层：IC 均值收缩 (James–Stein) + t / p 值 + BH FDR (q=0.10)。
2) 稳定层：滚动窗口 (20 / 60) 计算 roll_ic、roll_ir、负IC占比，构造稳定性得分:
    `Stab = 0.4*z(|roll_ic|) + 0.3*z(roll_ir) + 0.3*(1 - neg_ic_ratio)`；z 为截面标准化。
    半衰期：指数拟合失败时用自相关阈值法 (acf < 0.5) 回退；再失败标记 `tau=NaN`。
3) 实施层：改进净 IR (`net_ret = gross_ret - (commission+slippage+impact)*turnover`)，真实容量后续替换 mock；换手惩罚 `turnover_penalty = min(1, turnover / ref)`；实施得分 `ImplScore = net_ir_adj * (1 - turnover_penalty) * capacity_norm`。

综合总分：`TotalScore = 0.5*SignalScore + 0.3*Stab + 0.2*ImplScore`；
标签：alive (`TotalScore>0 & p_adj<0.10 & net_ir_adj>0`)，warning (`-0.5 < TotalScore ≤ 0` 或 `0.10 ≤ p_adj < 0.20`)，dead 其余；反向通过 → `alive_reverse` / `warning_reverse`。

### 4.6 新增字段
`ic_mean_shrink, p_value, p_value_adj, stab_score, impl_score, total_score, consistency_bonus, overall_score, improvement_reverse, estimation_capacity_flag`。

### 4.7 多周期融合更新
每周期得分 `TotalScore_p`，`OverallScore = mean(TotalScore_p) + 0.1 * (alive周期数/|P|)`；最终标签按 OverallScore 决定并与 FDR 结果一致性校验。

### 4.8 短窗口（60~100 日）自适应与每日监控
背景：当前仅有 ~60 个交易日，继续拉长到“一整年”会掩盖风格切换，且因子多数带有短期行为特征。需要一个**短窗口优先 + Regime 分段 + 每日增量更新**的机制。

1) 样本规模分级：
    - N < 40：仅提供探索标签（`exploratory`），不做正式 alive；允许记录方向与噪声特征。
    - 40 ≤ N < 60：使用收缩估计 (James–Stein) 和更宽松的 t 阈值：`t_alive_threshold(N) = 1.8 * sqrt(60 / N)`。
    - 60 ≤ N ≤ 100：使用标准 t 阈值 (≈2.0)；若波动异常（IC_std > 截面 80% 分位）则提高到 2.2。

2) 增量统计：每日新数据到来时使用在线更新：
    - IC_mean 采用 Welford 算法或简单 `μ_new = μ_old + (x_new - μ_old)/k`。
    - IC_std 同步更新，避免全量重算，支持快速回测迭代。
    - EWMA_IC：`EWMA_IC_t = λ * EWMA_IC_{t-1} + (1-λ) * IC_t`，λ 推荐 0.90（短窗口更“敏感”）。

3) 动态显著性与可靠度：
    - 可靠度分数 `Reliability = min(1, N / 80) * (1 - |Drift|)`；Drift = 最近 10 日 IC 均值 - 最近 30 日 IC 均值。
    - 当 Reliability < 0.5 时，不提升到正式白名单，只做观察；>=0.7 才允许权重放大。

4) 自适应阈值表示法：文档中的静态阈值在 UI / CLI 中呈现基础版，同时显示“动态调整后阈值”。示例：
    - 原始 t_alive = 2.0；若 N=50 → 动态 t_alive ≈ 2.0 * sqrt(60/50) ≈ 2.19。
    - 原始 IC_alive = 0.025；若 N<60 → 使用 `IC_alive_dyn = 0.025 * sqrt(60 / N)`（限制不超 0.040）。

5) 稳健性快速验证：
    - Block Bootstrap：块长 5，重复 200 次，计算 bootstrap IC 分布的 2.5% / 97.5% 分位。
    - 若置信区间跨 0 且 `Reliability ≥ 0.7`，标记 `borderline`; 若区间完全为正/负且 Reliability ≥ 0.6 → `direction_confirmed`。

6) 输出新增字段：
    `sample_size, dynamic_t_alive, dynamic_ic_alive, ewma_ic, reliability, drift_short_long, bootstrap_ic_low, bootstrap_ic_high, direction_confirmed_flag, borderline_flag`。

7) 每日健康仪表盘：以 `factor_health_daily.csv` 输出：
```
trade_date, factor, IC_today, IC_mean_online, IC_std_online, ewma_ic, reliability, dynamic_t_alive, alive_flag, reverse_flag
```
    可用于 Streamlit 面板实时观察（颜色：alive=绿，borderline=黄，dead=红，reverse=蓝）。

8) 权重延迟激活：当一个因子首次达到 alive 条件后需“确认期” K 天（建议 5~10）保持 Reliability ≥0.6 且未触发结构突变 (Drift 超阈值)，才加入正式组合；否则进入 `pending_activation`。

9) 与 Regime 整合：短窗口统计仍分 Regime; `Reliability_regime = Reliability * regime_presence_ratio`；若 Regime 稀有（出现天数 <15），标记 `regime_sparse`，延迟决策。

10) 风险提示：若换手在短窗口内持续处于样本顶端 10% 分位，且净 IR 改善有限（<0.05），触发 `turnover_warning`，自动降低实施层权重贡献 30%。

通过上述机制，60~100 日的因子仍可被快速评估并进入“观察→确认→激活”轨迹，而不是简单等待长期样本；同时避免短期噪声因子误判为稳定 Alpha。

---
## 5. 候选白名单选择规则
白名单 (Whitelist) 条件：
1. `OverallStatus = alive`。
2. `ConsistencyAliveRatio ≥ 0.50`。
3. `Net_IR_after_cost` 在每个 alive 周期非负；若个别周期略为 -0.05 内可保留并标记 `watch`。
4. `Top_Turnover` 满足其标签对应的阈值（普通 ≤0.35，高频标签 ≤0.60）。
5. `Capacity ≥ 500` 或满足主策略目标资金体量（可参数化）。
6. 非风险因子（规模/β等）若进入白名单则需同时打 `risk_factor` 标记，在组合构建中只做约束或正交化后使用。

附加过滤（可选）：
- `Q5-Q1_Sharpe > 1.0`。
- `NegReturnRatio < 50%`。（留一定包容度）
- `HalfLife` 非 NaN（若 NaN 进入 `data_insufficient` 二级观察列表）。

白名单输出 CSV 示例：
```
factor, OverallStatus, ConsistencyAliveRatio, mean_IC, mean_t, mean_Monotonicity, mean_NetIR, avg_Turnover, avg_Capacity, avg_NegRatio, HalfLife_mode, risk_flag, reverse_flag
```

统计聚合方法：
- 默认使用 *加权平均*：权重 = 周期的样本天数（当前三周期权重近似均等，可设为 1）。
- `HalfLife_mode` 取众数或中位数，NaN 优先替换为可用数值最靠近的周期值。
- 可附 `worst_case_NetIR` 与 `best_case_NetIR` 以识别不稳定性。

---
## 6. 相关性与冗余处理设计
目标：减少高度相关因子重复，提高组合有效维度。

### 6.1 相关性矩阵
- 使用 Spearman 截面相关（在联合样本日期上计算因子值对）。
- 对白名单因子构建相关矩阵 R。（规模 > 100 因子需分块）
- 缺失值处理：按交集日期；若有效配对样本比 < 60%，标记为低可信度，不参与直接剔除。

### 6.2 聚类方法
1. 将相关性转距离：`D_ij = 1 - |R_ij|`。
2. 使用层次聚类（Ward 或平均链接）得到簇分配。
3. 阈值剪枝：`|R_ij| > 0.7` 视为高度冗余；`0.5~0.7` 视为中度冗余。

### 6.2.1 FDR 相关性显著性
对高相关因子对计算相关系数 t 检验 p 值，应用 Benjamini–Hochberg 校正 (q=0.05)。校正后显著且 |R|>0.7 → 确认冗余；不显著 → 标记 `corr_uncertain` 暂缓剔除。中度冗余对若经济主题不同 → 保留并打 `cross_theme`。

### 6.3 代表因子选择（Medoid）
在每个簇内：
- 计算 `Score = ConsistencyAliveRatio * mean_IC * (1 - avg_Turnover)`（可归一化）。
- 得分最高作为 `cluster_medoid`。
- 其他成员若有独特经济含义 → 标记 `theme_member`。
若簇内存在 `alive_reverse` 且改进率 (IC_rev - |IC_orig|)/|IC_orig| > 0.5，则优先选择反向版本为 medoid；原始版本标记 `reverse_origin`。

### 6.4 输出
```
cluster_id, medoid_factor, members, max_abs_corr, medoid_score, theme_tag
```

### 6.5 合成主题因子（可选）
- 对簇内标准化因子 (z-score) 求平均或 PCA 第一主成分。
- 标记为 `THEME_[名称]`（如 PROFIT_QUALITY, GROWTH_ACC, SIZE_LIQ）。

---
## 7. 多因子综合打分框架
### 7.1 基础步骤
1. 输入：白名单 + 代表因子集合。
2. 标准化：每日对因子值进行行业中性与去极值（如中位数±3 MAD 或分位 Winsor 1%/99%）。
3. 风险正交：对因子矩阵 X 回归风险载体（市值、Beta、行业哑变量、流动性），使用残差作为净 Alpha。
4. 权重分配：
   - 静态：`w_i ∝ mean_IC_i`（截断在 [0.01, 0.10]）
   - 动态：`w_i ∝ EWMA_IC_i`（λ=0.94）与稳定性惩罚项 `(1 - turnover_penalty)`。
5. 综合得分：`Score = Σ w_i * Factor_i_resid`。
6. 排序分组：分位数（如 10 分位），构建多空或长端组合。

新增综合权重：
`w_i = (IC_shrink_i^+ * Stab_i^+ * (1 - TurnoverPen_i) * Bonus_i) / Σ(...)`；`IC_shrink_i^+ = max(IC_shrink_i,0)`；`Bonus_i = 1.1` 若 `p_adj<0.02`，`0.7` 若 `0.08≤p_adj<0.10`。
稳定性平滑：`Stab_i = 0.5*Stab_short_i + 0.5*Stab_long_i` (短 20 / 长 60)。

### 7.2 约束与归一化
- 权重归一：Σ|w_i| = 1。
- 单一主题簇权重上限：≤ 30%。
- 单因子最大权重：≤ 10%。
- 行业中性：对最终持仓约束行业权重差异在 ±5%。

### 7.3 回测模式
| 模式 | 描述 |
|------|------|
| Short-cycle | 5 天调仓，测试反应速度 |
| Mid-cycle | 10 天中性折中 |
| Long-cycle | 15/20 天 降低成本 |

比较指标：年化收益、信息比率、换手率、净 IR、最大回撤、组合半衰期稳定性、分位单调性。

### 7.4 稳健性检验
- 留一法：剔除一个重要因子重新计算 Score 看性能下降幅度。
- Permutation Test：随机打乱单一因子序列验证是否显著贡献。
- Block Bootstrap：对时间段抽样重新估计 IC 分布。

---
## 8. 反向测试计划
### 8.1 因子筛选
满足：
- `IC_mean < -0.025` 且 `t < -2.0` 多周期稳定。
- Monotonicity 绝对值 > 0.6。
示例（当前报告中候选）：`PLRC24`, `TRIX5`, 部分 `VROC`, `BIAS` 系列（需再验证）。

### 8.2 流程
1. 对满足条件因子取 `factor_rev = -factor_raw`。
2. 重新计算 IC, t, Monotonicity, Net_IR_after_cost。
3. 若反向后进入 alive → 标记 `alive_reverse` 并允许进入白名单二级。
4. 输出对照表：原 / 反向 指标对比。

补充验证：Block Bootstrap (块长=5) 对原/反向 IC 差异估计 95% 置信区间；改进率 `improvement = (IC_rev - abs(IC_orig))/abs(IC_orig)` > 0.3 且波动压缩 `vol_reduction = 1 - std(IC_rev)/std(IC_orig) > 0.1` → 入 `whitelist_reverse`；反向未改进且 `p_value_adj ≥ 0.15` → 标记 `drop_reverse`。

### 8.3 输出格式
```
factor, original_IC, reversed_IC, original_NetIR, reversed_NetIR, original_status, reversed_status
```

---
## 9. 扩展与稳健性验证
| 模块 | 目标 | 工具 |
|------|------|------|
| 时间扩展 | ≥ 1 年数据 | 重新计算全部统计 |
| Regime 分段 | 牛/熊/震荡 | 日期标注 + 分段 IC |
| 行业中性化 | 剥离行业暴露 | 回归 + 残差矩阵 |
| 风险暴露分析 | 与市场/风格因子相关性 | 多元回归、VIF |
| 成本敏感度 | 不同滑点与费率 | 参数扫描 |
| 极端日表现 | 大涨/大跌事件分析 | 事件窗口统计 |

---
## 10. 实施阶段路线
| 阶段 | 内容 | 产出 |
|------|------|------|
| Phase 1 | 存活判定实现 + 白名单 CSV | `factor_whitelist.csv` |
| Phase 2 | 相关性聚类 + 主题因子构造 | `factor_clusters.csv`, `theme_factors.csv` |
| Phase 3 | 综合打分模块 + 基础回测 | `multi_factor_score.csv`, 回测报告 |
| Phase 4 | 反向测试回溯 | `reverse_factor_eval.csv` |
| Phase 5 | 稳健性与扩展窗口 | 更新所有报告 |

---
## 11. 数据结构与输出规范
### 11.1 单周期原始指标输出 (per_period_metrics.csv)
```
factor, period_days, IC_mean, IC_t, IR, Monotonicity, Net_IR_after_cost, Top_Turnover, Capacity, NegReturnRatio, HalfLife, Q5Q1_Sharpe
```

### 11.2 白名单 (factor_whitelist.csv)
```
factor, OverallStatus, ConsistencyAliveRatio, mean_IC, mean_t, mean_Monotonicity, mean_NetIR, avg_Turnover, avg_Capacity, avg_NegRatio, HalfLife_mode, risk_flag, reverse_flag, candidate_tag
```

### 11.3 聚类 (factor_clusters.csv)
```
cluster_id, medoid_factor, members, medoid_score, max_abs_corr, theme_tag
```

### 11.4 多因子得分 (multi_factor_score_daily.csv)
```
trade_date, factor_score_raw, factor_score_resid, position_group, rebalance_flag
```

### 11.5 回测结果 (multi_factor_backtest_summary.csv)
```
config_id, start_date, end_date, rebalance_period, annual_return, sharpe, net_IR_after_cost, turnover, max_drawdown, IC_mean, capacity_estimate
```

### 11.6 反向测试 (reverse_factor_eval.csv)
```
factor, original_IC, reversed_IC, original_t, reversed_t, original_NetIR, reversed_NetIR, original_status, reversed_status
```

---
## 12. 伪代码示例
```python
CORE_ITEMS = ["IC_mean", "t_or_IR", "Monotonicity", "Net_IR_after_cost"]

for factor in factors:
    period_stats = load_period_metrics(factor)
    alive_core = warning_core = dead_core = 0
    reverse_periods = 0

    for stats in period_stats:
        status = classify_single_period(stats)
        if qualifies_reverse(stats):
            reverse_periods += 1
        update_counts(status, stats)

    overall = classify_overall(alive_core, warning_core, dead_core, len(period_stats))
    consistency = alive_core / (len(CORE_ITEMS) * len(period_stats))

    if overall == 'alive' and consistency >= 0.5 and turnover_ok(factor) and net_ir_ok(factor):
        tag = 'whitelist'
    elif reverse_periods >= math.ceil(len(period_stats)/2):
        tag = 'reverse_candidate'
    else:
        tag = 'drop'

    write_summary_row(factor, overall, consistency, tag)
```

---
## 13. 后续迭代点
1. 半衰期 NaN 处理：建立最少点数过滤与指数拟合失败回退方案（自相关近似）。
2. 成本模型细化：区分买入/卖出冲击成本 + 盘口厚度估计。
3. 行业中性：采用滚动回归（每月重估贝塔）。
4. 波动/风险控制：加入动态杠杆或波动目标（如日度波动年化控制在 12%）。
5. 在线监控：对综合得分的滚动 IC、风格暴露热力图、风险偏移警报。
6. 自适应权重：强化学习 / 贝叶斯更新（长期目标）。
7. 多重比较控制：对因子 IC / 收益显著性统一 FDR 校正防止过拟合。
8. 半衰期备选：AR(1) φ 参数转换为 e-fold 时间作为 tau 备份。
9. 数据质量层：缺失率、异常值比、重复值加入前置过滤。
10. 稳定性仪表盘：实时曲线显示 `total_score` 与 `stab_score` 变化。
11. 主题解释文本：基于簇特征自动生成因子经济叙事。
12. 容量模型升级：引入真实 ADV、冲击成本回归、市场份额上限。

---
## 14. 总结
本设计为多因子研究管线建立统一生存判定与选择、降维、综合评分、反向验证以及稳健性框架。当前短样本的结果仅作为初筛，核心后续工作是扩展时间窗口、完成风险与行业中性化，并在稳定性验证后再投入真实资金模拟。

---
**附录：参数可调建议**
| 参数 | 默认 | 可调范围 | 备注 |
|------|------|----------|------|
| IC_alive_threshold | 0.025 | 0.02 ~ 0.04 | 随市场噪声调整 |
| t_alive_threshold | 2.0 | 1.8 ~ 2.5 | 样本长度变化 |
| monotonicity_alive | 0.60 | 0.55 ~ 0.70 | 因子分位数数量影响 |
| netIR_dead_threshold | -0.3 | -0.2 ~ -0.5 | 成本模型精细化后再定 |
| turnover_limit | 0.35 | 0.30 ~ 0.50 | 根据费率与滑点 |

(结束)
