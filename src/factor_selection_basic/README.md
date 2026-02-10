# 稀疏事件信号选股最小验证示例（Base + 事件融合）

> 目标：提供一个**最小、可迭代**的验证骨架，聚焦“稀疏事件型/标签型选股信号”与已有 Base 选股规则的融合，快速判断其是否具备独立 Alpha 与增量价值。当前仅文档设计，后续可补齐脚本与 Notebook。

---
## 1. 场景与目标
- 你有一个“事件/标签”类选股想法：只有少量股票在某天满足条件（例如公告、异常资金流、结构变化等），希望验证这些触发股票未来是否更容易上涨。
- 采用日频数据，每日事件集合规模较小（例如 5~30 支股票）。
- 与一个基础的 Base 选股规则（如动量 TopK）融合，观察增量收益与风险改善。
- 输出：事件组合收益、命中率、事件后窗口超额、与 Base 融合后的提升。

## 2. 基本假设与问题拆解
| 假设 | 描述 | 验证方式 |
|------|------|----------|
| A | 事件触发股票集合未来 N 日收益或超额显著为正 | 事件后窗口收益、命中率、对照组分析 |
| B | 事件信号与 Base 融合后整体年化/IR/回撤指标改善 | 融合前后差异分析 |
| C | 事件效应稳定非偶然 | 多时间段、滚动窗口、Bootstrap、对照组 |
| D | 集中度风险可控 | 权重集中度、单股占比、容量评估 |

## 3. 数据与股票池
- 股票池：可用沪深300（流动性较好）或全 A 股后加流动性过滤（过去 20 日平均成交额 > 阈值）。
- 时间区间：最近 1~2 年做快速验证；通过后再扩展全历史。
- 使用字段：收盘价、开盘价、成交额、市值（用于匹配对照组）、行业标签。
- Point-In-Time 原则：事件所需信息必须在触发当天实际可获得；避免用未来公告落地数据。
- 执行假设：T 日收盘确定事件集合，T+1 开盘建仓，持有设定周期（初始 1 日）。

## 4. Base 选股规则示例
- 示例：20 日动量 `momentum_20 = close / close.shift(20) - 1` 排序取 TopK（如 100）。
- 初始持仓构建：等权分配或按动量分数归一化。
- 调仓频率：每日；后续可测试每周/每 5 日。
- 目标：提供一个稳定、广覆盖的基线，用于衡量事件信号增量价值。

## 5. 稀疏事件信号说明与构建
### 5.1 定义
事件信号是一个触发集合：在日期 t ，满足特定条件的少量股票被标记为 1（或给出一个正分值），其它股票缺失（NaN）或不出现。

### 5.2 触发条件示例（占位）
- 公告/财报发布首日（真实发布日期）。
- 成交额激增：`volume_today / Mean(volume_5) > X` 且价格稳健不暴涨。
- 资金流指标：`net_inflow_ratio > threshold`。
- 技术结构：多周期均线某种排列首次出现。

### 5.3 数据结构
`event_df`: MultiIndex (date, instrument) -> value (1 或 分数)。未触发则不写入或值为 NaN。

### 5.4 构建原则
- 不强行填充 0 参与全市场排序。
- 明确可获得性（字段在触发日真实存在）。
- 去重复：同一事件连续多天是否只记录首日或扩展窗口（需策略定义）。

### 5.5 特征扩展（可选）
若后续想区分强弱，可为事件附加权重：如资金流强度分段；暂时可全部视为等权。

### 5.6 融合方式（至少测试 2 种）
1. 过滤：Base TopK ∩ 事件集合；若事件数量不足 K，则补齐 Base 排序剩余。
2. 加权提升：事件触发股票权重 = 原权重 * (1+α)；α 例如 0.2, 0.5。
3. 双层资金分配：总资金拆为 Base (70~90%) + 事件组合 (10~30%) 等权。
4. 延长持仓：事件触发股票持有周期从 1D 延长到 N 日（如 3/5 日）。
5. 加仓策略：若事件触发且股票已在 Base 持仓内，则权重翻倍但不超单股上限。

### 5.7 评估指标
| 指标 | 描述 |
|------|------|
| 事件后1日平均收益 | 触发日后第 1 个交易日收益均值 |
| 事件后N日累计收益 | N=3/5/10 的累计收益均值与分布 |
| 命中率 | 事件后第 1/N 日收益 >0 的比例 |
| 超额收益 | 相对指数或全池等权的差值 |
| 对照组差异 | 行业+市值匹配随机组的收益差异统计显著性 |
| 稳定性 | 按月份/季度或滚动事件窗口统计收益均值曲线 |
| 集中度 | 单股最大权重、Herfindahl 指数、事件日总持仓数 |
| 成本敏感性 | 加交易成本后收益是否仍为正 |

## 6. 最小验证流程步骤
1. 设定时间区间与股票池。  
2. 计算 Base 选股分数（如动量），形成 Base 排序与初始持仓。  
3. 生成 `event_df`（稀疏事件集合）。  
4. 选择融合策略（过滤 / 加权提升 / 双层资金 / 延长持仓）。  
5. 构建当日目标持仓：记录权重与执行价格假设。  
6. 回测执行：T 日收盘生成事件 → T+1 开盘建仓 → 持有周期结束平仓。  
7. 统计事件后窗口收益、命中率、总组合收益曲线。  
8. 做对照组（匹配行业+市值的随机集合）差异显著性检验。  
9. 计算增量指标：年化、IR、最大回撤变化、集中度、成本后净效果。  
10. 稳健性：按月份/季度与滚动窗口分析；Bootstrap 构建置信区间。  

## 7. 指标与报告
- 收益类：累计收益、年化收益、最大回撤、Sharpe、Sortino。  
- 超额类：事件组合与基准差值，融合组合与 Base 差值。  
- 事件专属：事件后 1/3/5/10 日平均收益与衰减曲线、命中率曲线、滚动事件窗口收益。  
- 对照组差异：事件集合 vs 匹配随机集合收益差异、t 检验 p 值或 Bootstrap 置信区间。  
- 增量贡献：年化提升 Δ、IR 提升、最大回撤变化、集中度变化、成本后净效果。  
- 稳健性：月份/季度分段表现，滚动 50 次事件收益均值与方差。  

## 8. 判定标准（Go / No-Go）
| 维度 | 基线要求（示例，可调整） |
|------|---------------------------|
| 基础有效性 | 事件后 1 日平均超额 >0 且 p<0.05；3/5 日累计超额仍为正 |
| 命中率 | 事件后 1 日正收益比例 > 市场基准正收益概率 + (5~10%) |
| 融合增量 | 融合组合年化或 IR 提升 >5%，回撤不显著恶化 |
| 稳健性 | 多月份/季度均为正；滚动窗口收益稳定无持续塌陷 |
| 成本敏感性 | 加 5~10 bps 成本后仍保持正向超额 |
| 集中度风险 | 单股权重不超过总资金的 5%~10%；Herfindahl 指数不过度集中 |
| 对照组显著性 | 事件 vs 匹配随机组收益差异 p<0.05 或 Bootstrap 置信区间不含 0 |

> 阈值仅示例，可按市场实际与内部经验调整。

## 9. 后续扩展路径
- 多持仓周期对比：1D vs 3D vs 5D vs 10D。
- 交易成本/冲击模型引入：滑点随成交额比例变化。
- 行业/风格中性化：事件集合与 Base 按行业权重配平。
- 参数敏感性：触发阈值上下浮动、加权提升系数 α 变化。
- 容量评估：模拟扩大资金下的成交量占比与收益侵蚀。
- 风险预算：对事件与 Base 分配动态资金比例（波动或夏普加权）。
- 持续监控：滚动 50/100 次事件平均收益低于阈值触发降权或停用。

## 10. 文件结构规划（未来补充）
```
examples/factor_selection_basic/
  README.md              # 本说明
  build_base_and_events.py  # 生成 Base 排序与事件集合（占位）
  run_event_backtest.py     # 构造事件/融合组合并回测（占位）
  analysis.ipynb         # 事件后收益衰减与命中率可视化（可选）
  output/                # 回测与分析结果输出路径（占位）
```

## 11. 使用方式（预期流程占位）
```bash
# 1. 初始化 Qlib 数据（示例命令占位）
python scripts/get_data.py --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 2. 构建 Base 与事件（未来补充脚本）
python examples/factor_selection_basic/build_base_and_events.py

# 3. 运行事件融合回测（未来补充脚本）
python examples/factor_selection_basic/run_event_backtest.py

# 4. 查看 output/ 下的指标与报告
```

## 12. FAQ 简版
**Q: 为什么要用 T+1 执行？**  
A: 因为信号在 T 日收盘后才能完全获得，避免使用未来信息。  

**Q: 为什么先用 1 天持仓？**  
A: 这是最短反馈周期，可快速确认方向性；后续再测更长持有期的稳定性。  

**Q: 事件信号要不要标准化？**  
A: 若事件只做“是否触发”二值，可不标准化；若附带强度分数，可在触发子集中做归一化。  

**Q: 如何避免过拟合？**  
A: 控制参数数量、使用多时间段验证、做参数扰动、留独立测试区间。  

## 13. 后续将补充
- 事件触发构建脚本示例。
- 融合策略不同模式回测对比结果。
- 事件后收益衰减与稳定性可视化截图。

## 14. 新增：快速验证与最终预测输出脚本

已补充两个核心脚本，便于将策略从“想法”推进到“可落地”：

### 14.1 事件信号数据有效性验证 (`validate_event_signal.py`)
用途：判断事件本身是否有统计显著性与对 Base 的增量潜力。
输出文件：
- `validation_report.json` 包含 usability 与 incremental_lift 两大结构：
  - usability: 覆盖度、事件稀疏性、窗口收益、命中率、随机对照窗口收益、t 检验近似 t 值
  - incremental_lift: 事件组合 vs Base TopK 组合的年化差值、信息比率(IR)、日度差均值/波动
- `validation_daily_diff.csv`：每日 (事件组合 - Base) 收益差，便于画滚动与衰减曲线。

运行示例：
```bash
python examples/factor_selection_basic/validate_event_signal.py \
  --topk 50 --start 2023-01-01 --end 2024-12-31 --control_seed 42
```
解读要点：
- window_1_avg_cum_ret 与 control_window_1_avg_cum_ret 的差异及 t_value_window_1 是否显著 (>2 或 < -2)。
- info_ratio_diff 为正且 annual_return_diff >0：事件集合具备增量价值。
- cv_event_count 过高需关注稳定性，可能引入事件强度分层。  

### 14.2 最终融合预测输出 (`generate_final_prediction.py`)
用途：将 Base 排序与事件集合根据选择的模式融合成可供下游执行/多因子集成的统一预测文件。
输出文件：
- `final_prediction.csv`：字段包含 `date,instrument,final_score,final_weight,meta`
  - final_score：融合后排序得分（可继续进入统一因子框架或直接做权重归一化）
  - final_weight：该模式下计算得到的权重（若下游还有风险层，会再调整）
  - meta：JSON 字符串记录模式与参数（如 alpha、sleeve_ratio、事件当日数量等）

支持模式：
1. boost：事件股票得分 * (1+alpha) 后归一化；适合事件强度中等场景。
2. filter：事件数达到 min_events 时优先事件；不足补齐 Base 剩余；适合高质量少量事件。
3. extend_hold：事件股票延长 N 天加权 (bonus_beta)；适合持续性衰减逻辑。
4. two_layer：Base 与事件拆层资金分配；适合事件信号独立 Alpha 监控场景。

运行示例：
```bash
# Boost 模式
python /Users/fengzhi/Downloads/git/stock-backtesting-system/src/factor_selection_basic/generate_final_prediction.py --mode boost --alpha 0.3 --topk 200 --start 2023-01-01 --end 2024-12-31

# Filter 模式
python /Users/fengzhi/Downloads/git/stock-backtesting-system/src/factor_selection_basic/generate_final_prediction.py --mode filter --min_events 5 --topk 200 --start 2023-01-01 --end 2024-12-31

# Extend Hold 模式
python /Users/fengzhi/Downloads/git/stock-backtesting-system/src/factor_selection_basic/generate_final_prediction.py --mode extend_hold --hold_days 3 --bonus_beta 0.5 --topk 200 --start 2023-01-01 --end 2024-12-31

# Two-layer 模式
python /Users/fengzhi/Downloads/git/stock-backtesting-system/src/factor_selection_basic/generate_final_prediction.py --mode two_layer --sleeve_ratio 0.25 --topk 200 --start 2023-01-01 --end 2024-12-31
```
使用建议：
- 先跑验证脚本，若显著性与增量指标通过，再决定采用哪种融合模式生成 `final_prediction.csv`。
- `final_weight` 可直接传给执行层；若需风险中性化，再进入二次处理环节（行业/风格/因子归一）。
- 若多个事件源，将它们各自生成的 final_score 合并（例如加权或模型融合）。

### 14.3 与回测脚本 (`run_event_backtest.py`) 的协同
- 回测脚本用于多融合策略的绩效表现比较（含换手与成本）。
- 验证脚本强调事件本体统计显著性与基础增量。
- 预测脚本提供面向生产的标准化输出结构。

### 14.4 进阶待扩展
- 对照组行业+市值匹配与更严谨的统计显著性 (p 值 / Bootstrap CI)。
- extend_hold 模式支持衰减函数：`weight *= decay^t`。
- final_prediction 增加 `confidence` 字段，用事件历史命中率或最近窗口收益估算。
- 统一多事件源的冲突处理（例如相同股票不同事件加权规则）。

---
**版本**: v0.4 增加验证与最终预测脚本说明
