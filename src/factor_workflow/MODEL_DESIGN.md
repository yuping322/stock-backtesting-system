# 模型层设计（简化版）

本设计说明聚焦于 `examples/factor_workflow` 的模型层实现，采用「固定两款核心模型 + 动态权重融合」思路，在保持依赖轻量的同时兼顾稳健性和可维护性。

---

## 1. 输入 / 输出契约

- **输入数据来源**
   - `features_panel.pkl`：`MultiIndex(datetime, instrument)` 的因子矩阵。
   - `label_panel.pkl`：与之对齐的未来收益标签。
   - `meta_series.pkl`：包含行业、流通市值等元数据，供后续中性化使用。
- **输出制品**
   - `workflow_main.py` 产出的模型预测（按模型、日期落地在 `workspace/` 下）。
   - `backtest_evaluation.generate_final_signal()` 返回的最终信号（已行业/市值中性化）。
   - `export_scores.py` 导出的 `weights.csv`：列为 `date,code,weight`，保证每日权重和为 1。

---

## 2. 模型名录

| 角色 | 标识 | 说明 |
| --- | --- | --- |
| 稳定基线 | `ridge_core` / `ridge_short` | 自研 SVD Ridge（`local_models._RidgeSolver`），支持样本权重与数值稳定处理。 |
| 非线性增强 | `histgb_core` / `histgb_short` | `sklearn.ensemble.HistGradientBoostingRegressor`，捕捉适度非线性。 |

短期套件采用更快的 EMA 权重响应，其余超参数复用长期模型。引入新模型时只需在 `models_config.py` 添加 spec，并在 `local_models.py` 提供实现。

---

## 3. 训练与融合流程

1. **数据准备**：`DatasetH` 自动推断日期区间（留足测试天数），并串联 `SafeRobustZScoreNorm → SafeFillna → SafeCSRankNorm`，彻底消除 Pandas `SettingWithCopyWarning`。
2. **模型训练**：每个 spec 独立 `fit → predict`，训练集默认包含 `train + valid` 片段。
3. **效果评估**：`model_pipeline` 计算逐日 Spearman IC、EMA 权重及其稳定度，写入 qlib workflow。
4. **动态融合**：依据 EMA IC、阈值与最小权重 (`min_weight=0.05`) 生成长短期融合预测。
5. **信号后处理**：`combine_signal` 对融合得分做标准化、行业/市值中性化（Ridge 退化时有 lstsq 兜底）、分位数裁剪，并再次归一化。
6. **权重导出**：`export_scores` 调整列名、按日正向化和归一化，并由回归测试锁定格式。

---

## 4. 稳健性设计

- **数值安全**：
   - Ridge 解算采用 SVD，对极端奇异值自动衰减；如遇失败，退回 `lstsq` 并记录日志。
   - 所有预测流程使用 `np.nan_to_num` 约束 NaN/Inf，避免传播异常。
- **数据防护**：
   - 预处理器强制浅拷贝，杜绝链式赋值告警。
   - 中性化前移除常量列与异常值，空样本时返回未处理信号。
- **回归测试**：`tests/examples/test_factor_workflow.py` 在 CI 中执行（`pytest -m slow`），验证训练→导出流程与权重归一性。

---

## 5. 运行节奏与监控建议

| 项目 | 建议频率 | 说明 |
| --- | --- | --- |
| 模型重训 | 每周/双周 | 样本有限时可扩展训练窗口；短期模型可更高频。 |
| 指标巡检 | 每日 | 关注 `daily_ic_*`、模型权重、预测分布漂移。 |
| 回测复核 | 每月 | 使用最新信号跑回测，观察收益 / 风险与实盘偏差。 |

触发调参或回滚的条件包括：30 日滚动 IC 持续低于阈值、实盘与回测出现系统性偏离、或新增因子需要灰度验证。

---

## 6. 文件/配置速查

| 文件 | 作用 |
| --- | --- |
| `dataset_config.py` | 推断时间窗口 & 构造 `DatasetH`，使用安全处理器。 |
| `models_config.py` | 声明长/短期模型清单与融合超参。 |
| `local_models.py` | sklearn 模型封装与自研 Ridge solver。 |
| `model_pipeline.py` | 训练、IC 评估、动态融合逻辑。 |
| `combine_signal.py` | 信号中性化、裁剪、再标准化。 |
| `export_scores.py` | 最终权重导出，确保列名与归一化一致。 |
| `tests/examples/test_factor_workflow.py` | 回归测试，防止导出格式退化。 |

---

> 若未来需要 AutoML 或更复杂的 stacking，可在 `model_pipeline.py` 中增设试验套件并将输出纳入融合流程，同时扩充测试覆盖面。