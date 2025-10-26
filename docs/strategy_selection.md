## 策略选择指南（结构化）

目的：当你的预测文件为 CSV（列：`date,code,weight`）时，帮助你选择合适的策略、理解引擎在 `backtest_engine.py` 中的具体行为，并列出常见边界/测试建议。

本文件按以下部分组织：
- 一览与快速推荐
- 输入与数据契约（DataLoader）
- 每个策略的实现细节、代码位置与行为差异
- 参数映射（config.py）
- CLI / UI 使用示例
- 常见陷阱与单元/烟雾测试建议
- 建议改进（小改动、可快速实现）

---

### 一览与快速推荐
- direct_execution：当每天的 CSV 已经是目标组合（每条 `weight` 是目标占比、且你希望“忠实执行”）→ 推荐使用。
- weighted_top_n：当 CSV 提供“得分/权重”但可能不归一且你希望按得分比例分配（并通常想归一化）→ 推荐使用或先截断为 top-N 后传入。
- equal_weight：当你只相信候选名单，不相信数值权重 → 使用等权分配。
- momentum：不依赖 CSV 的 weight（以历史价格动量为信号），当你想回测基于价格因子的策略时使用。

---

## 输入与数据契约

- DataLoader: `backtest_engine.py::DataLoader.load_prediction_data`
	- 期望最小列：`date`, `code`。
	- 会执行：
		- `code` 标准化为 6 位字符串 (zfill)
		- `date` 标准化为 date（去掉时间）
		- 若缺 `weight` 列，则会添加默认 `weight=1.0`。

- 市场行情：由 `data.py::load_bt_stocks` 加载（优先 OSS 快照、可回退），返回 Backtrader 的 `PandasData` feed；若某只股票无数据或 `close` 列含 NaN，会被跳过。

---

## 策略实现细节（代码映射与行为）

说明：下面对每个策略给出对应类、关键行为、和对你 `date,code,weight` 文件的具体影响。

1) direct_execution（类：`DirectExecutionStrategy`，文件：`backtest_engine.py`）
 - 关键方法：`DirectExecutionStrategy.execute_strategy`
 - 权重语义：把 `weight` 当作目标组合占比，计算 target_value = total_value * weight。
 - 下单方式：对每条 `today_df` 使用 `order_target_value(data, target_value)`。
 - 重要细节：
	 - 有 1% 全账户阈值避免微小调整（`if abs(target_value - current_value) / total_value > 0.01`）。
	 - **当前实现不会自动卖出当天 CSV 中未出现的旧持仓**（即缺席不等于卖出）。
 - 当你的 CSV 每天包含全仓且权重之和约为 1，效果就是把每只按所给目标占比持有。

2) weighted_top_n（类：`WeightedTopNStrategy`，文件：`backtest_engine.py`）
 - 关键方法：`WeightedTopNStrategy.execute_strategy`
 - 权重语义：把 CSV 的 `weight` 视为相对得分或权重，并先按该列降序排序（若存在），再以 `weight/total_weight` 的方式归一化分配资金。
 - 下单方式：target_value = total_value * (weight / total_weight)
 - 持仓保留：使用 `hold_days` 控制持仓天数（到期才卖出），所以并非仅凭当天是否出现来卖出。
 - 注意：实现中**并未自动对候选数量做 top-N 截断**（若所需行为是只持 top-N，请在生成 pred_df 时先截断或修改策略）。

3) equal_weight（类：`EqualWeightStrategy`，文件：`backtest_engine.py`）
 - 关键方法：`EqualWeightStrategy.execute_strategy`
 - 权重语义：忽略 `weight` 字段，直接对当日候选集合做等权分配。
 - top-N 行为：若 `len(today_df) > top_n_stocks`，会用 `today_df.head(top_n_stocks)`（注意按 CSV 顺序截取）决定持仓。
 - 清仓行为：会把不在当天集合中的旧持仓卖出（有移出/卖出逻辑）。

4) momentum（配置存在于 `config.py`）
 - 说明：动量策略基于历史价格计算动量（`momentum_period`），不以 CSV 的 `weight` 直接作为目标占比；CSV 可作为候选池（实现可变）。

---

## 参数与配置（在哪里改）

- 全局/系统参数：`config.SystemConfig`（`config.py`）——包含 `initial_cash`, `commission_rate`, `slippage_rate` 等。
- 策略参数：`config.STRATEGY_PARAMS`（`config.py`）中定义每个策略的 UI 显示名与参数（例如 `hold_days`, `top_n_stocks`, `momentum_period`）。
- 运行时参数：`BacktestEngine.get_strategy_params` 会把 `SystemConfig` 与 `StrategyConfig` 合并为最终运行参数（见 `backtest_engine.py`）。

---

## CLI / UI 使用示例

在 CLI（`main.py`）或 Streamlit UI (`app.py`) 中选择策略并运行。

示例：
```
python main.py --data-file data/3_30_ah_top.csv --strategy direct_execution --benchmark sh000300 --output-dir results/sample_run

python main.py --data-file data/3_30_ah_top.csv --strategy weighted_top_n --hold-days 3 --output-dir results/sample_run

python main.py --data-file data/3_30_ah_top.csv --strategy equal_weight --top-n 10 --output-dir results/sample_run
```

备注：Streamlit UI 会从 `config.list_strategies()`/`StrategyFactory.list_strategies()` 获取策略选项并显示 `get_strategy_info()` 中的参数表单。

---

## 常见陷阱与检测（对你当前文件的重点）

- 权重和 ≠ 1：
	- `direct_execution` 会按原样把每条 weight 乘以总市值；若权重和小于 1 会留下现金，若大于 1 会尝试超额配置（可能失败或导致保证金问题）。
	- `weighted_top_n` 会归一化（weight/total_weight），因此更健壮于非归一化得分输入。

- 缺席和卖出：
	- `equal_weight` 会卖出当天未在名单的持仓；`weighted_top_n` 使用到期（hold_days）卖出；`direct_execution` **当前不会**自动卖出未出现的旧持仓（需注意）。

- 行情缺失：`data.load_bt_stocks` 会跳过没有行情或含 NaN 的股票。回测开始前应校验 `pred_df` 与可用行情的交集（BacktestEngine 中已做部分过滤）。

---

## 测试建议（最小 smoke tests）

1. 准备一个小 CSV（3 天 × 3 股票），确保每天权重和为 1：用 `direct_execution` 运行并检查 `logs/trade_log.txt` 是否有按权重下单记录。
2. 用相同 CSV 改成每条 weight=1（未归一），用 `weighted_top_n` 运行，确认分配被归一化。
3. 用 `equal_weight` 并把 CSV 顺序打乱，确认 top-N 是按 CSV 前 N 行截取（若你希望按 weight 排序，需先排序 CSV）。

---

## 推荐改进（可快速实现）

1. `DirectExecutionStrategy`：增加“卖出未出现在当日 CSV 的持仓”选项（默认可通过 config 开关启用）。这是最常见的需求 —— 如果你把 CSV 当作每日“最终组合”，缺席应该表示卖出。
	 - 变更点：在 `DirectExecutionStrategy.execute_strategy` 的开始或末尾遍历 `self.holdings`，对不在 `today_df` 的 code 调用 `order_target_value(data, 0)`。
2. `WeightedTopNStrategy`：如果预期行为是只持 top-N，应在策略内部对 `today_df` 做 `head(top_n)`（或按 weight 排序后取 top_n）。当前实现依赖 pred_df 预处理。

---

## 参考代码位置
- 策略实现：`backtest_engine.py`（类名：`WeightedTopNStrategy`, `EqualWeightStrategy`, `DirectExecutionStrategy`）
- 预测数据加载与标准化：`backtest_engine.py::DataLoader.load_prediction_data`
- 行情数据加载：`data.py::load_bt_stocks` / `data.py::load_bt_oss_stocks`
- 配置与参数：`config.py::STRATEGY_PARAMS`, `config.SystemConfig`, `config.StrategyConfig`

---

如果你要我把第 1 条“DirectExecution 缺席清仓”实现成代码补丁并运行一次 smoke test（使用当前 `data/3_30_ah_top.csv`），我可以直接做并在修改完成后运行一轮回测（短时间 smoke 测试）。

文档更新完毕。

---

### 前提说明（DataLoader 期望）
- 文件最小列：`date`, `code`。
- 推荐包含：`weight`（若缺省，`DataLoader` 会默认设为 1.0）。
- `code` 会被标准化为 6 位字符串（`'1'` → `'000001'`）。
- `date` 会被标准化为日期（无时间部分）。

示例行（来自 `data/3_30_ah_top.csv`）：
```
2025-08-19,601633,0.1
2025-08-19,688279,0.1
... 每天约 10 条，每条 weight=0.1
```

如果每天每条权重加总约为 1（例如 10 条每条 0.1），那说明上游已经输出“目标组合”。

---

### 策略逐条行为（基于 `backtest_engine.py` 实现）

#### 1) direct_execution（直接执行）
- 权重语义：把 `weight` 当作目标占比（target_value = total_portfolio_value * weight）。
- 行为要点：遍历当天 `pred_df` 的每一行，按权重计算目标市值并使用 `order_target_value` 调整仓位；有 1% 全账户变动阈值避免无意义下单。
- 清仓行为：当前实现不会显式把“未出现在当天 CSV 的旧持仓”自动卖掉（注释提到但未实现）。
- 适用场景：上游直接输出每日最终组合（含权重）且希望“忠实执行”的情况。
- 风险：若每天权重和 ≠ 1，会产生现金残留或超额配置；若希望缺席表示卖出，需要额外实现卖出逻辑。

#### 2) weighted_top_n（按权重分配）
- 权重语义：把 `weight` 视为得分或相对权重；策略先按该列排序（降序），再把每条的权重归一化为当日总权重的比例（weight/total_weight），按比例分配资金。
- 行为要点：现有实现并不自动把候选数限制为 top-N（除非 `pred_df` 事先被截断）；使用 `hold_days` 管理到期卖出。
- 适用场景：模型输出评分，想按评分大小分配资金且自动归一化时。
- 风险：若上游权重本身就是目标占比并且已经和为 1，会与 direct_execution 效果一致；若权重和不为 1，策略会把它们强制归一化。

#### 3) equal_weight（等权）
- 权重语义：忽略 `weight` 字段，仅把当天候选集合平均分配。
- 行为要点：若候选行数 > `top_n_stocks`，策略会取 `today_df.head(top_n_stocks)`（注意：按文件顺序截取），然后把资金等分到这些股票；会显式卖出不在当天名单的旧持仓。
- 适用场景：当你只需要候选名单并想要稳定、简单的等权配置时。
- 风险：若你期望按 `weight` 排序或按数值选 top-N，需在生成 CSV 时先做好排序或修改策略代码。

#### 4) momentum（动量）
- 权重语义：通常不使用 CSV 的 `weight`；核心依据是历史价格动量（`momentum_period`）。
- 行为要点：按动量排名选股并按规则分配权重/等权；适用于信号来自价格因子的策略回测。
- 适用场景：你想用价格因子模拟策略或与模型结果做对比时。

---

### 对比要点（一目了然）
- 权重是否被当作“目标占比”：only `direct_execution`。
- 权重是否会被归一化：`weighted_top_n` 会归一化，`direct_execution` 不会。
- 是否卖出未在当天 CSV 的旧仓：`equal_weight` 会，`weighted_top_n` 使用 hold_days，`direct_execution` 当前不会。
- top-N 截断在哪里发生：`equal_weight` 内置截断（按 head），`weighted_top_n` 需在 `pred_df` 预处理或代码修改，`direct_execution` 完全由 `pred_df` 决定。

---

### 针对你当前数据（每天 10 条，每条 weight=0.1）的建议
- 如果你的每条 weight 已经表示“目标组合占比且合计为 1” → 用 `direct_execution`（最少改动）。但注意：若希望缺席等于卖出，请在 `DirectExecutionStrategy` 加入清仓逻辑。
- 如果你的 `weight` 只是“分数/信号值”，不要当作绝对占比；用 `weighted_top_n`（或先取 top-N 再传入）。
- 如果你不信任 `weight` 数值，希望简单等权 → 用 `equal_weight`（并设置 `top_n_stocks`）。

---

### 快速 CLI 示例
```
# 直接执行（忠实按 weight 乘总值）
python main.py --data-file data/3_30_ah_top.csv --strategy direct_execution --benchmark sh000300 --output-dir results/sample_run

# 按权重归一化分配（如果你想先截断为 top-N，可先生成一个新的 CSV）
python main.py --data-file data/3_30_ah_top.csv --strategy weighted_top_n --output-dir results/sample_run

# 等权（取前 N 行，按 CSV 顺序）
python main.py --data-file data/3_30_ah_top.csv --strategy equal_weight --top-n 10 --output-dir results/sample_run
```

---

### 推荐的下一个小改进（可选）
- 如果你偏好 `direct_execution` 且希望“缺席等于卖出”，建议：在 `DirectExecutionStrategy.execute_strategy` 中加入一段把 `self.holdings` 中今天未出现在 `today_df` 的持仓全部卖出的逻辑；我可以帮你实现并做一次 smoke test（用 `data/3_30_ah_top.csv`）。

文档结束 — 若要我把“清仓缺席持仓”的修改提交为代码 patch 并跑一次回测，请回复“实现清仓并测试”。
