# 因子检验与训练策略指南

## 整体策略

### 阶段1: 因子有效性验证（3个月）
用最近3个月数据快速验证因子是否有效

```bash
# 最近3个月验证
python main_factor.py \
  --start 2024-10-01 \
  --end 2024-12-31 \
  --roll-win 60 \
  --factors VOL10 VSTD10 MAC20 PEG \
  --output-dir results/validation_3m
```

**筛选标准**: 只看 🟢 alive 的因子
- `roll_ir` ≥ 0.3（滚动IC比率）
- 胜率 ≥ 52%
- 净IR ≥ 0.2
- 最近5天 IC 不能全为负
- Q5-Q1 Sharpe ≥ 0.5

### 阶段2: 历史训练数据（6-12个月）
用更长历史数据训练模型

```bash
# 用过去6个月训练
python main_factor.py \
  --start 2024-07-01 \
  --end 2024-12-31 \
  --roll-win 60 \
  --factors VOL10 VSTD10 \
  --output-dir results/training_6m
```

**或使用1年数据**:
```bash
# 用过去1年训练
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --roll-win 60 \
  --factors VOL10 VSTD10 \
  --output-dir results/training_1y
```

### 阶段3: 生成预测信号
使用筛选出的因子生成预测

```bash
python scripts/build_predictions.py \
  --factor-file data/factor_values_sample.csv \
  --factors VOL10 VSTD10 \
  --output-dir data/predictions \
  --mode composite
```

## 关键指标说明

### 3个月验证的关键指标

1. **roll_ir** (滚动IC比率): 最近ROLL_WIN天的IC均值/标准差
   - ≥ 0.3: 好
   - 0.2-0.3: 一般
   - < 0.2: 差

2. **win_rate** (IC胜率): IC为正的天数占比
   - ≥ 52%: 好

3. **net_ir** (扣费净IR): 考虑换手成本后的净IC比率
   - ≥ 0.2: 好

4. **状态标志**:
   - 🟢 alive: 因子有效
   - 🟡 warning: 需要观察
   - 🔴 dead: 因子失效

## 推荐工作流

### 每周/每月检查因子

1. **运行3个月验证**
   ```bash
   python main_factor.py \
     --start $(date -v-3m +%Y-%m-01) \
     --end $(date +%Y-%m-%d) \
     --factors factor1 factor2 factor3
   ```

2. **查看监控CSV**
   ```bash
   cat monitor.csv
   ```

3. **筛选有效因子**: 只看 🟢 alive

4. **用6-12个月历史训练**: 选择更长窗口重新跑

5. **生成预测**: 用筛选出的因子

## 参数调优建议

### 调整ROLL_WIN（滚动窗口）
- `roll-win 30`: 1.5个月，更灵敏，容易波动
- `roll-win 60`: 3个月（默认），平衡灵敏度和稳定性
- `roll-win 90`: 4个月，更稳定但反应慢

### 调整时间窗口
- **3个月**: 快速验证，适合频繁检查
- **6个月**: 平衡稳定性和响应速度
- **12个月**: 更稳定的长期有效性评估

## 输出结果解读

### summary.csv
包含所有因子和周期的汇总数据

### 各因子的period_N文件夹
- `ic_series.csv`: IC序列，分析因子与收益的相关性
- `ret_series.csv`: 收益序列，分析因子收益
- `scores.csv`: 各项指标得分

### 状态判断
- 🟢 状态: 因子健康，可以使用
- 🟡 状态: 需要关注，可能开始衰减
- 🔴 状态: 因子失效，建议停止使用

## 注意事项

1. **IC衰减**: 因子通常会随时间衰减，定期检查很重要
2. **市场环境**: 不同市场环境下因子表现可能差异很大
3. **样本外测试**: 用过去6个月训练，用最近3个月验证
4. **多重检验**: 因子多了要调整p值阈值

## 示例脚本

### 完整工作流脚本

```bash
#!/bin/bash
# factor_workflow.sh

# 1. 最近3个月验证
python main_factor.py \
  --start 2024-10-01 \
  --end 2024-12-31 \
  --factors VOL10 VSTD10 MAC20 PEG \
  --output-dir results/validation_3m

# 2. 查看结果，提取有效因子（手动筛选）
# 从输出中找 🟢 alive 的因子

# 3. 用6个月历史训练有效因子
python main_factor.py \
  --start 2024-07-01 \
  --end 2024-12-31 \
  --factors VOL10 VSTD10 \
  --output-dir results/training_6m

# 4. 生成预测
python scripts/build_predictions.py \
  --factor-file data/factor_values_sample.csv \
  --factors VOL10 VSTD10 \
  --mode composite \
  --output-dir data/predictions
```

