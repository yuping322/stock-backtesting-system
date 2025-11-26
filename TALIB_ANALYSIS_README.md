# 基于 TALIB 因子测试结果的后续分析指南

## 概述

本指南展示了如何基于大规模 TALIB 因子测试结果进行后续分析，从筛选优秀因子到生成可交易的投资组合权重。

## 分析流程

### 1. 分析测试结果并筛选优秀因子

```bash
# 运行分析脚本，自动筛选优秀因子
python talib_analysis_workflow.py
```

或者手动分析：

```python
import pandas as pd

# 读取测试结果
summary = pd.read_csv('results/factor_test_talib_202509/summary.csv')

# 筛选优秀因子（等级为'优秀'且状态不为'dead'）
good_factors = summary[
    (summary['level'] == '优秀') &
    (summary['status_flag'] != '🔴 dead')
]['factor_name'].unique()

print("优秀因子:", good_factors)
```

### 2. 使用优秀因子进行详细分析

```bash
# 使用筛选出的优秀因子进行更详细的检验
python main_factor.py \
    --start 2025-09-01 \
    --end 2025-11-23 \
    --factors TALIB_MACD_12_26_9,TALIB_RSI_14,TALIB_ADX_14 \
    --plot true \
    --plot-mode save \
    --output-dir results/talib_detailed_analysis \
    --quantiles 10 \
    --periods 5 10 20 \
    --roll-win 60
```

### 3. 转换为 factor_workflow 格式

```bash
# 将因子数据转换为 QLib panel 格式
python convert_factor_to_workflow.py \
    --factor-csv results/talib_detailed_analysis \
    --output-dir exported_data_all
```

### 4. 运行机器学习流程生成预测权重

```bash
# 进入 factor_workflow 目录
cd factor_workflow

# 训练模型
python workflow_main.py

# 评估和回测
python backtest_evaluation.py

# 导出预测权重
python export_scores.py

# 返回上级目录
cd ..
```

## 输出文件说明

- `results/good_talib_factors.txt`: 筛选出的优秀因子列表
- `results/talib_detailed_analysis/`: 详细因子分析报告和图表
- `exported_data_all/features_panel.pkl`: QLib 格式的因子数据
- `results/factor_workflow/scores.csv`: 最终预测权重（date,code,weight）

## 使用预测权重

生成的 `results/factor_workflow/scores.csv` 包含：
- `date`: 交易日期
- `code`: 股票代码（6位数字）
- `weight`: 标准化权重（每日和为1）

可以用于：
1. 投资组合构建
2. 回测分析
3. 实盘交易

## 快速开始脚本

运行 `run_talib_analysis.sh` 脚本执行完整流程：

```bash
chmod +x run_talib_analysis.sh
./run_talib_analysis.sh
```

## 注意事项

1. **数据依赖**: 需要 TALIB 因子测试结果在 `results/factor_test_talib_202509/`
2. **因子值**: 当前实现使用模拟数据，实际使用时需要真实的因子计算结果
3. **参数调整**: 可以根据需要调整分位数、周期、滚动窗口等参数
4. **性能**: 大规模因子分析可能需要较长时间

## 自定义因子分析

如果你有自己的因子数据，可以：

1. 将因子数据保存为 CSV 格式：`date,code,factor_value`
2. 直接使用 `convert_factor_to_workflow.py` 转换
3. 然后运行 factor_workflow 流程

```bash
# 示例：使用自定义因子
python convert_factor_to_workflow.py \
    --factor-csv my_factors.csv \
    --output-dir exported_data_all
```