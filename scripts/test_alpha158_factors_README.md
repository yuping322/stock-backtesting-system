# Alpha158因子测试脚本说明

## 脚本功能

`scripts/test_alpha158_factors.sh` 用于测试Alpha158因子文件中的前10个因子。

## 使用方法

```bash
# 运行测试
./scripts/test_alpha158_factors.sh
```

## 测试内容

1. **因子文件**: `factors_test/Alpha158_20251004_20251103.csv`
2. **测试因子**: 前10个因子（KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2, OPEN0）
3. **日期范围**: 从因子文件中自动提取（2025-10-09 到 2025-10-31）
4. **股票池**: 使用因子文件中已有的股票代码（自动提取）
5. **调仓周期**: 5天和10天
6. **分位数**: 5

## 输出结果

- **监控文件**: `results/alpha158_test_first10/monitor.csv`
- **测试摘要**: 自动显示各因子的IC、IR等指标

## 测试结果示例

### 最佳表现因子（IC > 0.3）

| 因子 | IC | IR | 状态 |
|------|-----|-----|------|
| KMID2 | 1.000 | N/A | 🟢 |
| KSFT2 | 1.000 | N/A | 🟢 |
| KMID | 0.714 | 0.945 | 🟢 |
| KSFT | 0.429 | 0.439 | 🟢 |

### 负向因子（IC < -0.3）

| 因子 | IC | IR | 状态 |
|------|-----|-----|------|
| KLEN | -0.714 | -0.945 | 🔴 |
| KUP | -0.714 | -0.945 | 🔴 |
| OPEN0 | -0.714 | -0.945 | 🔴 |
| KUP2 | -0.429 | -0.439 | 🔴 |

## 注意事项

1. 脚本会自动从因子文件中提取日期范围和股票代码
2. 如果股票池获取失败，会使用因子文件中的股票代码
3. 测试结果会保存到 `results/alpha158_test_first10/monitor.csv`
4. 测试过程可能需要几分钟时间（取决于数据量）

## 查看结果

```bash
# 查看监控文件
cat results/alpha158_test_first10/monitor.csv

# 查看测试摘要
python -c "
import pandas as pd
df = pd.read_csv('results/alpha158_test_first10/monitor.csv')
print('测试因子数:', df['factor'].nunique())
print('调仓周期:', sorted(df['period'].unique()))
print('总测试数:', len(df))
"
```
