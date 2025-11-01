# Scripts 使用说明

本目录包含所有因子测试相关的脚本。

## 📊 汇总功能

### 生成所有结果的汇总报告

```bash
bash scripts/generate_summary_readme.sh
```

这将：
1. 扫描所有 `results/` 目录
2. 找到所有有 README.md 的测试批次
3. 生成一个汇总文件 `results/ALL_RESULTS_SUMMARY.md`

**查看汇总**:
```bash
cat results/ALL_RESULTS_SUMMARY.md
```

## 🔧 测试脚本

### 1. 测试所有因子

```bash
bash scripts/run_all_factors.sh
```

- 测试 `available_factors.txt` 中的所有因子
- 日期范围: 2025-07-26 ~ 2025-10-24
- 输出目录: `results/factor_test_all_oss/`

### 2. 测试前10个因子

```bash
bash scripts/run_first10_factors.sh
```

快速测试前10个因子，适合快速验证。

### 3. 继续运行剩余因子

```bash
bash scripts/run_remaining_factors.sh
```

- 自动跳过已完成的因子
- 只运行剩余未完成的因子
- 支持断点续跑

### 4. 测试指定范围的因子

```bash
bash scripts/run_factor_range.sh
```

测试指定索引范围的因子。

### 5. 重新测试失败的因子

```bash
bash scripts/run_failed_factors.sh
```

只测试上次失败的因子。

## 📈 工作流程

### 标准流程

1. **首次测试**
   ```bash
   bash scripts/run_all_factors.sh
   ```

2. **如果中断，继续运行**
   ```bash
   bash scripts/run_remaining_factors.sh
   ```

3. **生成汇总报告**
   ```bash
   bash scripts/generate_summary_readme.sh
   ```

4. **查看结果**
   ```bash
   cat results/ALL_RESULTS_SUMMARY.md
   ```

### 查看具体批次结果

```bash
# 查看具体因子的测试结果
cat results/factor_test_first10/README.md

# 查看所有因子汇总
cat results/ALL_RESULTS_SUMMARY.md
```

## 📂 输出结构

```
results/
├── ALL_RESULTS_SUMMARY.md       # 汇总报告（所有批次）
├── factor_test_first10/          # 第一批次
│   ├── README.md                 # 此批次摘要
│   ├── summary.csv               # CSV汇总
│   └── [因子名]/                 # 各因子详细结果
├── factor_test_all_oss/          # 所有因子
│   └── ...
└── ...
```

## 🔍 结果解读

### README.md 内容

每个批次的README包含：

1. **测试配置**
   - 日期范围
   - 股票池
   - 因子列表
   - 分位数、调仓周期

2. **结果汇总表格**
   - 因子名
   - 调仓周期
   - 等级（优秀/良好/一般）
   - 状态（🟢 alive / 🟡 warning / 🔴 dead）
   - 通过指标数

3. **详细得分**
   - IC均值、ICIR
   - 因子收益
   - 分位数收益
   - 单调性等

### 状态说明

- 🟢 **alive**: 因子表现优秀，可继续使用
- 🟡 **warning**: 因子表现一般，需要监控
- 🔴 **dead**: 因子失效，不推荐使用

## 🛠️ 自定义

### 修改测试配置

编辑脚本中的配置：

```bash
# 日期范围
START_DATE="2025-07-26"
END_DATE="2025-10-24"

# 输出目录
OUTPUT_DIR="results/factor_test_all_oss"
```

### 添加新的测试批次

1. 创建新的脚本文件
2. 设置唯一的 OUTPUT_DIR
3. 运行并生成结果
4. 自动被汇总脚本收录

## 💡 最佳实践

1. **定期生成汇总**
   ```bash
   bash scripts/generate_summary_readme.sh
   ```

2. **查看最新汇总**
   ```bash
   cat results/ALL_RESULTS_SUMMARY.md
   ```

3. **按需运行子集**
   - 第一次只测试前10个
   - 确认流程正确后再跑全部

4. **利用断点续跑**
   - 运行 `run_remaining_factors.sh`
   - 节省时间，避免重复

## 📊 对比分析

汇总报告帮你：

1. **快速浏览**所有批次的因子表现
2. **对比**不同因子在各周期下的表现
3. **筛选**最佳因子用于策略构建
4. **追踪**因子有效性随时间的变化

## 🎯 下一步

1. 查看汇总报告找出最佳因子
2. 在ML模块中使用有效因子
3. 定期重新测试验证因子有效性

