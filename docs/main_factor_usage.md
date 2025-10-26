# main_factor.py 使用指南

## 概述

`main_factor.py` 是因子检验的主程序入口，提供了画图、结果保存等扩展功能。它将命令行参数处理、画图控制等功能从核心的 `factor.py` 模块中分离出来。

## 功能特点

- ✅ 命令行的画图开关（默认不画图）
- ✅ 画图模式选择（弹窗显示或保存到文件）
- ✅ 结果输出目录管理
- ✅ 自定义因子文件支持
- ✅ 所有核心能力保留在 `factor.py` 中

## 基本使用

### 不画图（默认）

```bash
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10
```

### 画图并弹窗显示

```bash
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --plot-mode popup
```

### 画图并保存到文件

```bash
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --plot-mode save --output-dir results/my_test
```

## 参数说明

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start` | str | '2024-09-25' | 回测开始日期 |
| `--end` | str | '2025-10-14' | 回测结束日期 |
| `--stock-pool` | str | '000510.XSHG' | 股票池 |
| `--factors` | list | ['VOL10', 'single_day_VPT_12'] | 因子列表 |
| `--quantiles` | int | 10 | 分组数量 |
| `--periods` | list | [5, 10, 15] | 调仓周期 |
| `--roll-win` | int | 60 | 滚动窗口交易日数 |
| `--monitor-csv` | str | 'monitor.csv' | 监控结果CSV文件路径 |

### 画图和输出参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--plot` | choice | 'false' | 是否画图 (true/false) |
| `--plot-mode` | choice | 'popup' | 画图模式 (popup/save) |
| `--output-dir` | str | None | 结果输出目录 |

### 自定义因子参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--custom-factor-file` | str | None | 自定义因子文件路径 |
| `--custom-factor-name` | str | None | 自定义因子列名 |

## 使用示例

### 示例 1: 基本使用（无画图）

```bash
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10
```

### 示例 2: 多个因子 + 画图弹窗

```bash
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --factors VOL10 VSTD10 single_day_VPT_12 \
  --plot true \
  --plot-mode popup
```

### 示例 3: 保存图表到指定目录

```bash
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --factors VOL10 \
  --plot true \
  --plot-mode save \
  --output-dir results/vol10_test_20250101
```

### 示例 4: 使用自定义因子文件

```bash
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --custom-factor-file data/my_factors.csv \
  --custom-factor-name MY_FACTOR \
  --factors MY_FACTOR \
  --plot true \
  --plot-mode save \
  --output-dir results/custom_factor_test
```

### 示例 5: 完整配置

```bash
python main_factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10 RSI_14 \
  --quantiles 5 \
  --periods 5 10 15 20 \
  --roll-win 30 \
  --monitor-csv custom_monitor.csv \
  --plot true \
  --plot-mode save \
  --output-dir results/comprehensive_test
```

## 输出说明

### 返回结果

`main_factor.py` 会返回完整的检验结果，包括：

- ✅ 各项得分指标（IC、IR、收益等）
- ✅ IC 和收益的时间序列数据
- ✅ 滚动监控指标
- ✅ 状态标识（🟢/🟡/🔴）
- ✅ `get_clean_factor_and_forward_returns` 的完整数据

### 不画图模式（--plot false）

- 控制台输出因子检验结果
- 自动保存结果到输出目录（如果指定）
- 生成 summary.csv 汇总文件
- 生成各因子的详细数据文件
- 不显示图表

### 画图 + 弹窗模式（--plot true --plot-mode popup）

- 控制台输出因子检验结果
- 弹出 matplotlib 窗口显示图表
- 关闭窗口后继续

### 画图 + 保存模式（--plot true --plot-mode save）

- 控制台输出因子检验结果
- 图表保存到指定目录（或自动生成目录）
- 结果数据保存到指定目录
- 保存的文件包括：
  - `summary.csv`：汇总结果
  - `{factor}_{period}/ic_series.csv`：IC 序列
  - `{factor}_{period}/ret_series.csv`：收益序列
  - `{factor}_{period}/clean_data.csv`：完整数据
  - `plot_1.png`, `plot_2.png`, ... （各个图表）

### 结果数据结构

```
results/
└── factor_test_20250101_143025/
    ├── summary.csv                    # 汇总结果（所有因子+周期）
    ├── VOL10_period5/
    │   ├── ic_series.csv             # IC 时间序列
    │   ├── ret_series.csv            # 收益时间序列
    │   └── clean_data.csv            # clean factor data
    ├── VOL10_period10/
    │   └── ...
    └── plot_*.png                     # 图表文件
```

## 与 factor.py 的区别

| 特性 | factor.py | main_factor.py |
|------|-----------|----------------|
| 定位 | 核心模块 | 命令行工具 |
| 画图 | 总是画图 | 可选画图 |
| 画图模式 | 弹窗 | 弹窗或保存 |
| 输出目录 | 不支持 | 支持 |
| 命令行参数 | 基础参数 | 扩展参数 |
| 自定义因子 | 代码实现 | 命令行参数 |

## 架构说明

```
main_factor.py (命令行入口)
  ├── 解析命令行参数
  ├── 创建 CFG 配置
  ├── 创建 FactorTester
  └── 调用 tester.run(plot=True/False)
  
factor/factor.py (核心模块)
  ├── FactorTester.run(plot) ← 接受画图参数
  ├── 因子计算
  ├── Alphalens 检验
  └── 条件画图 (if plot: ...)
```

## 常见问题

### Q: 为什么默认不画图？

A: 因为画图会阻塞执行，在服务器或无图形环境运行时不方便。需要时可以显式开启。

### Q: 如何批量运行并保存结果？

A: 使用 `--plot true --plot-mode save --output-dir <目录>` 即可自动保存。

### Q: 可以同时弹窗和保存吗？

A: 目前一次只能选择一种模式。可以运行两次来实现。

### Q: 如何从脚本调用？

A: 
```python
from main_factor import parse_main_args, main
# 或者直接导入 FactorTester
from factor.factor import FactorTester, CFG
```

## 输出目录结构

```
results/
└── factor_test_20250101_143025/
    ├── plot_1.png
    ├── plot_2.png
    └── ...
```

## 参考

- [factor/factor.py](../factor/factor.py): 核心因子检验模块
- [factor/README.md](../factor/README.md): 因子模块文档
- [docs/factor_command_line.md](factor_command_line.md): 因子命令行文档
