# 因子层 (Factor Module) 指南

本文档说明因子生成框架 (Generator V2) 的设计理念与操作规范。

## 1. 核心理念

**“接口统一，计算分离”**
- 所有的因子计算器都对外暴露相同的 3 参数接口：`calculate(codes, start, end)`。
- 支持 Builtin (内置)、TA-Lib、Custom (自定义) 三类计算引擎。

## 2. 因子目录结构

- `src/factor/generator_v2/`: 核心计算框架（生产级）。
- `src/factor/generator/`: 旧版计算逻辑（归档中）。
- `research/ml/`: 机器学习因子建模与训练。

## 3. 核心计算器类型

| 类型 | 说明 | 示例 |
| :--- | :--- | :--- |
| **Builtin** | 纯 Python 实现的常见指标 | `VOL10`, `MA20` |
| **TA-Lib** | C++ 底层加速的 200+ 技术指标 | `TALIB_RSI`, `TALIB_MACD` |
| **Custom** | 用户通过独立 `.py` 脚本定义的逻辑 | 任何复杂的非线性因子 |

## 4. 质量检查 (Quality Verification)

系统内置 7 项因子质量检查，确保数据在进入策略前是准确且一致的：
1.  **缺失值 (NaNs)**
2.  **无穷大 (Infs)**
3.  **常数值 (Constant)**
4.  **异常值 (Outliers)**
5.  **分布偏移 (Shift)**
6.  **覆盖率 (Coverage)**
7.  **时间完整性 (Timeline)**

## 5. 绩效评估 (Alphalens)

通过 `analyze_factor_performance` 接口，系统可以自动生成 Alphalens 的 Tear Sheet，包含：
- **IC 分析**: 提供因子与未来收益的相关性。
- **分层收益分析**: 验证因子的单调性。
- **换手率分析**: 评估因子的交易频率要求。
