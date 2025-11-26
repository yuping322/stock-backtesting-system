# 因子数据生成脚本

这个脚本用于生成因子数据并保存到文件中，不进行因子检验。

## 功能特性

- 支持内置因子（VOL10, RSI_14等）
- 支持TALIB因子（TALIB_RSI_14等）
- 支持从文件加载因子
- 支持批量处理多个因子
- 保存为标准CSV格式（date, code, factor_value）

## 使用方法

### 基本用法

```bash
# 生成单个因子
python generate_factors.py --factors VOL10 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

# 生成多个因子
python generate_factors.py --factors VOL10 RSI_14 TALIB_RSI_14 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

# 从文件加载因子
python generate_factors.py --factor-file data/factor_values_sample.csv --factor-name my_factor --start 2024-01-01 --end 2024-12-31 --output-dir factors_output

# 指定股票池和最大股票数
python generate_factors.py --factors VOL10 --stock-pool small --max-stocks 100 --start 2024-01-01 --end 2024-12-31 --output-dir factors_output
```

### 参数说明

- `--factors`: 要生成的因子列表（多个因子用空格分隔）
- `--factor-file`: 因子文件路径（当因子来自文件时使用）
- `--factor-name`: 因子列名（与 --factor-file 一起使用）
- `--factor-dir`: 因子文件目录，会自动查找包含指定因子的CSV文件
- `--start`: 开始日期 (YYYY-MM-DD)
- `--end`: 结束日期 (YYYY-MM-DD)
- `--stock-pool`: 股票池
  - `small`: 小盘股（默认）
  - `stock`: 全市场（目前使用沪深300作为示例）
  - 指数代码: 如 `000300` (沪深300), `000905` (中证500)
- `--max-stocks`: 最大股票数量限制（用于测试）
- `--output-dir`: 输出目录（默认: factor_data）
- `--overwrite`: 覆盖已存在的文件

### 输出文件

脚本会在输出目录中生成以下文件：

- `{因子名}_{时间戳}.csv`: 因子数据文件，包含 date, code, factor_value 三列
- `generation_summary.txt`: 生成汇总报告

### 示例输出文件内容

```csv
date,code,factor_value
2024-01-02,000001,123.45
2024-01-02,000002,67.89
2024-01-03,000001,124.56
2024-01-03,000002,68.90
```

## 支持的因子类型

### 内置因子

- VOL10: 10日成交量均值
- VOL20: 20日成交量均值
- VPT_12: 12日价量趋势
- RSI_14: 14日相对强弱指数
- MA_5/MA_10/MA_20: 简单移动平均
- 等等...

### TALIB因子

所有TA-Lib技术指标，格式为 `TALIB_{指标名}_{参数}`

例如：
- TALIB_RSI_14: RSI指标，周期14
- TALIB_MACD_12_26_9: MACD指标
- TALIB_BBANDS_20_2_2: 布林带
- TALIB_ADXR_14: ADXR指标

### 文件因子

从CSV文件加载因子数据，文件需要包含：
- date: 日期列
- code: 股票代码列
- {因子名}: 因子值列

## 注意事项

1. 确保有足够的OSS访问权限来获取股票数据
2. TALIB因子可能需要较长的预热期（warmup period）
3. 大量股票和长时间的数据生成可能需要较长时间
4. 生成的文件会按因子名称和时间戳命名，避免覆盖</content>
<parameter name="filePath">/Users/fengzhi/Downloads/git/stock-backtesting-system/README_generate_factors.md