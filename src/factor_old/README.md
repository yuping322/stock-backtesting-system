# 因子检验模块

本模块用于单因子/多因子的 Alphalens 一键检验和自动打分。

## 文件说明

- `factor.py`: 主程序文件
- `factor_calculator.py`: 因子计算器接口（支持自定义因子）
- `example_custom_factor.py`: 自定义因子使用示例
- `export_data.py`: **数据导出工具**（导出价格和因子数据用于其他平台建模）
- `export_example.py`: 数据导出使用示例
- `docs/factor_command_line.md`: 命令行使用详细文档
- `docs/factor_refactoring_summary.md`: 重构总结文档

## 快速开始

### 基本使用

```bash
# 从项目根目录运行
python factor/factor.py

# 或使用绝对路径
python /path/to/stock-backtesting-system/factor/factor.py
```

### 常用命令

```bash
# 查看帮助
python factor/factor.py --help

# 指定回测区间
python factor/factor.py --start 2024-01-01 --end 2024-12-31

# 指定因子
python factor/factor.py --factors VOL10 VSTD10

# 自定义调仓周期
python factor/factor.py --periods 5 10 15
```

## 命令行参数

详见 [docs/factor_command_line.md](../docs/factor_command_line.md)

## 主要功能

1. **因子计算**: 支持多种因子类型的计算
2. **Alphalens 检验**: 自动运行因子有效性检验
3. **自动打分**: 根据 IC、IR、收益等指标自动打分
4. **滚动监控**: 实时监控因子表现
5. **可视化报告**: 生成完整的 tear-sheet

## 使用示例

```bash
# 完整配置示例
python factor/factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10 single_day_VPT_12 \
  --quantiles 10 \
  --periods 5 10 15 \
  --roll-win 60 \
  --monitor-csv monitor.csv
```

## 输出说明

运行后会输出：

1. **配置信息**: 显示当前使用的配置参数
2. **因子检验结果**: IC、IR、收益、单调性等指标
3. **打分结果**: 对每个调仓周期的因子表现进行打分
4. **滚动监控**: 显示滚动 IC、IR、波动率等指标
5. **状态标识**: 
   - 🟢 alive: 因子表现良好
   - 🟡 warning: 因子表现一般，需注意
   - 🔴 dead: 因子失效

## 自定义因子

本模块支持自定义因子计算函数，可以通过以下方式使用：

### 方式 1: 使用内置因子

```python
from factor.factor_calculator import create_factor_calculator

# 创建内置因子计算器
calc = create_factor_calculator(factor_name='VOL10')
```

### 方式 2: 自定义 OHLCV 因子函数

```python
import pandas as pd

def my_factor(ohlcv):
    """自定义因子：当前收盘价 / 20日均价"""
    return ohlcv['close'] / ohlcv['close'].rolling(20).mean()

# 创建因子计算器
calc = create_factor_calculator(factor_func=my_factor)
```

### 方式 3: 完全自定义的因子计算

```python
def custom_calc(stock_code, start_date, end_date):
    """可以从任意数据源读取数据"""
    # 实现自己的数据读取逻辑
    return pd.Series(...)

calc = create_factor_calculator(factor_func=custom_calc)
```

### 方式 4: 从文件加载因子

```python
# 从已计算好的文件加载因子
calc = create_factor_calculator(
    file_path='data/my_factors.csv',
    factor_name='MY_FACTOR'
)
```

### 完整示例

查看 `example_custom_factor.py` 了解详细用法：

```bash
python factor/example_custom_factor.py --example 1  # 内置因子
python factor/example_custom_factor.py --example 2  # OHLCV 因子
python factor/example_custom_factor.py --example 3  # 完全自定义
python factor/example_custom_factor.py --example 4  # 多因子组合
python factor/example_custom_factor.py --example 5  # 从文件加载
```

### 内置因子列表

- `VOL10`, `VOL20`: 成交量移动平均
- `VPT_12`: 价量趋势
- `RSI_14`: 相对强弱指标
- `MA_5`, `MA_10`, `MA_20`: 均线
- `VOLUME_RATIO`: 成交量比率
- `PRICE_CHANGE`: 价格变化率
- `HIGH_LOW_RATIO`: 高低价比率

更多详情参见 `factor_calculator.py`

## 数据导出功能

如果你需要在其他平台进行建模和测试，可以使用 `export_data.py` 导出价格和因子数据。

### 快速开始

```bash
# 导出最近3个月的数据（自动计算日期范围）
python factor/export_data.py --stocks 000001 000002 600000 --factors VOL10 VSTD10

# 指定日期范围
python factor/export_data.py --stocks 000001 000002 --factors VOL10 \
    --start 2024-01-01 --end 2024-03-31

# 指定输出目录
python factor/export_data.py --stocks 000001 --factors VOL10 --output ./my_data
```

### 导出模式

- `separate`: 分别导出价格数据和因子数据（生成 `price_data.csv` 和 `factor_data.csv`）
- `combined`: 导出合并的宽表数据（生成 `combined_data.csv`，包含所有字段）
- `both`: 同时导出分开和合并的数据（默认）

```bash
# 只导出合并数据
python factor/export_data.py --stocks 000001 --factors VOL10 --mode combined
```

### 输出文件格式

**price_data.csv**: 
- 列：`date`, `code`, `open`, `high`, `low`, `close`, `volume`, `amount`, ...
- 格式：长格式（每行一个日期+股票+字段值的组合）

**factor_data.csv**:
- 列：`date`, `code`, `因子1`, `因子2`, ...
- 格式：宽格式（每个因子一列）

**combined_data.csv**:
- 列：`date`, `code`, `open`, `high`, `low`, `close`, `volume`, `因子1`, `因子2`, ...
- 格式：宽格式（所有数据在一张表中）

### 使用示例

查看 `export_example.py` 了解详细的用法示例：

```bash
python factor/export_example.py
```

或在 Python 代码中直接调用：

```python
from export_data import export_combined_data, get_last_3_months

# 获取最近3个月的日期范围
start_date, end_date = get_last_3_months()

# 导出数据
export_combined_data(
    codes=['000001', '000002', '600000'],
    factors=['VOL10', 'VSTD10'],
    start_date=start_date,
    end_date=end_date,
    output_dir='./exported_data'
)
```

## 更多信息

- [详细使用文档](../docs/factor_command_line.md)
- [重构说明](../docs/factor_refactoring_summary.md)
- [自定义因子接口](../docs/factor_custom_factors.md)
- [从文件加载因子](../docs/factor_file_loading.md)
- [测试文档](../docs/factor_testing.md)