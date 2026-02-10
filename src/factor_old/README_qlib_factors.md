# Qlib因子生成与使用指南

## 概述

本模块提供了从qlib提取Alpha158/Alpha360因子并生成文件的功能，支持在`factor_calculator`中直接使用。

## 生成因子文件

### 使用方法

```bash
# 生成Alpha158因子文件（指定股票代码）
python factor/generate_qlib_factors.py \
    --factor-set Alpha158 \
    --codes 000001 000002 600000 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors

# 生成Alpha158因子文件（使用股票池）
python factor/generate_qlib_factors.py \
    --factor-set Alpha158 \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors

# 生成Alpha360因子文件
python factor/generate_qlib_factors.py \
    --factor-set Alpha360 \
    --stock-pool HS300 \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --output ./factors \
    --rebuild  # 强制重建qlib数据集
```

### 参数说明

- `--factor-set`: 因子集名称，`Alpha158` 或 `Alpha360`
- `--codes`: 股票代码列表（空格分隔）
- `--stock-pool`: 股票池名称（如HS300），与`--codes`二选一
- `--start`: 开始日期，格式 `YYYY-MM-DD`
- `--end`: 结束日期，格式 `YYYY-MM-DD`
- `--output`: 因子文件输出目录（默认：`./factors`）
- `--qlib-cache`: qlib数据集缓存目录（可选，默认在output目录下）
- `--rebuild`: 强制重建qlib数据集

### 输出文件

生成的因子文件格式：
- 文件名：`Alpha158_20240101_20241231.csv`（因子集_开始日期_结束日期）
- 格式：CSV，包含列：`date`, `code`, `ROC5`, `MA10`, ...（所有因子列）
- 数据格式：MultiIndex (date, code)，每列是一个因子

## 使用因子文件

### 在factor_calculator中使用

```python
from factor.factor_calculator import create_factor_calculator

# 方式1：从目录自动查找因子文件（推荐）
calc = create_factor_calculator(
    factor_name='ROC5',  # 因子名称
    factor_dir='./factors'  # 因子文件目录
)

# 方式2：直接指定文件路径
calc = create_factor_calculator(
    factor_name='ROC5',
    file_path='./factors/Alpha158_20240101_20241231.csv'
)

# 使用因子计算器
factor_values = calc.calculate(
    stock_code='000001',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
print(factor_values)
```

### 支持的因子名称

Alpha158包含158个因子，常见的有：
- `ROC5`, `ROC10`, `ROC20`: 收益率（不同窗口）
- `MA5`, `MA10`, `MA20`: 移动平均
- `STD5`, `STD10`: 标准差
- `RSV5`, `RSV10`: 相对强弱值
- `KMID`: K线中间价
- `KLEN`: K线长度
- `OPEN0`, `HIGH0`, `LOW0`, `CLOSE0`, `VWAP0`: 当前价格
- ... 共158个因子

Alpha360包含360个因子，更多样化。

## 工作流程

```
1. 从data.py获取股票数据
   ↓
2. 构建qlib数据集（calendars/instruments/features）
   ↓
3. 使用Alpha158/Alpha360 handler提取因子
   ↓
4. 保存为CSV文件（MultiIndex: date, code）
   ↓
5. factor_calculator从文件加载因子
```

## 注意事项

1. **数据依赖**: 需要确保`data.py`可以正常获取股票数据（OSS配置等）
2. **qlib版本**: 需要安装qlib: `pip install pyqlib`
3. **日期格式**: 因子文件中的日期为`date`类型（不是datetime）
4. **代码格式**: 股票代码统一为6位数字字符串（如`000001`）
5. **文件查找**: `factor_dir`会自动查找包含指定因子的最新CSV文件

## 示例

完整示例：

```python
# 1. 生成因子文件
# python factor/generate_qlib_factors.py --factor-set Alpha158 --stock-pool HS300 --start 2024-01-01 --end 2024-12-31 --output ./factors

# 2. 使用因子
from factor.factor_calculator import create_factor_calculator

# 创建ROC5因子计算器
roc5_calc = create_factor_calculator(factor_name='ROC5', factor_dir='./factors')

# 计算某只股票的ROC5因子值
roc5_values = roc5_calc.calculate('000001', '2024-01-01', '2024-01-31')
print(f"ROC5因子值:\n{roc5_values}")

# 创建MA10因子计算器
ma10_calc = create_factor_calculator(factor_name='MA10', factor_dir='./factors')
ma10_values = ma10_calc.calculate('000001', '2024-01-01', '2024-01-31')
print(f"MA10因子值:\n{ma10_values}")
```

