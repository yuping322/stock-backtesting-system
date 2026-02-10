# 从文件加载因子功能说明

## 概述

因子检验模块现已支持从已计算好的文件中加载因子数据，这对于：
- 预先计算好的因子数据
- 外部因子文件
- 大规模因子分析

非常有用。

## 功能特点

1. **自动缓存**: 文件内容只加载一次，后续调用使用缓存
2. **日期范围过滤**: 自动过滤到指定的日期范围
3. **股票代码标准化**: 自动处理 `.XSHG` / `.XSHE` 后缀
4. **错误处理**: 优雅处理文件不存在或格式错误的情况

## 文件格式要求

### CSV 格式

文件必须是 CSV 格式，包含以下列：

- `date`: 日期列（必需）
- `code`: 股票代码列（必需）
- 因子列: 一个或多个因子列（必需）

### 示例文件

```csv
date,code,MY_FACTOR,FACTOR_2,VOLUME_MA
2024-01-01,000001,1.23,4.56,1000000
2024-01-02,000001,1.25,4.58,1050000
2024-01-01,000002,2.34,5.67,2000000
2024-01-02,000002,2.36,5.69,2100000
```

### 股票代码格式

支持两种格式：
- 纯数字: `000001`, `000002`
- 带后缀: `000001.XSHG`, `000002.XSHE`

系统会自动标准化处理。

## 使用方法

### 方式 1: 使用 FileFactorCalculator

```python
from factor.factor_calculator import FileFactorCalculator

# 创建文件因子计算器
calc = FileFactorCalculator(
    file_path='data/my_factors.csv',
    factor_name='MY_FACTOR'
)

# 获取因子值
factor_values = calc.calculate('000001', '2024-01-01', '2024-01-31')
```

### 方式 2: 使用 create_factor_calculator

```python
from factor.factor_calculator import create_factor_calculator

# 创建文件因子计算器
calc = create_factor_calculator(
    file_path='data/my_factors.csv',
    factor_name='MY_FACTOR'
)
```

### 方式 3: 在 FactorTester 中使用

```python
from factor.factor import parse_args, CFG, FactorTester
from factor.factor_calculator import create_factor_calculator

# 解析命令行参数
args = parse_args()
args.factors = ['MY_FACTOR']
cfg = CFG(args)

# 创建文件因子计算器
custom_factors = {
    'MY_FACTOR': create_factor_calculator(
        file_path='data/my_factors.csv',
        factor_name='MY_FACTOR'
    ),
}

# 运行检验
tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

## 完整示例

### 示例 1: 基本使用

```python
import pandas as pd
from factor.factor import parse_args, CFG, FactorTester
from factor.factor_calculator import create_factor_calculator

# 创建示例因子文件
def create_sample_file():
    data = {
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'] * 2,
        'code': ['000001'] * 3 + ['000002'] * 3,
        'MY_FACTOR': [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
    }
    df = pd.DataFrame(data)
    df.to_csv('data/sample_factors.csv', index=False)

# 创建因子文件
create_sample_file()

# 运行因子检验
args = parse_args()
args.factors = ['MY_FACTOR']
cfg = CFG(args)

custom_factors = {
    'MY_FACTOR': create_factor_calculator(
        file_path='data/sample_factors.csv',
        factor_name='MY_FACTOR'
    ),
}

tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

### 示例 2: 使用 data.py 的 factor_for_al

也可以直接使用 `data.py` 中的 `factor_for_al` 函数：

```python
import data

# 从文件加载因子
factor_series = data.factor_for_al(
    codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_name='MY_FACTOR',
    file_path='data/my_factors.csv'
)

print(factor_series)
```

## 运行示例脚本

使用提供的示例脚本：

```bash
python factor/example_custom_factor.py --example 5
```

这会演示如何从文件加载因子。

## 优势

### 1. 性能优化

- 文件只加载一次并缓存
- 避免重复计算
- 适合大规模数据分析

### 2. 灵活性

- 支持任意 CSV 格式的因子文件
- 可以包含多个因子列
- 日期范围自动过滤

### 3. 易用性

- 简单的 API
- 自动处理常见问题
- 清晰的错误信息

## 常见问题

### Q: 文件格式不对怎么办？

A: 确保文件包含 `date` 和 `code` 列，以及因子列。

### Q: 如何处理多个因子？

A: 为每个因子创建单独的计算器：

```python
custom_factors = {
    'FACTOR_1': create_factor_calculator(
        file_path='data/factors.csv',
        factor_name='FACTOR_1'
    ),
    'FACTOR_2': create_factor_calculator(
        file_path='data/factors.csv',
        factor_name='FACTOR_2'
    ),
}
```

### Q: 文件很大会影响性能吗？

A: 文件只加载一次并缓存，后续调用非常快。

### Q: 如何与 data.py 的 factor_for_al 结合使用？

A: `factor_for_al` 现在支持 `file_path` 参数：

```python
factor_series = data.factor_for_al(
    codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_name='MY_FACTOR',
    file_path='data/my_factors.csv'  # 添加文件路径
)
```

## 参考

- [factor_calculator.py](../factor/factor_calculator.py): 核心实现
- [data.py](../data.py): `factor_for_al` 函数
- [example_custom_factor.py](../factor/example_custom_factor.py): 完整示例
- [自定义因子接口](factor_custom_factors.md): 详细文档
