# 自定义因子接口说明

## 概述

因子检验模块现已支持自定义因子计算，提供了灵活的接口来：
1. 使用内置因子
2. 传入自定义 OHLCV 因子计算函数
3. 实现完全自定义的数据读取和因子计算
4. **从文件加载已计算好的因子数据**

## 核心接口

### FactorCalculator 基类

所有因子计算器都继承自 `FactorCalculator` 基类：

```python
class FactorCalculator(ABC):
    @abstractmethod
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算因子值"""
        pass
```

### 四种因子计算器类型

#### 1. BuiltinFactorCalculator - 内置因子

使用预定义的内置因子：

```python
from factor.factor_calculator import create_factor_calculator

calc = create_factor_calculator(factor_name='VOL10')
```

支持的内置因子：
- `VOL10`, `VOL20`: 成交量移动平均
- `VPT_12`: 价量趋势（12日累积）
- `RSI_14`: 相对强弱指标（14周期）
- `MA_5`, `MA_10`, `MA_20`: 简单移动平均
- `VOLUME_RATIO`: 成交量比率（当前成交量/20日均成交量）
- `PRICE_CHANGE`: 价格变化率
- `HIGH_LOW_RATIO`: 高低价比率

#### 2. OHLCVFactorCalculator - OHLCV 因子函数

接受一个函数，该函数接收 OHLCV DataFrame 并返回因子值：

```python
def my_factor(ohlcv):
    """自定义因子函数"""
    return ohlcv['close'] / ohlcv['close'].rolling(20).mean()

calc = create_factor_calculator(factor_func=my_factor)
```

**函数签名要求**：
- 输入：一个参数 `ohlcv`（DataFrame，包含 open, high, low, close, volume 列）
- 输出：`pd.Series`（因子值序列，索引为日期）

#### 3. CustomFactorCalculator - 完全自定义

实现完全自定义的数据读取和因子计算：

```python
def custom_calc(stock_code, start_date, end_date):
    """完全自定义的因子计算"""
    # 可以从任意数据源读取数据
    # 文件、数据库、API 等
    data = your_data_loader(stock_code, start_date, end_date)
    factor_values = your_calculation(data)
    return pd.Series(factor_values, index=...)

calc = create_factor_calculator(factor_func=custom_calc)
```

**函数签名要求**：
- 输入：三个参数 `(stock_code, start_date, end_date)`
- 输出：`pd.Series`（因子值序列，索引为日期）

#### 4. FileFactorCalculator - 从文件加载

从已计算好的文件中加载因子数据：

```python
# 从文件加载因子
calc = create_factor_calculator(
    file_path='data/my_factors.csv',
    factor_name='MY_FACTOR'
)
```

**文件格式要求**：
- CSV 格式
- 必须包含 `date` 和 `code` 列
- 必须包含因子列（由 `factor_name` 指定）
- 日期会自动转换为 datetime
- 股票代码会自动处理 `.XSHG` / `.XSHE` 后缀

**示例文件格式**：
```csv
date,code,MY_FACTOR,FACTOR_2
2024-01-01,000001,1.23,4.56
2024-01-02,000001,1.25,4.58
2024-01-01,000002,2.34,5.67
2024-01-02,000002,2.36,5.69
```

## 使用方法

### 创建单个因子计算器

```python
from factor.factor_calculator import create_factor_calculator

# 方式 1: 内置因子
calc1 = create_factor_calculator(factor_name='VOL10')

# 方式 2: OHLCV 函数
def momentum(ohlcv):
    return ohlcv['close'].pct_change(10)
calc2 = create_factor_calculator(factor_func=momentum)

# 方式 3: 完全自定义
def my_calc(code, start, end):
    # 自定义逻辑
    return pd.Series(...)
calc3 = create_factor_calculator(factor_func=my_calc)

# 方式 4: 从文件加载
calc4 = create_factor_calculator(
    file_path='data/my_factors.csv',
    factor_name='MY_FACTOR'
)
```

### 在 FactorTester 中使用

```python
from factor.factor import parse_args, CFG, FactorTester
from factor.factor_calculator import create_factor_calculator

# 解析命令行参数
args = parse_args()
cfg = CFG(args)

# 定义自定义因子
custom_factors = {
    'MY_FACTOR': create_factor_calculator(factor_func=my_factor_func),
    'VOL10': create_factor_calculator(factor_name='VOL10'),
}

# 创建测试器并运行
tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

## 完整示例

### 示例 1: 价格动量因子

```python
import pandas as pd
from factor.factor import parse_args, CFG, FactorTester
from factor.factor_calculator import create_factor_calculator

def price_momentum(ohlcv):
    """10日价格动量"""
    return ohlcv['close'].pct_change(10)

args = parse_args()
args.factors = ['PRICE_MOMENTUM']
cfg = CFG(args)

custom_factors = {
    'PRICE_MOMENTUM': create_factor_calculator(factor_func=price_momentum),
}

tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

### 示例 2: 多个因子组合

```python
def rsi_factor(ohlcv):
    """自定义 RSI 计算"""
    delta = ohlcv['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def volume_momentum(ohlcv):
    """成交量动量"""
    return ohlcv['volume'] / ohlcv['volume'].rolling(20).mean()

args = parse_args()
args.factors = ['RSI_CUSTOM', 'VOLUME_MOMENTUM', 'VOL10']
cfg = CFG(args)

custom_factors = {
    'RSI_CUSTOM': create_factor_calculator(factor_func=rsi_factor),
    'VOLUME_MOMENTUM': create_factor_calculator(factor_func=volume_momentum),
    'VOL10': create_factor_calculator(factor_name='VOL10'),
}

tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

### 示例 3: 从文件读取因子

```python
import pandas as pd

def load_from_file(stock_code, start_date, end_date):
    """从文件读取因子值"""
    file_path = f'factors/{stock_code}.csv'
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df = df.loc[start_date:end_date]
    return df['factor_value']

args = parse_args()
args.factors = ['FILE_FACTOR']
cfg = CFG(args)

custom_factors = {
    'FILE_FACTOR': create_factor_calculator(factor_func=load_from_file),
}

tester = FactorTester(cfg, custom_factors=custom_factors)
tester.run()
```

## 运行示例脚本

提供了一个完整的示例脚本 `factor/example_custom_factor.py`：

```bash
# 运行不同示例
python factor/example_custom_factor.py --example 1  # 内置因子
python factor/example_custom_factor.py --example 2  # OHLCV 因子
python factor/example_custom_factor.py --example 3  # 完全自定义
python factor/example_custom_factor.py --example 4  # 多因子组合
```

## 数据格式要求

### OHLCV DataFrame 格式

当使用 `OHLCVFactorCalculator` 时，数据加载器应返回以下格式的 DataFrame：

```python
DataFrame:
    index: 日期 (datetime)
    columns: ['open', 'high', 'low', 'close', 'volume']
    
示例:
              open    high     low   close     volume
date                                                  
2024-01-01   10.5    10.8    10.3    10.6    1000000
2024-01-02   10.6    10.9    10.5    10.7    1200000
```

### 因子值 Series 格式

所有因子计算器必须返回以下格式的 Series：

```python
Series:
    index: 日期 (datetime)
    values: 因子值 (float)
    
示例:
date
2024-01-01    1.05
2024-01-02    1.08
2024-01-03    1.12
```

## 最佳实践

1. **性能优化**: 对于相同的数据，使用缓存机制避免重复计算
2. **错误处理**: 在自定义函数中添加适当的错误处理
3. **数据验证**: 确保返回的 Series 索引是日期格式
4. **文档说明**: 为自定义因子添加清晰的文档字符串
5. **测试**: 在投入生产前测试自定义因子的正确性

## 常见问题

### Q: 如何添加新的内置因子？

A: 在 `factor_calculator.py` 的 `BuiltinFactorCalculator.BUILTIN_FACTORS` 字典中添加：

```python
BUILTIN_FACTORS = {
    ...
    'MY_NEW_FACTOR': lambda ohlcv: ohlcv['close'].rolling(30).mean(),
}
```

### Q: 数据加载失败怎么办？

A: 确保返回空的 DataFrame 或 Series，函数会处理缺失数据：

```python
def my_factor(ohlcv):
    if ohlcv.empty:
        return pd.Series(dtype=float)
    return ohlcv['close'].rolling(10).mean()
```

### Q: 可以使用多个数据源吗？

A: 使用 `CustomFactorCalculator`，可以在函数内部实现多数据源逻辑：

```python
def multi_source_factor(code, start, end):
    data1 = load_from_source1(code, start, end)
    data2 = load_from_source2(code, start, end)
    combined = merge_data(data1, data2)
    return calculate_factor(combined)
```

## 参考

- [factor_calculator.py](../factor/factor_calculator.py): 核心接口实现
- [example_custom_factor.py](../factor/example_custom_factor.py): 完整示例
- [factor_command_line.md](factor_command_line.md): 命令行使用说明
