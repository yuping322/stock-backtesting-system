# V2 因子生成框架 - 快速开始指南

## 概述

`generator_v2` 是一个重新设计的因子生成框架，相比 V1 版本有以下改进：

| 方面 | V1 | V2 |
|------|-----|-----|
| **接口一致性** | 不同计算器接口不同 | 所有计算器统一 3 参数接口 |
| **错误处理** | 零散的异常处理 | 体系化异常层次结构 |
| **数据质量** | 无质量检查 | 7 项自动质量检查 |
| **可维护性** | 单文件 ~3000 行 | 模块化结构，职责清晰 |
| **测试性** | 难以单元测试 | 支持依赖注入，易于测试 |

## 文件结构

```
generator_v2/
├── __init__.py           # 模块入口，导出公共接口
├── exceptions.py         # 异常定义（5个自定义异常）
├── calculator.py         # 计算器实现（4种计算器类型）
├── generator.py          # 生成器实现（编排层）
├── quality.py            # 质量检查框架
├── utils.py              # 工具函数（数据加载、处理、配置）
└── examples.py           # 使用示例
```

## 核心概念

### 1. 分层架构

```
用户代码
   ↓
Generator (编排层)
   ├── 参数验证
   ├── 流程控制
   ├── 错误处理
   └── 结果报告
   ↓
Calculator (计算层)
   ├── BuiltinFactorCalculator
   ├── TalibFactorCalculator
   ├── CustomFunctionCalculator
   └── FileFactorCalculator
   ↓
Quality Checker (验证层)
   └── 7项数据质量检查
```

### 2. 统一的计算器接口

所有计算器都实现相同的接口：

```python
class FactorCalculator(ABC):
    @abstractmethod
    def calculate(
        self, 
        stock_code: str,      # 股票代码
        start_date: str,      # 开始日期 (YYYY-MM-DD)
        end_date: str         # 结束日期 (YYYY-MM-DD)
    ) -> pd.Series:           # 返回因子值序列
        pass
```

这样做的好处：
- ✅ 接口简单清晰
- ✅ 易于测试和扩展
- ✅ 支持多种因子来源
- ✅ 容易进行参数传递

### 3. 异常体系

```
FactorGenerationException (基类)
├── DataNotAvailableError       # 数据不可用
├── FactorCalculationError      # 计算失败
├── FactorValidationError       # 验证失败
└── PartialResultError          # 部分失败
```

每个异常都提供详细的上下文信息。

## 快速开始

### 方式 1：使用单个计算器

```python
from src.factor.generator_v2 import create_factor_calculator

# 创建计算器
calculator = create_factor_calculator('VOL10')

# 计算因子
result = calculator.calculate(
    stock_code='000001',
    start_date='2024-01-01', 
    end_date='2024-12-31'
)

# 结果是一个 pd.Series，索引为日期
print(result)
```

### 方式 2：使用内置因子生成器

```python
from src.factor.generator_v2 import BuiltinFactorGenerator

# 创建生成器
generator = BuiltinFactorGenerator(
    stock_codes=['000001', '000002', '000858'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_names=['VOL10', 'RSI_14', 'MA_20'],
    output_dir='./data/factor_tasks'
)

# 生成因子
df = generator.generate()

# 结果保存到 CSV，包含自动生成的元数据
print(df)
```

### 方式 3：创建自定义计算器

```python
from src.factor.generator_v2 import create_factor_calculator

# 定义自定义函数
def my_custom_factor(ohlcv: pd.DataFrame) -> pd.Series:
    """计算自定义因子"""
    return (ohlcv['close'] - ohlcv['close'].rolling(20).mean()) / ohlcv['close'].rolling(20).std()

# 创建计算器
calculator = create_factor_calculator(
    factor_name='MY_CUSTOM_FACTOR',
    factor_func=my_custom_factor
)

# 使用
result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
```

### 方式 4：从文件加载因子

```python
from src.factor.generator_v2 import create_factor_calculator

# 创建计算器
calculator = create_factor_calculator(
    file_path='./data/my_factors.csv'
)

# 使用
result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
```

## 支持的内置因子

| 因子名 | 描述 | 参数 |
|--------|------|------|
| `VOL10` | 10日成交量均值 | window=10 |
| `RSI_14` | 14日相对强弱指数 | period=14 |
| `MA_20` | 20日收盘价均线 | window=20 |
| `MACD_12_26_9` | MACD 指标 | fast=12, slow=26, signal=9 |

## 支持的 Talib 因子

可以直接使用 Talib 的任何指标，格式为 `TALIB_指标名_参数`：

```python
# RSI 指标，周期 14
calculator = create_factor_calculator('TALIB_RSI_14')

# 布林带，周期 20，标准差倍数 2
calculator = create_factor_calculator('TALIB_BBANDS_20_2')
```

## 错误处理

### 处理特定错误

```python
from src.factor.generator_v2 import (
    DataNotAvailableError,
    FactorCalculationError,
    PartialResultError,
)

try:
    df = generator.generate()
except PartialResultError as e:
    print(f"成功: {e.successful_count}")
    print(f"失败: {e.failed_count}")
    print(f"失败详情: {e.get_failure_summary()}")
except DataNotAvailableError as e:
    print(f"数据不可用: {e.stock_code}")
    print(f"原因: {e.reason}")
except FactorCalculationError as e:
    print(f"计算失败: {e.factor_name}")
    print(f"原因: {e.reason}")
```

## 数据质量检查

生成器会自动执行 7 项质量检查：

1. **必需列检查** - 检查是否有 date, stock_code 等必需列
2. **数据类型检查** - 检查日期、数值等数据类型
3. **NaN 比例检查** - 如果 NaN > 70% 报错，> 20% 警告
4. **日均股票数检查** - 检查每日股票数是否一致
5. **异常值检查** - 使用 3倍 IQR 方法检测离群值
6. **日期连续性检查** - 检查是否有超过 5 天的数据间隙
7. **股票代码规范化** - 确保股票代码为 6 位数字格式

示例：

```python
from src.factor.generator_v2 import DataQualityChecker

# 手动执行质量检查
result = DataQualityChecker.check_factor_output(df, ['VOL10', 'RSI_14'])

# 打印检查结果
DataQualityChecker.print_check_result(result, verbose=True)

if not result['passed']:
    print(f"检查失败，共 {len(result['issues'])} 个问题")
```

## 工具函数

### 数据加载

```python
from src.factor.generator_v2.utils import DataLoader

# 从 OSS 加载数据
df = DataLoader.load_ohlcv('000001', '2024-01-01', '2024-12-31')

# 从 CSV 加载数据
df = DataLoader.load_from_csv('./data/ohlcv.csv', stock_code='000001')
```

### 数据处理

```python
from src.factor.generator_v2.utils import DataProcessor

# 规范化股票代码
code = DataProcessor.normalize_stock_code('1')  # '000001'

# 填充 NaN 值
series = DataProcessor.fill_na_forward(series, limit=5)

# 移除异常值
series = DataProcessor.remove_outliers(series, method='iqr', threshold=3.0)

# 标准化
series = DataProcessor.standardize(series, method='zscore')
```

### 配置管理

```python
from src.factor.generator_v2.utils import ConfigManager

# 获取因子参数
params = ConfigManager.get_builtin_params('VOL10')  # {'window': 10}

# 加载配置文件
config = ConfigManager.load_config_file('./config.json')
```

## 性能优化建议

1. **批量处理** - 使用生成器而不是单个计算器，可以共享数据加载

```python
# ✅ 好 - 一次加载所有股票的数据
generator = BuiltinFactorGenerator(stock_codes=all_codes, ...)
df = generator.generate()

# ❌ 差 - 每只股票单独加载数据
for code in all_codes:
    calc = create_factor_calculator('VOL10')
    result = calc.calculate(code, ...)
```

2. **缓存** - 对于重复计算的因子，可以缓存结果

```python
# 可以在计算器中实现缓存
calculator = create_factor_calculator('VOL10')
result1 = calculator.calculate('000001', '2024-01-01', '2024-12-31')
result2 = calculator.calculate('000001', '2024-01-01', '2024-12-31')  # 使用缓存
```

3. **并行处理** - 使用多进程处理多个股票

```python
from multiprocessing import Pool

def compute_factor(args):
    stock_code, start_date, end_date = args
    calculator = create_factor_calculator('VOL10')
    return calculator.calculate(stock_code, start_date, end_date)

# 使用 Pool 并行处理
with Pool() as pool:
    results = pool.map(compute_factor, [...])
```

## 从 V1 迁移

### 关键变化

| 功能 | V1 | V2 |
|------|-----|-----|
| 创建计算器 | 直接实例化类 | 使用工厂函数 |
| 计算接口 | 不统一 | `calculate(code, start, end)` |
| 错误处理 | `try/except Exception` | 特定异常类型 |
| 参数传递 | 通过类属性 | 通过函数参数 |

### 迁移步骤

1. 将 `from src.factor.generator import ...` 改为 `from src.factor.generator_v2 import ...`
2. 使用 `create_factor_calculator()` 替代直接实例化
3. 更新异常处理逻辑
4. 使用生成器进行批量处理

## 常见问题

### Q: 如何添加新的计算器类型？

A: 继承 `FactorCalculator` 并实现 `calculate()` 方法：

```python
from src.factor.generator_v2 import FactorCalculator

class MyFactorCalculator(FactorCalculator):
    def calculate(self, stock_code, start_date, end_date):
        # 实现计算逻辑
        return pd.Series(...)
```

### Q: 如何自定义质量检查？

A: 扩展 `DataQualityChecker.check_factor_output()` 方法或创建自己的检查函数。

### Q: 计算器如何获取数据？

A: 默认从 `src.data.data.load_oss_complex_stocks()` 加载。可以通过 `set_data_loader()` 注入自定义数据加载函数。

## 完整工作流示例

```python
from src.factor.generator_v2 import BuiltinFactorGenerator, PartialResultError

# 1. 定义参数
stock_codes = ['000001', '000002', '000858', ...]
start_date = '2024-01-01'
end_date = '2024-12-31'
factor_names = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']

# 2. 创建生成器
generator = BuiltinFactorGenerator(
    stock_codes=stock_codes,
    start_date=start_date,
    end_date=end_date,
    factor_names=factor_names,
    output_dir='./data/factor_tasks'
)

# 3. 生成因子
try:
    df = generator.generate()
    print(f"✅ 生成成功: {len(df)} 条数据")
except PartialResultError as e:
    print(f"⚠️  部分失败: {e}")
except Exception as e:
    print(f"❌ 生成失败: {e}")

# 4. 使用结果
print(df.head())
print(df.describe())
```

## 参考文档

- [异常处理指南](./exceptions.py)
- [计算器实现](./calculator.py)
- [生成器实现](./generator.py)
- [质量检查](./quality.py)
- [工具函数](./utils.py)
- [使用示例](./examples.py)
