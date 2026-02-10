# 因子生成系统 - 快速开始指南

## 🚀 5分钟快速入门

### 1. 导入模块

```python
from src.factor import generate_builtin_factors
```

### 2. 生成内置因子

```python
result = generate_builtin_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14']
)
```

### 3. 查看结果

```python
# 返回 3 个文件路径
print(result['factor_file'])      # 因子数据 CSV
print(result['metadata_file'])    # 元信息 JSON
print(result['readme_file'])      # 说明文档 MD
```

---

## 📋 四种使用方式

### 方式 1: 内置因子（4 个）

```python
from src.factor import generate_builtin_factors

result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
)
```

**支持的内置因子：**
- `VOL10`: 10日成交量比值
- `RSI_14`: 14日相对强弱指标
- `MA_20`: 20日移动平均比值
- `MACD_12_26_9`: MACD 指标

### 方式 2: TA-Lib 因子（200+）

```python
from src.factor import generate_talib_factors

result = generate_talib_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9', 'TALIB_BBANDS_20_2_2']
)
```

**因子命名格式：**
```
TALIB_{FUNCTION_NAME}_{PARAM1}_{PARAM2}_...

例如：
- TALIB_RSI_14          → RSI, 周期 14
- TALIB_MACD_12_26_9    → MACD, 快速 12, 慢速 26, 信号 9
- TALIB_BBANDS_20_2_2   → 布林带, 周期 20, 标准差 2
```

### 方式 3: 文件因子（自定义）

```python
from src.factor import generate_file_factors

result = generate_file_factors(
    factor_file_paths={
        'my_factor1': './factors/factor1.csv',
        'my_factor2': './factors/factor2.csv'
    },
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

**输入文件格式：**
```csv
date,code,factor_value
2024-01-15,000001,1.23
2024-01-15,000002,1.45
```

### 方式 4: 指数代码（需要先转换）

```python
from src.factor import generate_builtin_factors
from data import load_stock_pool

# 第一步：获取指数成分股
index_stocks = load_stock_pool('000001')['code'].tolist()

# 第二步：生成因子
result = generate_builtin_factors(
    stock_codes=index_stocks,  # 使用转换后的股票列表
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

---

## 📊 常见问题

### Q: 为什么要先调用 `load_stock_pool()` 获取成分股？

**A:** 设计是为了明确和避免歧义。直接传递指数代码可能导致混淆：
- 是指数本身？
- 是指数成分股？
- 是指数的加权组合？

显式调用 `load_stock_pool()` 让意图更清楚。

### Q: 参数 `factor_names=None` 时会怎样？

**A:** 会使用默认的因子列表：
- 内置因子：使用全部 4 个
- TA-Lib：使用常见的 4 个（RSI, MACD, BBANDS, ATR）
- 文件因子：使用所有文件中的因子

### Q: 如何加载大量股票（>1000）？

**A:** 系统会自动按批处理（默认 100 只），无需特殊配置：

```python
all_stocks = ['000001', '000002', ...]  # 5000 只股票

result = generate_builtin_factors(
    stock_codes=all_stocks,  # 会自动分批处理
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

### Q: 输出的 CSV 文件在哪里？

**A:** 在 `data/factor_tasks/` 目录下：

```
data/factor_tasks/
└── task_20250129_153000/
    ├── factors_20250129_153000.csv
    ├── task_metadata_20250129_153000.json
    └── README_task_20250129_153000.md
```

### Q: 如何自定义输出目录？

**A:** 使用 `output_dir` 参数：

```python
result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    output_dir='./my_factors'  # 自定义输出目录
)
```

---

## 🔍 内置因子详解

### VOL10 - 10日成交量比值

```
VOL10 = 今日成交量 / 10日平均成交量
```

衡量当日成交量相对于最近 10 天平均成交量的比值。值越大，说明当日成交量越活跃。

### RSI_14 - 14日相对强弱指标

```
RSI = 100 * (上升平均值 / (上升平均值 + 下降平均值))
```

范围 0-100。>70 表示超买，<30 表示超卖。衡量价格上升和下降的相对力度。

### MA_20 - 20日移动平均比值

```
MA_20 = 今日收盘价 / 20日移动平均
```

当前价格相对于 20 日平均价格的比值。>1 说明价格在均线上方，<1 说明在均线下方。

### MACD_12_26_9 - MACD 直方图

```
DIF = 12日EMA - 26日EMA
DEA = DIF的9日EMA
MACD = 2 * (DIF - DEA)
```

MACD 直方图的值。正值表示上升势头，负值表示下降势头。

---

## 💡 最佳实践

### 1. 验证股票代码

```python
# ✅ 正确：6位数字
generate_builtin_factors(stock_codes=['000001'])

# ❌ 错误：指数代码
generate_builtin_factors(stock_codes=['000001'])  # 如果这是指数，需要先转换
```

### 2. 合理设置日期范围

```python
# ✅ 推荐：最多 1 年数据
result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# ⚠️  注意：过长的时间范围可能导致计算缓慢
```

### 3. 指定需要的因子避免浪费计算

```python
# ✅ 推荐：只计算需要的因子
result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10']  # 只计算这个
)

# ❌ 避免：计算不需要的因子
# factor_names 留为 None 会计算全部
```

### 4. 处理错误

```python
try:
    result = generate_builtin_factors(...)
except ValueError as e:
    print(f"参数错误: {e}")  # 股票代码、日期等格式错误
except Exception as e:
    print(f"计算失败: {e}")  # 数据加载或计算错误
```

---

## 📚 进阶用法

### 加载结果数据

```python
import pandas as pd

result = generate_builtin_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# 读取因子数据
df = pd.read_csv(result['factor_file'])
print(df.shape)  # (行数, 列数)
print(df.head())
```

### 查看元信息

```python
import json

result = generate_builtin_factors(...)

# 读取元信息
with open(result['metadata_file']) as f:
    metadata = json.load(f)
    print(f"股票数: {metadata['stocks']['total']}")
    print(f"日期范围: {metadata['date_range']}")
    print(f"记录数: {metadata['total_records']}")
```

### 在回测中使用

```python
from src.factor import generate_builtin_factors

# 生成因子
result = generate_builtin_factors(
    stock_codes=portfolio_stocks,
    start_date=backtest_start,
    end_date=backtest_end,
    factor_names=['VOL10', 'RSI_14']
)

# 读取因子数据
factors_df = pd.read_csv(result['factor_file'])

# 使用因子进行回测
# ... 回测逻辑 ...
```

---

## 🆘 故障排查

### 问题 1: "TA-Lib 未安装"

**解决方案：**
```bash
pip install TA-Lib
```

### 问题 2: "未能加载到 OHLCV 数据"

**原因：** `data.py` 中的 `load_ohlcv()` 接口未正确配置

**解决方案：** 检查 `data.py` 是否实现了 `load_ohlcv()` 函数

### 问题 3: "股票代码无效"

**原因：** 股票代码格式不正确

**解决方案：**
```python
# ✅ 正确格式：6位数字字符串
stock_codes=['000001', '000002']

# ❌ 错误格式
stock_codes=['SH000001']  # 不需要前缀
stock_codes=[1, 2]         # 不是字符串
```

---

## 📞 获取帮助

- 完整文档：`docs/FACTOR_SYSTEM.md`
- 文件结构：`docs/FACTOR_FILE_STRUCTURE.md`
- 模块文档：`src/factor/README.md`
- 演示脚本：`src/factor/tests/test_demo.py`

---

**快速链接：**
- [完整系统文档](../FACTOR_SYSTEM.md)
- [文件结构设计](../FACTOR_FILE_STRUCTURE.md)
- [模块说明](./README.md)
