# Qlib 因子生成器集成说明

## 概述

已成功将 Qlib 因子生成功能集成到新的因子生成系统中。现在用户可以使用统一的 API 生成和加载 Qlib 因子（Alpha158、Alpha360 等）。

## 文件位置

### 新创建的文件

1. **`src/factor/generator/qlib.py`** - Qlib 因子生成器模块
   - `QlibDatasetBuilder`: 从 data.py 构建 Qlib 数据集
   - `QlibFactorExtractor`: 从 Qlib 数据集提取因子
   - `QlibFactorGenerator`: 统一的因子生成器接口
   - `generate_qlib_factors()`: 快捷函数

2. **`tests/test_qlib_integration.py`** - 集成测试套件
   - Qlib 生成器测试
   - 生成 -> 文件加载的完整流程测试
   - CSV 格式检测测试

### 修改的文件

1. **`src/factor/generator/_base.py`**
   - 更新 `load_ohlcv_data()` 使用真实的 `data.load_oss_complex_stocks()` 接口
   - ✅ 所有 4 个生成器现在都使用真实的市场数据

2. **`src/factor/generator/file.py`**
   - 增强 CSV 格式检测，支持：
     - 标准格式: `date, code, factor_value`
     - Qlib 格式: `date, code, Alpha005, Alpha010, ...(100+ 列)`
     - 多列格式: `date, code, factor1, factor2, ...`
   - 添加 `_detect_csv_format()` 方法自动识别 CSV 类型
   - 添加 `_load_csv_file()` 方法处理不同格式

3. **`src/factor/generator/__init__.py`**
   - 添加 `generate_qlib_factors` 到导出列表

## 使用方式

### 1. 生成 Qlib 因子

```python
from src.factor.generator.qlib import generate_qlib_factors

# 生成 Alpha158 因子
df = generate_qlib_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_set='Alpha158',  # 也支持 Alpha360, Alpha158vwap, Alpha360vwap
    output_file='./data/alpha158_factors.csv'
)
```

### 2. 使用 File 加载器加载 Qlib CSV

```python
from src.factor.generator.file import generate_file_factors

# 自动检测并加载 Qlib CSV
result = generate_file_factors(
    factor_file_paths={
        'qlib_alpha158': './data/Alpha158_20240101_20241231.csv'
    },
    stock_codes=['000001'],  # 可选过滤
    start_date='2024-01-01',  # 可选过滤
    end_date='2024-01-31'     # 可选过滤
)

# 返回字典
# {
#     'factor_file': '/path/to/output/factors_*.csv',
#     'metadata_file': '/path/to/output/task_metadata_*.json',
#     'readme_file': '/path/to/output/README_task_*.md'
# }
```

### 3. 完整的生成 -> 加载流程

```python
from src.factor.generator.qlib import generate_qlib_factors
from src.factor.generator.file import generate_file_factors
import pandas as pd

# 步骤 1: 生成 Qlib 因子文件
print("第一步: 生成 Alpha158 因子...")
df_generated = generate_qlib_factors(
    stock_codes=['000001', '000002', '000003'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_set='Alpha158',
    output_file='./alpha158_factors.csv'
)

# 步骤 2: 用 File 加载器加载这个文件
print("第二步: 用 File 加载器加载 CSV...")
result = generate_file_factors(
    factor_file_paths={
        'alpha158': './alpha158_factors.csv'
    },
    stock_codes=['000001'],  # 只加载第一只股票
    start_date='2024-06-01',  # 过滤日期范围
    end_date='2024-06-30'
)

# 步骤 3: 读取最终输出
df_final = pd.read_csv(result['factor_file'])
print(f"最终数据形状: {df_final.shape}")
print(f"因子数: {len(df_final.columns) - 2}")
```

## 支持的 Qlib 因子集

| 因子集名称 | 因子数 | 描述 |
|-----------|-------|------|
| `Alpha158` | 158 | 标准 Alpha158 因子集 |
| `Alpha360` | 360 | 标准 Alpha360 因子集（更多因子） |
| `Alpha158vwap` | 158 | 基于成交量加权平均价格的 Alpha158 变体 |
| `Alpha360vwap` | 360 | 基于成交量加权平均价格的 Alpha360 变体 |

## CSV 格式说明

### 标准格式（单个因子）
```
date,code,factor_value
2024-01-01,000001,1.23
2024-01-02,000001,4.56
2024-01-02,000002,2.34
```

### Qlib Alpha 格式（多个因子）
```
date,code,Alpha005,Alpha010,Alpha015,...(100+ 列)
2024-01-01,000001,0.5,0.3,0.8,...
2024-01-02,000001,0.6,0.4,0.9,...
2024-01-02,000002,0.7,0.5,0.2,...
```

### 多列格式
```
date,code,factor1,factor2,factor3
2024-01-01,000001,1.23,4.56,7.89
2024-01-02,000001,2.34,5.67,8.90
```

## 数据流向

```
Qlib 数据集构建
    ↓
数据源: data.load_oss_complex_stocks()
    ↓
QlibDatasetBuilder
    ↓
Qlib 二进制格式 (calendars/, instruments/, features/)
    ↓
QlibFactorExtractor
    ↓
MultiIndex DataFrame (date, code)
    ↓
QlibFactorGenerator
    ↓
标准格式 DataFrame (date, stock_code, factor1, factor2, ...)
    ↓
CSV 文件或内存 DataFrame
    ↓
FileFactorGenerator (CSV 格式检测)
    ↓
最终输出 CSV
```

## 关键特性

### 1. 实时数据加载
- Qlib 生成器使用 `data.load_oss_complex_stocks()` 加载真实市场数据
- 支持自动缓存机制，避免重复计算

### 2. 灵活的 CSV 格式支持
- 自动检测 CSV 格式（标准/Qlib/多列）
- 自动处理列名变异（code/stock_code, date/Date 等）
- 支持代码格式标准化（去除交易所后缀，补齐 6 位）

### 3. 数据过滤
- 可选的股票代码过滤
- 可选的日期范围过滤
- 缺失数据处理

### 4. 完整的集成链
- Qlib 生成器生成的 CSV 可直接用 File 加载器加载
- 统一的输出格式和接口
- 一致的性能警告（但不影响功能）

## 测试结果

### 集成测试覆盖

✅ **测试 1: Qlib 因子生成器**
- 158 个 Alpha 因子成功生成
- 14 条记录（2 只股票 × 7 天）
- 160 列（date + stock_code + 158 个因子）

✅ **测试 2: Qlib 生成 -> File 加载集成**
- Qlib 生成 Alpha158 CSV 文件
- File 加载器正确加载和处理
- 过滤功能正常（股票、日期）
- 最终输出 4 条记录（1 只股票 × 4 天）

✅ **测试 3: CSV 格式检测**
- 标准格式检测正确
- Qlib 格式自动识别
- 多列格式支持

## 命令行用法

```bash
# 生成 Alpha158 因子
python src/factor/generator/qlib.py \
  --factor-set Alpha158 \
  --codes 000001 000002 000003 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output ./factors

# 生成 Alpha360 因子
python src/factor/generator/qlib.py \
  --factor-set Alpha360 \
  --codes 000001 000002 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output ./factors \
  --qlib-cache ./qlib_cache
```

## 常见问题

### Q: 如何从已有的 Qlib CSV 文件加载？
```python
result = generate_file_factors(
    factor_file_paths={
        'existing_factors': './path/to/Alpha158_20240101_20241231.csv'
    }
)
```

### Q: 可以混合加载多个源的因子吗？
```python
result = generate_file_factors(
    factor_file_paths={
        'qlib_alpha158': './qlib_factors.csv',
        'custom_factors': './my_custom.csv'
    }
)
```

### Q: Qlib 数据集会缓存吗？
是的，构建过的 Qlib 数据集会自动缓存，避免重复计算。使用 `rebuild=True` 可强制重建。

### Q: 性能警告怎么处理？
这些是 pandas 的性能建议，不影响功能。可以忽略，也可以修改 `_load_csv_file()` 中的数据构造方式。

## 依赖需求

- **Qlib**: `pip install pyqlib` (可选，仅在生成 Qlib 因子时需要)
- **Pandas**: 用于 CSV 处理
- **NumPy**: 用于数值计算

## 后续改进建议

1. **性能优化**: 改进 `_load_csv_file()` 的 DataFrame 构造方式
2. **并行处理**: 支持多线程/多进程处理多个 CSV 文件
3. **增量更新**: 支持追加新日期数据而不需要重新计算
4. **缓存管理**: 添加缓存清理和版本管理工具
5. **监控告警**: 添加数据质量检查和异常告警

## 相关文件参考

- 原始实现: `src/factor_old/generate_qlib_factors.py`
- 数据接口: `src/data/data.py`
- 基类: `src/factor/generator/_base.py`
- 配置: `src/factor/utils/`
