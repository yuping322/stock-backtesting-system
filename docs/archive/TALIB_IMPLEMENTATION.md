<!-- markdownlint-disable MD033 -->
# TA-Lib 因子生成 - 完整实现说明

## 概述

✅ **TA-Lib 因子生成模块已完全重写，与 `factor_old/generate_talib_factors.py` 完全一致**

## 关键特性

### 1. 216 个因子，参数完全对齐

新的 `src/factor/generator/talib.py` 生成的 TA-Lib 因子列表与 `factor_old` **完全一致**（都是 216 个）：

```
✅ 新实现: 216 个因子
✅ factor_old: 216 个因子
✅ 列表完全相同（无差异）
```

### 2. 参数生成完全匹配

所有参数组合与 `factor_old` 完全相同：

- **RSI**: `[6, 14, 21]`
- **SMA/EMA/WMA** 等趋势指标: `[5, 10, 14, 20, 21, 26, 30, 50, 60]`
- **MACD**: `[[12, 26, 9]]`
- **BBANDS**: `[[5,2,2], [10,2,2], [20,2,2], [21,2,2]]`
- **ATR/NATR**: `[14, 21]`
- 其他 160+ 个函数的参数组合完全匹配

### 3. 因子命名规范

统一的命名规范，使得因子易于识别和管理：

```
TALIB_{FUNCTION}_{PARAM1}_{PARAM2}_...

例如：
- TALIB_RSI_14
- TALIB_SMA_20  
- TALIB_MACD_12_26_9
- TALIB_BBANDS_20_2_2
```

### 4. 智能跳过处理

自动跳过不需要的函数：

- **K线形态识别** (80+): CDL2CROWS, CDL3BLACKCROWS, ... (跳过)
- **数学函数** (15+): CEIL, FLOOR, SIN, COS, ... (跳过)
- **统计函数**: LINEARREG, VAR, STDDEV, CORREL, ... (跳过)
- **其他复杂函数**: SAR, SAREXT, HT_TRENDLINE, ... (跳过)

## 实现架构

### 核心类

#### 1. `TalibFactorListGenerator`
生成 TA-Lib 因子列表的工具类：

```python
# 获取所有 TA-Lib 函数
functions = TalibFactorListGenerator.get_talib_functions()

# 为指定函数生成参数组合
params = TalibFactorListGenerator.generate_common_parameters('RSI')
# 返回: [[6], [14], [21]]

# 生成完整的因子列表
factors = TalibFactorListGenerator.generate_talib_factors()
# 返回: ['TALIB_AD', 'TALIB_ADX_14', 'TALIB_ADX_21', ...]
```

#### 2. `TalibFactorCalculator`
计算具体的 TA-Lib 因子值：

```python
# 计算单个因子
result = TalibFactorCalculator.calculate('TALIB_RSI_14', ohlcv_df)
# 返回: Series，索引为日期，值为因子值
```

#### 3. `TalibFactorGenerator`
继承 `FactorGenerator` 基类，完整的因子生成流程：

```python
generator = TalibFactorGenerator(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9']
)
df = generator.generate()
```

## 支持的 TA-Lib 指标

### 趋势指标 (10+)
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3, ACCBANDS

### 动量指标 (15+)
RSI, STOCHRSI, MOM, ROC, ROCP, ROCR, ROCR100, TRIX, WILLR, CCI, CMO, PPO, APO, STOCH, STOCHF

### 波动率指标 (10+)
ATR, NATR, TRANGE, ADX, ADXR, DX, PLUS_DI, PLUS_DM, MINUS_DI, MINUS_DM

### 成交量指标 (5+)
AD, ADOSC, OBV, MFI, KDJ

### 价格变换 (4+)
AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

### Hilbert 变换 (5+)
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

### 其他指标 (5+)
AROON, AROONOSC, BBANDS, MACD, AVGDEV

**总计：约 160 个函数，生成 216 个因子组合**

## 文件结构

```
src/factor/
├── generator/
│   ├── __init__.py           # 导出 TalibFactorGenerator 等
│   ├── _base.py              # FactorGenerator 基类
│   ├── talib.py             # ✅ TA-Lib 因子生成器（新实现）
│   ├── builtin.py           # 内置因子生成器
│   ├── file.py              # 文件因子加载器
│   └── oss.py               # OSS 因子加载器
├── utils/
│   ├── __init__.py          # 导出所有工具函数
│   ├── helpers.py           # 辅助函数
│   ├── validation.py        # 参数验证
│   └── constants.py         # 常量定义
├── merger/                  # 合并层（待实现）
├── analyzer/                # 分析层（待实现）
└── __init__.py              # 主导出
```

## 使用示例

### 示例 1: 使用默认因子

```python
from src.factor.generator import generate_talib_factors

result = generate_talib_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(result['factor_file'])  # 输出因子文件路径
```

### 示例 2: 指定具体因子

```python
factors = [
    'TALIB_RSI_14',
    'TALIB_MACD_12_26_9',
    'TALIB_BBANDS_20_2_2',
    'TALIB_ATR_14',
    'TALIB_SMA_20',
]

result = generate_talib_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_names=factors
)
```

### 示例 3: 获取所有可用因子

```python
from src.factor.generator.talib import TalibFactorListGenerator

all_factors = TalibFactorListGenerator.generate_talib_factors()
print(f"可用因子数: {len(all_factors)}")
print(all_factors[:10])  # 显示前 10 个
```

## 输出格式

因子生成后，会保存为标准的 CSV 格式：

```
date,stock_code,TALIB_RSI_14,TALIB_MACD_12_26_9,TALIB_BBANDS_20_2_2,...
2024-01-02,000001,42.5,0.123,100.5,...
2024-01-03,000001,45.2,0.145,101.2,...
...
```

同时生成的文件：

```
task_20250129_153000/
├── factors_20250129_153000.csv          # 因子数据
├── task_metadata_20250129_153000.json   # 元信息
└── README_task_20250129_153000.md       # 说明文档
```

## 兼容性验证

通过 `tests/test_talib_compatibility.py` 完整验证：

```bash
✅ 测试 1: TA-Lib 可用性 - 通过
✅ 测试 2: 函数列表 (161 个函数) - 通过
✅ 测试 3: 参数生成 (RSI, MACD, BBANDS, ATR, SMA, EMA) - 通过
✅ 测试 4: 因子列表 (216 个因子完全一致) - 通过
✅ 测试 5: 因子计算 (RSI, SMA, ATR) - 通过

🎉 所有兼容性测试通过！
```

## 与 factor_old 的关键差异

| 特性 | factor_old | 新实现 | 备注 |
|-----|-----------|---------|-----|
| 架构 | 脚本式函数 | 类+继承 | 更模块化 |
| 参数生成 | 函数式 | 类方法 | 更易扩展 |
| 因子列表 | 生成函数 | 类方法 | 相同的列表 |
| 输出 | 多个 CSV | 单个 CSV + 元信息 | 更易管理 |
| 集成 | 独立脚本 | 因子框架的一部分 | 更好的协作 |

**重要**: 因子名称、参数组合、跳过列表完全相同，确保向后兼容性。

## 后续改进计划

- [ ] 支持更多的参数组合选项
- [ ] 添加 TA-Lib 函数的中文文档
- [ ] 性能优化（并行计算）
- [ ] 缓存已计算的因子
- [ ] 支持增量更新

## 相关文件

- `src/factor/generator/talib.py` - TA-Lib 生成器主实现
- `src/factor_old/generate_talib_factors.py` - 原始实现（参考）
- `tests/test_talib_compatibility.py` - 兼容性测试套件
- `docs/FACTOR_SYSTEM.md` - 因子系统总体文档
- `src/factor/README.md` - 模块文档
