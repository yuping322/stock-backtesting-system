# 因子生成系统 - 实现完成

## 📋 完成进度

### ✅ 已完成的模块

#### 1. 工具层 (`utils/`)
- ✅ `__init__.py`: 模块导出
- ✅ `helpers.py`: 辅助函数（时间戳、目录创建、数据加载）
- ✅ `validation.py`: 参数验证函数
- ✅ `constants.py`: 常量定义（因子列表、配置）

#### 2. 生成层 (`generator/`)
- ✅ `__init__.py`: 模块导出
- ✅ `_base.py`: 基类 `FactorGenerator` 和公共函数
- ✅ `builtin.py`: 内置因子生成（VOL10, RSI_14, MA_20, MACD_12_26_9）
- ✅ `talib.py`: TA-Lib 因子生成（200+ 技术指标）
- ✅ `file.py`: 文件因子加载
- ✅ `oss.py`: OSS 因子框架（Alpha158/360）

#### 3. 合并层 (`merger/`)
- ✅ `__init__.py`: 合并层核心实现
- ✅ `merge_factor_files()`: 合并多个因子文件
- ✅ `merge_factor_directory()`: 合并整个目录的因子文件
- ✅ `check.py`: 合并验证工具
- ✅ `m.py`: 合并示例脚本

#### 4. 分析层 (`analyzer/`)
- ✅ `__init__.py`: 分析层模块导出
- ✅ `core.py`: 完整的 `FactorAnalyzer` 类实现
- ✅ 集成 Alphalens 因子分析
- ✅ 支持多周期打分和滚动监控
- ✅ 图表生成和保存功能

#### 5. 顶层接口
- ✅ `src/factor/__init__.py`: 导出所有公共接口

#### 6. 文档和示例
- ✅ `tests/test_demo.py`: 使用示例和测试脚本
- ✅ `tests/factor/test_merger.py`: 合并层测试
- ✅ 多个测试文件覆盖核心功能

### ⏳ 待完成的模块

#### 1. 元信息保存
- ⏳ `utils/metadata.py`: `save_task_metadata()` 函数

#### 2. 完整单元测试
- ⏳ `tests/test_builtin.py`: 内置因子测试
- ⏳ `tests/test_talib.py`: TA-Lib 因子测试
- ⏳ `tests/test_analyzer.py`: 分析层测试

#### 3. 性能优化和监控
- ⏳ 批量处理优化
- ⏳ 内存使用监控
- ⏳ 错误恢复机制

---

## 📁 目录结构

```
src/factor/
├── __init__.py                    # 顶层导出接口
├── generator/                      # 生成层
│   ├── __init__.py
│   ├── _base.py                   # 基类（✅ 完成）
│   ├── builtin.py                 # 内置因子（✅ 完成）
│   ├── talib.py                   # TA-Lib 因子（✅ 完成）
│   ├── file.py                    # 文件因子（✅ 完成）
│   └── oss.py                     # OSS 因子（✅ 完成）
├── merger/                         # 合并层
│   ├── __init__.py                # 核心合并函数（✅ 完成）
│   ├── check.py                   # 合并验证工具（✅ 完成）
│   └── m.py                       # 合并示例脚本（✅ 完成）
├── analyzer/                       # 分析层
│   ├── __init__.py                # 模块导出（✅ 完成）
│   └── core.py                    # FactorAnalyzer类（✅ 完成）
├── utils/                         # 工具层
│   ├── __init__.py
│   ├── helpers.py                 # 辅助函数（✅ 完成）
│   ├── validation.py              # 参数验证（✅ 完成）
│   ├── constants.py               # 常量定义（✅ 完成）
│   └── metadata.py                # 元信息保存（待实现）
└── tests/                         # 测试
    ├── __init__.py
    ├── test_demo.py               # 演示脚本（✅ 完成）
    └── test_merger.py             # 合并层测试（✅ 完成）
```

---

## 🚀 快速开始

### 1. 生成内置因子

```python
from src.factor import generate_builtin_factors

result = generate_builtin_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14', 'MA_20']
)

# 输出：
# {
#     'factor_file': 'data/factor_tasks/task_YYYYMMDD_HHMMSS/factors_YYYYMMDD_HHMMSS.csv',
#     'metadata_file': '..._metadata_....json',
#     'readme_file': '...README_....md'
# }
```

### 2. 生成 TA-Lib 因子

```python
from src.factor import generate_talib_factors

result = generate_talib_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9']
)
```

### 3. 加载文件因子

```python
from src.factor import generate_file_factors

result = generate_file_factors(
    factor_file_paths={
        'custom_factor1': './factors/my_factor1.csv',
        'custom_factor2': './factors/my_factor2.csv'
    },
    stock_codes=['000001', '000002']
)
```

### 5. 合并因子文件

```python
from src.factor.merger import merge_factor_files, merge_factor_directory

# 合并多个因子文件
result_df = merge_factor_files([
    './data/factor_tasks/task1/factors_20250101.csv',
    './data/factor_tasks/task2/factors_20250102.csv'
], output_file='./data/merged_factors.csv')

# 合并整个目录的因子文件
result_df = merge_factor_directory(
    factor_dir='./data/factor_tasks',
    pattern='**/factors_*.csv',
    exclude_factors=['noise'],
    output_file='./data/merged_all_factors.csv'
)
```

### 6. 因子分析

```python
from src.factor.analyzer.core import FactorAnalyzer
import pandas as pd

# 准备因子数据
factor_data = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
    'asset': ['000001', '000001', '000002', '000002'],
    'factor_value': [1.2, 1.5, 2.1, 1.8]
})

# 创建分析器
analyzer = FactorAnalyzer(
    factor_df=factor_data,
    start_date='2024-01-01',
    end_date='2024-01-31',
    quantiles=10,
    periods=[5, 10, 15]
)

# 执行因子分析
results = analyzer.analyze_factor(factor_name='my_factor', plot=True)

# 查看分析结果
for result in results:
    print(f"因子: {result.factor_name}, 周期: {result.period}天, 等级: {result.level}")
```

---

## 📊 内置因子

| 因子名称 | 描述 | 计算方式 |
|---------|------|--------|
| VOL10 | 10日成交量比值 | 今日成交量 / 10日平均成交量 |
| RSI_14 | 14日相对强弱指标 | RSI = 100 * (上升平均 / (上升平均+下降平均)) |
| MA_20 | 20日移动平均比值 | 今日收盘价 / 20日移动平均 |
| MACD_12_26_9 | MACD指标 | 2 * (DIF - DEA)，其中 DIF=EMA(12)-EMA(26) |

---

## 🔧 技术细节

### 参数验证

所有生成函数都进行严格的参数验证：
- `stock_codes`: 必须是 6 位数字的股票代码列表
- `start_date`, `end_date`: 必须是有效的日期，且 start < end
- `factor_names`: 必须是有效的因子名称

### 输出格式

所有因子都输出为标准格式：
```
date,stock_code,VOL10,RSI_14,MA_20,...
2024-01-15,000001,1.23,45.67,100.2,...
2024-01-15,000002,1.45,46.10,100.5,...
```

### 任务管理

每个生成任务会自动创建：
```
data/factor_tasks/task_YYYYMMDD_HHMMSS/
├── factors_YYYYMMDD_HHMMSS.csv       # 因子文件
├── task_metadata_YYYYMMDD_HHMMSS.json # 元信息
└── README_task_YYYYMMDD_HHMMSS.md    # 说明文档
```

---

## 💡 设计特点

### 1. 模块化设计
- 每个因子生成函数一个文件
- 避免单个文件过大
- 易于维护和扩展

### 2. 继承体系清晰
```
FactorGenerator (基类)
├── BuiltinFactorGenerator
├── TalibFactorGenerator
├── FileFactorGenerator
└── OSSFactorGenerator
```

### 3. 明确的参数处理
- `stock_codes` 只接收股票代码（不自动识别指数）
- 需要指数时，用户显式调用 `load_stock_pool()` 获取成分股

### 4. 完整的输出
- 因子 CSV 文件
- 元信息 JSON 文件
- 说明文档 Markdown 文件

---

## 📚 相关文档

- `docs/FACTOR_SYSTEM.md`: 完整的系统文档
- `docs/FACTOR_FILE_STRUCTURE.md`: 文件结构设计说明

---

## 🧪 测试

运行演示脚本：
```bash
cd /path/to/stock-backtesting-system
python src/factor/tests/test_demo.py
```

---

## ⚠️ 注意事项

### 1. TA-Lib 安装

如果要使用 TA-Lib 因子，需要先安装：
```bash
pip install TA-Lib
```

### 2. 数据依赖

需要 `data.py` 模块提供的接口：
- `load_ohlcv()`: 加载 OHLCV 数据
- `load_stock_pool()`: 获取指数成分股

### 3. 性能考虑

- 大批量股票生成因子时，会按照默认批处理大小（100）处理
- 可调整 `utils/constants.py` 中的 `DEFAULT_BATCH_SIZE`

---

## 🔄 后续开发

### 第一阶段（已完成）✅
- 生成层的 4 个函数 + 基类
- 工具层的验证、常量、辅助函数

### 第二阶段（已完成）✅
- 合并层（`merge_factor_files`, `merge_factor_directory`）
- 分析层（`FactorAnalyzer` 类，集成 Alphalens）
- 基本的单元测试

### 第三阶段（进行中）⏳
- 元信息保存（`save_task_metadata`）
- 完整的单元测试套件
- 性能优化和错误处理改进
- 更多分析指标和报告功能

---

## 📝 更新日志

### v0.2.0 (2025-12-03)
- ✅ 合并层：`merge_factor_files()` 和 `merge_factor_directory()` 函数
- ✅ 分析层：完整的 `FactorAnalyzer` 类，支持 Alphalens 集成
- ✅ 多周期因子打分和滚动监控
- ✅ 图表生成和保存功能
- ✅ 合并层测试用例

### v0.1.0 (2025-01-29)
- ✅ 生成层：4 个因子生成函数
- ✅ 工具层：参数验证、辅助函数、常量
- ✅ 基类：`FactorGenerator` 基类
- ✅ 文档和示例

---

**最后更新:** 2025-12-03  
**版本:** v0.2.0 - 合并层和分析层完成

切到 src/factor
python main_factor.py --factor-file /Users/fengzhi/Downloads/git/stock-backtesting-system/data/merge_tasks/output_file
