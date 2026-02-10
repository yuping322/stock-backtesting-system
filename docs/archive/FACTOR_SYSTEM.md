# 因子系统 - 完整文档

> **最终版本** - 函数式设计，简单高效

---

## 📖 目录

1. [核心理念](#核心理念)
2. [系统架构](#系统架构)
3. [文件输出规范](#文件输出规范)
4. [API 接口](#api-接口)
5. [使用流程](#使用流程)
6. [设计优势](#设计优势)
7. [常见问题](#常见问题)

---

## 核心理念

**不用类，就用函数。**

- 简单直接，每个因子来源一个函数
- 4 个独立的生成函数 + 元信息保存 + 合并工具
- 支持自动识别股票代码和指数代码
- 一个任务生成一个文件夹，一个 CSV 包含所有因子

### 关键特性

✨ **函数式设计**
- 不用类继承，就用函数
- 每个因子来源一个生成函数
- 易于理解、测试、扩展

✨ **明确的参数处理**
```python
# 方式 1: 直接指定股票代码
stock_codes=['000001', '000002', '000003']

# 方式 2: 如果是指数，先获取成分股再调用
from data import load_stock_pool
stock_codes = load_stock_pool('000001')['code'].tolist()
# 然后传入到生成函数
```

✨ **统一的任务输出**
```
data/factor_tasks/task_YYYYMMDD_HHMMSS/
├── factors_YYYYMMDD_HHMMSS.csv          # 所有因子（多列）
├── task_metadata_YYYYMMDD_HHMMSS.json   # 元信息
└── README_task_YYYYMMDD_HHMMSS.md       # 任务说明
```

✨ **支持 100+ 因子**
- 一个 CSV 文件可包含 100+ 因子列
- 无需 100 个单独的文件

---

## 系统架构

```
┌────────────────────────────────────────────────────────┐
│             Factor Generation Layer                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  generate_builtin_factors()    - 内置因子      │  │
│  │  generate_talib_factors()      - TA-Lib因子   │  │
│  │  generate_file_factors()       - CSV文件因子   │  │
│  │  generate_oss_factors()        - Alpha158/360  │  │
│  │  save_task_metadata()          - 保存任务元信息 │  │
│  └────────────────────────────────────────────────┘  │
│                      ↓                                 │
│      data/factor_tasks/task_YYYYMMDD_HHMMSS/         │
│      ├── factors_YYYYMMDD_HHMMSS.csv                 │
│      ├── task_metadata_YYYYMMDD_HHMMSS.json          │
│      └── README_task_YYYYMMDD_HHMMSS.md              │
└────────────────────────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│             Factor Merger Layer                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  merge_factor_files()          - 合并因子文件   │  │
│  │  merge_factor_directory()      - 合并目录       │  │
│  └────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│             Factor Analysis Layer                     │
│  ┌────────────────────────────────────────────────┐  │
│  │  analyze_factors()             - 因子分析     │  │
│  │  export_analysis_report()      - 导出报告     │  │
│  └────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 文件输出规范

### 目录结构

```
data/
├── factor_tasks/                              # 因子任务根目录
│   ├── task_20250129_153000/                  # 一个任务一个文件夹
│   │   ├── factors_20250129_153000.csv        # 所有因子在一个文件（多列）
│   │   ├── task_metadata_20250129_153000.json # 任务元信息
│   │   └── README_task_20250129_153000.md     # 任务说明
│   │
│   ├── task_20250130_100000/                  # 另一个任务
│   │   ├── factors_20250130_100000.csv
│   │   ├── task_metadata_20250130_100000.json
│   │   └── README_task_20250130_100000.md
│   │
│   └── ...

results/
├── analysis_task_20250129_153000.csv          # 分析结果
└── plot_task_20250129_153000.png              # 图表
```

### 文件命名规范

**因子文件（一个文件包含所有因子）:**
```
factors_{YYYYMMDD}_{HHMMSS}.csv

例如:
factors_20250129_153000.csv   # 包含 VOL10, RSI_14, MA_20, TALIB_RSI_14 等多个因子
factors_20250130_100000.csv   # 另一个任务的因子
```

### CSV 格式（多列因子）

```
date,stock_code,VOL10,RSI_14,MA_20,TALIB_RSI_14,TALIB_MACD_12_26_9,ALPHA158_001,...
2024-01-15,000001,1.23,45.67,100.2,45.50,0.12,0.001,...
2024-01-15,000002,1.45,46.10,100.5,46.20,0.13,0.002,...
2024-01-15,000003,0.98,44.50,99.8,44.80,0.11,0.001,...
2024-01-16,000001,1.25,46.20,100.5,46.30,0.14,0.001,...
```

### 任务元信息格式

**文件:** `task_metadata_{YYYYMMDD}_{HHMMSS}.json`

```json
{
    "task_id": "task_20250129_153000",
    "timestamp": "2025-01-29T15:30:00.000000",
    "factors": [
        {
            "name": "VOL10",
            "type": "builtin",
            "description": "10日成交量比值"
        },
        {
            "name": "RSI_14",
            "type": "builtin",
            "description": "14日相对强弱指标"
        },
        {
            "name": "TALIB_RSI_14",
            "type": "talib",
            "description": "TA-Lib RSI 指标"
        }
    ],
    "stocks": {
        "total": 1000,
        "type": "custom",
        "source": "000001, 000002, ..."
    },
    "date_range": ["2024-01-15", "2024-01-31"],
    "total_records": 15000,
    "file": "factors_20250129_153000.csv"
}
```

### 任务说明文档格式

**文件:** `README_task_{YYYYMMDD}_{HHMMSS}.md`

```markdown
# Factor Generation Task - 2025-01-29 15:30:00

## Task Overview
- **Task ID**: task_20250129_153000
- **Created**: 2025-01-29 15:30:00
- **Factors**: 4 (VOL10, RSI_14, MA_20, TALIB_RSI_14)
- **Stocks**: 1000 (from index 000001)
- **Total Records**: 15,000

## Factors Generated

| Factor | Type | Description |
|--------|------|-------------|
| VOL10 | builtin | 10日成交量比值 |
| RSI_14 | builtin | 14日相对强弱指标 |
| TALIB_RSI_14 | talib | TA-Lib RSI 指标 |
| ALPHA158_001 | oss | Alpha158 因子 |

## Stock Pool
- **Type**: Index
- **Code**: 000001 (Shanghai Composite)
- **Count**: 1000 stocks

## Output Files
- factors_20250129_153000.csv
- task_metadata_20250129_153000.json
- README_task_20250129_153000.md
```

---

## API 接口

### 1. generate_builtin_factors()

生成内置因子（技术指标）

**函数签名:**
```python
def generate_builtin_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `stock_codes` | List[str] | 股票代码列表（必须是股票，不是指数） | `['000001', '000002']` |
| `start_date` | str | 开始日期 | `'2024-01-01'` |
| `end_date` | str | 结束日期 | `'2024-01-31'` |
| `factor_names` | Optional[List[str]] | 因子列表，None 表示全部 | `['VOL10', 'RSI_14']` |
| `output_dir` | str | 输出目录 | `'./data/factor_tasks'` |

**返回值:**
```python
{
    'factor_file': 'data/factor_tasks/task_YYYYMMDD_HHMMSS/factors_YYYYMMDD_HHMMSS.csv',
    'metadata_file': 'data/factor_tasks/task_YYYYMMDD_HHMMSS/task_metadata_YYYYMMDD_HHMMSS.json',
    'readme_file': 'data/factor_tasks/task_YYYYMMDD_HHMMSS/README_task_YYYYMMDD_HHMMSS.md'
}
```

**支持的因子:**
- `VOL10`: 10日成交量比值
- `RSI_14`: 14日相对强弱指标
- `MA_20`: 20日移动平均比值
- `MACD_12_26_9`: MACD指标

**使用示例:**
```python
from src.factor.generator import generate_builtin_factors
from data import load_stock_pool

# 方式 1: 直接指定股票代码
result = generate_builtin_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14']
)

# 方式 2: 如果需要指数的成分股，先获取再传入
index_stocks = load_stock_pool('000001')['code'].tolist()  # 上证指数成分股
result = generate_builtin_factors(
    stock_codes=index_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14']
)
```

---

### 2. generate_talib_factors()

生成 TA-Lib 因子

**函数签名:**
```python
def generate_talib_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `stock_codes` | List[str] | 股票代码列表（必须是股票，不是指数） | `['000001', '000002']` |
| `start_date` | str | 开始日期 | `'2024-01-01'` |
| `end_date` | str | 结束日期 | `'2024-01-31'` |
| `factor_names` | Optional[List[str]] | TA-Lib 因子列表 | `['TALIB_RSI_14', 'TALIB_MACD_12_26_9']` |
| `output_dir` | str | 输出目录 | `'./data/factor_tasks'` |

**返回值:** 同 `generate_builtin_factors()`

**因子命名规范:**
```
TALIB_{FUNCTION_NAME}_{PARAM1}_{PARAM2}_...

例如:
- TALIB_RSI_14          # RSI 指标，周期 14
- TALIB_MACD_12_26_9    # MACD，快速周期 12，慢速周期 26，信号周期 9
- TALIB_STOCH_14_3_3    # Stochastic，周期 14，K 线 3，D 线 3
- TALIB_BBANDS_20_2_2   # Bollinger Bands，周期 20，标准差倍数 2
```

**支持 200+ TA-Lib 指标，常用包括:**
- RSI, MACD, STOCH, BBANDS, ATR, ADX, CCI, ROC, 等

**使用示例:**
```python
from src.factor.generator import generate_talib_factors
from data import load_stock_pool

# 方式 1: 直接指定股票代码
result = generate_talib_factors(
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9']
)

# 方式 2: 如果需要指数的成分股
index_stocks = load_stock_pool('000001')['code'].tolist()
result = generate_talib_factors(
    stock_codes=index_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['TALIB_RSI_14']
)
```

---

### 3. generate_file_factors()

从 CSV 文件加载因子

**函数签名:**
```python
def generate_file_factors(
    factor_file_paths: Dict[str, str],
    stock_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `factor_file_paths` | Dict[str, str] | 因子文件路径字典 | `{'custom_factor1': './factors/my_factor1.csv'}` |
| `stock_codes` | Optional[List[str]] | 股票代码过滤 | `['000001', '000002']` |
| `start_date` | Optional[str] | 日期开始过滤 | `'2024-01-01'` |
| `end_date` | Optional[str] | 日期结束过滤 | `'2024-01-31'` |
| `output_dir` | str | 输出目录 | `'./data/factor_tasks'` |

**返回值:** 同上

**输入文件格式要求:**
```
必须包含 date 和 stock_code 列，以及至少一列因子值
date,stock_code,factor_value
2024-01-15,000001,1.23
2024-01-15,000002,1.45
```

**使用示例:**
```python
from src.factor.generator import generate_file_factors

result = generate_file_factors(
    factor_file_paths={
        'custom_factor1': './factors/my_factor1.csv',
        'custom_factor2': './factors/my_factor2.csv'
    },
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

---

### 4. generate_oss_factors()

从 OSS（Alpha158/Alpha360）加载因子

**函数签名:**
```python
def generate_oss_factors(
    factor_names: List[str],
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `factor_names` | List[str] | Alpha158/360 因子列表 | `['ALPHA158_001', 'ALPHA158_002']` |
| `stock_codes` | List[str] | 股票代码列表（必须是股票，不是指数） | `['000001', '000002']` |
| `start_date` | str | 开始日期 | `'2024-01-01'` |
| `end_date` | str | 结束日期 | `'2024-01-31'` |
| `output_dir` | str | 输出目录 | `'./data/factor_tasks'` |

**返回值:** 同上

**使用示例:**
```python
from src.factor.generator import generate_oss_factors
from data import load_stock_pool

# 方式 1: 直接指定股票代码
result = generate_oss_factors(
    factor_names=['ALPHA158_001', 'ALPHA158_002'],
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# 方式 2: 如果需要指数的成分股
index_stocks = load_stock_pool('000001')['code'].tolist()
result = generate_oss_factors(
    factor_names=['ALPHA158_001'],
    stock_codes=index_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

---

### 5. save_task_metadata()

保存任务元信息和 README

**函数签名:**
```python
def save_task_metadata(
    factors: List[str],
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    task_dir: str,
    timestamp: str,
    notes: str = ""
) -> str
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `factors` | List[str] | 因子名称列表 | `['VOL10', 'RSI_14']` |
| `stock_codes` | List[str] | 股票代码列表 | `['000001', '000002']` |
| `start_date` | str | 开始日期 | `'2024-01-01'` |
| `end_date` | str | 结束日期 | `'2024-01-31'` |
| `task_dir` | str | 任务文件夹路径 | `'./data/factor_tasks/task_20250129_153000'` |
| `timestamp` | str | 时间戳 | `'20250129_153000'` |
| `notes` | str | 任务备注 | `'Test task'` |

**返回值:**
```
'data/factor_tasks/task_YYYYMMDD_HHMMSS/task_metadata_YYYYMMDD_HHMMSS.json'
```

同时生成:
- `task_metadata_*.json` - 元信息文件
- `README_task_*.md` - 任务说明文档

**使用示例:**
```python
from src.factor.generator import save_task_metadata

metadata_file = save_task_metadata(
    factors=['VOL10', 'RSI_14'],
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    task_dir='./data/factor_tasks/task_20250129_153000',
    timestamp='20250129_153000',
    notes="Test task for builtin factors"
)
```

---

### 6. merge_factor_files()

合并多个因子文件

**函数签名:**
```python
def merge_factor_files(
    factor_files: List[str],
    output_file: Optional[str] = None,
    how: str = 'outer'
) -> pd.DataFrame
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `factor_files` | List[str] | 因子文件路径列表 | `['factors_1.csv', 'factors_2.csv']` |
| `output_file` | Optional[str] | 输出文件路径，None 仅返回 DataFrame | `'merged_factors.csv'` |
| `how` | str | 合并方式：'outer' 或 'inner' | `'outer'` |

**返回值:** `pd.DataFrame` - 合并后的数据

**合并方式:**
- `'outer'` (默认): 保留所有日期和股票，缺失值为 NaN
- `'inner'`: 只保留共同的日期和股票

**使用示例:**
```python
from src.factor.merger import merge_factor_files

# 合并两个文件
merged = merge_factor_files(
    factor_files=['factors_1.csv', 'factors_2.csv'],
    output_file='merged_factors.csv'
)

# 仅返回 DataFrame
df = merge_factor_files(
    factor_files=['factors_1.csv', 'factors_2.csv']
)
```

---

### 7. merge_factor_directory()

合并整个目录中的所有因子文件

**函数签名:**
```python
def merge_factor_directory(
    factor_dir: str = './data/factor_tasks',
    pattern: str = 'factors_*.csv',
    output_file: Optional[str] = None,
    exclude_factors: Optional[List[str]] = None,
    how: str = 'outer'
) -> pd.DataFrame
```

**参数说明:**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `factor_dir` | str | 因子文件目录 | `'./data/factor_tasks'` |
| `pattern` | str | 文件匹配模式 | `'factors_*.csv'` |
| `output_file` | Optional[str] | 输出文件路径 | `'merged_all.csv'` |
| `exclude_factors` | Optional[List[str]] | 排除的因子名称 | `['TEMP_001']` |
| `how` | str | 合并方式 | `'outer'` |

**返回值:** `pd.DataFrame`

**使用示例:**
```python
from src.factor.merger import merge_factor_directory

# 合并任务目录中的所有因子
merged = merge_factor_directory(
    factor_dir='./data/factor_tasks/task_20250129_153000',
    output_file='./data/factor_tasks/task_20250129_153000/factors_all.csv'
)

# 合并所有任务，排除某些因子
merged = merge_factor_directory(
    factor_dir='./data/factor_tasks',
    pattern='task_*/factors_*.csv',
    exclude_factors=['TEMP_001'],
    how='inner'
)
```

---

## 使用流程

### 流程 1: 生成单一来源因子

```python
from src.factor.generator import generate_builtin_factors

# 生成内置因子
result = generate_builtin_factors(
    stock_codes=['000001'],  # 指数或股票，自动识别
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14']
)

# 返回值
# {
#     'factor_file': 'data/factor_tasks/task_20250129_153000/factors_20250129_153000.csv',
#     'metadata_file': 'data/factor_tasks/task_20250129_153000/task_metadata_20250129_153000.json',
#     'readme_file': 'data/factor_tasks/task_20250129_153000/README_task_20250129_153000.md'
# }
```

### 流程 2: 多来源因子合并

```python
from src.factor.generator import generate_builtin_factors, generate_talib_factors
from src.factor.merger import merge_factor_directory

# 1. 生成内置因子
result1 = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10', 'RSI_14']
)

# 2. 生成 TA-Lib 因子
result2 = generate_talib_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['TALIB_RSI_14', 'TALIB_MACD_12_26_9']
)

# 3. 合并所有因子
task_dir = result1['factor_file'].rsplit('/', 1)[0]
merged = merge_factor_directory(
    factor_dir=task_dir,
    output_file=f"{task_dir}/factors_all.csv"
)

# 生成的合并文件
# data/factor_tasks/task_20250129_153000/factors_all.csv
# 包含列: date, stock_code, VOL10, RSI_14, TALIB_RSI_14, TALIB_MACD_12_26_9
```

### 流程 3: 从文件加载因子

```python
from src.factor.generator import generate_file_factors

result = generate_file_factors(
    factor_file_paths={
        'custom_factor1': './factors/my_factor1.csv',
        'custom_factor2': './factors/my_factor2.csv'
    },
    stock_codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

---

## 设计优势

### ✨ 简单直接

- **不用类，就用函数** - 无复杂的类继承和多态
- **4 个独立的生成函数** - 每个因子来源一个函数
- **单一职责** - 每个函数做好一件事

### ✨ 灵活高效

- **自动识别股票/指数** - `stock_codes` 参数智能处理
- **支持 100+ 因子** - 一个 CSV 文件，无需多个文件
- **完整的工具链** - 生成、合并、分析一体化

### ✨ 清晰的组织

- **一个任务一个文件夹** - 方便追踪和管理
- **自动生成元信息** - JSON 和 README 一键生成
- **版本管理清晰** - 时间戳记录每个任务

### ✨ 易于扩展

- **新增因子源** - 添加一个新的生成函数即可
- **新增合并方式** - 修改 merge 函数的逻辑
- **新增分析功能** - 独立的分析模块

---

## 常见问题

### Q1: 怎么处理指数代码？

**A:** 使用 `load_stock_pool()` 先获取成分股，再传入生成函数：

```python
from data import load_stock_pool
from src.factor.generator import generate_builtin_factors

# 获取指数的所有成分股
index_stocks = load_stock_pool('000001')['code'].tolist()

# 传入生成函数
result = generate_builtin_factors(
    stock_codes=index_stocks,
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

### Q2: 为什么用一个 CSV 而不是多个文件？

**A:** 一个 CSV 文件的优势：
- **可扩展性** - 100 个因子不需要 100 个文件
- **易管理** - 一个任务一个文件，清晰明确
- **易分析** - 所有因子数据在一起，便于联合分析
- **易合并** - 多个因子源直接合并，无需复杂操作

### Q3: 可以组合多来源因子吗？

**A:** 完全支持。生成各自的因子后用 `merge_factor_directory()` 合并：

```python
# 生成内置因子
result1 = generate_builtin_factors(...)

# 生成 TA-Lib 因子
result2 = generate_talib_factors(...)

# 合并
task_dir = result1['factor_file'].rsplit('/', 1)[0]
merged = merge_factor_directory(factor_dir=task_dir)
```

### Q4: 如何处理缺失值？

**A:** 使用 `merge_factor_files()` 的 `how` 参数：

```python
# 保留所有数据，缺失值为 NaN（推荐用于大多数场景）
merged = merge_factor_files(files, how='outer')

# 只保留共同的日期和股票（严格要求）
merged = merge_factor_files(files, how='inner')
```

### Q5: 如何自定义输出目录？

**A:** 所有生成函数都支持 `output_dir` 参数：

```python
result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    output_dir='./my_factors'  # 自定义输出目录
)
```

### Q6: 任务元信息和 README 有什么用？

**A:** 这两个文件用于：
- **task_metadata.json** - 结构化记录任务信息，便于程序读取
- **README.md** - 人类可读的任务说明，便于回顾和文档化

---

## 快速参考

### 生成因子

```python
from src.factor.generator import (
    generate_builtin_factors,
    generate_talib_factors,
    generate_file_factors,
    generate_oss_factors
)

# 内置因子
result = generate_builtin_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# TA-Lib 因子
result = generate_talib_factors(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# 文件因子
result = generate_file_factors(
    factor_file_paths={'custom': './factors/my_factor.csv'},
    stock_codes=['000001']
)

# OSS 因子
result = generate_oss_factors(
    factor_names=['ALPHA158_001'],
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

### 合并因子

```python
from src.factor.merger import merge_factor_directory

# 合并任务目录
merged = merge_factor_directory(
    factor_dir='./data/factor_tasks/task_20250129_153000',
    output_file='./data/factor_tasks/task_20250129_153000/factors_all.csv'
)
```

### 返回值

所有生成函数返回：
```python
{
    'factor_file': 'path/to/factors_*.csv',
    'metadata_file': 'path/to/task_metadata_*.json',
    'readme_file': 'path/to/README_task_*.md'
}
```

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/factor/generator.py` | 因子生成核心模块 |
| `src/factor/merger.py` | 因子合并工具 |
| `src/factor/analyzer.py` | 因子分析模块 |
| `src/data.py` | 数据加载接口 |

---

## 后续开发

### 即将实现

- [ ] `generator.py` - 4 个生成函数 + 元信息保存
- [ ] `merger.py` - 2 个合并函数
- [ ] `analyzer.py` - 因子分析模块
- [ ] 单元测试和集成测试

### 预计工期

12 个工作日

---

**最后更新:** 2025-01-29  
**版本:** 1.0 Final

