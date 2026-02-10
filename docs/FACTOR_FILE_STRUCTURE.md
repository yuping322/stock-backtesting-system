# 因子系统 - 代码文件组织方案

> **目标:** 模块化设计，每个函数一个文件，避免单个文件过大

---

## 📁 推荐目录结构

```
src/factor/
├── __init__.py                    # 包初始化，导出所有公共接口
├── 
├── # ========== 生成层 (Generator Layer) ==========
├── generator/
│   ├── __init__.py
│   ├── builtin.py                 # generate_builtin_factors()
│   ├── talib.py                   # generate_talib_factors()
│   ├── file.py                    # generate_file_factors()
│   ├── oss.py                     # generate_oss_factors()
│   └── _base.py                   # 公共基类和工具函数
│
├── # ========== 合并层 (Merger Layer) ==========
├── merger/
│   ├── __init__.py
│   ├── merge.py                   # merge_factor_files()
│   └── merge_directory.py          # merge_factor_directory()
│
├── # ========== 分析层 (Analysis Layer) ==========
├── analyzer/
│   ├── __init__.py
│   ├── factor_analyzer.py         # 因子分析核心
│   └── report.py                  # 生成分析报告
│
├── # ========== 工具层 (Utility Layer) ==========
├── utils/
│   ├── __init__.py
│   ├── metadata.py                # save_task_metadata() + JSON/MD 生成
│   ├── validation.py              # 参数验证
│   ├── constants.py               # 常量定义 (因子列表等)
│   └── helpers.py                 # 辅助函数 (时间戳、目录创建等)
│
└── tests/
    ├── __init__.py
    ├── test_generator_builtin.py
    ├── test_generator_talib.py
    ├── test_generator_file.py
    ├── test_generator_oss.py
    ├── test_merger.py
    └── test_analyzer.py
```

---

## 📋 文件详细说明

### 生成层 (generator/)

#### `generator/_base.py` - 基类和公共工具

```python
# 公共基类
class FactorGenerator:
    """因子生成器基类"""
    def __init__(self, stock_codes, start_date, end_date, output_dir):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.task_dir = None
        self.timestamp = None
    
    def validate_params(self):
        """参数验证"""
        pass
    
    def create_task_dir(self):
        """创建任务目录"""
        pass
    
    def save_factors(self, df):
        """保存因子到 CSV"""
        pass

# 公共工具函数
def load_ohlcv(stock_codes, start_date, end_date):
    """从数据源加载 OHLCV 数据"""
    pass

def generate_timestamp():
    """生成时间戳"""
    pass
```

#### `generator/builtin.py` - 内置因子

```python
# 包含 generate_builtin_factors() 函数
# - VOL10, RSI_14, MA_20, MACD_12_26_9 等

def generate_builtin_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """生成内置因子"""
    pass
```

#### `generator/talib.py` - TA-Lib 因子

```python
# 包含 generate_talib_factors() 函数
# - 支持 200+ TA-Lib 指标

def generate_talib_factors(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    factor_names: Optional[List[str]] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """生成 TA-Lib 因子"""
    pass
```

#### `generator/file.py` - 文件因子

```python
# 包含 generate_file_factors() 函数
# - 从 CSV 文件加载因子

def generate_file_factors(
    factor_file_paths: Dict[str, str],
    stock_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """从文件加载因子"""
    pass
```

#### `generator/oss.py` - OSS 因子

```python
# 包含 generate_oss_factors() 函数
# - Alpha158 / Alpha360 因子

def generate_oss_factors(
    factor_names: List[str],
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """生成 OSS 因子"""
    pass
```

---

### 合并层 (merger/)

#### `merger/merge.py` - 合并因子文件

```python
def merge_factor_files(
    factor_files: List[str],
    output_file: Optional[str] = None,
    how: str = 'outer'
) -> pd.DataFrame:
    """合并多个因子文件"""
    pass
```

#### `merger/merge_directory.py` - 合并目录

```python
def merge_factor_directory(
    factor_dir: str = './data/factor_tasks',
    pattern: str = 'factors_*.csv',
    output_file: Optional[str] = None,
    exclude_factors: Optional[List[str]] = None,
    how: str = 'outer'
) -> pd.DataFrame:
    """合并整个目录的因子文件"""
    pass
```

---

### 分析层 (analyzer/)

#### `analyzer/factor_analyzer.py` - 因子分析

```python
class FactorAnalyzer:
    """因子分析核心类"""
    
    def __init__(self, factor_df):
        self.df = factor_df
    
    def calculate_statistics(self):
        """计算统计指标"""
        pass
    
    def analyze_correlation(self):
        """分析因子相关性"""
        pass
    
    def analyze_stability(self):
        """分析因子稳定性"""
        pass
```

#### `analyzer/report.py` - 生成报告

```python
def export_analysis_report(analyzer, output_dir):
    """导出分析报告"""
    pass
```

---

### 工具层 (utils/)

#### `utils/metadata.py` - 元信息保存

```python
def save_task_metadata(
    factors: List[str],
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    task_dir: str,
    timestamp: str,
    notes: str = ""
) -> str:
    """保存任务元信息"""
    pass

def generate_metadata_json(factors_dict, stocks_dict, date_range, output_file):
    """生成 JSON 元信息"""
    pass

def generate_readme_md(task_info, output_file):
    """生成 README Markdown"""
    pass
```

#### `utils/validation.py` - 参数验证

```python
def validate_stock_codes(stock_codes):
    """验证股票代码是否有效"""
    pass

def validate_date_range(start_date, end_date):
    """验证日期范围"""
    pass

def validate_factor_names(factor_names, source_type):
    """验证因子名称是否存在"""
    pass
```

#### `utils/constants.py` - 常量定义

```python
# 内置因子列表
BUILTIN_FACTORS = {
    'VOL10': '10日成交量比值',
    'RSI_14': '14日相对强弱指标',
    'MA_20': '20日移动平均比值',
    'MACD_12_26_9': 'MACD指标'
}

# TA-Lib 因子列表 (示例)
TALIB_FACTORS = {
    'RSI': '相对强弱指标',
    'MACD': 'MACD指标',
    # ... 200+ 指标
}

# OSS 因子列表
OSS_FACTORS = {
    'ALPHA158_001': 'Alpha158 因子1',
    # ... 158个因子
}

# 文件输出配置
DEFAULT_OUTPUT_DIR = './data/factor_tasks'
DEFAULT_TASK_DIR_PATTERN = 'task_{timestamp}'
DEFAULT_FACTOR_FILE_PATTERN = 'factors_{timestamp}.csv'
```

#### `utils/helpers.py` - 辅助函数

```python
def generate_timestamp() -> str:
    """生成时间戳 YYYYMMDD_HHMMSS"""
    pass

def create_task_directory(base_dir, timestamp) -> str:
    """创建任务目录，返回路径"""
    pass

def get_stock_data_from_cache(stock_codes, start_date, end_date):
    """从缓存获取股票数据"""
    pass
```

---

## 🔌 顶层接口 (__init__.py)

```python
# src/factor/__init__.py

# ========== 生成函数 ==========
from .generator.builtin import generate_builtin_factors
from .generator.talib import generate_talib_factors
from .generator.file import generate_file_factors
from .generator.oss import generate_oss_factors

# ========== 元信息 ==========
from .utils.metadata import save_task_metadata

# ========== 合并函数 ==========
from .merger.merge import merge_factor_files
from .merger.merge_directory import merge_factor_directory

# ========== 分析函数 ==========
from .analyzer.factor_analyzer import FactorAnalyzer
from .analyzer.report import export_analysis_report

# ========== 导出所有公共接口 ==========
__all__ = [
    # 生成
    'generate_builtin_factors',
    'generate_talib_factors',
    'generate_file_factors',
    'generate_oss_factors',
    # 元信息
    'save_task_metadata',
    # 合并
    'merge_factor_files',
    'merge_factor_directory',
    # 分析
    'FactorAnalyzer',
    'export_analysis_report',
]
```

---

## 💡 使用示例

### 导入方式 1: 从顶层导入

```python
from src.factor import (
    generate_builtin_factors,
    generate_talib_factors,
    merge_factor_files,
    FactorAnalyzer
)

# 使用
result = generate_builtin_factors(...)
```

### 导入方式 2: 从子模块导入

```python
from src.factor.generator.builtin import generate_builtin_factors
from src.factor.merger.merge import merge_factor_files

# 使用
result = generate_builtin_factors(...)
```

---

## 📊 文件大小预估

| 模块 | 文件 | 估计行数 | 说明 |
|------|------|--------|------|
| generator | _base.py | 200-300 | 基类和公共工具 |
| generator | builtin.py | 300-400 | 内置因子生成 |
| generator | talib.py | 200-300 | TA-Lib 因子生成 |
| generator | file.py | 150-200 | 文件因子加载 |
| generator | oss.py | 150-200 | OSS 因子加载 |
| merger | merge.py | 150-200 | 因子合并 |
| merger | merge_directory.py | 100-150 | 目录合并 |
| analyzer | factor_analyzer.py | 300-400 | 分析核心 |
| analyzer | report.py | 150-200 | 报告生成 |
| utils | metadata.py | 200-250 | 元信息保存 |
| utils | validation.py | 150-200 | 参数验证 |
| utils | constants.py | 200-300 | 常量定义 |
| utils | helpers.py | 150-200 | 辅助函数 |
| **总计** | **13 个文件** | **2500-3500** | **模块化，易维护** |

---

## ✅ 优势总结

1. **模块化清晰** - 每个生成函数一个文件，职责单一
2. **易于维护** - 单个文件 200-400 行，易于阅读和修改
3. **易于测试** - 每个文件可独立测试
4. **易于扩展** - 添加新的因子来源只需新增文件
5. **易于协作** - 多人可并行开发不同模块
6. **清晰的层次** - Generator → Merger → Analyzer 三层分离

---

## 🚀 实现顺序建议

1. **第 1-2 天** - 创建文件结构 + utils 模块
2. **第 3-5 天** - 实现 generator 层 (4 个生成函数)
3. **第 6-7 天** - 实现 merger 层
4. **第 8-10 天** - 实现 analyzer 层
5. **第 11-12 天** - 编写测试 + 集成验证

---

**推荐采用此方案 ✅**

这个结构既避免了单文件过大，又保持了模块间的清晰关系。
