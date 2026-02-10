# Qlib因子整合方案（依赖qlib实现）

## 一、整体流程设计

```
数据源(data.py) 
  ↓
构建qlib数据集（bin格式）
  ↓
初始化qlib和handler（Alpha158/Alpha360）
  ↓
从handler提取因子 → DataFrame格式
  ↓
用于factor模块的Alphalens检验
```

## 二、核心组件设计

### 2.1 Qlib数据集构建器

**功能**：将data.py的数据转换为qlib数据集格式

**位置**：`factor/qlib_data_builder.py` (新增)

**职责**：
- 从data.py获取OHLCV数据
- 转换为qlib要求的目录结构（calendars/instruments/features）
- 生成bin文件

**与alpha_test的区别**：
- alpha_test：一次性构建，用于模型训练
- factor模块：可复用，用于因子提取

### 2.2 Qlib因子提取器

**功能**：从qlib数据集中提取Alpha158/Alpha360因子

**位置**：`factor/qlib_factor_extractor.py` (新增)

**职责**：
- 初始化qlib环境
- 创建Alpha158/Alpha360 handler
- 从handler中提取特征
- 转换为factor模块可用的DataFrame格式

### 2.3 Qlib因子计算器

**功能**：封装为FactorCalculator接口

**位置**：`factor/qlib_factor_calculator.py` (新增)

**职责**：
- 实现FactorCalculator接口
- 内部使用qlib_data_builder和qlib_factor_extractor
- 对用户隐藏qlib细节

---

## 三、详细设计

### 3.1 数据流转

```python
# 步骤1：数据获取
price_data = data.load_oss_complex_stocks(codes, start, end, fields=['open','high','low','close','volume'])

# 步骤2：构建qlib数据集
qlib_data_dir = qlib_data_builder.build_from_dataframe(price_data, output_dir)

# 步骤3：初始化qlib
qlib.init(provider_uri=qlib_data_dir, region='cn')

# 步骤4：创建handler并提取因子
handler = Alpha158(...)
dataset = DatasetH(handler=handler, segments={...})
factors_df = qlib_factor_extractor.extract_factors(dataset, segment='train')

# 步骤5：用于factor模块
# factors_df格式：(date, code) MultiIndex, columns为因子名称
```

### 3.2 接口设计

#### A. Qlib数据集构建器

```python
# factor/qlib_data_builder.py

class QlibDataBuilder:
    """将DataFrame数据构建为qlib数据集格式"""
    
    @staticmethod
    def build_from_dataframe(
        df: pd.DataFrame,
        output_dir: Path,
        date_col: str = 'date',
        symbol_col: str = 'symbol',
        rebuild: bool = False
    ) -> Path:
        """
        从DataFrame构建qlib数据集
        
        Args:
            df: DataFrame with columns [date, symbol, open, high, low, close, volume, ...]
            output_dir: 输出目录
            date_col: 日期列名
            symbol_col: 股票代码列名
            rebuild: 是否重建（如果已存在）
        
        Returns:
            qlib数据集目录路径
        """
        # 实现逻辑（参考alpha_test/build_data）
        pass
    
    @staticmethod
    def build_from_data_module(
        codes: List[str],
        start_date: str,
        end_date: str,
        output_dir: Path,
        rebuild: bool = False
    ) -> Path:
        """
        直接从data模块获取数据并构建qlib数据集
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            output_dir: 输出目录
            rebuild: 是否重建
        
        Returns:
            qlib数据集目录路径
        """
        # 1. 从data.py获取数据
        data_dict = data.load_oss_complex_stocks(...)
        
        # 2. 转换为DataFrame
        df = convert_to_dataframe(data_dict)
        
        # 3. 构建qlib数据集
        return QlibDataBuilder.build_from_dataframe(df, output_dir, rebuild=rebuild)
```

#### B. Qlib因子提取器

```python
# factor/qlib_factor_extractor.py

class QlibFactorExtractor:
    """从qlib数据集中提取因子"""
    
    def __init__(self, qlib_data_dir: Path, region: str = 'cn'):
        """
        Args:
            qlib_data_dir: qlib数据集目录路径
            region: 地区（'cn'等）
        """
        self.qlib_data_dir = qlib_data_dir
        self.region = region
        self._initialized = False
    
    def initialize(self):
        """初始化qlib环境"""
        if not self._initialized:
            qlib.init(provider_uri=str(self.qlib_data_dir), region=self.region)
            self._initialized = True
    
    def extract_alpha158_factors(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        segment: str = 'train',
        feature_subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        提取Alpha158因子
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            segment: 数据段（'train', 'valid', 'test'）
            feature_subset: 因子子集，None表示全部
        
        Returns:
            DataFrame: MultiIndex (date, code), columns为因子名称
        """
        self.initialize()
        
        # 创建handler
        handler_conf = {
            'class': 'Alpha158',
            'module_path': 'qlib.contrib.data.handler',
            'kwargs': {
                'start_time': start_date,
                'end_time': end_date,
                'fit_start_time': start_date,
                'fit_end_time': end_date,
                'instruments': codes,
            },
        }
        
        # 创建dataset
        dataset_conf = {
            'class': 'DatasetH',
            'module_path': 'qlib.data.dataset',
            'kwargs': {
                'handler': handler_conf,
                'segments': {
                    segment: [start_date, end_date],
                },
            },
        }
        
        dataset = init_instance_by_config(dataset_conf)
        
        # 提取特征
        df = dataset.prepare(segment, col_set="feature")
        
        # 转换为factor模块格式
        # qlib格式: MultiIndex (datetime, instrument), columns为 ('feature', 'factor_name')
        # 需要转换为: MultiIndex (date, code), columns为 factor_name
        
        factors_df = self._convert_to_factor_format(df)
        
        if feature_subset:
            factors_df = factors_df[[col for col in feature_subset if col in factors_df.columns]]
        
        return factors_df
    
    def extract_alpha360_factors(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        segment: str = 'train',
        feature_subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """提取Alpha360因子（类似实现）"""
        pass
    
    def _convert_to_factor_format(self, qlib_df: pd.DataFrame) -> pd.DataFrame:
        """
        将qlib的特征DataFrame转换为factor模块格式
        
        qlib格式：
          - Index: MultiIndex (datetime, instrument)
          - Columns: MultiIndex (('feature', 'KMID'), ('feature', 'KLEN'), ...)
        
        factor格式：
          - Index: MultiIndex (date, code)
          - Columns: ['KMID', 'KLEN', ...]
        """
        # 提取因子名称
        if isinstance(qlib_df.columns, pd.MultiIndex):
            factor_names = [col[1] for col in qlib_df.columns if col[0] == 'feature']
            # 只保留feature列
            qlib_df = qlib_df.loc[:, ('feature', slice(None))]
            qlib_df.columns = factor_names
        
        # 确保index是MultiIndex (datetime, instrument)
        if not isinstance(qlib_df.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex (datetime, instrument)")
        
        # 重命名index levels
        qlib_df.index.names = ['date', 'code']
        
        return qlib_df
```

#### C. Qlib因子计算器（封装）

```python
# factor/qlib_factor_calculator.py

from .factor_calculator import FactorCalculator
from .qlib_data_builder import QlibDataBuilder
from .qlib_factor_extractor import QlibFactorExtractor
from pathlib import Path
from typing import List, Optional

class Alpha158FactorCalculator(FactorCalculator):
    """
    Alpha158因子计算器（依赖qlib实现）
    
    使用流程：
    1. 从data.py获取数据
    2. 构建qlib数据集
    3. 从qlib handler提取Alpha158因子
    4. 输出为factor模块格式
    """
    
    def __init__(
        self,
        qlib_data_dir: Optional[Path] = None,
        feature_subset: Optional[List[str]] = None,
        rebuild_dataset: bool = False
    ):
        """
        Args:
            qlib_data_dir: qlib数据集目录（如果已存在）
            feature_subset: 要计算的因子子集，None表示全部158个
            rebuild_dataset: 是否重建数据集（如果已存在）
        """
        self.qlib_data_dir = qlib_data_dir
        self.feature_subset = feature_subset
        self.rebuild_dataset = rebuild_dataset
        self.extractor = None
    
    def calculate(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        计算单个股票的因子（返回第一个因子）
        
        注意：Alpha158是因子集，单个股票有158个因子值
        这个方法返回第一个因子，如需全部因子请使用calculate_all
        """
        factors_df = self.calculate_all([stock_code], start_date, end_date)
        if factors_df.empty:
            return pd.Series(dtype=float)
        # 返回第一个因子
        first_factor = factors_df.columns[0]
        return factors_df[first_factor].xs(stock_code, level='code')
    
    def calculate_all(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        segment: str = 'train'
    ) -> pd.DataFrame:
        """
        批量计算所有股票的Alpha158因子
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            segment: 数据段
        
        Returns:
            DataFrame: MultiIndex (date, code), columns为因子名称
        """
        # 1. 确保qlib数据集存在
        if self.qlib_data_dir is None or not self.qlib_data_dir.exists():
            # 需要构建数据集
            self.qlib_data_dir = self._build_dataset(codes, start_date, end_date)
        
        # 2. 提取因子
        self.extractor = QlibFactorExtractor(self.qlib_data_dir)
        factors_df = self.extractor.extract_alpha158_factors(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            segment=segment,
            feature_subset=self.feature_subset
        )
        
        return factors_df
    
    def _build_dataset(
        self,
        codes: List[str],
        start_date: str,
        end_date: str
    ) -> Path:
        """构建qlib数据集"""
        # 使用临时目录或配置的目录
        output_dir = Path(f"./qlib_data_cache/{start_date}_{end_date}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return QlibDataBuilder.build_from_data_module(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            rebuild=self.rebuild_dataset
        )


class Alpha360FactorCalculator(FactorCalculator):
    """Alpha360因子计算器（类似实现）"""
    pass


class SingleAlpha158FactorCalculator(FactorCalculator):
    """
    单个Alpha158因子计算器
    用于提取Alpha158特征集中的某个特定因子（如ROC5）
    """
    
    def __init__(self, factor_name: str, qlib_data_dir: Optional[Path] = None):
        """
        Args:
            factor_name: 因子名称，如 'ROC5', 'MA10' 等
            qlib_data_dir: qlib数据集目录
        """
        self.factor_name = factor_name
        self.base_calculator = Alpha158FactorCalculator(
            qlib_data_dir=qlib_data_dir,
            feature_subset=[factor_name]
        )
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算单个因子"""
        factors_df = self.base_calculator.calculate_all([stock_code], start_date, end_date)
        if factors_df.empty or self.factor_name not in factors_df.columns:
            return pd.Series(dtype=float)
        return factors_df[self.factor_name].xs(stock_code, level='code')
```

---

## 四、集成到factor模块

### 4.1 更新因子注册表

```python
# factor/factor_calculator.py (修改)

from .qlib_factor_calculator import (
    Alpha158FactorCalculator,
    Alpha360FactorCalculator,
    SingleAlpha158FactorCalculator
)

def create_factor_calculator(
    factor_name: Optional[str] = None,
    factor_func: Optional[Callable] = None,
    file_path: Optional[str] = None,
    qlib_data_dir: Optional[Path] = None,
    **kwargs
):
    """
    创建因子计算器
    
    新增参数：
        qlib_data_dir: qlib数据集目录路径（用于Alpha158/Alpha360）
    
    新增支持：
        - 'ALPHA158': Alpha158因子集
        - 'ALPHA360': Alpha360因子集
        - 'ROC5', 'MA10'等: 单个Alpha158因子
    """
    # 检查是否为Alpha158特征集中的单个因子
    alpha158_factors = get_alpha158_feature_names()
    if factor_name in alpha158_factors:
        return SingleAlpha158FactorCalculator(factor_name, qlib_data_dir=qlib_data_dir)
    
    # 检查是否为Alpha158/Alpha360因子集
    if factor_name == 'ALPHA158':
        return Alpha158FactorCalculator(
            qlib_data_dir=qlib_data_dir,
            feature_subset=kwargs.get('feature_subset')
        )
    
    if factor_name == 'ALPHA360':
        return Alpha360FactorCalculator(
            qlib_data_dir=qlib_data_dir,
            feature_subset=kwargs.get('feature_subset')
        )
    
    # 原有逻辑...
    pass
```

### 4.2 辅助函数

```python
# factor/qlib_utils.py (新增)

from qlib.contrib.data.loader import Alpha158DL

def get_alpha158_feature_names(full: bool = False) -> List[str]:
    """
    获取Alpha158特征名称列表
    
    Args:
        full: 是否使用完整配置（158个特征）
    
    Returns:
        因子名称列表
    """
    if full:
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {"windows": [0, 1, 2, 3, 4]},
            "rolling": {
                "windows": [5, 10, 20, 30, 60],
                "include": None,
                "exclude": [],
            },
        }
    else:
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {},
        }
    
    fields, names = Alpha158DL.get_feature_config(conf)
    return names
```

---

## 五、使用示例

### 5.1 使用Alpha158因子集

```python
from factor.factor_calculator import create_factor_calculator

# 方式1：指定qlib数据集目录（如果已存在）
calc = create_factor_calculator(
    factor_name='ALPHA158',
    qlib_data_dir=Path('./qlib_data_cache')
)

# 方式2：不指定目录，自动构建
calc = create_factor_calculator('ALPHA158')

# 计算所有Alpha158因子
factors_df = calc.calculate_all(
    codes=['000001', '000002'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
# 返回: (date, code) MultiIndex, 158列因子

# 用于Alphalens检验
from factor.factor import FactorTester
# 可以选择单个因子检验
roc5_df = factors_df['ROC5']
# 或者检验整个因子集
```

### 5.2 使用单个Alpha158因子

```python
# 只计算ROC5因子
roc5_calc = create_factor_calculator('ROC5')
roc5_series = roc5_calc.calculate('000001', '2024-01-01', '2024-12-31')
# 返回: Series with date index

# 可以直接用于Alphalens检验
from factor.factor import FactorTester
cfg = ...  # 配置
tester = FactorTester(cfg)
# 使用roc5_calc作为因子源
```

### 5.3 在factor.py中使用

```python
# factor/factor.py (修改)

class FactorTester:
    def get_factors(self):
        """获取因子数据"""
        factors = {}
        
        for factor_name in self.cfg.FACTORS:
            # 支持Alpha158/Alpha360
            if factor_name in ['ALPHA158', 'ALPHA360']:
                calc = create_factor_calculator(factor_name, qlib_data_dir=...)
                factors_df = calc.calculate_all(...)
                # 可以选择单个因子或整个因子集进行检验
                factors[factor_name] = factors_df
            else:
                # 原有逻辑
                calc = create_factor_calculator(factor_name)
                # ...
```

---

## 六、文件结构

```
factor/
├── __init__.py
├── factor.py                          # 主程序（修改）
├── factor_calculator.py               # 因子计算器接口（修改，添加qlib支持）
├── qlib_data_builder.py               # 【新增】Qlib数据集构建器
├── qlib_factor_extractor.py           # 【新增】Qlib因子提取器
├── qlib_factor_calculator.py          # 【新增】Qlib因子计算器封装
├── qlib_utils.py                      # 【新增】Qlib工具函数
└── README.md                          # 更新文档
```

---

## 七、实现步骤

### Step 1: 提取alpha_test中的数据构建逻辑

从 `alpha_test/run_alpha_minimal.py` 的 `build_data()` 函数提取：
- 数据集构建逻辑
- bin文件生成逻辑
- 目录结构创建

### Step 2: 实现QlibDataBuilder

创建 `factor/qlib_data_builder.py`：
- 实现 `build_from_dataframe()`
- 实现 `build_from_data_module()`
- 复用alpha_test中的构建逻辑

### Step 3: 实现QlibFactorExtractor

创建 `factor/qlib_factor_extractor.py`：
- 实现 `extract_alpha158_factors()`
- 实现数据格式转换
- 支持因子子集提取

### Step 4: 实现QlibFactorCalculator

创建 `factor/qlib_factor_calculator.py`：
- 实现 `Alpha158FactorCalculator`
- 实现 `SingleAlpha158FactorCalculator`
- 集成QlibDataBuilder和QlibFactorExtractor

### Step 5: 集成到factor模块

修改 `factor/factor_calculator.py`：
- 更新 `create_factor_calculator()` 支持Alpha158
- 添加因子注册

### Step 6: 测试验证

- 测试数据构建
- 测试因子提取
- 测试与factor模块集成
- 验证结果正确性

---

## 八、关键设计决策

### 决策1: qlib数据集缓存策略

**方案**：按日期范围缓存数据集
- 目录命名：`qlib_data_cache/{start_date}_{end_date}`
- 避免重复构建
- 支持手动指定已有数据集路径

### 决策2: 因子提取粒度

**支持两种粒度**：
1. **因子集级别**：一次提取所有158/360个因子
2. **单个因子级别**：只提取某个特定因子（如ROC5）

### 决策3: 与factor模块的集成方式

**方案**：通过FactorCalculator接口
- Alpha158作为特殊的因子计算器
- 保持与现有因子计算器的接口一致
- 支持在factor.py中无缝使用

---

## 九、注意事项

1. **数据依赖**：需要先构建qlib数据集，可能需要一些时间
2. **内存占用**：Alpha158有158个特征，批量计算时注意内存
3. **日期范围**：某些特征需要足够的历史数据（如60日滚动）
4. **qlib版本**：确保qlib版本兼容Alpha158/Alpha360 handler

---

## 十、与alpha_test的关系

整合后的关系：
- **alpha_test模块**：保留模型训练功能（后续可迁移）
- **factor模块**：获得Alpha158/Alpha360因子计算能力
- **数据构建逻辑**：提取到factor模块，可复用

alpha_test模块后续可以：
- 移除数据构建逻辑（使用factor模块的）
- 移除因子提取逻辑（使用factor模块的）
- 专注于模型训练部分

