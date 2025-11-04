# Alpha_Test模块因子功能提取与整合方案

## 一、Alpha_Test模块中的因子相关功能总结

### 1.1 核心因子功能

#### A. Alpha158特征集
- **位置**：`alpha_test/quick_start_alpha_workflows.py` 的 `get_alpha158_features()`
- **功能**：获取Alpha158特征集的字段和名称
- **特点**：
  - 支持简化版和完整版两种配置
  - 简化版：基础K线、价格、滚动特征
  - 完整版：包含多窗口、多滚动算子的158个特征
- **依赖**：`qlib.contrib.data.loader.Alpha158DL`

**配置示例**：
```python
# 简化版配置
conf = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}

# 完整版配置
conf = {
    "kbar": {},
    "price": {
        "windows": [0, 1, 2, 3, 4],
        "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
    },
    "volume": {"windows": [0, 1, 2, 3, 4]},
    "rolling": {
        "windows": [5, 10, 20, 30, 60],
        "include": None,  # 使用默认算子
        "exclude": [],
    },
}
```

#### B. Alpha360特征集
- **位置**：`alpha_test/quick_start_alpha_workflows.py` 的 `build_task()` 中
- **功能**：使用qlib的Alpha360 handler
- **特点**：360个特征的更大特征集

#### C. Qlib Handler特征提取
- **位置**：`alpha_test/run_alpha_minimal.py` 的 `build_task()`
- **功能**：通过Alpha158 handler从qlib数据集提取特征
- **特点**：
  - 需要先构建qlib数据集
  - 通过handler自动计算所有特征
  - 特征数量：158个（Alpha158）或360个（Alpha360）

### 1.2 特征计算流程

```
数据源(data.py) 
  ↓
构建qlib数据集（calendars/instruments/features）
  ↓
初始化Alpha158/Alpha360 Handler
  ↓
Handler自动计算特征（基于OHLCV数据）
  ↓
输出特征DataFrame（MultiIndex: date, instrument）
```

### 1.3 特征命名规则

Alpha158特征命名示例：
- `KMID`: K线中间价
- `KLEN`: K线长度
- `OPEN0`: 当前开盘价
- `HIGH0`: 当前最高价
- `LOW0`: 当前最低价
- `VWAP0`: 当前VWAP
- `ROC5`, `ROC10`, `ROC20`: 收益率（不同窗口）
- `MA5`, `MA10`, `MA20`: 移动平均
- `STD5`, `STD10`: 标准差
- `RSV5`, `RSV10`: 相对强弱值
- ... 共158个特征

### 1.4 与factor模块的差异

| 特性 | factor模块 | alpha_test模块 |
|------|-----------|---------------|
| **因子定义方式** | 手动定义因子计算函数 | 使用qlib预定义handler |
| **因子数量** | 单个或少量因子 | 大量特征集（158/360） |
| **计算方式** | 逐个股票、逐个日期计算 | 批量计算，利用qlib优化 |
| **依赖框架** | 无（纯Pandas/NumPy） | 依赖qlib框架 |
| **输出格式** | Pandas DataFrame | Qlib Dataset → DataFrame |
| **特征类型** | 自定义技术指标 | 标准化Alpha因子集 |

---

## 二、可抽象的功能点

### 2.1 可直接提取的功能

#### A. Alpha158/Alpha360特征名称获取
```python
def get_alpha158_feature_names(full=False) -> List[str]:
    """获取Alpha158特征名称列表"""
    from qlib.contrib.data.loader import Alpha158DL
    conf = {...}  # 配置
    fields, names = Alpha158DL.get_feature_config(conf)
    return names

def get_alpha360_feature_names() -> List[str]:
    """获取Alpha360特征名称列表"""
    # 类似实现
```

#### B. 从qlib数据集提取特征为DataFrame
```python
def extract_features_from_qlib_dataset(
    dataset, 
    segment='train',
    feature_names=None
) -> pd.DataFrame:
    """从qlib数据集提取特征，转换为factor模块可用的DataFrame格式"""
    df = dataset.prepare(segment, col_set="feature")
    # 转换为 (date, code, factor_name, value) 格式
```

#### C. 基于data.py数据源直接计算Alpha158特征
```python
def compute_alpha158_factors_from_data(
    codes: List[str],
    start_date: str,
    end_date: str,
    feature_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    不依赖qlib数据集，直接从data.py数据源计算Alpha158特征
    输出格式：(date, code) MultiIndex, columns为因子名称
    """
```

### 2.2 需要适配的功能

#### A. Qlib数据集构建逻辑
- **现状**：alpha_test中需要先构建qlib格式数据集（bin文件）
- **问题**：依赖qlib数据格式，不够灵活
- **方案**：提取核心特征计算逻辑，不依赖bin文件格式

#### B. Handler的特征计算逻辑
- **现状**：特征计算封装在qlib的handler中
- **问题**：无法单独提取某个特征
- **方案**：实现轻量级的Alpha158特征计算器，基于Pandas实现

---

## 三、整合设计方案

### 3.1 在factor模块中新增Alpha158因子计算器

#### 设计思路

```
factor/
  ├── factor_calculator.py (现有)
  ├── qlib_factor_calculator.py (新增)
  │   ├── Alpha158FactorCalculator
  │   ├── Alpha360FactorCalculator
  │   └── QlibHandlerFactorCalculator
  └── builtin_qlib_factors.py (新增)
      ├── ALPHA158_FEATURES
      ├── ALPHA360_FEATURES
      └── 特征计算函数实现
```

#### A. Qlib因子计算器类

```python
# factor/qlib_factor_calculator.py

from typing import List, Optional
import pandas as pd
import numpy as np
from .factor_calculator import FactorCalculator

class Alpha158FactorCalculator(FactorCalculator):
    """
    Alpha158因子计算器
    支持两种模式：
    1. 通过qlib handler计算（需要qlib数据集）
    2. 直接基于OHLCV数据计算（不依赖qlib）
    """
    
    def __init__(self, use_qlib_handler=False, feature_subset=None):
        """
        Args:
            use_qlib_handler: 是否使用qlib handler（需要qlib数据集）
            feature_subset: 要计算的因子子集，None表示全部
        """
        self.use_qlib_handler = use_qlib_handler
        self.feature_subset = feature_subset
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算单个股票的Alpha158因子（单个因子）"""
        # 如果支持单个因子提取，可以实现
        pass
    
    def calculate_all(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量计算所有股票的Alpha158因子
        返回：(date, code) MultiIndex DataFrame，columns为因子名称
        """
        if self.use_qlib_handler:
            return self._calculate_via_qlib(codes, start_date, end_date)
        else:
            return self._calculate_directly(codes, start_date, end_date)
    
    def _calculate_via_qlib(self, codes, start_date, end_date):
        """通过qlib handler计算（需要现有qlib数据集）"""
        # 使用alpha_test中的逻辑
        pass
    
    def _calculate_directly(self, codes, start_date, end_date):
        """直接基于OHLCV数据计算（不依赖qlib数据集）"""
        # 实现轻量级Alpha158特征计算
        pass
```

#### B. 轻量级Alpha158特征计算实现

```python
# factor/builtin_qlib_factors.py

"""
Alpha158/Alpha360特征集的计算实现
基于Pandas实现，不依赖qlib框架
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class Alpha158FeatureCalculator:
    """Alpha158特征的轻量级计算实现"""
    
    # 特征定义配置
    FEATURE_CONFIG = {
        "price_windows": [0, 1, 2, 3, 4],
        "price_features": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
        "volume_windows": [0, 1, 2, 3, 4],
        "rolling_windows": [5, 10, 20, 30, 60],
    }
    
    @staticmethod
    def get_feature_names(full=True) -> List[str]:
        """获取Alpha158特征名称列表"""
        names = []
        
        # K线特征
        names.extend(["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"])
        
        # 价格特征（多窗口）
        for window in ([0] if not full else [0, 1, 2, 3, 4]):
            for feat in ["OPEN", "HIGH", "LOW", "VWAP"]:
                if window == 0:
                    names.append(feat + "0")
                else:
                    names.append(f"{feat}{window}")
        
        # 滚动特征（ROC, MA, STD等）
        if full:
            for window in [5, 10, 20, 30, 60]:
                names.extend([
                    f"ROC{window}", f"MA{window}", f"STD{window}",
                    f"BETA{window}", f"RSQR{window}", f"RESI{window}",
                    f"MAX{window}", f"MIN{window}", f"QTLU{window}", f"QTLD{window}",
                    f"RANK{window}", f"RSV{window}", f"CORR{window}",
                    # ... 更多特征
                ])
        
        return names
    
    @classmethod
    def calculate_features(cls, ohlcv: pd.DataFrame, feature_subset=None) -> pd.DataFrame:
        """
        基于OHLCV数据计算Alpha158特征
        
        Args:
            ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'vwap']
                   index为日期
            feature_subset: 要计算的因子子集
        
        Returns:
            DataFrame with columns为因子名称
        """
        result = {}
        
        # 基础K线特征
        result['KMID'] = (ohlcv['high'] + ohlcv['low']) / 2
        result['KLEN'] = ohlcv['high'] - ohlcv['low']
        result['KMID2'] = result['KMID'] ** 2
        result['KUP'] = ohlcv['high'] - result['KMID']
        result['KLOW'] = result['KMID'] - ohlcv['low']
        # ... 更多K线特征
        
        # 价格特征（当前）
        result['OPEN0'] = ohlcv['open']
        result['HIGH0'] = ohlcv['high']
        result['LOW0'] = ohlcv['low']
        result['VWAP0'] = ohlcv['vwap']
        
        # 滚动特征
        for window in [5, 10, 20, 30, 60]:
            close = ohlcv['close']
            # ROC (Rate of Change)
            result[f'ROC{window}'] = close.pct_change(window)
            # MA (Moving Average)
            result[f'MA{window}'] = close.rolling(window).mean()
            # STD (Standard Deviation)
            result[f'STD{window}'] = close.rolling(window).std()
            # ... 更多滚动特征
        
        df = pd.DataFrame(result)
        
        if feature_subset:
            df = df[[col for col in feature_subset if col in df.columns]]
        
        return df
```

#### C. 适配器：Qlib Handler → Factor Calculator

```python
# factor/qlib_adapter.py

"""
Qlib Handler到Factor Calculator的适配器
允许factor模块使用qlib的Alpha158/Alpha360 handler
"""

class QlibHandlerAdapter:
    """适配qlib handler为factor计算器"""
    
    @staticmethod
    def extract_factors_from_qlib_handler(
        handler,
        codes: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        从qlib handler提取因子，转换为factor模块格式
        
        Returns:
            DataFrame: MultiIndex (date, code), columns为因子名称
        """
        # 实现逻辑：通过handler获取特征，转换为DataFrame
        pass
```

### 3.2 集成到factor模块的因子注册表

```python
# factor/factor_registry.py (修改现有或新增)

from .qlib_factor_calculator import Alpha158FactorCalculator, Alpha360FactorCalculator

# 扩展因子注册表
FACTOR_REGISTRY = {
    # 现有因子
    'VOL10': ...,
    'VSTD10': ...,
    
    # 新增：Alpha158特征集（作为单个"因子组"）
    'ALPHA158': Alpha158FactorCalculator,
    'ALPHA360': Alpha360FactorCalculator,
    
    # 或者支持单个Alpha158特征
    'ROC5': lambda: Alpha158FactorCalculator(feature_subset=['ROC5']),
    'ROC10': lambda: Alpha158FactorCalculator(feature_subset=['ROC10']),
    'MA5': lambda: Alpha158FactorCalculator(feature_subset=['MA5']),
    # ... 更多
}
```

### 3.3 使用示例

#### 方式1：通过factor模块使用Alpha158特征集
```python
from factor.factor_calculator import create_factor_calculator

# 创建Alpha158计算器（不依赖qlib数据集）
calc = create_factor_calculator('ALPHA158')

# 计算所有Alpha158因子
factors_df = calc.calculate_all(['000001', '000002'], '2024-01-01', '2024-12-31')
# 返回：(date, code) MultiIndex, 158列特征

# 用于Alphalens检验
from factor.factor import FactorTester
tester = FactorTester(cfg)
tester.run()  # 自动使用Alpha158因子
```

#### 方式2：提取单个Alpha158特征作为独立因子
```python
# 只计算ROC5因子
roc5_calc = create_factor_calculator(factor_name='ROC5', source='alpha158')
roc5_series = roc5_calc.calculate('000001', '2024-01-01', '2024-12-31')
```

#### 方式3：混合使用Alpha158和自定义因子
```python
# Alpha158因子 + 自定义因子
factors = {
    'ALPHA158': Alpha158FactorCalculator(),
    'MY_CUSTOM': MyCustomFactorCalculator(),
}
```

---

## 四、实现优先级

### Phase 1: 基础抽象（高优先级）
1. ✅ **提取特征名称获取功能**
   - `get_alpha158_feature_names()`
   - `get_alpha360_feature_names()`
   - 添加到 `factor/builtin_qlib_factors.py`

2. ✅ **实现轻量级Alpha158计算器**
   - 基于Pandas实现核心特征计算
   - 不依赖qlib数据集格式
   - 实现常用的滚动特征（ROC, MA, STD等）

3. ✅ **集成到factor模块**
   - 添加到因子注册表
   - 支持作为因子组或单个因子使用

### Phase 2: 完整适配（中优先级）
1. ✅ **Qlib Handler适配器**
   - 如果已有qlib数据集，直接使用handler提取
   - 避免重复计算

2. ✅ **特征子集支持**
   - 支持只计算部分Alpha158特征
   - 提高计算效率

### Phase 3: 优化增强（低优先级）
1. ✅ **性能优化**
   - 批量计算优化
   - 缓存机制

2. ✅ **完整特征集实现**
   - 实现Alpha158的所有158个特征
   - 实现Alpha360的所有360个特征

---

## 五、代码结构建议

```
factor/
├── __init__.py
├── factor.py                    # 主程序（现有）
├── factor_calculator.py         # 因子计算器接口（现有）
├── qlib_factor_calculator.py    # 【新增】Qlib因子计算器
│   ├── Alpha158FactorCalculator
│   ├── Alpha360FactorCalculator
│   └── QlibHandlerFactorCalculator
├── builtin_qlib_factors.py      # 【新增】Alpha158/Alpha360特征实现
│   ├── Alpha158FeatureCalculator
│   ├── Alpha360FeatureCalculator
│   └── feature_definitions.py  # 特征定义配置
├── qlib_adapter.py              # 【新增】Qlib适配器
│   └── QlibHandlerAdapter
└── README.md                     # 更新文档，说明Alpha158/Alpha360支持
```

---

## 六、关键设计决策

### 决策1：是否依赖qlib框架？

**方案A：完全独立实现（推荐）**
- ✅ 优点：不依赖qlib，可独立使用
- ✅ 优点：更灵活，可自定义特征计算
- ❌ 缺点：需要重新实现所有特征计算逻辑

**方案B：依赖qlib handler（备选）**
- ✅ 优点：复用qlib的成熟实现
- ❌ 缺点：必须构建qlib数据集，耦合度高

**建议**：采用方案A，但提供方案B作为可选路径

### 决策2：因子粒度选择

**方案A：特征集级别（推荐）**
- `ALPHA158`作为一个因子组，一次计算所有158个特征
- 适合批量使用和模型训练

**方案B：单个特征级别**
- `ROC5`, `MA10`等作为独立因子
- 适合单独检验和筛选

**方案C：两者都支持**
- 默认特征集级别
- 支持提取单个特征作为独立因子
- **推荐采用方案C**

### 决策3：输出格式

**统一输出格式**：
```python
# MultiIndex DataFrame
# Index: (date, code)
# Columns: factor_name
# Values: factor_value

           ROC5    ROC10   MA5     MA10    ...
date       code
2024-01-01 000001  0.02    0.05    10.5    10.3    ...
           000002  0.01    0.03    20.1    20.0    ...
```

---

## 七、迁移路径

### 步骤1：在factor模块中实现Alpha158计算器
- 创建 `factor/qlib_factor_calculator.py`
- 实现基础特征计算（ROC, MA, STD等）

### 步骤2：集成测试
- 测试Alpha158因子计算是否正确
- 与qlib handler输出对比验证

### 步骤3：更新factor模块文档
- 说明如何使用Alpha158/Alpha360因子
- 提供使用示例

### 步骤4：alpha_test模块简化
- 移除因子相关代码（如果已经整合到factor模块）
- 只保留模型训练部分（后续可迁移到factor_workflow）

---

## 八、预期收益

1. **统一因子接口**：Alpha158/Alpha360可以通过factor模块统一管理
2. **复用性提升**：Alpha158特征可以在Alphalens检验中使用
3. **降低耦合**：不依赖qlib数据集格式，更灵活
4. **功能完整**：factor模块支持更丰富的因子类型
5. **代码简化**：alpha_test模块可以专注于模型训练

---

## 九、注意事项

1. **计算准确性**：确保轻量级实现的Alpha158特征与qlib handler结果一致
2. **性能考虑**：Alpha158有158个特征，批量计算时注意性能
3. **数据要求**：某些特征可能需要较长的历史数据（如60日滚动）
4. **向后兼容**：确保不影响现有factor模块功能

