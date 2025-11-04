# Factor模块与Alpha_Test模块整合架构设计

## 一、功能对比总结

### 1. **factor模块** - 因子生成与检验层

#### 核心职责
- **因子计算**：支持内置因子、自定义OHLCV因子、完全自定义因子
- **因子检验**：使用Alphalens进行单因子/多因子有效性检验
- **因子评估**：计算IC、IR、收益、单调性等指标
- **因子监控**：滚动窗口监控因子表现，自动打分和状态标识
- **数据导出**：导出价格和因子数据用于其他平台

#### 技术栈
- Alphalens：因子有效性检验
- Pandas/NumPy：数据处理
- data.py：数据源接口（OSS）

#### 输入/输出
- **输入**：股票池、日期范围、因子列表、调仓周期
- **输出**：因子检验报告、IC/IR指标、tear-sheet、因子数据CSV

#### 数据流
```
数据源(data.py) → 因子计算(factor_calculator) → Alphalens检验 → 评估报告
```

---

### 2. **alpha_test模块** - Alpha策略回测层

#### 核心职责
- **数据构建**：从CSV或真实数据源构建qlib格式数据集
- **特征工程**：使用Alpha158/Alpha360等预定义特征集
- **模型训练**：LightGBM、线性模型等
- **信号生成**：模型预测生成交易信号
- **组合回测**：使用qlib进行组合回测分析

#### 技术栈
- Qlib：量化研究和回测框架
- LightGBM：梯度提升模型
- Alpha158/Alpha360：预定义特征集
- data.py：数据源接口（OSS）

#### 输入/输出
- **输入**：股票池、日期范围、模型配置
- **输出**：qlib数据集、训练模型、预测信号、IC分析、回测结果

#### 数据流
```
数据源(data.py) → qlib数据集构建 → Alpha158特征提取 → 模型训练 → 信号生成 → 回测分析
```

---

### 3. **factor_workflow模块** - 因子建模工作流层

#### 核心职责
- **模型套件训练**：Ridge、HistGB等模型组合
- **预测融合**：基于IC的加权融合多个模型预测
- **信号后处理**：行业中性化、极值裁剪
- **组合构建**：生成可交易的信号权重
- **回测评估**：运行样本回测并导出权重

#### 技术栈
- Qlib：量化框架
- Scikit-learn：Ridge、HistGB模型
- Pandas：数据处理

#### 输入/输出
- **输入**：因子面板数据（features_panel.pkl）、标签数据（label_panel.pkl）
- **输出**：融合预测、模型权重、信号权重CSV、回测结果

#### 数据流
```
因子数据 → 模型训练套件 → IC加权融合 → 信号后处理 → 组合回测 → 权重导出
```

---

## 二、功能重叠与差异分析

### 重叠部分

| 功能 | factor模块 | alpha_test模块 | factor_workflow模块 |
|------|-----------|---------------|-------------------|
| 数据获取 | ✅ 使用data.py | ✅ 使用data.py | ❌ 从pkl文件读取 |
| 因子计算 | ✅ 核心功能 | ❌ 使用预定义特征 | ❌ 使用已有因子数据 |
| 因子检验 | ✅ Alphalens检验 | ❌ 无 | ❌ 使用IC但无独立检验 |
| 模型训练 | ❌ 无 | ✅ LightGBM/Linear | ✅ Ridge/HistGB |
| 信号生成 | ❌ 无 | ✅ 模型预测 | ✅ 融合预测 |
| 回测分析 | ❌ 无 | ✅ qlib回测 | ✅ qlib回测 |
| IC计算 | ✅ Alphalens | ✅ SigAnaRecord | ✅ 滚动IC |

### 关键差异

1. **因子视角 vs 模型视角**
   - `factor模块`：从因子出发，检验因子有效性
   - `alpha_test模块`：从特征工程出发，训练预测模型
   - `factor_workflow模块`：从已有因子出发，建模和融合

2. **检验方式**
   - `factor模块`：使用Alphalens做因子分层回测
   - `alpha_test模块`：使用qlib的SignalRecord做信号分析
   - `factor_workflow模块`：使用滚动IC做模型融合

3. **数据格式**
   - `factor模块`：Pandas DataFrame（date, code, factor_value）
   - `alpha_test模块`：Qlib数据集（按股票组织，特征工程）
   - `factor_workflow模块`：MultiIndex DataFrame（features_panel.pkl）

---

## 三、整合架构设计

### 3.1 整体架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                    统一入口层 (Unified Entry)                  │
│  - CLI接口                                                    │
│  - Python API接口                                             │
│  - 配置文件管理                                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   数据适配层 (Data Adapter Layer)              │
│  - 统一数据接口（基于data.py）                                 │
│  - 数据格式转换器                                              │
│    ├─ CSV → Pandas                                           │
│    ├─ Pandas → Qlib Dataset                                  │
│    └─ Qlib Dataset → MultiIndex Panel                       │
│  - 数据缓存管理                                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                因子计算层 (Factor Computation Layer)           │
│  - 因子计算器（factor_calculator.py）                          │
│    ├─ 内置因子库                                               │
│    ├─ OHLCV因子计算                                            │
│    └─ 自定义因子接口                                           │
│  - 因子数据管理                                                │
│    ├─ 因子存储（OSS/本地）                                     │
│    └─ 因子版本管理                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              因子检验层 (Factor Evaluation Layer)              │
│  - Alphalens检验（factor.py）                                 │
│    ├─ 单因子检验                                               │
│    ├─ 多因子检验                                               │
│    └─ 滚动监控                                                 │
│  - Qlib特征工程（alpha_test）                                  │
│    ├─ Alpha158特征集                                           │
│    ├─ Alpha360特征集                                           │
│    └─ 自定义特征集                                             │
│  - 因子筛选器                                                   │
│    ├─ IC阈值筛选                                               │
│    ├─ IR阈值筛选                                               │
│    └─ 综合打分筛选                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 模型训练层 (Model Training Layer)               │
│  - 轻量级模型（alpha_test）                                    │
│    ├─ LinearModel                                             │
│    ├─ LGBModel                                                │
│    └─ 自定义模型                                               │
│  - 模型套件（factor_workflow）                                 │
│    ├─ Ridge套件                                               │
│    ├─ HistGB套件                                              │
│    └─ 模型融合策略                                             │
│  - 模型管理                                                     │
│    ├─ 模型版本控制                                             │
│    ├─ 超参数管理                                               │
│    └─ 模型评估                                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                信号生成层 (Signal Generation Layer)             │
│  - 单模型信号（alpha_test）                                    │
│  - 多模型融合（factor_workflow）                               │
│    ├─ IC加权融合                                               │
│    ├─ 等权重融合                                               │
│    └─ 自适应权重                                               │
│  - 信号后处理                                                  │
│    ├─ 行业中性化                                               │
│    ├─ 风格中性化                                               │
│    ├─ 极值裁剪                                                 │
│    └─ 标准化                                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               回测评估层 (Backtest Evaluation Layer)            │
│  - Qlib回测引擎                                                │
│    ├─ 组合回测                                                 │
│    ├─ 单策略回测                                               │
│    └─ 风险分析                                                 │
│  - 绩效评估                                                     │
│    ├─ 收益指标                                                 │
│    ├─ 风险指标                                                 │
│    └─ 风险调整收益                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块设计

#### A. 统一数据接口层

```python
# unified_data_adapter.py

class UnifiedDataAdapter:
    """统一数据适配器，提供统一的数据访问接口"""
    
    def get_price_data(self, codes, start_date, end_date, fields=['open','high','low','close','volume']):
        """
        获取价格数据，返回统一格式
        - 支持格式：Pandas DataFrame、Qlib Dataset、MultiIndex Panel
        """
        pass
    
    def get_factor_data(self, codes, start_date, end_date, factor_names):
        """
        获取因子数据
        """
        pass
    
    def convert_to_qlib_dataset(self, price_df, factor_df):
        """转换为Qlib数据集格式"""
        pass
    
    def convert_to_factor_panel(self, factor_df):
        """转换为因子面板格式（MultiIndex）"""
        pass
```

#### B. 因子计算引擎

```python
# factor_engine.py

class FactorEngine:
    """因子计算引擎，整合factor模块的因子计算能力"""
    
    def __init__(self, calculator_registry):
        self.calculators = calculator_registry  # 因子计算器注册表
    
    def compute_factor(self, factor_name, codes, start_date, end_date):
        """计算单个因子"""
        calculator = self.calculators.get(factor_name)
        return calculator.calculate(codes, start_date, end_date)
    
    def compute_factors_batch(self, factor_names, codes, start_date, end_date):
        """批量计算因子"""
        pass
    
    def evaluate_factor(self, factor_name, codes, start_date, end_date):
        """评估因子有效性（使用Alphalens）"""
        pass
```

#### C. 模型训练引擎

```python
# model_engine.py

class ModelEngine:
    """模型训练引擎，整合alpha_test和factor_workflow的模型能力"""
    
    def train_single_model(self, dataset, model_config):
        """训练单个模型（alpha_test风格）"""
        pass
    
    def train_model_suite(self, dataset, model_specs, fusion_config):
        """训练模型套件（factor_workflow风格）"""
        pass
    
    def predict(self, model, dataset, segment='test'):
        """模型预测"""
        pass
```

#### D. 信号生成引擎

```python
# signal_engine.py

class SignalEngine:
    """信号生成引擎"""
    
    def generate_single_signal(self, model, dataset):
        """生成单模型信号"""
        pass
    
    def fuse_signals(self, signals, weights, fusion_method='ic_weighted'):
        """融合多个信号"""
        pass
    
    def post_process_signal(self, signal, neutralize_industry=True, neutralize_size=True):
        """信号后处理"""
        pass
```

#### E. 回测引擎

```python
# backtest_engine.py

class BacktestEngine:
    """回测引擎，整合qlib回测能力"""
    
    def run_alphlens_backtest(self, factor, price):
        """运行Alphalens因子回测"""
        pass
    
    def run_qlib_backtest(self, signal, config):
        """运行Qlib组合回测"""
        pass
    
    def evaluate_performance(self, backtest_result):
        """评估回测绩效"""
        pass
```

---

## 四、整合工作流设计

### 4.1 端到端工作流

```
1. 数据准备阶段
   ├─ 获取价格数据（data.py）
   ├─ 计算因子（factor_calculator）
   └─ 数据格式转换（统一适配器）

2. 因子筛选阶段（可选）
   ├─ Alphalens因子检验（factor.py）
   ├─ IC/IR评估
   └─ 因子筛选（IC阈值、综合打分）

3. 特征工程阶段（可选）
   ├─ Alpha158/Alpha360特征提取（alpha_test）
   ├─ 自定义特征工程
   └─ 特征选择

4. 模型训练阶段
   ├─ 单模型训练（alpha_test风格）
   └─ 模型套件训练（factor_workflow风格）

5. 信号生成阶段
   ├─ 模型预测
   ├─ 信号融合（多模型）
   └─ 信号后处理（中性化等）

6. 回测评估阶段
   ├─ Alphalens回测（因子视角）
   ├─ Qlib回测（组合视角）
   └─ 绩效评估

7. 结果导出阶段
   ├─ 信号权重导出
   ├─ 回测报告生成
   └─ 模型和因子保存
```

### 4.2 三种典型使用场景

#### 场景1：因子发现和筛选（factor模块主导）
```
数据获取 → 因子计算 → Alphalens检验 → 因子筛选 → 因子导出
```

#### 场景2：快速Alpha策略回测（alpha_test模块主导）
```
数据获取 → Qlib数据集构建 → Alpha158特征 → 模型训练 → 信号生成 → 回测分析
```

#### 场景3：生产级因子建模（factor_workflow模块主导）
```
因子面板数据 → 模型套件训练 → IC加权融合 → 信号后处理 → 组合回测 → 权重导出
```

---

## 五、数据流转设计

### 5.1 统一数据模型

```python
# unified_data_models.py

@dataclass
class PriceData:
    """统一价格数据模型"""
    data: pd.DataFrame  # MultiIndex (date, code) 或 DatetimeIndex
    fields: List[str]   # ['open', 'high', 'low', 'close', 'volume']
    metadata: Dict      # 元数据信息

@dataclass
class FactorData:
    """统一因子数据模型"""
    data: pd.DataFrame  # MultiIndex (date, code)
    factor_names: List[str]
    metadata: Dict

@dataclass
class FeatureData:
    """统一特征数据模型（用于模型训练）"""
    data: pd.DataFrame  # Qlib Dataset格式
    feature_names: List[str]
    label: pd.Series
    metadata: Dict
```

### 5.2 数据转换矩阵

| 源格式 | 目标格式 | 转换器 | 使用场景 |
|--------|---------|--------|----------|
| data.py → DataFrame | Alphalens格式 | `to_alphalens_format` | factor模块检验 |
| data.py → DataFrame | Qlib Dataset | `to_qlib_dataset` | alpha_test训练 |
| DataFrame → MultiIndex Panel | factor_workflow格式 | `to_factor_panel` | factor_workflow训练 |
| Qlib Dataset → DataFrame | 标准DataFrame | `from_qlib_dataset` | 结果导出 |

---

## 六、配置管理设计

### 6.1 统一配置结构

```python
# unified_config.py

@dataclass
class DataConfig:
    """数据配置"""
    source: str  # 'oss', 'local', 'qlib'
    codes: List[str]  # 或股票池标识
    start_date: str
    end_date: str
    fields: List[str]

@dataclass
class FactorConfig:
    """因子配置"""
    factor_names: List[str]
    calculator_config: Dict  # 因子计算器配置
    
@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str  # 'single', 'suite'
    model_specs: Dict
    fusion_config: Optional[Dict]
    
@dataclass
class BacktestConfig:
    """回测配置"""
    backtest_type: str  # 'alphalens', 'qlib', 'both'
    config: Dict

@dataclass
class WorkflowConfig:
    """完整工作流配置"""
    data: DataConfig
    factor: Optional[FactorConfig]
    feature: Optional[FeatureConfig]  # Alpha158等
    model: ModelConfig
    signal: SignalConfig
    backtest: BacktestConfig
```

### 6.2 配置文件示例

```yaml
# config/workflow_example.yaml

data:
  source: oss
  stock_pool: HS300
  start_date: 2024-01-01
  end_date: 2024-12-31
  fields: [open, high, low, close, volume]

factor:
  enabled: true
  factors: [VOL10, VSTD10, RSI_14]
  evaluation:
    method: alphalens
    quantiles: 10
    periods: [5, 10, 15]
    ic_threshold: 0.05
    ir_threshold: 0.5

feature:
  enabled: true
  feature_set: Alpha158  # 或 Alpha360, custom
  custom_features: []

model:
  type: suite  # 或 single
  specs:
    - name: Ridge
      params: {alpha: 1.0}
    - name: HistGB
      params: {max_depth: 5}
  fusion:
    method: ic_weighted
    window: 60

signal:
  post_process:
    neutralize_industry: true
    neutralize_size: true
    clip_extremes: true
    clip_threshold: 0.05

backtest:
  type: both  # alphalens, qlib, both
  qlib_config:
    account: 100000000
    commission: 0.0005
  alphalens_config:
    quantiles: 10
    periods: [5, 10, 15]
```

---

## 七、整合优先级建议

### Phase 1: 数据层整合（基础）
1. ✅ **统一数据适配器**：封装data.py接口，提供统一数据访问
2. ✅ **数据格式转换器**：实现DataFrame ↔ Qlib Dataset ↔ MultiIndex Panel转换
3. ✅ **数据缓存机制**：避免重复数据加载

### Phase 2: 因子层整合（核心）
1. ✅ **因子引擎封装**：统一factor_calculator接口
2. ✅ **因子存储系统**：统一因子数据的存储和版本管理
3. ✅ **因子筛选器**：整合Alphalens检验结果，自动筛选有效因子

### Phase 3: 模型层整合（增强）
1. ✅ **模型训练引擎**：支持单模型和模型套件两种模式
2. ✅ **模型注册机制**：可插拔的模型注册表
3. ✅ **超参数管理**：统一的超参数配置和版本管理

### Phase 4: 信号层整合（高级）
1. ✅ **信号生成引擎**：统一单模型和多模型信号生成
2. ✅ **信号融合策略**：支持多种融合方法（IC加权、等权、自适应）
3. ✅ **信号后处理流水线**：统一的中性化、裁剪等处理

### Phase 5: 回测层整合（完整）
1. ✅ **统一回测接口**：封装Alphalens和Qlib回测
2. ✅ **绩效评估统一**：统一的指标计算和报告生成
3. ✅ **结果导出统一**：统一的权重和报告导出格式

---

## 八、关键技术决策

### 8.1 数据格式选择
- **内部统一格式**：MultiIndex DataFrame `(date, code)`
- **Qlib交互**：按需转换为Qlib Dataset
- **Alphalens交互**：按需转换为Alphalens格式

### 8.2 因子存储策略
- **实时计算**：轻量级因子（如VOL10）
- **预计算存储**：复杂因子（如技术指标）
- **缓存机制**：避免重复计算

### 8.3 模型训练策略
- **快速验证**：使用轻量级模型（Linear、单LGB）
- **生产部署**：使用模型套件（Ridge+HistGB融合）
- **实验管理**：使用Qlib Workflow记录所有实验

### 8.4 信号融合策略
- **因子层面**：基于Alphalens的IC加权
- **模型层面**：基于滚动IC的EMA权重
- **自适应权重**：根据近期表现动态调整

---

## 九、待解决问题

1. **数据格式不统一**
   - factor模块：Pandas DataFrame
   - alpha_test模块：Qlib Dataset
   - factor_workflow模块：MultiIndex Panel
   - **解决方案**：建立统一数据适配层

2. **因子定义不一致**
   - factor模块：自定义因子计算器
   - alpha_test模块：使用Alpha158预定义特征
   - **解决方案**：统一因子注册表，支持两种模式

3. **回测引擎分离**
   - factor模块：Alphalens回测
   - alpha_test/factor_workflow：Qlib回测
   - **解决方案**：统一回测接口，内部选择引擎

4. **配置管理分散**
   - 每个模块有独立的配置方式
   - **解决方案**：统一配置管理系统

5. **实验管理不统一**
   - factor模块：无实验管理
   - alpha_test/factor_workflow：使用Qlib Workflow
   - **解决方案**：统一实验跟踪系统

---

## 十、后续整合步骤（建议）

### 第一步：建立统一基础设施
1. 创建`core/`目录，存放统一接口
2. 实现统一数据适配器
3. 实现数据格式转换器
4. 建立统一配置管理系统

### 第二步：模块接口化
1. 将factor模块接口化（Factory模式）
2. 将alpha_test模块接口化
3. 将factor_workflow模块接口化
4. 建立模块注册机制

### 第三步：工作流编排
1. 实现统一工作流引擎
2. 支持YAML配置驱动
3. 支持Python API调用
4. 支持CLI命令行

### 第四步：结果统一
1. 统一结果格式
2. 统一报告生成
3. 统一导出接口

---

## 十一、预期收益

1. **提高复用性**：各模块可以独立使用，也可以组合使用
2. **降低耦合度**：通过统一接口层解耦各模块
3. **增强扩展性**：新功能可以通过注册机制接入
4. **统一用户体验**：一套配置、一套API、一套结果格式
5. **便于维护**：清晰的架构和职责划分

---

## 附录：相关文件路径

- factor模块：`/factor/`
- alpha_test模块：`/alpha_test/`
- factor_workflow模块：`/factor_workflow/`
- 数据源：`/data.py`
- 统一接口层：`/core/`（待创建）

