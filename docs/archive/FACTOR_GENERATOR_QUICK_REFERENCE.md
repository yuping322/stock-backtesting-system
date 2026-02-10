# 因子生成模块问题清单（Quick Reference）

**生成日期**: 2026-02-03  
**用途**: 快速查询问题、优先级、解决方案

---

## 📌 问题速查表

### 架构问题

| # | 问题 | 位置 | 严重性 | 解决方案 | 参考文档 |
|---|------|------|--------|---------|---------|
| A1 | 基类设计不完整 | `_base.py` | 🔴 P0 | 扩展基类功能，添加验证、对齐、异常恢复 | [改进指南 §8.2](#) |
| A2 | 子类实现方式混乱 | 各生成器 | 🔴 P0 | 采用统一的 Orchestration Pattern | [改进指南 §1.2](#) |
| A3 | 职责边界模糊（生成器 vs 计算器） | 各文件 | 🔴 P0 | 明确职责划分：生成器只编排，计算器只计算 | [改进指南 §P0-问题4](#) |

### 功能问题

| # | 问题 | 位置 | 严重性 | 解决方案 | 参考文档 |
|---|------|------|--------|---------|---------|
| F1 | 数据加载层多重依赖 | `_base.py` | 🟡 P1 | 实现 Adapter Pattern，统一依赖注入 | [分析报告 §2.1](#) |
| F2 | QLib 数据集构建复杂 | `qlib.py` | 🟡 P1 | 移到基础设施层 (`src/data/`) | [分析报告 §2.2](#) |
| F3 | Talib 参数硬编码 | `talib.py` | 🟡 P1 | 外部化到 `config/talib_parameters.yaml` | [改进指南 §P1-问题5](#) |
| F4 | 缺少数据质量检查 | 各生成器 | 🔴 P0 | 实现 `DataQualityChecker` 类 | [改进指南 §P0-问题3](#) |

### 集成问题

| # | 问题 | 位置 | 严重性 | 解决方案 | 参考文档 |
|---|------|------|--------|---------|---------|
| I1 | 计算器接口不一致 | `calculator.py` | 🔴 P0 | 统一为 3 参数接口 | [改进指南 §P0-问题1](#) |
| I2 | 没有统一的错误处理 | 各文件 | 🔴 P0 | 定义异常体系 (FactorGenerationException, etc.) | [改进指南 §P0-问题2](#) |
| I3 | 生成器与 `all_in_one.py` 不协调 | `all_in_one.py` | 🟡 P1 | 将验证逻辑移到基类 | [分析报告 §3.2](#) |

### 代码质量问题

| # | 问题 | 位置 | 严重性 | 解决方案 | 参考文档 |
|---|------|------|--------|---------|---------|
| Q1 | 代码重复和冗余 | `_base.py`, 各子类 | 🟡 P1 | 统一到 `DataFrameProcessor` 类 | [分析报告 §4.1](#) |
| Q2 | 异常处理不一致 | 各文件 | 🔴 P0 | 使用自定义异常类 | [改进指南 §P0-问题2](#) |
| Q3 | 类型提示不完整 | 各文件 | 🟢 P2 | 添加 TypedDict, dataclass | [分析报告 §4.3](#) |
| Q4 | 测试覆盖不足 | `tests/` | 🟡 P1 | 创建测试框架 (`tests/factor/`) | [改进指南 §P1-问题6](#) |

### 性能与维护问题

| # | 问题 | 位置 | 严重性 | 解决方案 | 参考文档 |
|---|------|------|--------|---------|---------|
| P1 | 数据转换低效 | `_base.py` | 🟡 P1 | 使用 pandas 向量化操作 | [分析报告 §6.1](#) |
| P2 | 没有缓存机制 | `_base.py` | 🟢 P2 | 实现 `OHLCVCache` | [分析报告 §6.2](#) |
| P3 | 进度跟踪不完整 | 各生成器 | 🟢 P2 | 添加进度回调机制 | [分析报告 §6.3](#) |

---

## 🔍 问题诊断流程

### 问题 1: 生成因子时出错

```
Q: 生成因子出错，怎么诊断？

A: 按以下步骤排查：

1️⃣  检查错误类型
   - DataNotAvailableError → 数据不可用（检查股票代码、日期范围）
   - FactorCalculationError → 计算失败（检查 OHLCV 数据质量）
   - FactorValidationError → 验证失败（检查数据质量报告）
   - PartialResultError → 部分失败（检查 failures 字典）

2️⃣  查看错误消息
   - 包含 stock_code? → 是该股票的问题
   - 包含 factor_name? → 是该因子的问题
   - 包含 date range? → 是日期范围的问题

3️⃣  检查日志
   - logger.warning() → 非致命问题，可以继续
   - logger.error() → 致命问题，需要修复

示例：
try:
    df = generator.generate()
except PartialResultError as e:
    print(f"失败的因子: {e.failures}")
    # 修复失败的因子，重试
except FactorValidationError as e:
    print(f"数据质量问题: {e.issue}")
    # 改进数据
```

### 问题 2: 计算器返回 NaN

```
Q: 为什么计算器返回全 NaN？

A: 检查以下几点：

1. OHLCV 数据是否加载成功？
   - 检查 data.load_oss_complex_stocks() 是否可用
   - 检查股票代码和日期范围是否正确

2. 因子计算函数是否有 bug？
   - 检查该因子的计算公式
   - 用样本数据手动测试

3. 结果是否被正确过滤？
   - 检查 clamp_dataframe_to_date_range() 是否过滤了所有结果
```

### 问题 3: 生成速度慢

```
Q: 生成因子非常慢，怎么优化？

A: 检查以下几点：

1. 是否在循环中重复加载 OHLCV 数据？
   ❌ for stock in stocks:
       ohlcv = load_ohlcv(stock)  # 重复 N 次加载
   
   ✅ all_ohlcv = load_ohlcv(stocks)  # 只加载 1 次
       for stock in stocks:
           use all_ohlcv[stock]

2. 是否在构建 DataFrame 时用了 append()？
   ❌ for stock in stocks:
       result = result.append(df)  # O(n²) 复杂度
   
   ✅ dfs = []
       for stock in stocks:
           dfs.append(df)
       result = pd.concat(dfs)  # O(n) 复杂度

3. 是否有并行化的可能？
   - 不同股票的计算相互独立 → 可以并行
   - 使用 multiprocessing 或 joblib
```

---

## 📋 修复优先级

### 🔴 今天必须做（P0）

```
□ [A1] 基类设计 - 添加 validate_output(), align_dates()
□ [I1] 计算器接口 - 统一所有计算器的 3 参数签名
□ [I2] 错误处理 - 定义异常体系，更新所有生成器
□ [F4] 数据质量 - 实现 DataQualityChecker，集成到生成器
```

**预计时间**: 8-16 小时

---

### 🟡 本周应该做（P1）

```
□ [F1] 数据加载 - 实现 Adapter Pattern 统一依赖
□ [F3] Talib 参数 - 创建外部化配置文件
□ [Q1] 代码重复 - 统一数据处理函数到 DataFrameProcessor
□ [Q4] 单元测试 - 创建测试框架
□ [P1] 数据转换 - 优化向量化操作
```

**预计时间**: 40-60 小时

---

### 🟢 后续可以做（P2）

```
□ [Q3] 类型提示 - 添加完整的类型提示
□ [P2] 缓存机制 - 实现 OHLCVCache
□ [P3] 进度跟踪 - 添加进度回调
□ [A2] 模式统一 - 采用 Orchestration Pattern
```

**预计时间**: 30-40 小时

---

## 🛠️ 快速修复命令

### 1. 查看当前文件结构
```bash
tree src/factor/generator -I '__pycache__'
```

### 2. 运行集成测试
```bash
cd /Users/fengzhi/Downloads/git/stock-backtesting-system
python -m src.factor.generator.all_in_one
```

### 3. 检查依赖
```bash
python -c "from src.data import data; print(dir(data))" | grep -E "(load|factor)"
```

### 4. 查看异常处理
```bash
grep -r "except Exception" src/factor/generator/ | wc -l
```

### 5. 检查类型提示
```bash
grep -r "-> pd.DataFrame" src/factor/generator/ | wc -l
```

---

## 📞 常见问题解答

### Q1: 我应该先修复哪个问题？
**A**: 按优先级：
1. 计算器接口 (I1) → 这是基础，很多地方依赖它
2. 错误处理 (I2) → 这样能更好地诊断其他问题
3. 数据质量 (F4) → 确保输出正确
4. 其他问题

### Q2: 修复 A1 问题会不会影响现有代码？
**A**: 不会。基类只是添加新方法，不删除现有方法。
现有的子类可以继续工作，逐步迁移到新接口。

### Q3: 我该如何测试我的修改？
**A**: 创建一个简单的测试脚本：
```python
from src.factor.generator import builtin

generator = builtin.BuiltinFactorGenerator(
    stock_codes=['000001'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    factor_names=['VOL10']
)

try:
    df = generator.generate()
    print(f"✅ 成功: {len(df)} 条数据")
except Exception as e:
    print(f"❌ 失败: {e}")
```

### Q4: 文档在哪里？
**A**: 
- 问题分析: `docs/FACTOR_GENERATOR_ISSUES_ANALYSIS.md`
- 改进指南: `docs/FACTOR_GENERATOR_IMPROVEMENT_GUIDE.md`
- 本文件: `docs/FACTOR_GENERATOR_QUICK_REFERENCE.md`

### Q5: 有没有示例代码？
**A**: 有，在改进指南中。每个问题都有代码示例。

---

## 📊 进度追踪

| 问题 | 状态 | 负责人 | 预计完成 | 备注 |
|------|------|--------|---------|------|
| I1 | ⏳ TODO | - | - | 计算器接口统一 |
| I2 | ⏳ TODO | - | - | 异常体系 |
| F4 | ⏳ TODO | - | - | 数据质量检查 |
| A1 | ⏳ TODO | - | - | 基类扩展 |
| F1 | ⏳ TODO | - | - | 数据加载统一 |
| F3 | ⏳ TODO | - | - | Talib 参数外部化 |
| Q1 | ⏳ TODO | - | - | 代码重复消除 |
| Q4 | ⏳ TODO | - | - | 单元测试框架 |
| P1 | ⏳ TODO | - | - | 性能优化 |

---

## 📞 技术支持

需要帮助？按以下方式获取支持：

1. **查看文档**: `docs/FACTOR_GENERATOR_*.md`
2. **查看代码示例**: `docs/FACTOR_GENERATOR_IMPROVEMENT_GUIDE.md` 中的代码示例
3. **运行测试**: `pytest tests/factor/ -v`
4. **查看日志**: 生成器会输出详细的 logger 信息

---

**最后更新**: 2026-02-03  
**文件位置**: `/docs/FACTOR_GENERATOR_QUICK_REFERENCE.md`

