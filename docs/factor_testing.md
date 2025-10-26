# 因子模块测试文档

## 概述

`tests/test_factor.py` 提供了因子模块的完整测试套件，验证了因子计算、文件加载、命令行解析等功能的正确性。

## 测试结构

### 1. TestFactorCalculator - 因子计算器测试

测试各种因子计算器的基本功能：

- ✅ `test_builtin_factor_calculator`: 内置因子计算器
- ✅ `test_ohlcv_factor_calculator`: OHLCV 因子函数
- ✅ `test_file_factor_calculator`: 从文件加载因子
- ✅ `test_custom_factor_calculator`: 完全自定义因子
- ✅ `test_create_factor_calculator`: 工厂函数

### 2. TestDataIntegration - 数据集成测试

测试与 `data.py` 的集成：

- ✅ `test_load_factor_from_file`: 从文件加载因子数据
- ✅ `test_load_factor_from_file_with_normalized_codes`: 代码标准化处理

### 3. TestFactorParser - 命令行解析测试

测试命令行参数解析：

- ✅ `test_parse_args_default`: 默认参数
- ✅ `test_parse_args_custom`: 自定义参数

### 4. TestCFG - 配置类测试

测试配置类初始化：

- ✅ `test_cfg_initialization`: 配置初始化

### 5. TestFactorHelperFunctions - 辅助函数测试

测试辅助函数：

- ✅ `test_rolling_monitor`: 滚动监控函数

### 6. TestFactorTester - 主类测试

测试 FactorTester 类：

- ✅ `test_factor_tester_initialization`: 初始化
- ✅ `test_factor_tester_with_custom_factors`: 自定义因子

### 7. TestBuiltinFactors - 内置因子测试

测试内置因子列表和函数：

- ✅ `test_builtin_factors_list`: 内置因子列表
- ✅ `test_builtin_factor_functions`: 内置因子函数

## 运行测试

### 运行所有测试

```bash
python -m pytest tests/test_factor.py -v
```

### 运行特定测试类

```bash
# 只测试因子计算器
python -m pytest tests/test_factor.py::TestFactorCalculator -v

# 只测试数据集成
python -m pytest tests/test_factor.py::TestDataIntegration -v
```

### 运行特定测试

```bash
# 测试文件因子计算器
python -m pytest tests/test_factor.py::TestFactorCalculator::test_file_factor_calculator -v
```

## 测试覆盖

### 功能覆盖

✅ 因子计算器接口
- BuiltinFactorCalculator（内置因子）
- OHLCVFactorCalculator（OHLCV 函数）
- FileFactorCalculator（文件加载）
- CustomFactorCalculator（完全自定义）

✅ 数据加载
- 从 CSV 文件加载
- 代码标准化（.XSHG/.XSHE 后缀处理）
- 日期范围过滤
- 多股票过滤

✅ 命令行接口
- 参数解析
- 默认值
- 自定义参数

✅ 配置管理
- CFG 类初始化
- 配置传递

✅ 辅助功能
- 滚动监控
- 因子工厂函数

## 测试数据

测试使用临时文件创建测试数据：

```python
# 示例因子文件格式
date,code,MY_FACTOR
2024-01-01,000001,1.23
2024-01-02,000001,1.25
```

## 关键测试场景

### 1. 代码标准化

测试确保股票代码正确标准化：
- `000001` → `000001`
- `000001.XSHG` → `000001`
- `1` → `000001`（补齐6位）

### 2. 文件加载

测试从文件加载因子时：
- 正确读取 CSV
- 正确设置 MultiIndex
- 正确过滤日期和股票

### 3. 缓存机制

测试 FileFactorCalculator 的缓存：
- 文件只加载一次
- 后续调用使用缓存

## 发现的 Bug 和修复

### Bug 1: 代码类型不匹配

**问题**: 读取 CSV 时，pandas 可能将代码列读取为数字而非字符串。

**修复**: 在 `pd.read_csv()` 时指定 `dtype={'code': str}`

### Bug 2: 代码长度不统一

**问题**: 代码被标准化为 `1` 而不是 `000001`。

**修复**: 添加 `str.zfill(6)` 补齐6位

### Bug 3: 代码后缀不一致

**问题**: `_normalize_codes` 添加后缀，但文件中已去掉后缀。

**修复**: 在过滤时也去掉后缀进行比较

## 测试结果

所有 15 个测试用例均通过：

```
============================== 15 passed in 1.52s ==============================
```

## 持续集成

建议在 CI/CD 流程中添加：

```yaml
# GitHub Actions 示例
- name: Run factor tests
  run: |
    pip install pytest
    python -m pytest tests/test_factor.py -v
```

## 扩展测试

可以添加更多测试：

1. **性能测试**: 大规模数据加载性能
2. **错误处理测试**: 异常情况处理
3. **边界测试**: 空数据、单条数据等
4. **集成测试**: 与 alphalens 的完整集成

## 参考

- [factor_calculator.py](../factor/factor_calculator.py): 因子计算器实现
- [factor.py](../factor/factor.py): 主程序
- [data.py](../data.py): 数据加载接口
- [factor_custom_factors.md](factor_custom_factors.md): 自定义因子文档
