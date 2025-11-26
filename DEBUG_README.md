# VS Code 调试配置说明

## 概述

为TALIB因子模型训练流程创建了完整的VS Code调试配置，可以手动逐步调试整个模型建模过程。

## 调试配置列表

### 1. 完整流程调试
- **配置名称**: `Python: 调试TALIB模型训练（手动）`
- **文件**: `debug_talib_model.py`
- **功能**: 逐步执行完整的模型训练流程，包括环境设置、数据验证、QLib初始化、数据集创建、模型训练和结果保存

### 2. 核心模块调试

#### 数据处理
- **配置名称**: `factor_workflow/convert_sample.py: 数据转换`
- **功能**: 调试数据格式转换过程

#### 模型训练
- **配置名称**: `factor_workflow/workflow_main.py: 模型训练`
- **功能**: 调试核心模型训练流程

#### 回测评估
- **配置名称**: `factor_workflow/backtest_evaluation.py: 回测评估`
- **功能**: 调试回测和性能评估

#### 权重导出
- **配置名称**: `factor_workflow/export_scores.py: 权重导出`
- **功能**: 调试最终权重生成

#### TALIB因子生成
- **配置名称**: `generate_talib_factors.py: 生成TALIB因子`
- **功能**: 调试TALIB因子计算

## 使用方法

### 方法1: 完整流程调试（推荐）

1. 在VS Code中打开 `debug_talib_model.py`
2. 设置断点（在关键函数或需要检查的地方）
3. 按 `F5` 或点击调试面板的运行按钮
4. 选择 `Python: 调试TALIB模型训练（手动）` 配置
5. 逐步执行，观察每个步骤的输出

### 方法2: 模块化调试

1. 选择对应的调试配置
2. 在目标文件中设置断点
3. 运行调试
4. 检查变量值和执行流程

## 调试步骤详解

`debug_talib_model.py` 执行以下步骤：

1. **环境设置**: 检查虚拟环境和路径配置
2. **数据文件检查**: 验证所需的数据文件是否存在
3. **数据加载验证**: 加载并分析特征、标签和IC数据
4. **QLib初始化**: 初始化量化库环境
5. **数据集创建**: 创建训练和测试数据集
6. **模型训练**: 训练Long和Short模型套件
7. **结果保存**: 保存预测结果和统计信息

## 关键断点建议

### 数据验证阶段
```python
# 在 load_and_validate_data() 函数中设置断点
features_df = pd.read_pickle(paths.FEATURES_FILE)  # 检查特征数据
ic_df = pd.read_pickle(paths.IC_FILE)              # 检查IC数据
```

### 模型训练阶段
```python
# 在 train_models() 函数中设置断点
long_suite = train_model_suite('long', long_dataset, long_model_specs, fusion_config['long'])
# 检查模型训练结果
```

### 结果分析阶段
```python
# 在训练完成后检查预测结果
pred_long = long_suite.fused_prediction
pred_short = short_suite.fused_prediction
```

## 输出文件

调试完成后会在 `debug_model_results/` 目录生成：
- `predictions_long.pkl`: Long模型预测结果
- `predictions_short.pkl`: Short模型预测结果
- `debug_summary.txt`: 训练统计摘要

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   错误: 数据文件不完整
   解决: 先运行数据准备流程生成所需文件
   ```

2. **QLib初始化失败**
   ```
   错误: QLib初始化失败
   解决: 检查虚拟环境和依赖安装
   ```

3. **内存不足**
   ```
   错误: 内存不足
   解决: 减少数据量或增加系统内存
   ```

### 调试技巧

- 使用VS Code的变量查看器检查DataFrame内容
- 在关键计算步骤设置断点
- 使用调试控制台执行临时代码
- 查看调用堆栈了解执行流程

## 环境要求

- Python 3.12+
- 激活的虚拟环境 (.venv)
- 已安装的依赖包 (qlib, pandas, numpy等)
- TALIB数据文件 (talib_export_small/ 目录)

## 相关文件

- `factor_workflow/README.md`: 详细的工作流说明
- `requirements.txt`: 项目依赖
- `talib_model_results/`: 之前的训练结果