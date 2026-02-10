# 完整使用指南

## 📦 已创建的文件

### 训练脚本

1. **`train_super_factor.py`** - 基础版
   - 简单快速
   - 仅LightGBM
   - 适合快速迭代

2. **`train_super_factor_v2.py`** ⭐ - 完整版
   - 完整实现notebook功能
   - 8种模型对比
   - 特征选择
   - 因子配置化

3. **`predict_stocks.py`** - 预测脚本
   - 支持两种模型格式
   - 自动选股
   - 标准输出格式

### 配置文件

4. **`factor_config.json`** - 因子配置
   - 可自定义因子列表
   - 支持动态加载

### 文档

5. **`README.md`** - 完整说明
6. **`QUICKSTART.md`** - 快速开始
7. **`VERSION_COMPARISON.md`** - 版本对比

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install lightgbm scikit-learn scipy matplotlib pandas numpy
```

### 2. 选择版本

**入门用户**: 使用基础版
```bash
python ml/train_super_factor.py
```

**进阶用户**: 使用完整版（推荐）
```bash
python ml/train_super_factor_v2.py
```

### 3. 预测选股

```bash
python ml/predict_stocks.py --top-n 10
```

## 📊 完整版特性

### 8种机器学习模型

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| KNN | 非参数，距离计算 | 局部模式识别 |
| 逻辑回归 | 线性，可解释 | 特征重要性明显 |
| 决策树 | 非线性，可解释 | 规则提炼 |
| 朴素贝叶斯 | 快速，概率输出 | 大数据集 |
| 随机森林 | 集成学习，鲁棒 | 复杂关系 |
| AdaBoost | 自适应提升 | 难分类样本 |
| SVM | 高维空间 | 小样本 |
| LightGBM | 梯度提升，高效 | 大规模数据 |

### 特征选择方法

1. **F检验** - 方差分析
2. **互信息** - 非线性关系
3. **随机森林** - 特征重要性
4. **RFE** - 递归特征消除

### 数据处理流程

```
原始数据 → Winsorize → 标准化 → 缺失值处理 → 特征选择
```

### 评估指标

- **Accuracy** - 准确率
- **AUC** - ROC曲线下面积
- **交叉验证** - 5折CV评估
- **模型对比** - 自动选择最佳

## 🎯 使用工作流

### 每周/每月

1. **训练模型**（完整版）
   ```bash
   python ml/train_super_factor_v2.py
   ```
   
2. **查看对比结果**
   ```bash
   cat ml/output/model_comparison.csv
   ```

3. **使用最佳模型预测**
   ```bash
   python ml/predict_stocks.py --top-n 10
   ```

4. **查看选股结果**
   ```bash
   cat data/factor_values_sample.csv
   ```

### 每日

只需运行预测脚本（模型已训练好）
```bash
python ml/predict_stocks.py
```

## 📝 自定义因子

编辑 `ml/factor_config.json`:

```json
{
  "factors": [
    "sales_growth",
    "operating_revenue_growth_rate",
    "your_custom_factor"
  ]
}
```

## 🔧 参数调整

### train_super_factor_v2.py

```python
# 日期范围
START_DATE = '2024-01-01'
END_DATE = '2024-10-24'

# 股票池
INDEX = '000510'  # 小盘股

# 标签策略
PERCENT_SELECT = [0.3, 0.3]  # 前30%为1，后30%为0

# 特征选择
RFE_N_FEATURES = 20  # RFE选择的特征数
```

## 📈 输出文件

### 训练输出

- `ml/models/*.pkl` - 各个模型文件
- `ml/models/feature_selector.pkl` - 特征选择器
- `ml/models/model_info.json` - 模型配置信息
- `ml/output/model_comparison.csv` - 模型对比结果

### 预测输出

- `data/factor_values_sample.csv` - 选股结果（追加模式）

## 🐛 常见问题

### Q1: 数据加载失败
**A**: 检查OSS连接和数据路径

### Q2: 训练太慢
**A**: 使用基础版或减少训练数据日期范围

### Q3: 模型准确率低
**A**: 
- 尝试更多因子
- 调整标签策略
- 使用特征选择

### Q4: 内存不足
**A**:
- 减少因子数量
- 缩短日期范围
- 增加服务器内存

## 💡 最佳实践

1. **首次使用**: 运行完整版了解各模型表现
2. **日常使用**: 用基础版快速训练
3. **定期更新**: 每月用完整版重新对比
4. **因子管理**: 使用factor_config.json统一配置
5. **版本控制**: 保存每次训练的model_info.json

## 📚 参考

- JoinQuant社区: 八大机器学习模型大比拼
- docs/factor_summary.md: 有效因子列表
- docs/factor_strategy_guide.md: 因子检验策略

