# ML超级因子构建

## 功能说明

将超级因子构建拆分为两个独立脚本：
1. **训练模型** - 使用历史数据训练超级因子模型
   - `train_super_factor.py` - 基础版（仅LightGBM）
   - `train_super_factor_v2.py` - 增强版（多模型对比+特征选择）
2. **预测选股** (`predict_stocks.py`) - 使用训练好的模型生成选股结果

## 版本选择

### 基础版 (`train_super_factor.py`)
- ✅ 简单快速
- ✅ 仅使用LightGBM
- ✅ 适合快速迭代

### 增强版 (`train_super_factor_v2.py`)
- ✅ 多模型对比（LightGBM、逻辑回归、随机森林、AdaBoost）
- ✅ 特征选择（F检验、互信息、随机森林重要性）
- ✅ 完整评估指标
- ✅ 自动选择最佳模型
- ⚠️ 训练时间较长

## 安装依赖

```bash
pip install lightgbm scikit-learn scipy
```

## 使用流程

### 步骤1: 训练模型

**基础版**（推荐新手）:
```bash
python ml/train_super_factor.py
```

**增强版**（推荐进阶用户）:
```bash
python ml/train_super_factor_v2.py
```

**输出**:
- `ml/models/super_factor_model.txt` - LightGBM模型
- `ml/models/ic_weights.parquet` - IC权重历史
- `ml/models/model_info.json` - 模型配置信息

### 步骤2: 预测选股（每日运行）

```bash
# 使用最新数据预测
python ml/predict_stocks.py --date 2024-10-24 --top-n 10

# 自动使用最新日期
python ml/predict_stocks.py --top-n 10

# 指定输出文件
python ml/predict_stocks.py --output data/my_predictions.csv
```

**输出**: `data/factor_values_sample.csv` (追加模式)

### 配置参数

#### 训练参数 (`train_super_factor.py`)

编辑脚本修改训练配置：

```python
# 训练日期范围（使用更长历史）
start_date = "2024-01-01"
end_date = "2024-10-24"

# 股票池
stock_pool = "small"

# 滚动窗口
ROLL_WIN = 50
```

#### 预测参数 (`predict_stocks.py`)

命令行参数：

```bash
--date YYYY-MM-DD    # 预测日期，默认最新交易日
--top-n N            # 选择股票数量，默认10
--stock-pool NAME    # 股票池，默认small
--output PATH        # 输出文件路径
```

## 使用的因子

脚本使用从 `docs/factor_summary.md` 中筛选的有效因子，包括：

### 增长类 (8个)
- sales_growth (8/9分 - 最强)
- operating_revenue_growth_rate
- total_profit_growth_rate
- 等

### 盈利类 (16个)
- operating_cost_ttm (7/9分)
- total_operating_revenue_ttm (7/9分)
- gross_profit_ttm (7/9分)
- 等

### 现金流类 (5个)
- cashflow_per_share_ttm
- cash_flow_to_price_ratio
- 等

### 规模类 (4个)
- market_cap (7/9分)
- size (7/9分)
- 等

### 技术指标 (7个)
- raw_beta (7/9分)
- beta (7/9分)
- boll_down (7/9分)
- MAC20 (7/9分)
- 等

**总计约40个有效因子**

## 输出格式

### 训练输出 (`train_super_factor.py`)

保存在 `ml/models/` 目录：
- `super_factor_model.txt` - LightGBM模型文件
- `ic_weights.parquet` - 历史IC权重
- `model_info.json` - 模型配置信息

### 预测输出 (`predict_stocks.py`)

保存为CSV格式，类似 `data/factor_values_sample.csv`：

```csv
date,code,weight
2024-10-24,000001,0.0919
2024-10-24,000002,0.0649
2024-10-24,000003,0.0776
...
```

- `date`: 预测日期
- `code`: 股票代码（6位）
- `weight`: 权重（总和为1）

## 工作流程

### 训练流程 (`train_super_factor.py`)

1. **加载历史数据** (6-12个月)
2. **计算收益率标签** (10天后)
3. **因子清洗**: winsorize + 市值中性化 + z-score
4. **滚动IC加权**: 50天滚动窗口，动态权重
5. **风格中性化**: 去除市值和beta暴露
6. **训练LightGBM**: 全量数据训练
7. **保存模型**: 模型文件 + IC权重 + 配置信息

### 预测流程 (`predict_stocks.py`)

1. **加载模型**: 读取训练好的模型和IC权重
2. **读取最新数据**: 当日因子数据
3. **因子清洗**: 与训练时保持一致
4. **计算超级因子**: 使用最新IC权重
5. **LightGBM预测**: 输入超级因子得分
6. **选股**: 按得分排序选择Top N
7. **归一化权重**: 转换为权重分布
8. **保存结果**: 追加到CSV文件

## 注意事项

1. **数据要求**: 需要OSS中有对应的因子数据和价格数据
2. **运行时间**: 根据数据量可能需要几分钟到十几分钟
3. **内存占用**: 较多因子时可能占用较大内存
4. **持续监控**: 建议定期运行（每月或每季度）更新超级因子

## 后续步骤

### 定期更新

**模型训练** (每月/每季度):
```bash
python ml/train_super_factor.py
```

**每日预测** (每天收盘后):
```bash
python ml/predict_stocks.py --top-n 10
```

### 与BacktestEngine集成

```python
import pandas as pd
from backtest_engine import BacktestEngine

# 读取预测结果
predictions = pd.read_csv("data/factor_values_sample.csv")

# 筛选最新日期
latest_date = predictions['date'].max()
latest_preds = predictions[predictions['date'] == latest_date]

# 运行回测
engine = BacktestEngine()
# ... 配置并使用latest_preds运行回测
```

### 自动运行脚本

创建 `scripts/daily_predict.sh`:

```bash
#!/bin/bash
cd /path/to/stock-backtesting-system
python ml/predict_stocks.py --top-n 10 --output data/predictions_$(date +%Y%m%d).csv
```

用cron定时运行：
```bash
# 每天16:00运行
0 16 * * * /path/to/daily_predict.sh
```

## 参数调优建议

### 1. 调整滚动窗口
- 短窗口(30天): 更敏感，响应市场变化快
- 长窗口(60天): 更稳定，减少噪声

### 2. 调整中性化变量
根据需求添加更多风格变量：
```python
NEUTRAL_STYLE = ["market_cap", "beta", "roe_ttm", "financial_liability"]
```

### 3. 调整LightGBM参数
根据过拟合情况调整：
- 增大 `max_depth`: 模型更复杂
- 减小 `learning_rate`: 训练更慢但更稳定
- 调整 `subsample` 和 `colsample_bytree`: 防止过拟合

## 故障排查

### 问题1: 数据读取失败
- 检查OSS连接配置
- 确认数据路径正确
- 检查日期范围内是否有数据

### 问题2: LightGBM训练失败
- 检查是否有足够的数据（建议至少60天）
- 检查标签数据是否有NaN值
- 尝试减小 `LGB_ITER` 或增大 `LGB_ETA`

### 问题3: 内存不足
- 减少因子数量
- 缩短日期范围
- 使用更大的内存服务器

## 版本历史

- v1.0: 初始版本，支持40个有效因子
- 测试日期: 2024-07-26 ~ 2024-10-24
- 股票池: 小盘股指数(000510)

