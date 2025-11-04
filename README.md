# 股票回测系统

## 环境配置

### 环境变量设置
在使用本系统前，需要配置OSS访问凭证。复制 `env.example` 文件为 `env.sh` 并填入您的配置信息：

```bash
cp env.example env.sh
```

编辑 `env.sh` 文件，填入您的OSS访问凭证：

```bash
# OSS配置
OSS_ACCESS_KEY_ID=your_access_key_id_here
OSS_ACCESS_KEY_SECRET=your_access_key_secret_here
OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
OSS_BUCKET_NAME=your_bucket_name_here

# AkShare OSS配置
AKSHARE_OSS_ACCESS_KEY_ID=your_akshare_access_key_id_here
AKSHARE_OSS_ACCESS_KEY_SECRET=your_akshare_access_key_secret_here
```

**注意**: `env.sh` 文件包含敏感信息，不会被提交到Git仓库中。

## 项目概述

这是一个完整的股票回测系统，基于Backtrader和Streamlit构建，支持多种交易策略、实时数据获取和丰富的分析功能。

## 功能特性

### 🎯 核心功能
- **多策略支持**: 加权TopN、等权重、动量策略
- **多基准对比**: 支持沪深300、上证指数、深证成指、创业板指、中证500、上证50
- **实时数据**: 集成AKShare获取实时股票和指数数据
- **灵活配置**: 参数可配置，支持配置文件管理
- **丰富可视化**: 多种图表展示方式

### 📊 分析功能
- **净值分析**: 策略与基准净值对比、相对收益
- **收益分析**: 日收益分布、月度/年度统计
- **风险分析**: 回撤分析、VaR、CVaR、夏普比率
- **持仓分析**: 每日持仓详情、权重分布
- **交易记录**: 详细交易历史
- **Alpha分析**: 集成Alphalens进行因子分析

### ⚙️ 技术特性
- **Backtrader引擎**: 基于成熟的Backtrader回测框架
- **配置分离**: 系统配置与策略配置分离
- **结果缓存**: 支持结果缓存，提高重复运行效率
- **参数化缓存**: 缓存键包含所有参数，确保参数变化时重新计算
- **真实数据验证**: 使用AKShare实时数据

## 文件结构

```
/stock-backtesting-system/
├── app.py                              # Streamlit 回测界面入口
├── backtrader_base_strategy.py         # Backtrader基础策略
├── data.py                             # 数据处理模块
├── config.py                           # 配置文件
├── main.py                             # 命令行回测接口
├── main_factor.py                      # 因子检验命令行入口（带画图功能）
├── factor/                             # 因子检验模块
│   ├── factor.py                      # 因子检验核心程序
│   ├── factor_calculator.py           # 因子计算器接口
│   ├── example_custom_factor.py      # 自定义因子示例
│   └── README.md                       # 使用说明
├── requirements.txt                    # 依赖包列表
├── scripts/                           # 运行脚本目录
│   ├── start.sh                      # 启动脚本
│   ├── run_all_factors.sh            # 运行所有因子测试
│   ├── run_first10_factors.sh        # 运行前10个因子测试
│   └── run_*.sh                      # 其他运行脚本
├── data/                               # 数据目录
│   ├── *.csv                         # 股票数据CSV文件
│   ├── best_config_*.json            # 最优配置缓存
│   ├── *predicted_tomorrow_result.json # 预测结果
│   └── selected_stocks_*_result.json  # 选股结果
├── logs/                               # 日志目录
├── docs/                               # 文档目录
│   ├── factor_command_line.md        # 因子检验命令行文档
│   ├── factor_refactoring_summary.md  # 因子模块重构说明
│   └── ...                            # 其他文档
└── README.md                           # 说明文档
```

## 数据源

- **AKShare**: 实时获取股票和指数数据
- **本地CSV**: 支持导入本地股票数据文件
- **缓存机制**: 自动缓存计算结果提高效率

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行系统
```bash
# 方法一：使用启动脚本（推荐）
./start.sh

# 方法二：直接运行
streamlit run app.py
```

### 3. 配置参数
- 选择回测文件或使用实时数据
- 设置初始资金和交易费用
- 选择策略和参数
- 选择基准指数
- 选择分析内容

### 4. 查看结果
- 综合摘要
- 详细分析（多标签页）
- 丰富的可视化图表
- Alpha因子分析

### 5. 因子检验（命令行）

#### 方式 1: 使用 main_factor.py（推荐）

`main_factor.py` 提供画图、结果保存等扩展功能：

```bash
# 基本使用（不画图）
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10

# 画图并弹窗显示
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --plot-mode popup

# 画图并保存到文件
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --plot true --plot-mode save --output-dir results/factor_test

# 查看帮助
python main_factor.py --help
```

#### 方式 2: 使用 factor/factor.py（核心模块）

```bash
# 基本使用
python factor/factor.py

# 指定参数
python factor/factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10

# 查看帮助
python factor/factor.py --help
```

详细使用说明请参考：
- [docs/main_factor_usage.md](docs/main_factor_usage.md) - main_factor.py 使用文档
- [factor/README.md](factor/README.md) - 因子模块文档

## 策略说明

### 加权TopN策略
- 按权重选择TopN股票
- 支持自定义权重列
- 定期调仓

### 等权重策略
- 等权重配置股票
- 限制股票数量
- 定期调仓

### 动量策略
- 基于动量因子选股
- 选择动量最强的股票
- 支持自定义动量周期

## 性能指标

### 收益指标
- 总收益率、年化收益率
- Alpha、Beta
- 信息比率

### 风险指标
- 最大回撤、波动率
- VaR (95%)、CVaR (95%)
- 跟踪误差

### 效率指标
- 夏普比率、卡玛比率
- 月度胜率
- 连续盈利/亏损

## 技术栈

- **Python 3.7+**: 主要编程语言
- **Backtrader**: 回测引擎
- **Streamlit**: Web界面
- **Pandas/NumPy**: 数据处理

## 机器学习因子建模管线（ML Pipeline）

新增支持特征预处理与分组子模型融合，命令行入口 `python -m ml.ml_pipeline`，核心步骤包括：

1. Baseline 建模：对全部因子训练单一模型并输出 SHAP 重要性。
2. Groups 分组：使用 Spearman 相关 + 层次聚类对高重要性因子进行分组，限制最大组数。
3. Submodels 分组子模型：针对每个因子组训练独立模型，按近期 IC (信息系数) 加权融合。
4. All：一次性执行上述全流程。

### 预处理功能
通过 `FeaturePreprocessor` 实现以下步骤（仅基于训练窗口统计，避免数据泄漏）：
- 缺失值填充：均值或中位数 (`--impute-method mean|median`)，使用 `--no-impute` 可禁用。
- 标准化：Z-score (`--no-standardize` 禁用)。
- 行业中性化：按行业去均值 (`--neutralize-industry`)，默认行业列名 `industry` 可用 `--industry-col` 修改。

### 新增命令行参数
```
--no-impute                禁用缺失值填充（默认启用 mean 填充）
--impute-method {mean,median}  选择填充方法（默认 mean）
--no-standardize           禁用标准化（默认启用）
--neutralize-industry      行业中性化（按行业去均值）
--industry-col INDUSTRY    指定行业列名（默认 industry）
```

### IC 计算修复
已修复子模型融合阶段 IC 误用训练集标签的问题，现使用测试窗口真实标签计算 Spearman IC，避免信息泄漏。

### 输出文件示例
- `baseline_predictions.csv` / `baseline_full_scores.csv`
- `shap_importance.csv`
- `factor_groups.json`
- `group_model_predictions.csv` / `group_model_full_scores.csv` / `group_component_scores.csv`

### 快速示例
```
python -m ml.ml_pipeline --factor-file data/factor_values_sample.csv --mode all --top-n 20 \
	--neutralize-industry --impute-method median
```

若缺少标签列可使用 `--synthetic-label` 生成合成标签用于流程验证。若未安装 xgboost，管线会回退到线性回归用于测试环境运行。

- **Matplotlib/Seaborn**: 数据可视化
- **AKShare**: 金融数据获取
- **Alphalens**: 因子分析

## 系统特点

1. **模块化设计**: 配置、策略、引擎分离
2. **可扩展性**: 易于添加新策略和分析器
3. **用户友好**: 直观的Streamlit界面
4. **功能完整**: 从数据加载到结果展示全流程
5. **实时数据**: 支持AKShare实时数据获取
6. **专业分析**: 集成Alphalens因子分析

## 运行环境

- Python 3.7+
- 参见requirements.txt完整依赖列表

## 注意事项

1. 需要网络连接以获取AKShare实时数据
2. 数据文件需要符合指定格式
3. 缓存功能默认开启，可通过参数关闭
4. 首次运行需要下载数据，可能较慢
5. 启动脚本包含OSS环境变量配置，可根据需要修改

## 扩展开发

可以通过以下方式扩展系统：
1. 继承 `BaseStrategy` 类添加新策略
2. 扩展配置文件添加新参数
3. 添加新的分析器和指标
4. 集成其他数据源（Tushare、Baostock等）
5. 添加机器学习模型
6. 支持更多技术指标和因子
7. 添加风险控制模块
8. 集成实时交易接口

---

**🎉 系统功能完整，支持实时数据回测和专业分析！**