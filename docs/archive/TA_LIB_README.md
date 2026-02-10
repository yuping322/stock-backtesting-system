# TA-Lib 技术指标检验程序

这个程序用于检验和可视化各种技术分析指标，使用 TA-Lib 库计算常见的技术指标。

## 文件结构

### Python 程序
- `talib_indicator_analysis.py` - 主要的 Python 程序，包含所有核心逻辑
  - 数据加载（使用 AKShare 获取上证指数数据）
  - 技术指标计算（10个常用指标）
  - 统计信息输出
  - 可视化图表生成
  - 详细报告生成

### Shell 脚本
- `scripts/run_talib_analysis.sh` - 启动脚本
  - 检查依赖项（TA-Lib 和 AKShare）
  - 调用 Python 程序

## 检验的技术指标

1. **SMA** - 简单移动平均线 (Simple Moving Average)
2. **EMA** - 指数移动平均线 (Exponential Moving Average)
3. **RSI** - 相对强弱指数 (Relative Strength Index)
4. **MACD** - MACD指标 (Moving Average Convergence Divergence)
5. **BBANDS** - 布林带 (Bollinger Bands)
6. **STOCH** - 随机震荡指标 (Stochastic Oscillator)
7. **WILLR** - 威廉指标 (Williams %R)
8. **CCI** - 顺势指标 (Commodity Channel Index)
9. **MFI** - 资金流量指标 (Money Flow Index)
10. **ROC** - 变动率指标 (Rate of Change)

## 使用方法

### 方式1：运行 Shell 脚本
```bash
./scripts/run_talib_analysis.sh
```

### 方式2：直接运行 Python 程序
```bash
python talib_indicator_analysis.py
```

## 输出结果

程序会在 `results/` 目录下创建一个时间戳命名的文件夹，包含：
- `talib_indicators_chart.png` - 技术指标可视化图表
- `README.md` - 详细的检验报告

## 依赖项

- TA-Lib: `pip install TA-Lib`
- AKShare: `pip install akshare`
- pandas, numpy, matplotlib

## 注意事项

- macOS 用户需要先安装 TA-Lib 系统依赖：`brew install ta-lib`
- 程序会自动获取上证指数的最近一年数据作为测试数据
- 如果网络问题导致数据获取失败，会生成模拟数据进行测试