# TA-Lib 因子生成器

这个程序用于生成所有 TA-Lib 支持的技术指标因子列表，格式类似于 `alpha158_factors.txt`。

## 生成的文件

- `talib_factors.txt` - 包含所有 TA-Lib 因子的列表文件

## 因子命名规范

```
TALIB_{INDICATOR}_{PARAMS}
```

**示例：**
- `TALIB_SMA_20` → SMA(20) 简单移动平均线
- `TALIB_RSI_14` → RSI(14) 相对强弱指数  
- `TALIB_MACD_12_26_9` → MACD(12,26,9) MACD指标
- `TALIB_BBANDS_20_2_2` → BBANDS(20,2,2) 布林带

## 包含的指标类型

### 📈 移动平均线
- SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3

### 📊 动量指标  
- RSI, STOCH, STOCHF, STOCHRSI, WILLR, CCI, CMO, MOM, ROC, PPO, APO

### 📉 波动率指标
- ATR, NATR, TRANGE, ADX, ADXR, DX, PLUS_DI, MINUS_DI

### 💰 成交量指标
- AD, ADOSC, OBV, MFI

### 🎯 其他技术指标
- BBANDS (布林带), AROON, AROONOSC, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

## 使用方法

### 生成因子列表
```bash
python generate_talib_factors.py
```

### 在因子检验中使用
```bash
python factor.py --factors TALIB_SMA_20 TALIB_RSI_14 TALIB_MACD_12_26_9
```

## 统计信息

- **总计因子数量**: 216个
- **覆盖指标**: 40+ 个 TA-Lib 技术指标
- **参数组合**: 为常用周期生成合理的参数组合

## 注意事项

- 过滤掉了数学函数（如 sin, cos, exp 等）
- 过滤掉了K线形态识别函数（CDL系列）
- 每个指标只生成最常用的参数组合
- 因子名称使用大写，以 `TALIB_` 开头