# main_factor.py 结果返回说明

## 概述

`main_factor.py` 和 `factor.py` 的 `run()` 方法现在都会返回检验结果，包含了所有计算出的指标和统计信息。

## 返回结果结构

### FactorTestResult 类

```python
class FactorTestResult:
    factor_name: str           # 因子名称
    period: int               # 调仓周期
    scores: dict              # 各项得分 {(指标名, (值, 是否通过))}
    level: str                # 等级（优秀/良好/一般）
    ic_series: pd.Series      # IC 序列
    ret_series: pd.Series     # 收益序列
    top_turnover: float       # Top 组换手率
    rolling_monitor: dict     # 滚动监控数据
    status_flag: str          # 状态标识（🟢/🟡/🔴）
    clean_data: pd.DataFrame  # clean factor data
```

## 使用方式

### 方式 1: 命令行使用

```bash
# 运行并保存结果
python main_factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10 --output-dir results/test
```

结果会自动保存到输出目录。

### 方式 2: Python 代码调用

```python
from factor.factor import FactorTester, CFG, parse_args

# 创建配置
args = parse_args()
cfg = CFG(args)

# 运行检验
tester = FactorTester(cfg)
results = tester.run(plot=False)

# 使用结果
for result in results:
    print(f"因子: {result.factor_name}")
    print(f"周期: {result.period}天")
    print(f"等级: {result.level}")
    print(f"状态: {result.status_flag}")
    
    # 访问得分
    for key, (value, passed) in result.scores.items():
        print(f"  {key}: {value:.3f} {'✅' if passed else '❌'}")
    
    # 访问时间序列数据
    ic_series = result.ic_series
    ret_series = result.ret_series
    
    # 访问滚动监控数据
    roll_ic = result.rolling_monitor.roll_ic
    roll_ir = result.rolling_monitor.roll_ir
```

## 结果保存结构

```
results/
└── factor_test_20250101_143025/
    ├── summary.csv                          # 汇总结果
    ├── VOL10_period5/
    │   ├── ic_series.csv                   # IC 序列
    │   ├── ret_series.csv                  # 收益序列
    │   └── clean_data.csv                  # 完整数据
    ├── VOL10_period10/
    │   └── ...
    └── plot_1.png                          # 图表（如果画图）
```

## summary.csv 格式

| factor_name | period | level | status_flag | IC均值_value | IC均值_passed | IR比率_value | ... |
|-------------|--------|-------|-------------|--------------|---------------|--------------|-----|
| VOL10 | 5 | 优秀 | 🟢 alive | 0.08 | True | 1.5 | ... |
| VOL10 | 10 | 良好 | 🟡 warning | 0.06 | False | 1.2 | ... |

## 示例：分析结果

```python
import pandas as pd
from factor.factor import FactorTester, CFG, parse_args

# 运行检验
args = parse_args()
cfg = CFG(args)
tester = FactorTester(cfg)
results = tester.run(plot=False)

# 分析结果
for result in results:
    # 筛选优秀的因子
    if result.level == '优秀':
        print(f"优秀因子: {result.factor_name} (周期{result.period}天)")
    
    # 分析 IC 序列
    ic_mean = result.ic_series.mean()
    ic_std = result.ic_series.std()
    print(f"IC均值: {ic_mean:.3f}, IC标准差: {ic_std:.3f}")
    
    # 检查最近表现
    recent_ic = result.ic_series.tail(10)
    positive_rate = (recent_ic > 0).mean()
    print(f"最近10期IC为正比例: {positive_rate:.1%}")
```

## 结果字段说明

### scores 字典

包含各项得分指标：

- `IC均值`: IC的均值
- `IR比率`: IC均值/IC标准差
- `多空年化`: Top-Bottom 年化收益
- `单调性`: 分位数收益的单调性
- `Top换手率`: Top组的换手率
- `IC半衰期(τ)`: IC的半衰期
- `Q5-Q1夏普`: Q5-Q1的夏普比率
- `扣费净IR`: 扣费后的净IR
- `容量(亿元)`: 容量指标

每个指标返回 `(值, 是否通过阈值)`

### rolling_monitor 字典

包含滚动监控数据：

- `roll_ic`: 滚动IC均值
- `roll_ir`: 滚动IR
- `roll_t`: 滚动t统计量
- `top_std`: Top组波动率
- `neg_day`: 负日比例

### status_flag

状态标识：

- `🟢 alive`: 因子表现良好
- `🟡 warning`: 因子表现一般，需注意
- `🔴 dead`: 因子失效

## 导出到 Excel

```python
import pandas as pd

# 运行检验
results = tester.run(plot=False)

# 转换为 DataFrame
summary_data = []
for result in results:
    row = {
        'factor_name': result.factor_name,
        'period': result.period,
        'level': result.level,
        'status_flag': result.status_flag,
    }
    for key, (value, passed) in result.scores.items():
        row[key] = value
    summary_data.append(row)

df = pd.DataFrame(summary_data)
df.to_excel('factor_results.xlsx', index=False)
```

## 批量分析多个因子

```python
factors_to_test = ['VOL10', 'VOL20', 'RSI_14', 'MA_5']

all_results = []
for factor in factors_to_test:
    args.factors = [factor]
    cfg = CFG(args)
    tester = FactorTester(cfg)
    results = tester.run(plot=False)
    all_results.extend(results)

# 分析所有结果
df = pd.DataFrame([r.to_dict() for r in all_results])
print(df.groupby('factor_name')['level'].value_counts())
```

## 参考

- [main_factor_usage.md](main_factor_usage.md): 使用文档
- [factor/factor.py](../factor/factor.py): 核心实现
