# Factor.py 命令行使用指南

## 概述

`factor.py` 已重构为命令行模式，所有配置参数都可以通过命令行参数指定，原有的配置值作为默认值。

## 基本用法

### 1. 使用默认配置

```bash
python factor.py
```

这会使用以下默认参数：
- 回测区间: 2024-09-25 ~ 2025-10-14
- 股票池: 000510.XSHG
- 因子: VOL10, single_day_VPT_12
- 分位数: 10
- 调仓周期: 5, 10, 15 天
- 滚动窗口: 60 天
- 监控文件: monitor.csv

### 2. 查看帮助信息

```bash
python factor.py --help
```

## 参数说明

### 回测区间

- `--start START`: 回测开始日期 (YYYY-MM-DD)
- `--end END`: 回测结束日期 (YYYY-MM-DD)

示例:
```bash
python factor.py --start 2024-01-01 --end 2024-12-31
```

### 股票池

- `--stock-pool STOCK_POOL`: 股票池，可以是指数代码或 "stock"（全市场）

示例:
```bash
python factor.py --stock-pool 000510.XSHG
python factor.py --stock-pool stock  # 全市场
```

### 因子

- `--factors FACTORS [FACTORS ...]`: 要检验的因子列表（可指定多个）

示例:
```bash
python factor.py --factors VOL10
python factor.py --factors VOL10 VSTD10 single_day_VPT_12
```

### Alphalens 参数

- `--quantiles QUANTILES`: 分组数量（默认: 10）
- `--periods PERIODS [PERIODS ...]`: 调仓周期（天），可指定多个（默认: 5 10 15）

示例:
```bash
python factor.py --quantiles 5
python factor.py --periods 5 10 15 20
```

### 数据清洗

- `--fillna FILLNA`: 是否填充缺失值 (0=否, 1=是，默认: 0)
- `--winsorize WINSORIZE`: 是否异常值处理 (0=否, 1=是，默认: 0)
- `--neutralize NEUTRALIZE`: 是否中性化 (0=否, 1=是，默认: 0)
- `--standardize STANDARDIZE`: 是否标准化 (0=否, 1=是，默认: 0)

示例:
```bash
python factor.py --fillna 1 --winsorize 1
```

### 滚动窗口和监控

- `--roll-win ROLL_WIN`: 滚动窗口交易日数（默认: 60）
- `--monitor-csv MONITOR_CSV`: 监控结果CSV文件路径（默认: monitor.csv）
- `--last-only`: 只输出最新一期（默认: False）

示例:
```bash
python factor.py --roll-win 30
python factor.py --monitor-csv my_monitor.csv
python factor.py --last-only
```

## 完整示例

### 示例 1: 基础使用

```bash
python factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10
```

### 示例 2: 多因子检验

```bash
python factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10 VSTD10 single_day_VPT_12 \
  --quantiles 10 \
  --periods 5 10 15
```

### 示例 3: 自定义参数

```bash
python factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool stock \
  --factors VOL10 \
  --quantiles 5 \
  --periods 3 5 10 \
  --roll-win 30 \
  --fillna 1 \
  --monitor-csv custom_monitor.csv
```

### 示例 4: 快速测试

```bash
python factor.py \
  --factors VOL10 \
  --periods 5 \
  --roll-win 20
```

## 输出说明

运行后会输出：

1. **配置信息**: 显示当前使用的配置参数
2. **因子检验结果**: 包括 IC、IR、收益、单调性等指标
3. **打分结果**: 对每个调仓周期的因子表现进行打分
4. **滚动监控**: 显示滚动 IC、IR、波动率等指标
5. **状态标识**: 
   - 🟢 alive: 因子表现良好
   - 🟡 warning: 因子表现一般，需注意
   - 🔴 dead: 因子失效

## 与旧版本对比

### 旧版本（硬编码配置）

```python
class CFG:
    START = '2024-09-25'
    END = '2025-10-14'
    STOCK_POOL = '000510.XSHG'
    FACTORS = ['VOL10', 'single_day_VPT_12']
    # ... 更多配置
```

需要修改代码才能更改配置。

### 新版本（命令行配置）

```bash
python factor.py --start 2024-01-01 --end 2024-12-31 --factors VOL10
```

无需修改代码，通过命令行参数即可更改配置。

## 注意事项

1. 所有日期格式必须为 `YYYY-MM-DD`
2. 股票池代码需要包含交易所后缀（如 `.XSHG` 或 `.XSHE`）
3. 因子名称必须与代码中定义的因子名称一致
4. 首次运行可能需要下载数据，请耐心等待
5. 回测区间越长，运行时间越长

## 故障排除

### 问题: 找不到模块

```bash
pip install -r requirements.txt
```

### 问题: 数据获取失败

检查网络连接，确保可以访问数据源。

### 问题: 因子计算失败

检查因子名称是否正确，以及是否有足够的历史数据。

## 更多信息

查看代码中的 `factor.py` 文件获取更多细节。
