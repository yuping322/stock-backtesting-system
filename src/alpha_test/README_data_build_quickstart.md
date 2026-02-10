# Build Minimal Qlib Data From Raw CSV & Run LightGBM Quick Start

本示例展示如何从原始的日频 CSV 构造一个最小可用的 qlib 数据目录，然后运行快速 LightGBM 例程。

## 1. 原始 CSV 格式
位于 `examples/data_raw/` 目录下，每行：
```
date,symbol,open,high,low,close,vwap,volume
2024-12-30,AAA,10.0,10.5,9.8,10.3,10.25,120000
...
```
如果缺少 vwap 列，会自动用 close 代替。

示例文件：`examples/data_raw/fake_prices.csv` 已提供两只股票 AAA / BBB 几天数据。

## 2. 转换为 qlib 数据结构
运行：
```bash
python examples/build_qlib_from_csv.py --src examples/data_raw/fake_prices.csv --dest examples/mini_qlib_data
```
或批量：
```bash
python examples/build_qlib_from_csv.py --src examples/data_raw --dest examples/mini_qlib_data
```
生成目录：
```
examples/mini_qlib_data/
  calendars/day.txt
  instruments/all.txt
  features/aaa/open.day.bin ...
  features/bbb/open.day.bin ...
```

## 3. 初始化并运行快速模型
```bash
python examples/quick_start_alpha_workflows.py --data examples/mini_qlib_data --show-alpha158
```
可选打印完整因子集合示例：
```bash
python examples/quick_start_alpha_workflows.py --data examples/mini_qlib_data --show-alpha158 --full-alpha158
```

输出包括：
- Alpha158 / Alpha360 两个任务的训练 & IC 摘要。
- 记录器 artifacts 路径。

## 4. 常见问题
| 问题 | 原因 | 解决 |
|------|------|------|
| IC 为 NaN | 数据天数太少或标签无波动 | 增加日期跨度或使用真实数据集 |
| 找不到 features | converter 未成功运行 | 检查 `examples/mini_qlib_data/features` 是否生成文件 |
| KeyError: symbol | CSV 中列名不匹配 | 确保列名为 `symbol` 而不是 `code` |
| ValueError 缺列 | CSV 缺少必要列 | 补齐 open/high/low/close/volume |

## 5. 后续扩展
- 增加更多股票与日期，重复运行转换脚本。
- 在 quick start 中加入 `PortAnaRecord` 做组合回测。
- 修改 `build_qlib_from_csv.py` 支持分钟频（需调整 calendars 文件名为 `1min.txt` 等）。

欢迎继续扩展！
