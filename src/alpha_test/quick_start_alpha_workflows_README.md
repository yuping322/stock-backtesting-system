# Quick Start: Alpha158 & Alpha360 LightGBM

本示例脚本 `quick_start_alpha_workflows.py` 用于快速在 Alpha158 与 Alpha360 两个数据集上训练一个 LightGBM 模型，输出基础信号分析(IC)。比 benchmarks 里的完整 workflow 更轻量。

## 前置条件
1. 已安装并初始化 qlib 数据（如未下载）：
```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```
2. 安装依赖：
```bash
pip install lightgbm qlib
```
   （如果在仓库根目录已执行过 `pip install -e .[all]` 可以跳过）

## 运行脚本
```bash
python examples/quick_start_alpha_workflows.py --data ~/.qlib/qlib_data/cn_data
```
可选参数：
- `--seed` 设置随机种子（当前脚本中主要影响 LightGBM 的内部随机性）。

## 输出内容
运行后会看到：
- Alpha158 任务日志：训练、IC 统计、artifact 路径。
- Alpha360 任务日志：训练、IC 统计、artifact 路径。
- 在默认 mlflow 跟踪目录（通常是 `mlruns/` 或 qlib 配置指定路径）下创建两个 experiment：`quick_lightgbm_alpha158` 与 `quick_lightgbm_alpha360`。

## 与 Benchmark Workflow 的差异
| 项目 | Benchmark YAML | Quick Start 脚本 |
|------|----------------|------------------|
| 回测/组合分析 | 包含 `PortAnaRecord` | 省略（更快） |
| 参数调优 | 预设超参 | 固定简化参数 |
| 多次运行统计 | 需要脚本批量运行 | 单次演示 |
| 记录内容 | 信号+组合+多指标 | 仅信号+IC |

## 扩展建议
- 增加 `PortAnaRecord` 以产生策略收益曲线：
```python
from qlib.workflow.record_temp import PortAnaRecord
# config 参考 benchmarks workflows
port_rec = PortAnaRecord(config=...)
port_rec.generate()
```
- 运行多次收集均值与标准差：外层写一个循环调用脚本。
- 替换模型为 `XGBModel`, `CatBoostModel` 或深度模型（需安装对应依赖）。

## 常见问题
1. IC 全是 NaN：检查数据是否下载完整；确保训练区间有标签。
2. ImportError: lightgbm：未安装 `lightgbm` 包。
3. 提示找不到数据：确认 `--data` 路径与 `qlib.init` 一致。

## 后续
若需要同时加入回测与收益指标，可直接改脚本加入 `PortAnaRecord`；或者使用 `qrun` 运行 benchmarks 目录下的 YAML。

祝使用愉快！
