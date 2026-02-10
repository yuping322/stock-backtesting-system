# 因子模块 main_factor.py 重构总结

## 概述

创建了 `main_factor.py` 作为因子检验的命令行入口，将画图、结果保存等非核心功能从 `factor.py` 中分离出来。

## 文件结构

```
/stock-backtesting-system/
├── main_factor.py                      # 主程序（命令行入口）
│   ├── 命令行参数解析
│   ├── 画图控制
│   ├── 结果保存
│   └── 调用 factor.py 核心功能
├── factor/
│   ├── factor.py                       # 核心模块（简化）
│   │   ├── FactorTester.run(plot)     # 接受画图参数
│   │   └── 核心因子检验能力
│   ├── factor_calculator.py          # 因子计算器
│   └── ...
```

## 主要改动

### 1. 创建 main_factor.py

**位置**: 项目根目录

**功能**:
- 完整的命令行参数解析
- 画图开关（默认 false）
- 画图模式选择（popup/save）
- 输出目录管理
- 自定义因子文件支持

**新增参数**:
- `--plot`: 是否画图 (true/false, 默认 false)
- `--plot-mode`: 画图模式 (popup/save, 默认 popup)
- `--output-dir`: 结果输出目录
- `--custom-factor-file`: 自定义因子文件路径
- `--custom-factor-name`: 自定义因子列名

### 2. 简化 factor.py

**改动**:
- `FactorTester.run()` 方法添加 `plot` 参数
- 画图功能变为可选：`if plot: al.tears.create_full_tear_sheet(...)`
- 保留所有核心因子检验能力

**核心能力保留**:
- 因子计算
- Alphalens 检验
- 自动打分
- 滚动监控
- IC/IR 计算

### 3. 架构分离

| 组件 | 职责 |
|------|------|
| `main_factor.py` | 命令行接口、画图控制、结果保存 |
| `factor/factor.py` | 核心因子检验能力 |
| `factor/factor_calculator.py` | 因子计算器接口 |

## 使用对比

### 旧方式（factor.py）

```bash
# 总是画图，无法控制
python factor/factor.py --start 2024-01-01 --end 2024-12-31
```

### 新方式（main_factor.py）

```bash
# 不画图（默认）
python main_factor.py --start 2024-01-01 --end 2024-12-31

# 画图弹窗
python main_factor.py --start 2024-01-01 --end 2024-12-31 --plot true --plot-mode popup

# 画图保存
python main_factor.py --start 2024-01-01 --end 2024-12-31 --plot true --plot-mode save --output-dir results/test
```

## 参数对比

### factor.py 支持的参数

```
--start, --end, --stock-pool, --factors, --quantiles, --periods,
--fillna, --winsorize, --neutralize, --standardize,
--roll-win, --monitor-csv, --last-only
```

### main_factor.py 支持的参数

**基础参数**（同 factor.py）:
```
--start, --end, --stock-pool, --factors, --quantiles, --periods,
--roll-win, --monitor-csv
```

**扩展参数**（新增）:
```
--plot           # 是否画图
--plot-mode      # 画图模式
--output-dir     # 输出目录
--custom-factor-file   # 自定义因子文件
--custom-factor-name   # 自定义因子列名
```

## 优势

### 1. 灵活性

- 默认不画图，适合批量运行
- 可按需开启画图功能
- 支持多种画图输出方式

### 2. 模块化

- 核心能力集中在 `factor.py`
- 扩展功能在 `main_factor.py`
- 职责清晰，易于维护

### 3. 用户体验

- 命令行参数更丰富
- 自动管理输出目录
- 更友好的文档和帮助信息

## 代码示例

### 从 Python 代码调用

```python
from factor.factor import FactorTester, CFG

# 创建配置
class Args:
    start = '2024-01-01'
    end = '2024-12-31'
    # ... 其他参数

cfg = CFG(Args())
tester = FactorTester(cfg)

# 不画图
tester.run(plot=False)

# 画图
tester.run(plot=True)
```

### 命令行调用

```bash
# 基本使用
python main_factor.py --start 2024-01-01 --end 2024-12-31

# 带画图
python main_factor.py --start 2024-01-01 --end 2024-12-31 --plot true

# 保存图表
python main_factor.py --start 2024-01-01 --end 2024-12-31 --plot true --plot-mode save --output-dir results/test
```

## 测试验证

- ✅ 命令行帮助信息正常
- ✅ 参数解析正确
- ✅ 画图控制功能正常
- ✅ 核心功能保持不变

## 文档

- [main_factor_usage.md](main_factor_usage.md): 使用文档
- [factor/README.md](../factor/README.md): 因子模块文档
- [factor_command_line.md](factor_command_line.md): 因子命令行文档

## 下一步

1. 添加更多画图选项（图表类型选择）
2. 支持结果导出到 Excel
3. 添加批量因子测试功能
4. 集成报告生成
