# Factor.py 重构总结

## 重构目标

将 `factor.py` 从硬编码配置模式重构为类似 `main.py` 的命令行模式，提高代码的灵活性和可维护性。

## 主要改动

### 1. 新增命令行参数解析

创建了 `parse_args()` 函数，使用 `argparse` 模块解析命令行参数：

```python
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='单因子/多因子 Alphalens 一键检验 + 自动打分',
        ...
    )
    # 添加各种参数
    ...
    return parser.parse_args()
```

### 2. 重构 CFG 类

**旧版本**（硬编码）:
```python
class CFG:
    START = '2024-09-25'
    END = '2025-10-14'
    STOCK_POOL = '000510.XSHG'
    FACTORS = ['VOL10', 'single_day_VPT_12']
    QUANTILES = 10
    PERIODS = [5, 10, 15]
    # ...
```

**新版本**（从命令行参数初始化）:
```python
class CFG:
    """配置类，从命令行参数初始化"""
    def __init__(self, args):
        self.START = args.start
        self.END = args.end
        self.STOCK_POOL = args.stock_pool
        self.FACTORS = args.factors
        self.QUANTILES = args.quantiles
        self.PERIODS = args.periods
        # ...
```

### 3. 更新所有函数签名

所有使用 CFG 配置的函数都更新为接受 `cfg` 参数：

- `rolling_monitor(factor_name, ic_series, tb_ret_series, period_days, cfg)`
- `quick_score(factor_name, factor_data, period_days, cfg)`

### 4. 更新 FactorTester 类

- `__init__` 方法现在接受 `cfg` 参数
- 所有方法中使用 `self.cfg` 访问配置，而不是全局 `CFG`
- 所有硬编码的分位数从 `10` 改为使用 `cfg.QUANTILES`

### 5. 新增主函数

```python
def main():
    """主函数"""
    args = parse_args()
    cfg = CFG(args)
    
    # 打印配置信息
    print("=" * 60)
    print("因子检验配置")
    print("=" * 60)
    # ...
    
    # 运行因子检验
    tester = FactorTester(cfg)
    tester.run()
```

## 支持的命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start` | str | '2024-09-25' | 回测开始日期 |
| `--end` | str | '2025-10-14' | 回测结束日期 |
| `--stock-pool` | str | '000510.XSHG' | 股票池 |
| `--factors` | list | ['VOL10', 'single_day_VPT_12'] | 因子列表 |
| `--quantiles` | int | 10 | 分组数量 |
| `--periods` | list | [5, 10, 15] | 调仓周期 |
| `--fillna` | int | 0 | 是否填充缺失值 |
| `--winsorize` | int | 0 | 是否异常值处理 |
| `--neutralize` | int | 0 | 是否中性化 |
| `--standardize` | int | 0 | 是否标准化 |
| `--roll-win` | int | 60 | 滚动窗口交易日数 |
| `--monitor-csv` | str | 'monitor.csv' | 监控结果CSV文件路径 |
| `--last-only` | flag | False | 只输出最新一期 |

## 使用示例

### 基本使用（默认配置）

```bash
python factor.py
```

### 自定义回测区间

```bash
python factor.py --start 2024-01-01 --end 2024-12-31
```

### 指定因子和调仓周期

```bash
python factor.py --factors VOL10 VSTD10 --periods 5 10 15
```

### 完整配置

```bash
python factor.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --stock-pool 000510.XSHG \
  --factors VOL10 single_day_VPT_12 \
  --quantiles 10 \
  --periods 5 10 15 \
  --roll-win 60 \
  --monitor-csv custom_monitor.csv
```

## 重构优势

1. **灵活性**: 无需修改代码即可更改配置
2. **可维护性**: 配置集中在命令行，代码更清晰
3. **一致性**: 与 `main.py` 的设计风格保持一致
4. **可复用性**: 可以轻松编写脚本批量运行不同配置
5. **用户友好**: 提供详细的帮助信息和示例

## 向后兼容性

- 所有原有配置值都作为默认值保留
- 功能完全保持不变，只是配置方式改变
- 运行 `python factor.py` 等同于使用旧的默认配置

## 测试验证

- ✅ 模块导入成功
- ✅ 命令行帮助信息正常显示
- ✅ 所有参数正确解析
- ✅ 没有 linter 错误

## 相关文件

- `factor.py`: 重构后的主文件
- `docs/factor_command_line.md`: 详细的使用文档
- `test_factor.py`: 测试示例脚本

## 下一步

1. 添加配置文件支持（如 YAML/JSON）
2. 添加更多因子类型
3. 支持因子组合和权重配置
4. 添加批量测试功能
5. 集成到 CI/CD 流程
