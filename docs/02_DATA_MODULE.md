# 数据层 (Data Module) 指南

本文档梳理了系统的数据流转、存储结构以及核心接口。

## 1. 数据来源

系统主要支持三类数据源：
1.  **OSS (阿里单对象存储)**: 存储海量历史日线行情、快照（Spot）以及聚宽/新浪转存的财务报表。
2.  **AKShare**: 用于实时获取最新交易日的行情数据，补充历史数据的空白。
3.  **ModelScope**: 作为备用或补充的外部日线行情源。

## 2. 存储转换 (DuckDB 2.0)

目前系统正在从“纯文本/OSS 实时读取”向“本地 DuckDB 缓存”迁移：
- **`market.db`**: 存储 OHLCV 数据。
- **`factors.db`**: 存储计算好的因子值（长表 EAV 结构）。

## 3. 核心接口 (`src/data/data.py`)

### 行情获取
- `load_oss_stocks(codes, start, end)`: 获取指定股票的收盘价宽表。
- `load_oss_complex_stocks(...)`: 获取包含 OHLCV 的字典或宽表。
- `get_index_stocks(symbol, date)`: 获取指数成分股列表。

### 财务数据
- `get_history_fundamentals(security, fields, ...)`: 获取跨多张财务报表的综合数据。

### 工具函数
- `get_trading_dates(start, end)`: 生成 A 股交易日序列。

## 4. 注意事项

- **代码归一化**: 系统内部统一使用 6 位数字代码（如 `000001`）。
- **市场后缀**: 在与 OSS 交互时会自动补全 `.XSHG` 或 `.XSHE`。
- **环境变量**: 必须配置 `OSS_ACCESS_KEY_ID` 和 `OSS_ACCESS_KEY_SECRET` 才能访问历史数据。
