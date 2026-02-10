# `data.py` 对象存储接口说明

本文档梳理 `data.py` 中与对象存储（OSS）及外部数据源交互的主要函数，帮助快速了解数据来源、文件格式、入参 / 出参及常见注意事项。

---

## 1. 通用配置与依赖

| 项目 | 描述 |
| --- | --- |
| 认证信息 | 需设置环境变量 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`。可选：`OSS_ENDPOINT`（默认 `https://oss-cn-hangzhou.aliyuncs.com`）、`OSS_BUCKET_NAME`（默认 `test123432`）。|
| 代码归一化 | `_normalize_code_arg` 将输入股票编码统一为 6 位数字字符串；`_ensure_exchange_prefix` / `_ensure_exchange_suffix` 在需要时补上 `sh/sz/bj` 前缀或 `.XSHG/.XSHE/.XBJ` 后缀。|
| 日历依赖 | 若安装 `chinese_calendar` 会使用其 `is_workday`；否则退化为简单工作日判定（周一至周五）。|
| 错误处理 | OSS 相关函数对 `NoSuchKey` 等异常提供日志记录；若文件缺失返回空 DataFrame；解析出错（空文件、列缺失）会抛出 `ValueError` 以便快速定位。|

---

## 2. 行情快照与日线数据

### 2.1 `load_new_stocks`
- **作用**：按日期区间拉取 `stock_zh_a_spot_em/` 目录下的每日快照（今开价），拼成宽表。
- **OSS 路径**：`stock_zh_a_spot_em/stock_zh_a_spot_em_YYYYMMDD_HHMM.csv`
- **文件格式**：CSV，至少包含 `代码`、`今开` 列。
- **入参**：
  - `codes`: 单个或多个股票代码（可为 6 位、带前缀等）；可选。
  - `start`, `end`: 支持 `str`/`date`/`datetime`；默认区间为 2000-01-01 至今天。
- **返回值**：`pd.DataFrame`，`index` 为日期，列为 6 位股票代码，值为对应的今开价。
- **备注**：
  - 若指定 `codes`，会在读取每个文件后根据 `代码` 字段过滤。
  - 同一天存在多个快照时取时间戳最大的文件。

### 2.2 `load_bt_oss_stocks`
- **作用**：同样遍历快照目录，但保留原始快照长表，便于 Backtrader 使用。
- **OSS 路径 / 文件格式**：同 `load_new_stocks`，要求 CSV 中至少包含 `代码`、`今开`、`最高`、`最低`、`最新价`、`成交量`。
- **入参**：同 `load_new_stocks`。
- **返回值**：`pd.DataFrame`，每行包含快照原始字段及 `date`（从文件名解析）。

### 2.3 `_wide_to_ohlcv`
- **作用**：将 `load_bt_oss_stocks` 的长表或宽表快照转成 Backtrader 需要的 OHLCV 结构。
- **输入**：
  - 支持两种 DataFrame：
    1. 快照 CSV 长表（列包含 `代码`、`今开`、`最高`、`最低`、`最新价`、`成交量`）。
    2. 宽表（index=日期, columns=股票代码, values=价格）。
- **输出**：`pd.DataFrame`，包含列 `date`、`asset`、`open`、`high`、`low`、`close`、`volume`。

### 2.4 `load_bt_stocks`
- **作用**：基于快照生成 Backtrader 的 `bt.feeds.PandasData`。
- **依赖**：调用 `load_bt_oss_stocks` → `_wide_to_ohlcv`，不直接访问 OSS。
- **入参**：股票代码列表、起止日期。
- **返回值**：`Dict[str, bt.feeds.PandasData]`，key 为 6 位代码。

### 2.5 `load_bt_pricing`
- **作用**：将 `load_bt_stocks` 得到的 feeds 拼成价格宽表。
- **返回值**：`pd.DataFrame`，列为股票代码，值为 `close`。

### 2.6 `load_oss_stocks`
- **作用**：从日线行情目录读取指定股票的收盘价，拼成宽表。
- **OSS 路径**：`hangqing/daily_data/{sh|sz|bj}{code}.csv`
- **文件格式**：CSV，需包含 `日期`、`close` 列。
- **入参**：
  - `codes`: 股票代码列表（必填）。
  - `start`, `end`: 起止日期，默认 2000-01-01 至今天。
- **返回值**：`pd.DataFrame`（宽表）。
- **备注**：
  - 目录中缺失某只股票时仅打印日志并跳过。
  - 同一日期重复记录会保留最后一次出现的数据。

### 2.7 `get_valuation`
- **作用**：读取单只股票估值（日线行情）并按日期过滤。
- **OSS 路径**：同 `load_oss_stocks`。
- **输出**：按日期倒序排序的 `pd.DataFrame`，并保持 `日期` 列为字符串。

### 2.8 `load_modelscope_stocks`（HTTP 数据源）
- **作用**：从 ModelScope 数据集下载 `{sh|sz|bj}{code}.npy`（内容为 CSV），获取 `date/close`。
- **文件格式**：CSV，常见列 `日期`、`symbol`、`open`、`high`、`low`、`close`。
- **返回值**：收盘价宽表。
- **备注**：不依赖 OSS bucket，但需要网络访问和 `requests`。

### 2.9 `load_modelscope_complex_stocks`
- **作用**：同上，但可返回多个字段或者完整字段字典。
- **返回值**：
  - `fields="close"`（默认）→ 收盘价宽表。
  - `fields="all"` → `dict[str, DataFrame]`。
  - `fields=list[str]` → `dict[str, DataFrame]`（指定字段）。

---

## 3. 因子与财务数据

### 3.1 `read_factor_data`
- **作用**：从 OSS `uploads/` 目录按日期逐日读取因子结果，拼成 `(date, code)` MultiIndex。
- **OSS 路径**：`uploads/{year}/factors_YYYYMMDD_all.csv`
- **文件格式**：CSV，行索引为股票代码（含 `.XSHG/.XSHE` 后缀），列为各因子。
- **入参**：
  - `codes`: 股票列表（可选，若提供会自动补后缀并过滤）。
  - `start_date`, `end_date`: 日期字符串。
  - `factors`: 欲保留的列名列表（可选）。
  - `base_path`: OSS 目录前缀（默认 `uploads`）。
- **返回值**：`pd.DataFrame`，`index` 为 `(date, code)`，列为因子。

### 3.2 `read_factor_data_loal`
- **作用**：从本地目录读取同样结构的因子 CSV，不经过 OSS。
- **文件格式**：同上。

### 3.3 `factor_for_al`
- **作用**：获取单个因子列并转换成 Alphalens 需要的 `(date, asset)` Series（资产代码为 6 位数字）。
- **依赖**：`read_factor_data`。
- **返回值**：`pd.Series`，`index.names == ["date", "asset"]`。

### 3.4 财报函数 `get_balance` / `get_income` / `get_cashflow`
- **作用**：读取三张财务报表（资产负债表、利润表、现金流量表）。
- **OSS 路径**：
  - 资产负债表：`jukuan/stock_financial_report_sina/{sh|sz|bj}{code}.csv`
  - 利润表：`jukuan/stock_financial_report_sina_lirun/{sh|sz|bj}{code}.csv`
  - 现金流量表：`jukuan/stock_financial_report_sina_xianjinliu/{sh|sz|bj}{code}.csv`
- **文件格式**：CSV，包含 `报告日`、`类型` 等中文列。
- **入参**：
  - `code`: 股票代码。
  - `report_type`: 默认 `"合并期末"`。
  - `date`: 可选，若传入则筛选 `报告日` 不晚于该日期的记录。
- **返回值**：按照报告日期倒序的 `pd.DataFrame`。

### 3.5 `get_history_fundamentals`
- **作用**：综合调用上面三个财务函数，将指定字段（例如 `balance.total_assets`）整合成 MultiIndex 表。
- **入参**：
  - `security`: 单个或多个股票代码。
  - `fields`: 形如 `balance.total_assets` 的字段列表。
  - `stat_date`: 例如 `"2020q1"`，可选。
  - `report_type`, `count`: 报表类型、取数条数。
- **返回值**：`pd.DataFrame`，`index` 为 `(code, statDate)`。

### 3.6 `print_table_columns`
- **作用**：打印指定财务表 CSV 的表头；用于排查字段命名。
- **依赖**：同财报函数。

---

## 4. 指数相关

### 4.1 `_load_index_df`
- **作用**：遍历 `index/` 目录下包含指数代码的 CSV，拼接成一个 DataFrame。
- **OSS 路径**：`index/{index_symbol}_*.csv`
- **文件格式**：CSV，需包含列 `品种代码`（或 `code`）、`指数纳入日期`（或 `in_date`）。

### 4.2 `get_index_stocks`
- **作用**：基于 `_load_index_df`，按给定日期返回已纳入指数的股票列表。
- **入参**：`index_symbol`, `as_of`（可选）。
- **返回值**：`List[str]`，6 位股票代码。

### 4.3 `get_index_daily`
- **作用**：读取指数日行情并计算净值序列（相对起始日归一化为 1）。
- **OSS 路径**：`stock_zh_index_daily/{index_symbol}_*.csv`
- **文件格式**：CSV，至少 `date`, `close` 列。
- **返回值**：`pd.Series`，索引为日期，值为净值。

---

## 5. 其他函数

### 5.1 `load_code2name`
- **作用**：读取本地或 OSS 中的代码-名称映射文件（默认 `data/merged_weights.csv`，可通过常量 `MAPPING_FILE` 指定）。
- **文件格式**：CSV，列 `code`、`name`。
- **返回值**：`Dict[str, str]`。

### 5.2 `get_trading_dates`
- **作用**：根据工作日规则生成交易日序列，不访问 OSS。
- **入参**：`start`, `end`, `as_str`。
- **返回值**：`List[date]` 或 `List[str]`。

### 5.3 `handler`
- **作用**：云函数入口示例。流程：
  1. 解析事件参数，确定股票列表与因子。
  2. 调用 `factor_for_al`、`load_modelscope_stocks`、Alphalens 指标函数计算 IC。
  3. 将结果传给 `save_result` 保存到 `daily_metrics/daily_metrics.csv`。
- **入参**：`event`, `context`（云函数标准签名）。
- **输出**：包含各类指标的 `dict`。

### 5.4 `save_result`
- **作用**：将一条指标结果追加/更新到 OSS 上的 `daily_metrics/daily_metrics.csv`。
- **文件格式**：CSV，以 `trade_date,factor_name` 为复合主键。
- **入参**：
  - `bucket`: 当前 OSS bucket 对象。
  - `date_tag`: 字符串形式的日期。
  - `res_dict`: 包含 `factor_name`、`IC_mean` 等字段的字典。
- **返回值**：无。

---

## 6. 常用排查流程
1. **验证凭证**：运行任一读取函数，若日志提示“OSS 访问凭证未配置”，说明环境变量缺失。
2. **列出对象**：可临时使用 `oss2.ObjectIterator(bucket, prefix=...)` 打印真实文件名，确认命名是否符合约定。
3. **检查字段**：一旦抛出 `ValueError`（例如“文件解析失败”“缺少列”），优先检查 CSV 表头是否覆盖文档中列名。
4. **网络依赖**：ModelScope 请求失败时会抛出 `ConnectionError`，需确认外网可用。

---

## 7. 术语速查
- **快照**：开盘前或盘中全市场行情快照，来自 `stock_zh_a_spot_em` 目录。
- **日线行情**：按交易日聚合的收盘价数据，来自 `hangqing/daily_data`。
- **因子数据**：每日预先计算的多因子指标，按日期切分存储在 `uploads/year/` 子目录。
- **财报**：来自聚宽/新浪接口的财务报表原始 CSV。

如需添加新的数据源，可参考以上规范，确保文件命名和列名与读取逻辑保持一致，或扩展 `_normalize_*` 方法以兼容新的命名规则。
