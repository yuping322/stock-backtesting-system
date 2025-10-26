# data.py 接口文档

## 概述
`data.py` 是数据处理核心模块，包含股票数据加载、财务报表获取、因子分析等功能。

---

## 数据结构类

### 1. `DateRange`
**功能**: 日期范围过滤
```python
@dataclass(frozen=True)
class DateRange:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    
    def apply(self, frame: pd.DataFrame, column: str = "date") -> pd.DataFrame
```

### 2. `OHLCVRecord`
**功能**: OHLCV数据记录
```python
class OHLCVRecord(TypedDict):
    date: pd.Timestamp
    asset: str
    open: float
    high: float
    low: float
    close: float
    volume: float
```

### 3. `FactorResultRow`
**功能**: 因子分析结果
```python
class FactorResultRow(TypedDict, total=False):
    trade_date: str
    factor_name: str
    IC_mean: float
    ICIR: float
    FactorReturn_mean: float
    QuantileMeanReturn: float
```

### 4. `FinancialQuery`
**功能**: 财务查询请求
```python
@dataclass(frozen=True)
class FinancialQuery:
    code: str
    report_type: str
    table: Literal["balance", "income", "cashflow"]
    date: Optional[pd.Timestamp] = None
```

---

## 工具函数

### 5. `_normalize_date_arg()`
**功能**: 日期参数规范化
```python
def _normalize_date_arg(
    value: Union[str, dt_date, dt_datetime, pd.Timestamp, None],
    *,
    default: Union[str, dt_date, dt_datetime, pd.Timestamp, None] = None,
    as_date: bool = False,
) -> Optional[Union[pd.Timestamp, dt.date]]
```

### 6. `_normalize_code_arg()`
**功能**: 股票代码规范化
```python
def _normalize_code_arg(
    codes: Union[str, int, Iterable[Union[str, int]], None],
    *,
    allow_none: bool = True,
    deduplicate: bool = True,
) -> Optional[List[str]]
```

### 7. `_ensure_exchange_prefix()`
**功能**: 补全交易所前缀 (sh/sz/bj)
```python
def _ensure_exchange_prefix(code: Union[str, int]) -> str
```

### 8. `_ensure_exchange_suffix()`
**功能**: 补全交易所后缀 (.XSHG/.XSHE/.XBJ)
```python
def _ensure_exchange_suffix(code: Union[str, int]) -> str
```

---

## OSS数据加载

### 9. `load_new_stocks()`
**功能**: 从OSS加载快照数据
```python
def load_new_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**返回**: DataFrame(index=date, columns=股票代码, values=今开)

### 10. `load_oss_stocks()`
**功能**: 从OSS加载日线行情
```python
def load_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**返回**: DataFrame(index=date, columns=股票代码, values=收盘价)

### 11. `read_factor_data()`
**功能**: 从OSS读取因子数据
```python
def read_factor_data(
    codes: Optional[List[str]] = None,
    start_date: str = None,
    end_date: str = None,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"
) -> pd.DataFrame
```
**返回**: MultiIndex DataFrame [(date, code), factor_columns]

### 12. `read_factor_data_loal()`
**功能**: 从本地读取因子数据
```python
def read_factor_data_loal(
    codes: List[str],
    start_date: str,
    end_date: str,
    factors: Optional[List[str]] = None,
    base_path: str = "/home/data/uploads"
) -> pd.DataFrame
```

---

## ModelScope数据加载

### 13. `load_modelscope_stocks()`
**功能**: 从ModelScope加载股票日线数据
```python
def load_modelscope_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**返回**: DataFrame(index=date, columns=股票代码, values=收盘价)

### 14. `load_modelscope_complex_stocks()`
**功能**: 从ModelScope加载多字段数据
```python
def load_modelscope_complex_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
    fields: Union[str, List[str]] = "close",
) -> pd.DataFrame | Dict[str, pd.DataFrame]
```
**返回**: 单个字段返回DataFrame，多个字段返回Dict

---

## 因子分析

### 15. `factor_for_al()`
**功能**: 获取Alphalens格式的因子Series
```python
def factor_for_al(
    codes: List[str],
    start_date: str,
    end_date: str,
    factor_name: str,
    *,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"
) -> pd.Series
```
**返回**: Series index=(date, asset)

---

## 财务报表

### 16. `get_balance()`
**功能**: 获取资产负债表
```python
def get_balance(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```

### 17. `get_income()`
**功能**: 获取利润表
```python
def get_income(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```

### 18. `get_cashflow()`
**功能**: 获取现金流量表
```python
def get_cashflow(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```

### 19. `get_valuation()`
**功能**: 获取估值数据
```python
def get_valuation(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None
) -> pd.DataFrame
```

### 20. `get_history_fundamentals()`
**功能**: 批量获取财务数据（聚宽风格）
```python
def get_history_fundamentals(
    security: Union[str, List[str]],
    fields: List[str],
    watch_date: Union[str, dt_date, dt_datetime, None] = None,
    stat_date: Union[str, None] = None,
    count: int = 1,
    interval: str = "1q",
    report_type: str = "合并期末",
) -> pd.DataFrame
```
**返回**: MultiIndex DataFrame [(code, statDate), field_columns]

---

## 指数相关

### 21. `get_index_stocks()`
**功能**: 获取指数成分股
```python
def get_index_stocks(
    index_symbol: str,
    date: Optional[Union[str, dt_date, dt_datetime]] = None
) -> List[str]
```

### 22. `get_index_daily()`
**功能**: 获取指数日线行情
```python
def get_index_daily(
    index_symbol: str,
    start: Union[str, dt_date, dt_datetime],
    end: Union[str, dt_date, dt_datetime],
) -> pd.Series
```
**返回**: Series(index=date, values=归一化净值)

---

## Backtrader适配

### 23. `load_bt_stocks()`
**功能**: 加载Backtrader数据格式
```python
def load_bt_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> Dict[str, bt.feeds.PandasData]
```
**返回**: {code: PandasData}

### 24. `load_bt_pricing()`
**功能**: 生成Alphalens价格数据
```python
def load_bt_pricing(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**返回**: DataFrame(index=date, columns=code, values=close)

### 25. `load_bt_oss_stocks()`
**功能**: 从OSS加载Backtrader原始数据
```python
def load_bt_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```

---

## 交易日历

### 26. `get_trading_dates()`
**功能**: 获取交易日列表
```python
def get_trading_dates(
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    as_str: bool = False
) -> List[dt.date] | List[str]
```

---

## 工具函数

### 27. `code2name`
**功能**: 股票代码到名称的映射字典
```python
code2name: Dict[str, str]  # {code: name}
```

### 28. `print_table_columns()`
**功能**: 打印财务报表字段列表
```python
def print_table_columns(
    table: Literal["balance", "income", "cashflow"],
    code: str = "000001"
) -> None
```

---

## 内部函数

### 29. `_collect_files()`
**功能**: 收集OSS文件
```python
def _collect_files(start: dt.date, end: dt.date) -> Dict[dt.date, str]
```

### 30. `_wide_to_ohlcv()`
**功能**: 宽表转OHLCV格式
```python
def _wide_to_ohlcv(wide: pd.DataFrame) -> pd.DataFrame
```

### 31. `_get_fin_df()`
**功能**: 统一拉取财务报表
```python
def _get_fin_df(
    code: str,
    date: Union[str, dt_date, dt_datetime, None],
    report_type: str,
    table: Literal["balance", "income", "cashflow"]
) -> pd.DataFrame
```

### 32. `save_result()`
**功能**: 保存结果到OSS
```python
def save_result(bucket, date_tag: str, res_dict: dict) -> None
```

### 33. `handler()`
**功能**: 云函数入口（因子分析任务）
```python
def handler(event, context) -> dict
```

---

## 总结

### 按功能分类
- **数据加载**: load_new_stocks, load_oss_stocks, load_modelscope_stocks
- **Backtrader**: load_bt_stocks, load_bt_pricing
- **财务报表**: get_balance, get_income, get_cashflow, get_valuation
- **因子分析**: read_factor_data, factor_for_al
- **指数**: get_index_stocks, get_index_daily
- **工具**: get_trading_dates, code2name
- **格式化**: _normalize_date_arg, _normalize_code_arg

### 数据源
1. **OSS**: load_new_stocks, load_oss_stocks, read_factor_data
2. **ModelScope**: load_modelscope_stocks, load_modelscope_complex_stocks
3. **本地**: read_factor_data_loal

### 输出格式
- **DataFrame**: 多数函数返回
- **Series**: get_index_daily, factor_for_al
- **Dict**: load_bt_stocks (返回 {code: PandasData})
- **List**: get_index_stocks, get_trading_dates

