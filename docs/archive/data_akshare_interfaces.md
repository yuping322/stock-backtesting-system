# data_akshare.py 接口文档

## 📋 概述

`data_akshare.py` 是基于AKShare库实现的数据模块，**接口参数与`data.py`完全一致**，但底层数据源使用AKShare。

---

## ✅ 核心特点

1. **接口一致性** - 与`data.py`完全相同的函数签名
2. **数据源** - 使用AKShare获取实时金融数据
3. **完全兼容** - 可直接替换`data.py`使用
4. **无需配置** - 不需要OSS、ModelScope等外部配置

---

## 📚 接口列表（与data.py完全一致）

### 一、数据结构类（4个）
1. `FactorResultRow` - 因子分析结果
2. `FinancialQuery` - 财务查询请求
3. `DateRange` - 日期范围过滤
4. `OHLCVRecord` - OHLCV数据记录

### 二、工具函数（8个）
5. `_normalize_date_arg()` - 日期参数规范化
6. `_normalize_code_arg()` - 股票代码规范化
7. `_ensure_exchange_prefix()` - 补全交易所前缀
8. `_ensure_exchange_suffix()` - 补全交易所后缀
9. `_add_prefix()` - 自动补前缀
10. `_parse_date()` - 统一日期格式
11. `_normalize_codes()` - 批量代码规范化
12. `code2name` - 代码到名称映射字典

### 三、数据加载接口（5个）

#### 13. `load_new_stocks()`
```python
def load_new_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**AKShare实现**: 使用`ak.stock_zh_a_spot_em()`获取实时快照

#### 14. `load_oss_stocks()`
```python
def load_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**AKShare实现**: 使用`ak.stock_zh_a_hist()`获取历史日线

#### 15. `load_modelscope_stocks()`
```python
def load_modelscope_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**AKShare实现**: 调用`load_oss_stocks()`

#### 16. `load_modelscope_complex_stocks()`
```python
def load_modelscope_complex_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
    fields: Union[str, List[str]] = "close",
) -> pd.DataFrame | Dict[str, pd.DataFrame]
```
**AKShare实现**: 使用`ak.stock_zh_a_hist()`获取多字段数据

#### 17. `read_factor_data()`
```python
def read_factor_data(
    codes: Optional[List[str]] = None,
    start_date: str = None,
    end_date: str = None,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"
) -> pd.DataFrame
```
**注意**: 需要外部因子数据源

### 四、财务报表接口（5个）

#### 18. `get_balance()`
```python
def get_balance(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```
**AKShare实现**: `ak.stock_balance_sheet_by_report_em()`

#### 19. `get_income()`
```python
def get_income(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```
**AKShare实现**: `ak.stock_profit_sheet_by_report_em()`

#### 20. `get_cashflow()`
```python
def get_cashflow(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame
```
**AKShare实现**: `ak.stock_cash_flow_sheet_by_report_em()`

#### 21. `get_valuation()`
```python
def get_valuation(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None
) -> pd.DataFrame
```
**AKShare实现**: `ak.stock_zh_a_hist()`

#### 22. `get_history_fundamentals()`
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
**AKShare实现**: 批量调用财务报表接口

### 五、指数相关接口（2个）

#### 23. `get_index_stocks()`
```python
def get_index_stocks(
    index_symbol: str,
    date: Optional[Union[str, dt_date, dt_datetime]] = None
) -> List[str]
```
**AKShare实现**: `ak.index_stock_cons()`

#### 24. `get_index_daily()`
```python
def get_index_daily(
    index_symbol: str,
    start: Union[str, dt_date, dt_datetime],
    end: Union[str, dt_date, dt_datetime],
) -> pd.Series
```
**AKShare实现**: `ak.stock_zh_index_daily()`

### 六、Backtrader适配接口（4个）

#### 25. `load_bt_stocks()`
```python
def load_bt_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> Dict[str, bt.feeds.PandasData]
```
**AKShare实现**: 基于`load_oss_stocks()`生成PandasData

#### 26. `load_bt_pricing()`
```python
def load_bt_pricing(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**AKShare实现**: 从Backtrader数据提取价格

#### 27. `load_bt_oss_stocks()`
```python
def load_bt_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame
```
**AKShare实现**: 调用`load_oss_stocks()`

#### 28. `_wide_to_ohlcv()`
```python
def _wide_to_ohlcv(wide: pd.DataFrame) -> pd.DataFrame
```
**功能**: 宽表转OHLCV格式

### 七、交易日历（1个）

#### 29. `get_trading_dates()`
```python
def get_trading_dates(
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    as_str: bool = False
) -> List[dt.date] | List[str]
```
**实现**: 使用工作日判断（可集成chinese_calendar）

### 八、其他接口（5个）

#### 30. `factor_for_al()` - Alphalens格式因子
#### 31. `read_factor_data_loal()` - 本地因子数据
#### 32. `save_result()` - 保存结果（需配置）
#### 33. `handler()` - 云函数入口（需配置）
#### 34. `print_table_columns()` - 打印表字段
#### 35. `load_code2name()` - 加载代码映射

---

## 🔍 AKShare API映射

| data_akshare接口 | AKShare API | 说明 |
|-----------------|-------------|------|
| `load_new_stocks()` | `ak.stock_zh_a_spot_em()` | 实时快照 |
| `load_oss_stocks()` | `ak.stock_zh_a_hist()` | 历史日线 |
| `get_balance()` | `ak.stock_balance_sheet_by_report_em()` | 资产负债表 |
| `get_income()` | `ak.stock_profit_sheet_by_report_em()` | 利润表 |
| `get_cashflow()` | `ak.stock_cash_flow_sheet_by_report_em()` | 现金流量表 |
| `get_index_stocks()` | `ak.index_stock_cons()` | 指数成分股 |
| `get_index_daily()` | `ak.stock_zh_index_daily()` | 指数日线 |

---

## 💡 使用示例

### 示例1: 加载股票日线数据
```python
from data_akshare import load_oss_stocks

# 加载平安银行和浦发银行的日线数据
df = load_oss_stocks(
    codes=["000001", "600000"],
    start="2024-01-01",
    end="2024-12-31"
)
print(df.head())
```

### 示例2: 获取财务报表
```python
from data_akshare import get_balance, get_income

# 获取平安银行资产负债表
balance = get_balance("000001")
print(balance.head())

# 获取利润表
income = get_income("000001")
print(income.head())
```

### 示例3: 获取指数数据
```python
from data_akshare import get_index_stocks, get_index_daily

# 获取沪深300成分股
stocks = get_index_stocks("000300")
print(f"成分股数量: {len(stocks)}")

# 获取指数净值
nav = get_index_daily("000300", "2024-01-01", "2024-12-31")
print(nav.head())
```

### 示例4: Backtrader回测
```python
from data_akshare import load_bt_stocks
import backtrader as bt

# 加载Backtrader数据
feeds = load_bt_stocks(
    codes=["000001"],
    start="2024-01-01",
    end="2024-12-31"
)

# 创建回测引擎
cerebro = bt.Cerebro()
for code, data in feeds.items():
    cerebro.adddata(data)

# 运行回测
cerebro.run()
```

---

## 📊 对比 data.py

| 特性 | data.py | data_akshare.py |
|------|---------|-----------------|
| 数据源 | OSS、ModelScope | AKShare |
| 配置需求 | 需要OSS凭证 | 无需配置 |
| 接口一致性 | - | ✅ 完全一致 |
| 实时数据 | ✅ | ✅ |
| 财务报表 | ✅ | ✅ |
| 指数数据 | ✅ | ✅ |
| 因子数据 | ✅ | ⚠️ 需外部源 |
| 适用场景 | 生产环境 | 开发/测试环境 |

---

## 🎯 使用建议

### 何时使用 data_akshare.py
1. ✅ 本地开发和测试
2. ✅ 无需OSS/ModelScope配置
3. ✅ 需要实时数据更新
4. ✅ 接口完全兼容现有代码

### 何时使用 data.py
1. ✅ 生产环境
2. ✅ 有OSS存储的因子数据
3. ✅ 需要高稳定性
4. ✅ 已经有现成配置

---

## 🔧 安装依赖

```bash
pip install akshare>=1.9.0
```

---

## ⚠️ 注意事项

1. **网络连接**: AKShare需要网络连接获取数据
2. **速率限制**: 大量请求可能被限流
3. **因子数据**: 需要外部数据源提供因子
4. **数据字段**: AKShare返回的字段名可能与OSS版本不同
5. **性能**: 实时请求可能比OSS慢

---

## 📝 测试

运行测试脚本：
```bash
python tests/test_data_akshare.py
```

测试覆盖：
- ✅ 数据加载接口
- ✅ 财务报表接口
- ✅ 指数接口
- ✅ Backtrader适配
- ✅ 工具函数

---

**最后更新**: 2025-01-XX  
**版本**: v1.0  
**状态**: ✅ 已实现并可测试

