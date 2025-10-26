"""
data_akshare.py - 基于AKShare的数据模块
接口参数与data.py完全一致，实现使用AKShare库
"""

from __future__ import annotations

import datetime as dt
from datetime import date as dt_date, datetime as dt_datetime
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union, Literal, TypedDict

import backtrader as bt
import pandas as pd
from alphalens.performance import (
    factor_returns,
    mean_information_coefficient,
    mean_return_by_quantile,
)
from alphalens.utils import get_clean_factor_and_forward_returns
from types import SimpleNamespace

# 使用缓存系统替换akshare
try:
    from akshare_wrapper_v2 import AkShareWrapperV2
    storage_config = {
        'type': 'oss',  # 指定使用阿里云OSS存储
        'access_key_id': os.getenv('AKSHARE_OSS_ACCESS_KEY_ID'),  # 从环境变量获取
        'access_key_secret': os.getenv('AKSHARE_OSS_ACCESS_KEY_SECRET'),  # 从环境变量获取
        'endpoint': 'https://oss-cn-hangzhou.aliyuncs.com',  # 替换为您的OSS endpoint
        'bucket': 'aksharecache',  # 替换为您的存储桶名称
        'min_file_size_mb': 1.0,  # 小文件合并阈值（MB）
    }
    ak = AkShareWrapperV2(storage_config)
except ImportError:
    import akshare as ak
except Exception as e:
    import akshare as ak
    logging.warning(f"使用AKShare缓存系统失败，回退到直连模式: {e}")

calendar = None
try:
    import chinese_calendar as calendar
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    calendar = SimpleNamespace(
        is_workday=lambda d: d.weekday() < 5,
    )

LOGGER = logging.getLogger(__name__)


# =========================
# 数据结构类（与data.py完全一致）
# =========================

class FactorResultRow(TypedDict, total=False):
    trade_date: str
    factor_name: str
    IC_mean: float
    ICIR: float
    FactorReturn_mean: float
    QuantileMeanReturn: float


@dataclass(frozen=True)
class FinancialQuery:
    code: str
    report_type: str
    table: Literal["balance", "income", "cashflow"]
    date: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class DateRange:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]

    def apply(self, frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
        result = frame
        if self.start is not None:
            result = result[result[column] >= self.start]
        if self.end is not None:
            result = result[result[column] <= self.end]
        return result


class OHLCVRecord(TypedDict):
    date: pd.Timestamp
    asset: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# =========================
# 工具函数（与data.py完全一致）
# =========================

def _normalize_date_arg(
    value: Union[str, dt_date, dt_datetime, pd.Timestamp, None],
    *,
    default: Union[str, dt_date, dt_datetime, pd.Timestamp, None] = None,
    as_date: bool = False,
) -> Optional[Union[pd.Timestamp, dt.date]]:
    """将日期参数统一转换为 Timestamp 或 date，兼容字符串和原始对象。"""
    if value is None:
        if default is None:
            return None
        value = default

    ts = pd.to_datetime(value)
    if pd.isna(ts):
        return None

    if as_date:
        return pd.Timestamp(ts).date()
    return pd.Timestamp(ts)


def _normalize_code_arg(
    codes: Union[str, int, Iterable[Union[str, int]], None],
    *,
    allow_none: bool = True,
    deduplicate: bool = True,
) -> Optional[List[str]]:
    """将股票代码统一转换为 6 位数字字符串，兼容多种输入格式。"""
    if codes is None:
        return None if allow_none else []

    if isinstance(codes, (str, bytes, int)):
        iterable = [codes]
    elif isinstance(codes, Iterable):
        iterable = list(codes)
    else:
        iterable = [codes]

    normalized: List[str] = []
    for raw_code in iterable:
        if raw_code is None:
            continue
        code_str = str(raw_code).strip()
        if not code_str:
            continue

        upper = code_str.upper()
        match = re.search(r"\d{6}", upper)
        if match:
            digits = match.group(0)
        elif upper.isdigit() and len(upper) <= 6:
            digits = upper
        else:
            digits = upper

        if digits.isdigit():
            digits = digits.zfill(6)

        normalized.append(digits)

    if deduplicate:
        seen = set()
        deduped: List[str] = []
        for code in normalized:
            if code not in seen:
                seen.add(code)
                deduped.append(code)
        normalized = deduped

    return normalized


def _ensure_exchange_prefix(code: Union[str, int]) -> str:
    """把股票代码统一转成带交易所前缀的形式 (sh/sz/bj)。"""
    normalized = _normalize_code_arg(code, allow_none=False, deduplicate=False)
    if not normalized:
        raise ValueError(f"无法识别股票代码: {code!r}")

    digits = normalized[0]
    if digits.startswith("6"):
        return f"sh{digits}"
    if digits.startswith(("0", "3")):
        return f"sz{digits}"
    if digits.startswith(("4", "8")):
        return f"bj{digits}"
    return digits


def _ensure_exchange_suffix(code: Union[str, int]) -> str:
    """把股票代码统一转成带交易所后缀的形式 (.XSHG/.XSHE/.XBJ)。"""
    code_str = str(code).strip()
    if not code_str:
        return code_str

    code_upper = code_str.upper()
    if code_upper.startswith(("SH", "SZ", "BJ")) and code_upper[2:].isdigit():
        return code_str
    if code_upper.endswith((".XSHG", ".XSHE", ".XBJ")):
        return code_upper

    normalized = _normalize_code_arg(code, allow_none=False, deduplicate=False)
    if not normalized:
        raise ValueError(f"无法识别股票代码: {code!r}")

    digits = normalized[0]
    if digits.startswith("6"):
        return f"{digits}.XSHG"
    if digits.startswith(("0", "3")):
        return f"{digits}.XSHE"
    if digits.startswith(("4", "8")):
        return f"{digits}.XBJ"
    return digits


def _add_prefix(code: str) -> str:
    """自动补 6 位并加交易所前缀"""
    return _ensure_exchange_prefix(code)


def _parse_date(d: Union[str, dt_date, dt_datetime, None]) -> pd.Timestamp:
    """统一转成 pandas.Timestamp"""
    if d is None:
        return pd.Timestamp.now()
    if isinstance(d, str):
        return pd.to_datetime(d)
    if isinstance(d, dt_date) and not isinstance(d, dt_datetime):
        return pd.Timestamp(d)
    if isinstance(d, dt_datetime):
        return pd.Timestamp(d)
    raise TypeError("date 必须是 str / datetime.date / datetime.datetime / None")


# =========================
# AKShare数据加载接口
# =========================

def load_new_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    使用AKShare加载快照数据，返回 DataFrame(index=date, columns=股票代码, values=今开)。
    """
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)
    normalized_codes = _normalize_code_arg(codes)

    LOGGER.debug("加载 AKShare 快照数据: codes=%s, start=%s, end=%s", normalized_codes, start_date, end_date)

    # 使用AKShare获取实时快照
    try:
        df = ak.stock_zh_a_spot_em()
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        
        if normalized_codes:
            df = df[df["代码"].isin(normalized_codes)]
        
        # 转换格式
        df = df[["代码", "今开"]].rename(columns={"代码": "asset", "今开": "close"})
        df["date"] = pd.to_datetime(pd.Timestamp.now().date())
        
        # 转成宽表
        prices = df.pivot(index="date", columns="asset", values="close")
        return prices
    except Exception as e:
        LOGGER.error(f"加载AKShare快照数据失败: {e}")
        return pd.DataFrame(dtype=float)


def load_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    使用AKShare加载日线行情，返回 DataFrame(index=date, columns=股票代码, values=收盘价)。
    """
    import time
    import random
    
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)
    normalized_codes = _normalize_code_arg(codes, allow_none=False)

    if not normalized_codes:
        LOGGER.warning("load_oss_stocks 未提供 codes，返回空结果")
        return pd.DataFrame(dtype=float)

    LOGGER.debug("加载 AKShare 日线数据: codes=%s, start=%s, end=%s", normalized_codes, start_date, end_date)

    frames = []
    for idx, code in enumerate(normalized_codes):
        # 添加延迟避免请求过快
        if idx > 0:
            delay = random.uniform(1.0, 2.0)  # 随机延迟1.0-2.0秒
            time.sleep(delay)
        
        # 重试机制
        max_retries = 5  # 增加重试次数
        for retry in range(max_retries):
            try:
                # 使用AKShare获取日线数据
                code_with_prefix = _ensure_exchange_prefix(code)
                df = ak.stock_zh_a_hist(
                    symbol=code_with_prefix,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq"  # 前复权
                )
                
                if df.empty:
                    LOGGER.warning("AKShare 未找到股票 %s 的日线数据", code)
                    break
                
                # 统一列名
                df["date"] = pd.to_datetime(df["日期"])
                df["close"] = pd.to_numeric(df["收盘"], errors="coerce")
                df["asset"] = code
                
                df = df.loc[:, ["date", "close", "asset"]]
                frames.append(df)
                break  # 成功获取数据，退出重试循环
                
            except Exception as e:
                if retry < max_retries - 1:
                    # 重试前等待更长时间
                    retry_delay = random.uniform(3, 6)  # 增加到3-6秒
                    LOGGER.warning(f"下载股票 {code} 失败 (尝试 {retry+1}/{max_retries})，{retry_delay:.1f}秒后重试: {e}")
                    time.sleep(retry_delay)
                else:
                    LOGGER.warning(f"下载股票 {code} 的日线数据失败 (已重试{max_retries}次): {e}")
                    continue

    if not frames:
        LOGGER.warning("未能获取任何股票的 AKShare 行情数据: %s", normalized_codes)
        return pd.DataFrame(dtype=float)

    df_all = pd.concat(frames, ignore_index=True)
    prices = (
        df_all
        .drop_duplicates(subset=["date", "asset"], keep="last")
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    
    LOGGER.info(f"成功加载 {len(prices.columns)} 只股票的数据")
    return prices


def load_modelscope_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    使用AKShare加载股票日线数据，返回 DataFrame(index=date, columns=股票代码, values=收盘价)。
    """
    return load_oss_stocks(codes=codes, start=start, end=end)


def load_modelscope_complex_stocks(
    codes: Union[str, List[str]],
    start: str = None,
    end: str = None,
    fields: Union[str, List[str]] = "close",
) -> pd.DataFrame:
    """
    使用AKShare加载多字段数据。
    
    fields:
        - "close" (默认): 收盘价
        - "all": 所有字段，返回 dict {字段名: DataFrame}
        - [字段列表]: 指定字段列表，返回 dict
    """
    import time
    import random
    
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)
    normalized_codes = _normalize_code_arg(codes, allow_none=False)

    if not normalized_codes:
        LOGGER.warning("未提供有效股票代码: %s", codes)
        if isinstance(fields, str) and fields.lower() == "all":
            return {}
        if isinstance(fields, list):
            return {}
        return pd.DataFrame()

    frames = []
    for idx, code in enumerate(normalized_codes):
        # 添加延迟避免请求过快
        if idx > 0:
            delay = random.uniform(1.0, 2.0)
            time.sleep(delay)
        
        # 重试机制
        max_retries = 5  # 增加重试次数
        for retry in range(max_retries):
            try:
                code_with_prefix = _ensure_exchange_prefix(code)
                df = ak.stock_zh_a_hist(
                    symbol=code_with_prefix,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq"
                )
                
                if df.empty:
                    break
                
                # 转换列名
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                })
                df["asset"] = code
                frames.append(df)
                break  # 成功获取数据，退出重试循环
                
            except Exception as e:
                if retry < max_retries - 1:
                    retry_delay = random.uniform(3, 6)  # 增加到3-6秒
                    LOGGER.warning(f"加载股票 {code} 失败 (尝试 {retry+1}/{max_retries})，{retry_delay:.1f}秒后重试: {e}")
                    time.sleep(retry_delay)
                else:
                    LOGGER.warning(f"加载股票 {code} 失败 (已重试{max_retries}次): {e}")
                    continue

    if not frames:
        if isinstance(fields, str) and fields.lower() == "all":
            return {}
        if isinstance(fields, list):
            return {}
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # 根据 fields 返回
    if isinstance(fields, str) and fields.lower() == "all":
        value_cols = [c for c in df_all.columns if c not in ["date", "asset", "日期"]]
        result = {col: df_all.pivot(index="date", columns="asset", values=col) for col in value_cols}
        return result
    elif isinstance(fields, str):
        return df_all.pivot(index="date", columns="asset", values=fields).sort_index()
    elif isinstance(fields, list):
        result = {col: df_all.pivot(index="date", columns="asset", values=col) for col in fields if col in df_all.columns}
        return result
    else:
        raise ValueError("fields 必须是 'close' / 'all' / [字段列表]")


# =========================
# 因子分析（保持与data.py一致，但需要外部数据源）
# =========================

def read_factor_data(
    codes: Optional[List[str]] = None,
    start_date: str = None,
    end_date: str = None,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"
) -> pd.DataFrame:
    """
    读取因子数据。注意：AKShare版本需要外部提供因子数据源。
    """
    LOGGER.warning("read_factor_data 需要外部因子数据源，当前返回空表")
    idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
    return pd.DataFrame(index=idx)


def read_factor_data_loal(
    codes: List[str],
    start_date: str,
    end_date: str,
    factors: Optional[List[str]] = None,
    base_path: str = "/home/data/uploads"
) -> pd.DataFrame:
    """
    读取本地因子数据。
    """
    # 读取本地因子数据需要实际的OSS实现
    # 这里暂时返回空表，如果需要可以调用data.py的实现
    LOGGER.warning("read_factor_data_loal 在 data_akshare 中未实现，返回空表")
    idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
    return pd.DataFrame(index=idx)


def factor_for_al(
    codes: List[str],
    start_date: str,
    end_date: str,
    factor_name: str,
    *,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"
) -> pd.Series:
    """
    返回 alphalens 所需的因子 Series。
    """
    df = read_factor_data(
        codes,
        start_date,
        end_date,
        factors=(factors or [factor_name]),
        base_path=base_path
    )

    if factor_name not in df.columns:
        raise KeyError(f"因子 '{factor_name}' 不在数据中，可用列：{df.columns.tolist()}")

    factor_series = df[factor_name].dropna()
    factor_series.index = factor_series.index.set_levels(
        [
            factor_series.index.levels[0],
            factor_series.index.levels[1].str.replace('.XSHG', '', regex=False)
                                          .str.replace('.XSHE', '', regex=False)
        ]
    )
    factor_series.index.names = ['date', 'asset']
    return factor_series


# =========================
# 财务报表接口（使用AKShare）
# =========================

def get_balance(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame:
    """
    使用AKShare获取资产负债表。
    """
    try:
        code_with_prefix = _ensure_exchange_prefix(code)
        df = ak.stock_balance_sheet_by_report_em(symbol=code_with_prefix)
        
        if df.empty:
            return pd.DataFrame()
        
        # 处理日期字段
        date_col = None
        for col in df.columns:
            if "日期" in str(col) or "date" in str(col).lower():
                date_col = col
                break
        
        if date_col and date:
            target_dt = _parse_date(date)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[date_col] <= target_dt]
        
        return df.sort_values(date_col if date_col else df.columns[0], ascending=False).reset_index(drop=True)
    except Exception as e:
        LOGGER.error(f"获取资产负债表失败: {e}")
        return pd.DataFrame()


def get_income(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame:
    """
    使用AKShare获取利润表。
    """
    try:
        code_with_prefix = _ensure_exchange_prefix(code)
        df = ak.stock_profit_sheet_by_report_em(symbol=code_with_prefix)
        
        if df.empty:
            return pd.DataFrame()
        
        # 处理日期字段
        date_col = None
        for col in df.columns:
            if "日期" in str(col) or "date" in str(col).lower():
                date_col = col
                break
        
        if date_col and date:
            target_dt = _parse_date(date)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[date_col] <= target_dt]
        
        return df.sort_values(date_col if date_col else df.columns[0], ascending=False).reset_index(drop=True)
    except Exception as e:
        LOGGER.error(f"获取利润表失败: {e}")
        return pd.DataFrame()


def get_cashflow(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None,
    *,
    report_type: str = "合并期末"
) -> pd.DataFrame:
    """
    使用AKShare获取现金流量表。
    """
    try:
        code_with_prefix = _ensure_exchange_prefix(code)
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=code_with_prefix)
        
        if df.empty:
            return pd.DataFrame()
        
        # 处理日期字段
        date_col = None
        for col in df.columns:
            if "日期" in str(col) or "date" in str(col).lower():
                date_col = col
                break
        
        if date_col and date:
            target_dt = _parse_date(date)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[date_col] <= target_dt]
        
        return df.sort_values(date_col if date_col else df.columns[0], ascending=False).reset_index(drop=True)
    except Exception as e:
        LOGGER.error(f"获取现金流量表失败: {e}")
        return pd.DataFrame()


def get_valuation(
    code: str,
    date: Union[str, dt_date, dt_datetime, None] = None
) -> pd.DataFrame:
    """
    使用AKShare获取估值数据。
    """
    try:
        code_with_prefix = _ensure_exchange_prefix(code)
        df = ak.stock_zh_a_hist(
            symbol=code_with_prefix,
            period="daily",
            adjust="qfq"
        )
        
        df["日期"] = pd.to_datetime(df["日期"])
        
        if date:
            target_dt = _parse_date(date)
            df = df[df["日期"] <= target_dt]
        
        return df.sort_values("日期", ascending=False).reset_index(drop=True)
    except Exception as e:
        LOGGER.error(f"获取估值数据失败: {e}")
        return pd.DataFrame()


def get_history_fundamentals(
    security: Union[str, List[str]],
    fields: List[str],
    watch_date: Union[str, dt_date, dt_datetime, None] = None,
    stat_date: Union[str, None] = None,
    count: int = 1,
    interval: str = "1q",
    report_type: str = "合并期末",
) -> pd.DataFrame:
    """
    批量获取财务数据（聚宽风格）。
    """
    import warnings
    
    if isinstance(security, str):
        security = [security]
    
    dfs = []
    for code in security:
        df = pd.DataFrame()
        
        # 根据字段前缀获取相应数据
        balance_fields = [f.split(".", 1)[1] for f in fields if f.startswith("balance.")]
        income_fields = [f.split(".", 1)[1] for f in fields if f.startswith("income.")]
        cashflow_fields = [f.split(".", 1)[1] for f in fields if f.startswith("cashflow.")]
        
        if balance_fields:
            try:
                balance_df = get_balance(code, date=watch_date, report_type=report_type)
                df = balance_df.head(count).copy()
                df["code"] = code
            except Exception as e:
                LOGGER.warning(f"获取资产负债表失败: {e}")
        
        if income_fields:
            try:
                income_df = get_income(code, date=watch_date, report_type=report_type)
                df = income_df.head(count).copy()
                df["code"] = code
            except Exception as e:
                LOGGER.warning(f"获取利润表失败: {e}")
        
        if cashflow_fields:
            try:
                cashflow_df = get_cashflow(code, date=watch_date, report_type=report_type)
                df = cashflow_df.head(count).copy()
                df["code"] = code
            except Exception as e:
                LOGGER.warning(f"获取现金流量表失败: {e}")
        
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame(columns=["code", "statDate"] + fields)
    
    result = pd.concat(dfs, ignore_index=True)
    result["statDate"] = result["报告日期"].dt.strftime("%Y-%m-%d")
    return result.set_index(["code", "statDate"])


# =========================
# 指数相关接口（使用AKShare）
# =========================

def get_index_stocks(
    index_symbol: str,
    date: Optional[Union[str, dt_date, dt_datetime]] = None
) -> List[str]:
    """
    获取指定指数在指定时刻的成分股代码列表。
    """
    try:
        # 指数代码映射
        index_map = {
            "000300": "sh000300",
            "000001": "sh000001",
            "399001": "sz399001",
            "399006": "sz399006",
            "000905": "sh000905",
            "000016": "sh000016",
        }
        
        index_code = index_map.get(index_symbol, index_symbol)
        # AKShare正确的参数名是symbol
        df = ak.index_stock_cons(symbol=index_code)
        
        if df.empty:
            return []
        
        # 提取股票代码
        codes = df["品种代码"].astype(str).str.zfill(6).tolist()
        return codes
    except Exception as e:
        LOGGER.error(f"获取指数成分股失败: {e}")
        return []


def get_index_daily(
    index_symbol: str,
    start: Union[str, dt_date, dt_datetime],
    end: Union[str, dt_date, dt_datetime],
) -> pd.Series:
    """
    获取指数日线行情，返回归一化净值序列。
    """
    try:
        # 指数代码映射
        index_map = {
            "000300": "sh000300",
            "000001": "sh000001",
            "399001": "sz399001",
            "399006": "sz399006",
            "000905": "sh000905",
            "000016": "sh000016",
        }
        
        index_code = index_map.get(index_symbol, index_symbol)
        start_str = pd.to_datetime(start).strftime("%Y%m%d")
        end_str = pd.to_datetime(end).strftime("%Y%m%d")
        
        # 获取指数数据
        df_index = ak.stock_zh_index_daily(symbol=index_code)
        
        if df_index.empty:
            raise ValueError(f"{index_symbol} 在 {start}~{end} 区间没有数据")
        
        # 统一日期格式为字符串进行比较
        df_index["date"] = pd.to_datetime(df_index["date"])
        df_index["date_str"] = df_index["date"].dt.strftime("%Y%m%d")
        
        # 使用字符串格式过滤
        mask = (df_index["date_str"] >= start_str) & (df_index["date_str"] <= end_str)
        df_index = df_index[mask]
        
        if df_index.empty:
            raise ValueError(f"{index_symbol} 在 {start}~{end} 区间没有数据")
        
        # 计算归一化净值
        df_index["nav"] = df_index["close"] / df_index["close"].iloc[0]
        
        return df_index.set_index("date")["nav"]
    except Exception as e:
        LOGGER.error(f"获取指数日线失败: {e}")
        return pd.Series(dtype=float)


# =========================
# Backtrader适配接口
# =========================

def _wide_to_ohlcv(wide: pd.DataFrame) -> pd.DataFrame:
    """
    将宽表或CSV快照转换成长表OHLCV。
    """
    df = wide.copy()
    snapshot_cols = {"代码", "今开", "最高", "最低", "最新价", "成交量"}

    if snapshot_cols.issubset(df.columns):
        if "date" not in df.columns:
            df["date"] = pd.to_datetime("today").normalize()
        ohlcv = df[["date", "代码", "今开", "最高", "最低", "最新价", "成交量"]].copy()
        ohlcv.rename(columns={
            "代码": "asset",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "最新价": "close",
            "成交量": "volume",
        }, inplace=True)
        return ohlcv[["date", "asset", "open", "high", "low", "close", "volume"]]

    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.name != "date":
        df.index.name = "date"

    long = (
        df.stack(dropna=False)
        .rename("close")
        .reset_index()
        .rename(columns={"level_1": "asset"})
    )

    long["open"] = long["close"]
    long["high"] = long["close"]
    long["low"] = long["close"]
    long["volume"] = 0.0

    ordered_cols = ["date", "asset", "open", "high", "low", "close", "volume"]
    return long[ordered_cols]


def load_bt_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    加载Backtrader原始数据。
    """
    return load_oss_stocks(codes=codes, start=start, end=end)


def load_bt_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> Dict[str, bt.feeds.PandasData]:
    """
    加载Backtrader数据格式。
    """
    if isinstance(codes, str):
        codes = [codes]

    # 读取日线数据
    wide = load_oss_stocks(codes=codes, start=start, end=end)
    if wide.empty:
        print("没有任何股票历史行情数据")
        return {}

    # 转OHLCV
    ohlcv = _wide_to_ohlcv(wide)

    feeds: Dict[str, bt.feeds.PandasData] = {}
    normalized_codes = _normalize_code_arg(codes, allow_none=False) or []
    
    for code in normalized_codes:
        sub = ohlcv[ohlcv["asset"] == code].copy()
        if sub.empty:
            print(f"跳过股票 {code}, 没有历史行情数据")
            continue

        if sub["close"].isna().any():
            print(f"跳过股票 {code}, close 列存在 NaN")
            continue

        sub.set_index("date", inplace=True)
        sub.sort_index(inplace=True)

        feeds[code] = bt.feeds.PandasData(
            dataname=sub,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=None,
            name=code,
        )

    print(f"成功加载 {len(feeds)} 支有效股票")
    return feeds


def load_bt_pricing(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    生成Alphalens价格数据。
    """
    feeds = load_bt_stocks(codes=codes, start=start, end=end)

    if not feeds:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    frames = []
    for code, data in feeds.items():
        df = data.params.dataname.copy()
        df = df[["close"]].rename(columns={"close": code})
        frames.append(df)

    pricing = pd.concat(frames, axis=1)

    if not isinstance(pricing.index, pd.DatetimeIndex):
        pricing.index = pd.to_datetime(pricing.index)
    pricing = pricing.sort_index()

    return pricing


# =========================
# 交易日历（与data.py一致）
# =========================

def get_trading_dates(
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    as_str: bool = False
) -> List[dt.date] | List[str]:
    """
    获取 [start, end] 区间内的所有 A 股交易日。
    """
    def _to_date(x):
        if isinstance(x, str):
            x = x.replace("-", "")
            return dt.datetime.strptime(x, "%Y%m%d").date()
        elif isinstance(x, dt.datetime):
            return x.date()
        elif isinstance(x, dt.date):
            return x
        else:
            raise ValueError(f"不支持的日期类型: {type(x)}")
    
    start_date, end_date = _to_date(start), _to_date(end)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    trading_days = []
    if CALENDAR_AVAILABLE and calendar is not None:
        is_workday = calendar.is_workday
    else:
        is_workday = lambda d: d.weekday() < 5

    current = start_date
    while current <= end_date:
        if is_workday(current):
            trading_days.append(current)
        current += dt.timedelta(days=1)
    
    if as_str:
        return [d.strftime("%Y%m%d") for d in trading_days]
    return trading_days


# =========================
# 代码映射（与data.py一致）
# =========================

MAPPING_FILE = "all_a_stocks.csv"

def load_code2name():
    """返回 {code: name} 的字典"""
    if not os.path.exists(MAPPING_FILE):
        return {}
    mapping = pd.read_csv(MAPPING_FILE)
    return dict(zip(mapping["code"].astype(str).str.zfill(6),
                    mapping["name"]))

code2name = None
if code2name is None:
    code2name = load_code2name()


# =========================
# 其他接口（保持签名一致）
# =========================

def save_result(bucket, date_tag: str, res_dict: dict):
    """保存结果。注意：AKShare版本可能需要不同的存储方式。"""
    LOGGER.warning("save_result 需要外部存储配置")
    pass


def handler(event, context):
    """云函数入口。注意：AKShare版本需要适配。"""
    LOGGER.warning("handler 需要外部配置")
    return {"status": "not_implemented"}


def print_table_columns(table: Literal["balance", "income", "cashflow"], code: str = "000001"):
    """打印表字段。注意：AKShare的字段可能与OSS不同。"""
    LOGGER.warning("print_table_columns 需要AKShare实际数据才能展示字段")
    pass


def _get_default_date():
    """获取默认日期。"""
    ctx = globals().get("context")
    if ctx and hasattr(ctx, "current_dt"):
        return ctx.current_dt
    return dt_datetime.now()


def _collect_files(start: dt.date, end: dt.date) -> Dict[dt.date, str]:
    """收集文件。注意：AKShare版本不需要OSS文件收集。"""
    return {}


def _normalize_codes(codes: List[str]) -> List[str]:
    """批量代码规范化。"""
    return [_ensure_exchange_suffix(code) for code in codes]


def _load_index_df(index_symbol: str) -> pd.DataFrame:
    """加载指数文件。注意：AKShare版本直接使用API。"""
    try:
        index_map = {
            "000300": "sh000300",
            "000001": "sh000001",
            "399001": "sz399001",
            "399006": "sz399006",
            "000905": "sh000905",
            "000016": "sh000016",
        }
        index_code = index_map.get(index_symbol, index_symbol)
        df = ak.index_stock_cons(index=index_code)
        df["in_date"] = pd.to_datetime(df.get("纳入日期", pd.Timestamp.now()))
        return df
    except Exception as e:
        LOGGER.error(f"加载指数文件失败: {e}")
        return pd.DataFrame()


def _get_fin_df(code: str,
                date: Union[str, dt_date, dt_datetime, None],
                report_type: str,
                table: Literal["balance", "income", "cashflow"]) -> pd.DataFrame:
    """统一拉取财务报表。"""
    func_map = {
        "balance": get_balance,
        "income": get_income,
        "cashflow": get_cashflow,
    }
    return func_map[table](code, date=date, report_type=report_type)

