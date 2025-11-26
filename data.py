from __future__ import annotations

import datetime as dt
from datetime import date as dt_date, datetime as dt_datetime
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union, Literal, TypedDict
import hashlib

import backtrader as bt
import pandas as pd
import requests
from alphalens.performance import (
    factor_returns,
    mean_information_coefficient,
    mean_return_by_quantile,
)
from alphalens.utils import get_clean_factor_and_forward_returns
import oss2
from types import SimpleNamespace
from pathlib import Path

# 尝试导入 dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

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

# ========== 内存缓存机制 ==========

# 全局缓存字典
_OSS_CACHE: Dict[str, bytes] = {}
_DF_CACHE: Dict[str, pd.DataFrame] = {}

def _get_cache_key(prefix: str, filename: str) -> str:
    """生成缓存键"""
    return f"{prefix}:{filename}"

def _get_from_cache(key: str) -> Optional[bytes]:
    """从缓存获取数据"""
    return _OSS_CACHE.get(key)

def _put_to_cache(key: str, data: bytes):
    """将数据放入缓存"""
    _OSS_CACHE[key] = data
    # 限制缓存大小，避免内存爆炸
    if len(_OSS_CACHE) > 1000:
        # 删除最旧的缓存
        _OSS_CACHE.pop(next(iter(_OSS_CACHE)), None)

def clear_cache():
    """清空缓存"""
    global _OSS_CACHE, _DF_CACHE
    _OSS_CACHE.clear()
    _DF_CACHE.clear()
    LOGGER.info("缓存已清空")


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
    if pd.isna(ts):  # pd.to_datetime(None) -> NaT
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


OSS_EXCEPTIONS = getattr(oss2, "exceptions", None)
NO_SUCH_KEY_ERROR = getattr(OSS_EXCEPTIONS, "NoSuchKey", FileNotFoundError)
OSS_GENERIC_ERROR = getattr(OSS_EXCEPTIONS, "OssError", Exception)


# ========== OSS 凭证加载（从 .env 文件读取，只读取一次并缓存） ==========

def _load_oss_credentials():
    """
    从 .env 文件加载 OSS 凭证，只读取一次并缓存
    
    Returns:
        tuple: (access_key_id, access_key_secret, endpoint, bucket_name)
    """
    # 查找 .env 文件（从当前文件所在目录向上查找，直到项目根目录）
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # 优先查找当前目录的 .env 文件
    env_file = current_dir / ".env"
    
    # 如果当前目录没有，向上查找项目根目录（查找包含 .git 或 README.md 的目录）
    if not env_file.exists():
        # 从当前目录向上查找，直到找到包含 .git 或 README.md 的目录
        search_dir = current_dir
        while search_dir.parent != search_dir:  # 直到根目录
            if (search_dir / ".git").exists() or (search_dir / "README.md").exists():
                env_file = search_dir / ".env"
                break
            search_dir = search_dir.parent
        else:
            # 如果没找到项目根目录，使用当前目录
            env_file = current_dir / ".env"
    
    # 优先从 .env 文件加载（强制从 .env 读取）
    if DOTENV_AVAILABLE and env_file.exists():
        # override=True 表示 .env 文件的值优先于环境变量
        load_dotenv(env_file, override=True)
        LOGGER.info(f"已从 .env 文件加载配置: {env_file}")
    elif env_file.exists():
        LOGGER.warning("未安装 python-dotenv，无法从 .env 文件读取配置，将使用环境变量")
    else:
        LOGGER.warning(f"未找到 .env 文件: {env_file}，将使用环境变量")
    
    # 从环境变量读取（.env 文件的值已通过 load_dotenv 加载到环境变量中）
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    endpoint = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
    bucket_name = os.getenv("OSS_BUCKET_NAME", "test123432")
    
    return access_key_id, access_key_secret, endpoint, bucket_name

# 在模块加载时读取一次并缓存
_OSS_ACCESS_KEY_ID, _OSS_ACCESS_KEY_SECRET, _OSS_ENDPOINT, _OSS_BUCKET_NAME = _load_oss_credentials()

# 初始化 OSS 认证和 Bucket（使用缓存的凭证）
if not _OSS_ACCESS_KEY_ID or not _OSS_ACCESS_KEY_SECRET:
    LOGGER.warning("OSS 访问凭证未配置，相关函数将无法访问 OSS 资源")
    LOGGER.warning("请检查 .env 文件或环境变量中的 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET")
    auth = None
    bucket = None
else:
    auth = oss2.Auth(_OSS_ACCESS_KEY_ID, _OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, _OSS_ENDPOINT, _OSS_BUCKET_NAME)
    LOGGER.info(f"OSS 已初始化: endpoint={_OSS_ENDPOINT}, bucket={_OSS_BUCKET_NAME}")




PREFIX = "stock_zh_a_spot_em/"      # OSS 目录




# 正则：stock_zh_a_spot_em_YYYYMMDD_HHMM.csv
FNAME_RE = re.compile(r'stock_zh_a_spot_em_(\d{8})_(\d{4})\.csv$')

def _collect_files(start: dt.date, end: dt.date) -> Dict[dt.date, str]:
    """
    返回 dict{date: 最新文件完整 key}，仅包含 start~end 之间的日期
    """
    candidates: Dict[dt.date, str] = {}
    for obj in oss2.ObjectIterator(bucket, prefix=PREFIX):
        m = FNAME_RE.search(obj.key)
        if not m:
            continue
        file_date = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        if start <= file_date <= end:
            # 同一天只保留时间戳最大的文件
            if file_date not in candidates or obj.key > candidates[file_date]:
                candidates[file_date] = obj.key
    return candidates


def load_new_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    从 OSS 目录 /stock_zh_a_spot_em/ 拉取 start~end 区间内的所有快照，
    返回 DataFrame(index=date, columns=股票代码, values=今开)。
    """
    # ---------------- 日期范围处理 ----------------
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)

    normalized_codes = _normalize_code_arg(codes)

    LOGGER.debug("加载 OSS 快照数据: codes=%s, start=%s, end=%s", normalized_codes, start_date, end_date)
    file_map = _collect_files(start_date, end_date)
    if not file_map:
        return pd.DataFrame(dtype=float)   # 无数据

    # ---------------- 股票代码过滤 ----------------
    if normalized_codes:
        LOGGER.debug("快照筛选股票代码: %s", normalized_codes)

    # ---------------- 逐文件读取并合并 ----------------
    frames = []
    for file_date, key in sorted(file_map.items()):
        local_path = f"/tmp/{key.split('/')[-1]}"
        try:
            bucket.get_object_to_file(key, local_path)
        except NO_SUCH_KEY_ERROR as exc:
            message = f"OSS 上找不到快照文件: {key}"
            LOGGER.error(message)
            raise FileNotFoundError(message) from exc
        except OSS_GENERIC_ERROR as exc:
            message = f"下载 OSS 快照文件失败: {key}"
            LOGGER.error(message, exc_info=exc)
            raise ConnectionError(message) from exc

        try:
            df = pd.read_csv(local_path, dtype={"代码": str})
        except pd.errors.EmptyDataError:
            LOGGER.warning("文件 %s 为空，已跳过", key)
            continue
        except pd.errors.ParserError as exc:
            message = f"文件 {key} 解析失败: {exc}"
            LOGGER.error(message)
            raise ValueError(message) from exc

        df = df[["代码", "今开"]].rename(columns={"代码": "asset", "今开": "close"})
        df["date"] = pd.to_datetime(file_date)
        if normalized_codes:
            df = df[df["asset"].isin(normalized_codes)]
        frames.append(df)

    if not frames:
        LOGGER.warning("区间 %s-%s 内未找到可用的快照数据", start_date, end_date)
        return pd.DataFrame(dtype=float)

    df_all = pd.concat(frames, ignore_index=False)

    # ---------------- 转成宽表 ----------------
    prices = (
        df_all
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    return prices

def load_oss_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    从 OSS 目录 hangqing/daily_data/ 拉取 start~end 区间内所有股票的日线行情，
    返回 DataFrame(index=date, columns=股票代码, values=收盘价)。
    """
    # ---------------- 日期范围处理 ----------------
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)

    # ---------------- 股票代码过滤 ----------------
    normalized_codes = _normalize_code_arg(codes)
    if normalized_codes is None:
        LOGGER.warning("load_oss_stocks 未提供 codes，返回空结果")
        return pd.DataFrame(dtype=float)

    # ---------------- 遍历 OSS 目录 ----------------
    prefix = "hangqing/daily_data/"
    frames = []

    for code in normalized_codes:
        try:
            fname = _ensure_exchange_prefix(code) + ".csv"
            object_name = prefix + fname
            
            # 尝试从缓存获取
            cache_key = _get_cache_key("daily_data", object_name)
            content = _get_from_cache(cache_key)
            
            if content is None:
                # 缓存未命中，从OSS读取
                content = bucket.get_object(object_name).read()
                # 存入缓存
                _put_to_cache(cache_key, content)
                LOGGER.debug("缓存未命中，从OSS读取: %s", object_name)
            else:
                LOGGER.debug("缓存命中: %s", object_name)
                
            # 检查内容是否为空
            if not content or len(content.strip()) == 0:
                LOGGER.warning("股票 %s 的 CSV 文件为空", code)
                continue
                
        except NO_SUCH_KEY_ERROR:
            LOGGER.warning("OSS 未找到股票 %s 的日线数据", code)
            continue
        except OSS_GENERIC_ERROR as exc:
            message = f"下载股票 {code} 的日线数据失败"
            LOGGER.error(message, exc_info=exc)
            raise ConnectionError(message) from exc

        try:
            df = pd.read_csv(io.BytesIO(content),
                             dtype={"代码": str, "日期": str})
        except pd.errors.ParserError as exc:
            message = f"股票 {code} 的 CSV 解析失败: {exc}"
            LOGGER.error(message)
            raise ValueError(message) from exc
        except pd.errors.EmptyDataError:
            LOGGER.warning("股票 %s 的 CSV 文件无数据", code)
            continue

        # 日期、价格列统一命名
        df["date"] = pd.to_datetime(df["日期"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        # 日期过滤
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask, ["date", "close"]]
        df["asset"] = code
        frames.append(df)

    if not frames:
        LOGGER.warning("未能获取任何股票的 OSS 行情数据: %s", normalized_codes)
        return pd.DataFrame(dtype=float)

    df_all = pd.concat(frames, ignore_index=True)
    prices = (
        df_all
        .drop_duplicates(subset=["date", "asset"], keep="last")
        .pivot(index="date", columns="asset", values="close")
        .sort_index()
    )
    # # ---------------- 转宽表 ----------------
    # prices = (
    #     df_all
    #     .pivot(index="date", columns="asset", values="close")
    #     .sort_index()
    # )
    return prices


def load_oss_complex_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
    fields: Union[str, List[str]] = "close",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    从 OSS 目录 hangqing/daily_data/ 拉取 start~end 区间内所有股票的日线行情，
    支持多字段返回。
    
    参数
    ----
    codes : str | List[str] | None
        股票代码列表
    start : str | None
        开始日期
    end : str | None
        结束日期
    fields : str | List[str]
        字段名，支持：
        - "close" (默认): 收盘价
        - "all": 所有字段，返回 dict {字段名: DataFrame}
        - [字段列表]: 指定字段列表，返回 dict
        
    返回
    ----
    pd.DataFrame | Dict[str, pd.DataFrame]
        单个字段返回DataFrame，多个字段返回Dict
    """
    # ---------------- 日期范围处理 ----------------
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)

    # ---------------- 股票代码过滤 ----------------
    normalized_codes = _normalize_code_arg(codes)
    if normalized_codes is None or (isinstance(normalized_codes, list) and len(normalized_codes) == 0):
        # 如果 codes 为空，取全量 A 股
        normalized_codes = get_index_stocks("A")

    # ---------------- 遍历 OSS 目录 ----------------
    prefix = "hangqing/daily_data/"
    frames = []

    for code in normalized_codes:
        try:
            fname = _ensure_exchange_prefix(code) + ".csv"
            object_name = prefix + fname
            
            # 尝试从缓存获取
            cache_key = _get_cache_key("complex_stocks", object_name)
            content = _get_from_cache(cache_key)
            
            if content is None:
                # 缓存未命中，从OSS读取
                content = bucket.get_object(object_name).read()
                # 存入缓存
                _put_to_cache(cache_key, content)
                LOGGER.debug("缓存未命中，从OSS读取: %s", object_name)
            else:
                LOGGER.debug("缓存命中: %s", object_name)
                
            # 检查内容是否为空
            if not content or len(content.strip()) == 0:
                LOGGER.warning("股票 %s 的 CSV 文件为空", code)
                continue
                
        except NO_SUCH_KEY_ERROR:
            LOGGER.warning("OSS 未找到股票 %s 的日线数据", code)
            continue
        except OSS_GENERIC_ERROR as exc:
            message = f"下载股票 {code} 的日线数据失败"
            LOGGER.error(message, exc_info=exc)
            continue
            # raise ConnectionError(message) from exc

        try:
            df = pd.read_csv(io.BytesIO(content),
                             dtype={"代码": str, "日期": str})
        except pd.errors.ParserError as exc:
            message = f"股票 {code} 的 CSV 解析失败: {exc}"
            LOGGER.error(message)
            # raise ValueError(message) from exc
            continue
        except pd.errors.EmptyDataError:
            LOGGER.warning("股票 %s 的 CSV 文件无数据", code)
            continue

        # 日期、价格列统一命名
        df["date"] = pd.to_datetime(df["日期"])
        
        # 转换数值列
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 日期过滤
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask]
        df["asset"] = code
        
        # 只保留需要的列
        keep_columns = ["date", "asset"] + [col for col in numeric_columns if col in df.columns]
        df = df[keep_columns]
        frames.append(df)

    if not frames:
        LOGGER.warning("未能获取任何股票的 OSS 行情数据: %s", normalized_codes)
        if isinstance(fields, str) and fields.lower() == "all":
            return {}
        if isinstance(fields, list):
            return {}
        return pd.DataFrame(dtype=float)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["date", "asset"], keep="last")

    # 根据 fields 返回
    if isinstance(fields, str) and fields.lower() == "all":
        # 全部字段转宽表（除了 'asset'、'date'）
        value_cols = [c for c in df_all.columns if c not in ["date", "asset"]]
        result = {col: df_all.pivot(index="date", columns="asset", values=col).sort_index() 
                 for col in value_cols if col in df_all.columns}
        return result

    elif isinstance(fields, str):
        # 单个字段
        if fields not in df_all.columns:
            LOGGER.warning("字段 %s 不存在，可用字段: %s", fields, list(df_all.columns))
            return pd.DataFrame(dtype=float)
        return df_all.pivot(index="date", columns="asset", values=fields).sort_index()

    elif isinstance(fields, list):
        # 多个字段 -> dict
        result = {}
        for col in fields:
            if col in df_all.columns:
                result[col] = df_all.pivot(index="date", columns="asset", values=col).sort_index()
            else:
                LOGGER.warning("字段 %s 不存在，可用字段: %s", col, list(df_all.columns))
        return result

    else:
        raise ValueError("fields 必须是 'close' / 'all' / [字段列表]")


# def load_modelscope_stocks(
#     codes: Union[str, List[str]],
#     start: str = None,
#     end: str = None,
# ) -> pd.DataFrame:
#     """
#     下载多只股票日线 CSV（伪装成 .npy），合并后按日期过滤，
#     返回只含 ['date', 'asset', 'close'] 三列的 DataFrame，
#     且以 'date' 作为 DatetimeIndex，可直接用作 alphalens 的 pricing_df。
#     """
#     base_url = "https://modelscope.cn/api/v1/datasets/yuping322/stock_zh_a_daily/repo"
#     params_tpl = {"Revision": "master", "FilePath": None}

#     normalized_codes = _normalize_code_arg(codes, allow_none=False)
#     if not normalized_codes:
#         LOGGER.warning("模型数据源未提供有效股票代码: %s", codes)
#         return pd.DataFrame(columns=["date", "asset", "close"])

#     date_range = DateRange(
#         start=_normalize_date_arg(start),
#         end=_normalize_date_arg(end),
#     )

#     frames = []
#     for code in normalized_codes:
#         prefixed = _ensure_exchange_prefix(code)
#         fname = f"{prefixed}.npy"
#         params = params_tpl.copy()
#         params["FilePath"] = fname

#         try:
#             resp = requests.get(base_url, params=params, timeout=30)
#             resp.raise_for_status()
#         except requests.RequestException as exc:
#             message = f"下载模型数据 {fname} 失败: {exc}"
#             LOGGER.error(message)
#             raise ConnectionError(message) from exc

#         try:
#             df = pd.read_csv(io.BytesIO(resp.content))
#         except pd.errors.ParserError as exc:
#             message = f"模型数据 {fname} 解析失败: {exc}"
#             LOGGER.error(message)
#             raise ValueError(message) from exc

#         # 统一列名并只保留我们需要的三列
#         df.columns = [c.strip().lower() for c in df.columns]
#         rename_map = {
#             "日期": "date",
#             "symbol": "asset",
#             "close": "close",
#         }
#         df = (
#             df.rename(columns=rename_map)
#               .assign(asset=code)  # 用原始纯数字代码做 asset
#               .loc[:, ["date", "asset", "close"]]
#         )

#         # 转日期
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = date_range.apply(df)
#         frames.append(df)


#     if not frames:
#         LOGGER.warning("模型数据源未返回任何记录: codes=%s", normalized_codes)
#         return pd.DataFrame(columns=["date", "asset", "close"])

#     df_all = pd.concat(frames, ignore_index=True)

#     # 日期过滤
#     df_all = date_range.apply(df_all)

#     # 转宽表：日期为 index，股票代码为列
#     prices = (
#         df_all
#         .pivot(index="date", columns="asset", values="close")
#         .sort_index()
#     )
#     return prices


# def load_modelscope_complex_stocks(
#     codes: Union[str, List[str]],
#     start: str = None,
#     end: str = None,
#     fields: Union[str, List[str]] = "close",
# ) -> pd.DataFrame:
#     """
#     下载多只股票日线 CSV（伪装成 .npy），按日期过滤。
#     默认只返回收盘价宽表，列为原始股票代码。
    
#     fields:
#         - "close" (默认): 收盘价
#         - "all": 所有字段，返回 dict {字段名: DataFrame}
#         - [字段列表]: 指定字段列表，返回 dict
#     """


#     base_url = "https://modelscope.cn/api/v1/datasets/yuping322/stock_zh_a_daily/repo"
#     params_tpl = {"Revision": "master", "FilePath": None}

#     normalized_codes = _normalize_code_arg(codes, allow_none=False)
#     if not normalized_codes:
#         LOGGER.warning("模型复杂数据源未提供有效股票代码: %s", codes)
#         if isinstance(fields, str) and fields.lower() == "all":
#             return {}
#         if isinstance(fields, list):
#             return {}
#         return pd.DataFrame()

#     date_range = DateRange(
#         start=_normalize_date_arg(start),
#         end=_normalize_date_arg(end),
#     )

#     frames = []
#     for code in normalized_codes:
#         prefixed = _ensure_exchange_prefix(code)
#         fname = f"{prefixed}.npy"
#         params = params_tpl.copy()
#         params["FilePath"] = fname

#         try:
#             resp = requests.get(base_url, params=params, timeout=30)
#             resp.raise_for_status()
#         except requests.RequestException as exc:
#             message = f"下载模型数据 {fname} 失败: {exc}"
#             LOGGER.error(message)
#             raise ConnectionError(message) from exc

#         try:
#             df = pd.read_csv(io.BytesIO(resp.content))
#         except pd.errors.ParserError as exc:
#             message = f"模型数据 {fname} 解析失败: {exc}"
#             LOGGER.error(message)
#             raise ValueError(message) from exc
#         df.columns = [c.strip().lower() for c in df.columns]

#         # 统一列名
#         rename_map = {"日期": "date", "symbol": "asset", "close": "close"}
#         df = df.rename(columns=rename_map)

#         # 用原始数字代码作为 asset
#         df["asset"] = code

#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = date_range.apply(df)
#         frames.append(df)

#     if not frames:
#         LOGGER.warning("模型复杂数据源未返回任何记录: codes=%s", normalized_codes)
#         if isinstance(fields, str) and fields.lower() == "all":
#             return {}
#         if isinstance(fields, list):
#             return {}
#         return pd.DataFrame()

#     df_all = pd.concat(frames, ignore_index=True)

#     # 日期过滤
#     df_all = date_range.apply(df_all)

#     # 根据 fields 返回
#     if isinstance(fields, str) and fields.lower() == "all":
#         # 全部字段转宽表（除了 'asset'、'date'）
#         value_cols = [c for c in df_all.columns if c not in ["date", "asset"]]
#         result = {col: df_all.pivot(index="date", columns="asset", values=col) for col in value_cols}
#         return result

#     elif isinstance(fields, str):
#         # 单个字段
#         return df_all.pivot(index="date", columns="asset", values=fields).sort_index()

#     elif isinstance(fields, list):
#         # 多个字段 -> dict
#         result = {col: df_all.pivot(index="date", columns="asset", values=col) for col in fields if col in df_all.columns}
#         return result

#     else:
#         raise ValueError("fields 必须是 'close' / 'all' / [字段列表]")




# # 测试
# if __name__ == "__main__":
#     # 默认（close 收盘价）
#     prices = load_modelscope_stocks(["000001", "600000"], start="2020-01-01")
#     print(prices)
#     # 指定某个字段
#     opens = load_modelscope_stocks(["000001", "600000"], start="2020-01-01", fields="open")
#     print(opens)
#     # 多个字段
#     data = load_modelscope_stocks(["000001", "600000"], fields=["open", "high", "low"])
#     print(data)
#     # 所有字段（dict 形式，每个字段一个 DataFrame）
#     all_data = load_modelscope_stocks(["000001", "600000"], fields="all")
#     print(all_data)
    
    


def _normalize_codes(codes: List[str]) -> List[str]:
    """
    把 6 位数字自动补全为带交易所后缀的标准格式，
    已经是完整格式的保持不变。
    """
    return [_ensure_exchange_suffix(code) for code in codes]



def read_factor_data(
    codes: Optional[List[str]] = None,
    start_date: str = None,
    end_date: str = None,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads"   # OSS 中的目录前缀，不要带 "/"
) -> pd.DataFrame:
    """
    从 OSS 读取因子数据，内部自动补全股票代码后缀。
    返回以 (date, code) 为 MultiIndex 的 DataFrame。

    参数
    ----
    codes : list[str] | None
        股票列表。None 或空列表表示读取文件中全部股票。
    start_date : str
    end_date   : str
    factors : list[str] | None
        因子列表。None 或空列表表示读取文件中全部因子。
    base_path : str
        OSS 中的目录前缀，不要带 "/"。
    """
    # 1. 自动补全后缀（None 时保持 None）
    if codes:
        codes = _normalize_codes(codes)

    # 2. 日期序列
    dates = pd.date_range(start_date, end_date, freq="D")

    # 3. 逐日读取
    frames = []
    for d in dates:
        object_name = f"{base_path}/{d.year}/factors_{d.strftime('%Y%m%d')}_all.csv"

        if not bucket.object_exists(object_name):
            continue

        try:
            # 尝试从缓存获取
            cache_key = _get_cache_key("factor_data", object_name)
            cached_df = _DF_CACHE.get(cache_key)
            
            if cached_df is not None:
                LOGGER.debug("因子数据缓存命中: %s", object_name)
                df = cached_df.copy()
            else:
                # 缓存未命中，从OSS读取
                result = bucket.get_object(object_name)
                df = pd.read_csv(result, index_col=0)
                # 存入缓存
                _DF_CACHE[cache_key] = df
                LOGGER.debug("因子数据缓存未命中，从OSS读取: %s", object_name)
        except Exception as e:
            print(f"读取失败 {object_name}: {e}")
            continue

        # 4. 过滤股票
        if codes:                       # 非空列表才过滤
            df = df[df.index.isin(codes)]
        # 5. 过滤因子
        if factors:                     # 非空列表才过滤
            cols = [c for c in factors if c in df.columns]
            df = df[cols]

        if df.empty:
            continue

        df = df.copy()
        df["date"] = d
        df["code"] = df.index
        frames.append(df)

    # 6. 无数据返回空表
    if not frames:
        # 这里无法提前知道列名，只能返回空表
        idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
        return pd.DataFrame(index=idx)

    # 7. 合并
    result = (
        pd.concat(frames, ignore_index=True)
          .set_index(["date", "code"])
          .sort_index()
    )
    return result


def read_factor_data_loal(
    codes: List[str],
    start_date: str,
    end_date: str,
    factors: Optional[List[str]] = None,
    base_path: str = "/home/data/uploads"
) -> pd.DataFrame:
    """
    读取因子数据，内部自动补全股票代码后缀。
    返回以 (date, code) 为 MultiIndex 的 DataFrame。
    """
    # 1. 自动补全后缀
    codes = _normalize_codes(codes)

    # 2. 日期序列
    dates = pd.date_range(start_date, end_date, freq="D")

    # 3. 逐日读取
    frames = []
    for d in dates:
        fpath = os.path.join(base_path,
                             str(d.year),
                             f"factors_{d.strftime('%Y%m%d')}_all.csv")
        if not os.path.isfile(fpath):
            continue
        try:
            df = pd.read_csv(fpath, index_col=0)
        except Exception as e:
            print(f"读取失败 {fpath}: {e}")
            continue

        # 过滤股票
        if codes:
            df = df[df.index.isin(codes)]
        # 过滤因子
        if factors is not None:
            cols = [c for c in factors if c in df.columns]
            df = df[cols]

        if df.empty:
            continue

        df = df.copy()
        df["date"] = d
        df["code"] = df.index
        frames.append(df)

    # 4. 无数据返回空表
    if not frames:
        cols = factors if factors is not None else []
        idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
        return pd.DataFrame(index=idx, columns=cols)

    # 5. 合并
    result = (
        pd.concat(frames, ignore_index=True)
          .set_index(["date", "code"])
          .sort_index()
    )
    return result




def factor_for_al(
    codes: List[str],
    start_date: str,
    end_date: str,
    factor_name: str,
    *,
    factors: Optional[List[str]] = None,
    base_path: str = "uploads",
    file_path: Optional[str] = None
) -> pd.Series:
    """
    返回 alphalens 所需的因子 Series，索引为 (date, asset)，
    asset 统一为纯 6 位数字字符串。
    
    参数
    ----
    codes : list[str]
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    factor_name : str
        因子名称
    factors : list[str] | None
        因子列表（兼容旧接口）
    base_path : str
        OSS 或本地目录前缀
    file_path : str | None
        如果提供文件路径，则直接从该文件加载（优先级最高）
        文件格式: CSV，需包含 date, code 列和因子列
    """
    # 优先从文件加载
    if file_path and os.path.exists(file_path):
        df = _load_factor_from_file(file_path, codes, start_date, end_date, factor_name)
    else:
        # 从 OSS 或本地目录加载
        df = read_factor_data(
            codes,
            start_date,
            end_date,
            factors=(factors or [factor_name]),
            base_path=base_path
        )

    if df.empty:
        # 返回空 Series
        idx = pd.MultiIndex.from_tuples([], names=['date', 'asset'])
        return pd.Series(index=idx, dtype=float)

    if factor_name not in df.columns:
        raise KeyError(f"因子 '{factor_name}' 不在数据中，可用列：{df.columns.tolist()}")

    # 取单因子列，转成 Series
    factor_series = df[factor_name].dropna()

    # === 核心改动：去掉 .XSHG / .XSHE 后缀，统一成 6 位纯数字 ===
    factor_series.index = factor_series.index.set_levels(
        [
            factor_series.index.levels[0],  # date 不变
            factor_series.index.levels[1].str.replace('.XSHG', '', regex=False)
                                          .str.replace('.XSHE', '', regex=False)
        ]
    )

    factor_series.index.names = ['date', 'asset']
    return factor_series


def _load_factor_from_file(
    file_path: str,
    codes: List[str],
    start_date: str,
    end_date: str,
    factor_name: str
) -> pd.DataFrame:
    """
    从单个文件加载因子数据
    
    参数
    ----
    file_path : str
        文件路径
    codes : list[str]
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    factor_name : str
        因子名称
        
    返回
    ----
    pd.DataFrame
        以 (date, code) 为 MultiIndex 的 DataFrame
    """
    try:
        # 读取文件，代码列保持字符串格式
        df = pd.read_csv(file_path, dtype={'code': str})
        
        # 检查必要的列
        if 'date' not in df.columns or 'code' not in df.columns:
            raise ValueError(f"文件必须包含 'date' 和 'code' 列")
        
        if factor_name not in df.columns:
            raise ValueError(f"文件必须包含因子列 '{factor_name}'")
        
        # 日期列转换为 datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # 标准化代码列（去掉后缀并补齐6位）
        df['code'] = df['code'].astype(str).str.replace('.XSHG', '', regex=False).str.replace('.XSHE', '', regex=False)
        df['code'] = df['code'].str.zfill(6)  # 补齐6位
        
        # 过滤日期范围
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # 过滤股票代码
        if codes:
            # 标准化代码（去掉后缀用于匹配）
            codes_normalized = [_normalize_code_arg([c], allow_none=False)[0] for c in codes]
            df = df[df['code'].isin(codes_normalized)]
        
        if df.empty:
            idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
            return pd.DataFrame(index=idx)
        
        # 设置 MultiIndex
        df = df.set_index(['date', 'code']).sort_index()
        
        return df
        
    except Exception as e:
        print(f"从文件加载因子数据失败 {file_path}: {e}")
        idx = pd.MultiIndex.from_tuples([], names=["date", "code"])
        return pd.DataFrame(index=idx)

# 新增：把结果存到 OSS
def save_result(bucket, date_tag: str, res_dict: dict):
    """把一行结果追加到 OSS 的 daily_metrics.csv。"""
    obj_name = "daily_metrics/daily_metrics.csv"

    # 构造一行 DataFrame
    row = pd.Series(res_dict, name=date_tag).to_frame().T
    row.index.name = "trade_date"

    # 如果文件已存在，先读下来再合并
    if bucket.object_exists(obj_name):
        old = pd.read_csv(
            io.BytesIO(bucket.get_object(obj_name).read()),
            index_col=["trade_date", "factor_name"],   # 以这两列为索引
            parse_dates=True
        )
        # 用 (trade_date, factor_name) 做联合主键去重/更新
        new = pd.concat([old, row.set_index("factor_name", append=True)])
        new = new[~new.index.duplicated(keep="last")].sort_index()
    else:
        new = row.set_index("factor_name", append=True)

    # 写回 OSS
    buf = io.BytesIO()
    new.to_csv(buf)
    buf.seek(0)
    bucket.put_object(obj_name, buf)


# -----------------------------------------------------
# 改造后的 handler：默认跑最近 30 天
# -----------------------------------------------------
def handler(event, context):

    logger = logging.getLogger()
    if isinstance(event, (bytes, str)):
        event = json.loads(event.decode() if isinstance(event, bytes) else event)
    event = event or {}

    # 1. 计算窗口：默认「今天-30」~「今天-1」
    today = pd.Timestamp(dt.date.today())
    start_date = (today - pd.Timedelta(days=3)).strftime("%Y%m%d")
    end_date   = (today - pd.Timedelta(days=1)).strftime("%Y%m%d")  # 昨天收盘

    codes       = event.get("codes", [])
    factor_name = event.get("factor_name", "net_profit_growth_rate")

    # 2. 因子 & 价格
    # 1. 取到因子
    factor_df = factor_for_al(codes, start_date, end_date, factor_name)

    # 2. 用因子里真实出现的代码去拉价格
    codes_in_factor = factor_df.index.get_level_values('asset').unique().tolist()
    codes_in_factor =codes_in_factor[0:20]
    print(codes_in_factor)
    prices_df = load_oss_stocks(codes_in_factor,
                            start=pd.to_datetime(start_date),
                            end=pd.to_datetime(end_date))

    if factor_df.empty or prices_df.empty:
        logger.warning(f"{start_date}~{end_date} 无数据，跳过")
        return {"status": "skip", "date": str(today.date())}

    factor_data = get_clean_factor_and_forward_returns(
        factor_df, prices_df, quantiles=5, periods=(1,), max_loss=1.0
    )
    if factor_data.empty:
        raise ValueError("factor_data 为空")

    # 3. 计算四指标
    ic       = mean_information_coefficient(factor_data)
    fr       = factor_returns(factor_data)
    mean_ret, _ = mean_return_by_quantile(factor_data)

    res = {
        "trade_date": today.date().isoformat(),
        "IC_mean": float(ic.mean()),
        "ICIR": float(ic.mean() / ic.std()),
        "FactorReturn_mean": float(fr.mean()),
        "QuantileMeanReturn": float(mean_ret.mean().diff().iloc[-1])  # 5-1
    }

    # 4. 存 OSS
    save_result(bucket, str(today.date()), res)
    logger.info(res)
    return res


# ---------- 默认日期策略 ----------
def _get_default_date():
    """
    根据运行环境返回默认日期时间：
    • 回测：context.current_dt
    • 研究：datetime.now()
    需要在入口把 context 注入进来，否则退化为 now()
    """
    # 回测框架会把 context 注入到 __builtins__ 或 globals
    ctx = globals().get("context")
    if ctx and hasattr(ctx, "current_dt"):
        return ctx.current_dt       # 回测
    return dt_datetime.now()        # 研究 / 本地

# ---------- 数据 IO ----------
def _load_index_df(index_symbol: str) -> pd.DataFrame:
    """
    从OSS加载指数成分股数据
    
    文件格式：index/{code}_{name}.csv (如: index/000300_沪深300.csv)
    """
    # 查找带中文名称的文件
    prefix = f"index/{index_symbol}_"
    
    # 添加重试机制
    max_retries = 3
    for retry in range(max_retries):
        try:
            file_found = False
            for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                if obj.key.endswith(".csv"):
                    content = bucket.get_object(obj.key).read()
                    df = pd.read_csv(io.BytesIO(content), dtype=str)
                    file_found = True
                    break
            
            if not file_found:
                # 如果没找到文件，继续下一次尝试
                if retry < max_retries - 1:
                    LOGGER.warning(f"未找到文件，重试 {retry + 1}/{max_retries}...")
                    import time
                    time.sleep(1)
                    continue
                raise FileNotFoundError(f"OSS 上找不到 {index_symbol} 相关的指数文件")
            
            # 如果成功找到并读取文件，跳出重试循环
            break
        except (oss2.exceptions.RequestError, requests.exceptions.SSLError, Exception) as e:
            if retry < max_retries - 1:
                LOGGER.warning(f"OSS连接失败，重试 {retry + 1}/{max_retries}... 错误: {e}")
                import time
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                # 最后一次重试失败，抛出异常
                raise
    
    # 重命名列
    rename = {"品种代码": "code", "纳入日期": "in_date", "指数纳入日期": "in_date"}
    # 只重命名存在的列，避免重复列名
    for old_name, new_name in rename.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # 如果有多个in_date列，只保留第一个（通过drop_duplicates处理列）
    if df.columns.tolist().count("in_date") > 1:
        # 删除重复的列名，保留第一个
        cols = df.columns.tolist()
        seen = set()
        new_cols = []
        for col in cols:
            if col == "in_date":
                if "in_date" not in seen:
                    new_cols.append("in_date")
                    seen.add("in_date")
            else:
                new_cols.append(col)
        df = df.iloc[:, [cols.index(c) for c in new_cols]]
        df.columns = new_cols
    
    # 手动转换日期，避免pandas的重复键问题
    if "in_date" not in df.columns:
        raise KeyError("缺少 in_date 列，请检查指数文件表头")
    # 确保in_date是单列Series
    if isinstance(df["in_date"], pd.DataFrame):
        # 如果返回DataFrame，取第一列
        date_col = df["in_date"].iloc[:, 0]
    else:
        date_col = df["in_date"].copy()
    df["in_date"] = pd.Series([pd.to_datetime(d, errors="coerce") for d in date_col], index=df.index)
    
    # 检查是否有无效日期
    na_mask = df["in_date"].isna()
    # .sum() 应该返回标量，但为了兼容性，确保是标量整数
    try:
        na_count = int(na_mask.sum())
    except (ValueError, TypeError):
        # 如果.sum()返回非标量，使用dropna并记录警告
        na_count = len(df) - len(df.dropna(subset=["in_date"]))
    if na_count > 0:
        LOGGER.warning(f"有 {na_count} 个无效日期，已过滤")
        df = df.dropna(subset=["in_date"])
    return df

# ---------- 主 API ----------
def get_index_stocks(
    index_symbol: str,
    date: Optional[Union[str, dt_date, dt_datetime]] = None
) -> List[str]:
    """
    获取指定指数在指定时刻已纳入的成分券代码列表。
    支持特殊股票池类型：
    - 'HS300': 沪深300
    - 'ZZ500': 中证500  
    - 'ZZ800': 中证800
    - 'CYBZ': 创业板指
    - 'ZXBZ': 中小板指
    - 'A': 全A股
    - 'AA': 全A股（排除科创板、创业板）
    - 'small': 小盘股
    - 'small_25': 小盘股（市值<25亿）
    - 'small_fengzhi': 小盘股（市值<80亿）
    
    参数
    ----
    index_symbol : str
        指数代码，如 '000300' 或特殊股票池类型
    date : str | datetime.date | datetime.datetime | None
        查询日期；None 时根据环境自动选择默认时间
    返回
    ----
    List[str]
        股票代码列表
    """
    # 1. 统一转换成 pandas.Timestamp
    if date is None:
        ts = pd.Timestamp(_get_default_date())
    elif isinstance(date, str):
        ts = pd.to_datetime(date)
    elif isinstance(date, dt_date) and not isinstance(date, dt_datetime):
        ts = pd.Timestamp(date)
    elif isinstance(date, dt_datetime):
        ts = pd.Timestamp(date)
    else:
        raise TypeError("date 必须是 str / datetime.date / datetime.datetime / None")

    def _get_stocks_for_index(symbol: str) -> List[str]:
        """内部函数：获取单个指数的成分股"""
        try:
            df = _load_index_df(symbol)
            # 先按日期过滤
            mask = df["in_date"] <= ts
            df = df.loc[mask]
            # 按股票代码分组，保留最新的记录
            df = df.sort_values("in_date").groupby("code").last().reset_index()
            return df["code"].tolist()
        except Exception as e:
            LOGGER.warning(f"获取指数 {symbol} 成分股失败: {e}")
            return []

    # 2. 处理特殊股票池类型
    if index_symbol == 'HS300':
        stockList = _get_stocks_for_index('000300')
    elif index_symbol == 'ZZ500':
        stockList = _get_stocks_for_index('399905')
    elif index_symbol == 'ZZ800':
        stockList = _get_stocks_for_index('399906')
    elif index_symbol == 'CYBZ':
        stockList = _get_stocks_for_index('399006')
    elif index_symbol == 'ZXBZ':
        stockList = _get_stocks_for_index('399005')
    elif index_symbol == 'A':
        stockList = _get_stocks_for_index('000002') + _get_stocks_for_index('399107')
        stockList = [stock for stock in stockList if not stock.startswith(('68', '4', '8'))]
    elif index_symbol == 'AA':
        stockList = _get_stocks_for_index('000985')
        stockList = [stock for stock in stockList if not stock.startswith(('3', '68', '4', '8'))]
    elif index_symbol == 'small':
        stockList = _get_stocks_for_index('399101')
        stockList = [stock for stock in stockList if not stock.startswith(('68', '4', '8'))]
    elif index_symbol == 'small_25':
        initial_list = _get_stocks_for_index('000002') + _get_stocks_for_index('399107')
        # 注意：这里需要实现 get_fundamentals 函数来获取市值数据
        # 暂时返回初始列表，实际使用时需要根据具体的数据源实现
        stockList = initial_list[:50]  # 临时实现
    elif index_symbol == 'small_fengzhi':
        initial_list = _get_stocks_for_index('000002') + _get_stocks_for_index('399107')
        # 注意：这里需要实现 get_fundamentals 函数来获取市值数据
        # 暂时返回过滤后的列表，实际使用时需要根据具体的数据源实现
        stockList = [stock for stock in initial_list if not stock.startswith(('3','68', '4', '8'))]
    else:
        # 3. 处理普通指数代码
        stockList = _get_stocks_for_index(index_symbol)
    
    # 4. 直接返回股票列表（过滤功能已移除）
    return stockList











# ---------- 工具 ----------
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

# # ---------- 主 API ----------
# def get_balance(
#     code: str,
#     date: Union[str, dt_date, dt_datetime, None] = None,
#     *,
#     report_type: str = "合并期末"
# ) -> pd.DataFrame:
#     """
#     拉取单只股票现金流量表数据（含资产负债表字段）。
#     OSS 文件名规则：{sh/sz/bj}xxxxxx.csv
#     """
#     ts = _parse_date(date)
#     key = _add_prefix(code) + ".csv"
#     key = "jukuan/stock_financial_report_sina/"+key
#     try:
#         content = bucket.get_object(key).read()
#     except oss2.exceptions.NoSuchKey:
#         raise FileNotFoundError(f"OSS 上找不到 {key}")

#     df = pd.read_csv(io.BytesIO(content), dtype=str)
#     df["报告日"] = pd.to_datetime(df["报告日"], errors="coerce")

#     mask = df["报告日"] <= ts
#     if "类型" in df.columns:
#         mask &= (df["类型"] == report_type)
#     df = df.loc[mask].sort_values("报告日", ascending=False).reset_index(drop=True)
#     return df
# ---------- 内部统一拉取 ----------
def _get_fin_df(code: str,
                date: Union[str, dt_date, dt_datetime, None],
                report_type: str,
                table: Literal["balance", "income", "cashflow"]) -> pd.DataFrame:
    """table 决定子目录"""
    key_map = {
        "balance": "jukuan/stock_financial_report_sina",
        "income":  "jukuan/stock_financial_report_sina_lirun",   # 如后续拆目录，直接改这里
        "cashflow":"jukuan/stock_financial_report_sina_xianjinliu",
    }
    key = f"{key_map[table]}/{_add_prefix(code)}.csv"
    try:
        content = bucket.get_object(key).read()
    except oss2.exceptions.NoSuchKey:
        raise FileNotFoundError(f"OSS 上找不到 {key}")

    df = pd.read_csv(io.BytesIO(content), dtype=str)
    df["报告日"] = pd.to_datetime(df["报告日"], errors="coerce")

    mask = df["报告日"] <= _parse_date(date)
    if "类型" in df.columns:
        mask &= (df["类型"] == report_type)
    return df.loc[mask].sort_values("报告日", ascending=False).reset_index(drop=True)


# ---------- 对外 API ----------
def get_balance(code: str,
                date: Union[str, dt_date, dt_datetime, None] = None,
                *,
                report_type: str = "合并期末") -> pd.DataFrame:
    return _get_fin_df(code, date, report_type, table="balance")

def get_income(code: str,
               date: Union[str, dt_date, dt_datetime, None] = None,
               *,
               report_type: str = "合并期末") -> pd.DataFrame:
    return _get_fin_df(code, date, report_type, table="income")

def get_cashflow(code: str,
                 date: Union[str, dt_date, dt_datetime, None] = None,
                 *,
                 report_type: str = "合并期末") -> pd.DataFrame:
    return _get_fin_df(code, date, report_type, table="cashflow")

def get_valuation(code: str,
                  date: Union[str, dt_date, dt_datetime, None] = None) -> pd.DataFrame:
    
    key = f"hangqing/daily_data/{_add_prefix(code)}.csv"
    try:
        content = bucket.get_object(key).read()
    except oss2.exceptions.NoSuchKey:
        # folder_prefix = "hangqing/daily_data/"
        # print(f"OSS 上找不到 {key}，该文件夹下文件有：")
        # for obj in oss2.ObjectIterator(bucket, prefix=folder_prefix):
        #     print(obj.key)
        raise FileNotFoundError(f"OSS 上找不到 {key}")
    
    df = pd.read_csv(io.BytesIO(content), dtype=str)

    # 解析日期列，兼容字符串格式
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

    if date:
        target_dt = _parse_date(date)
        if "日期" in df.columns:
            mask = df["日期"] <= target_dt
            df = df.loc[mask]
        else:
            raise KeyError("数据缺少 '日期' 列，无法按日期过滤")

    df = df.sort_values("日期", ascending=False).reset_index(drop=True)

    # 恢复日期格式为字符串，保持对外接口一致
    if "日期" in df.columns:
        df["日期"] = df["日期"].dt.strftime("%Y-%m-%d")

    return df

# ---------- 只改动 get_history_fundamentals ----------
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
    聚宽风格批量财报接口，输入/输出股票代码均为 6 位纯数字字符串。
    """
    import re, warnings

    # 1. 统一成 list
    if isinstance(security, str):
        security = [security]

    # 2. 解析 stat_date
    def parse_stat_date(sd):
        if sd is None:
            return None
        if (m := re.match(r"(\d{4})q([1-4])", str(sd), re.I)):
            y, q = m.groups()
            ends = ["03-31", "06-30", "09-30", "12-31"]
            return f"{y}-{ends[int(q)-1]}"
        return str(sd)

    stat_date_fmt = parse_stat_date(stat_date)

    # 3. 字段映射（按需补充）
    field_map: Dict[str, Dict[str, str]] = {
        # ---------- balance ----------
        "balance": {
            "report_date": "报告日",
            "total_assets": "资产总计",
            "total_liability": "负债合计",
            "total_equity": "负债及股东权益总计",
            "cash_and_deposit_in_cb": "现金及存放中央银行款项",
            "cash": "其中:现金",
            "deposit_in_cb": "存放中央银行款",
            "settlement_provision": "结算准备金",
            "precious_metals": "贵金属",
            "deposit_in_other_banks": "存放同业款项",
            "margin_paid": "存出保证金",
            "lending_to_other_banks": "拆出资金",
            "trading_金融资产": "交易性金融资产",
            "reverse_repo": "买入返售金融资产",
            "accounts_receivable": "应收账款",
            "notes_receivable": "应收票据",
            "receivable_financing": "应收款项融资",
            "prepayments": "预付账款",
            "interest_receivable": "应收利息",
            "dividend_receivable": "应收股利",
            "inventory": "存货",
            "non_current_asset_held_for_sale": "划分为持有待售的资产",
            "available_for_sale": "可供出售金融资产",
            "fair_value_through_oci": "以公允价值计量且其变动计入其他综合收益的金融资产",
            "amortized_cost": "以摊余成本计量的金融资产",
            "long_term_equity_investment": "长期股权投资",
            "debt_investment": "债权投资",
            "other_debt_investment": "其他债权投资",
            "investment_property": "投资性房地产",
            "derivative_financial_assets": "衍生金融工具资产",
            "other_equity_investment": "其他权益工具投资",
            "long_term_receivables": "长期应收款",
            "fixed_assets_original": "固定资产原值",
            "accumulated_depreciation": "减:累计折旧",
            "fixed_assets_net": "固定资产净值",
            "construction_in_progress": "在建工程",
            "fixed_assets_total": "固定资产合计",
            "right_of_use_asset": "使用权资产",
            "intangible_assets": "无形资产",
            "goodwill": "商誉",
            "long_term_prepaid_expenses": "长期待摊费用",
            "deferred_tax_assets": "递延所得税资产",
            "short_term_borrowing": "短期借款",
            "borrowing_from_cb": "向中央银行借款",
            "customer_deposits": "客户存款(吸收存款)",
            "placements_from_banks": "同业存入及拆入",
            "borrowing_from_banks": "拆入资金",
            "trading_financial_liabilities": "交易性金融负债",
            "repo_sold": "卖出回购金融资产款",
            "notes_payable": "应付票据",
            "accounts_payable": "应付账款",
            "advances_from_customers": "预收账款",
            "salaries_payable": "应付职工薪酬",
            "taxes_payable": "应交税费",
            "interest_payable": "应付利息",
            "dividends_payable": "应付股利",
            "other_payables": "其他应付款",
            "contract_liabilities": "合同负债",
            "lease_liabilities": "租赁负债",
            "long_term_borrowing": "长期借款",
            "bonds_payable": "应付债券",
            "deferred_tax_liabilities": "递延所得税负债",
            "share_capital": "股本",
            "capital_reserve": "资本公积",
            "other_comprehensive_income": "其他综合收益",
            "surplus_reserve": "盈余公积",
            "retained_earnings": "未分配利润",
        },

        # ---------- income ----------
        "income": {
            "report_date": "报告日",
            "total_operating_revenue": "营业收入",
            "net_interest_income": "净利息收入",
            "interest_income": "利息收入",
            "interest_expense": "利息支出",
            "net_fee_and_commission_income": "手续费及佣金净收入",
            "fee_and_commission_income": "手续费及佣金收入",
            "fee_and_commission_expense": "手续费及佣金支出",
            "net_trading_income": "净交易收入",
            "investment_income": "投资收益",
            "fair_value_change_income": "公允价值变动收益/(损失)",
            "operating_profit": "营业利润",
            "total_profit": "利润总额",
            "income_tax": "所得税",
            "net_profit": "净利润",
            "net_profit_parent": "归属于母公司的净利润",
            "basic_eps": "基本每股收益",
            "diluted_eps": "稀释每股收益",
            "other_comprehensive_income": "其他综合收益",
            "total_comprehensive_income": "综合收益总额",
        },

        # ---------- cashflow ----------
        "cashflow": {
            "report_date": "报告日",
            "net_cash_operating": "经营活动产生的现金流量净额",
            "net_cash_investing": "投资活动产生的现金流量净额",
            "net_cash_financing": "筹资活动产生的现金流量净额",
            "net_increase_cce": "现金及现金等价物净增加额",
            "cash_end_period": "期末现金及现金等价物余额",
            "cash_begin_period": "期初现金及现金等价物余额",
            "customer_deposits_net_increase": "客户存款和同业存放款项净增加额",
            "loans_and_advances_net_increase": "客户贷款及垫款净增加额",
            "interest_received": "收到利息",
            "interest_paid": "支付的利息",
            "taxes_paid": "支付的各项税费",
            "fixed_assets_purchase": "购建固定资产、无形资产和其他长期资产支付的现金",
            "debt_repayment": "偿还债务所支付的现金",
            "dividends_paid": "分配股利、利润或偿付利息支付的现金",
        }
    }

    # 4. 按表名分组
    table_fields = {"cashflow": [], "income": [], "balance": []}
    for f in fields:
        if f.startswith("cashflow."):
            table_fields["cashflow"].append(f.split(".", 1)[1])
        elif f.startswith("income."):
            table_fields["income"].append(f.split(".", 1)[1])
        elif f.startswith("balance."):
            table_fields["balance"].append(f.split(".", 1)[1])
        else:
            warnings.warn(f"字段 {f} 未指定表前缀，已跳过")

    # 5. 取数 & 拼接
    dfs = []
    for code in security:
        # 内部用 _add_prefix 拿数据，但返回仍用原 6 位数字
        ak_code = _add_prefix(code)          # 仅内部 OSS key 用
        out = pd.DataFrame()

        for table, sub_fields in table_fields.items():
            if not sub_fields:
                continue
            df = globals()[f"get_{table}"](ak_code,
                                           date=watch_date,
                                           report_type=report_type)

            if stat_date_fmt:
                df = df[df["报告日"] == stat_date_fmt]
            df = df.head(count)
            if df.empty:
                continue

            df = df.copy()
            df["code"] = code                # 保持 6 位数字
            df["statDate"] = df["报告日"].dt.strftime("%Y-%m-%d")

            for f in sub_fields:
                col = field_map[table].get(f, f)
                df[f"{table}.{f}"] = df[col] if col in df.columns else None

            keep_cols = ["code", "statDate"] + [f"{table}.{f}" for f in sub_fields]
            df = df[keep_cols]

            if out.empty:
                out = df
            else:
                out = pd.merge(out, df, on=["code", "statDate"], how="outer")

        if not out.empty:
            dfs.append(out)

    if not dfs:
        return pd.DataFrame(columns=["code", "statDate"] + fields)

    result = pd.concat(dfs, ignore_index=True).set_index(["code", "statDate"])
    return result

# ---------- 核心：拉取一张表的原始字段 ----------
def print_table_columns(table: Literal["balance", "income", "cashflow"], code: str = "000001"):
    dirs = {
        "balance": "jukuan/stock_financial_report_sina",
        "income":  "jukuan/stock_financial_report_sina_lirun",
        "cashflow":"jukuan/stock_financial_report_sina_xianjinliu",
    }
    key = f"{dirs[table]}/{_add_prefix(code)}.csv"
    try:
        content = bucket.get_object(key).read()
    except oss2.exceptions.NoSuchKey:
        print(f"{table} 表文件不存在：{key}")
        return

    df = pd.read_csv(io.BytesIO(content), nrows=0)  # 只读表头
    print(f"\n【{table.upper()} 原始字段列表】")
    for c in df.columns:
        print(f'"{c}",')


# # ---------- 依次打印三张表 ----------
# print_table_columns("balance")
# print_table_columns("income")
# print_table_columns("cashflow")

# ---------- 示例 ----------
# if __name__ == "__main__":
#     # 取平安银行 2024 年报 & 2025Q1 两期数据
#     df = get_history_fundamentals(
#         security=["000001"],
#         fields=["balance.total_assets", "income.net_profit", "cashflow.net_cash_operating"],
#         stat_date="2024q4",
#         count=2,
#     )
#     print(df)


# ---------- 示例 ----------
# if __name__ == "__main__":
#     # 以下三种写法等价，均能拿到 sh600000.csv
#     print(get_cashflow("600000").head())      # 自动补前缀
#     print(get_cashflow("sh600000").head())    # 已带前缀
#     print(get_cashflow(600000).head())        # 纯数字

# # ---------- 示例 ----------
# if __name__ == "__main__":
#     print(get_index_stocks("000300"))               # 默认当前时间
#     print(get_index_stocks("000012", datetime(2025, 8, 22, 15, 30, 0))) 
#     print(get_index_stocks("000300", "2024-12-30"))

    
# -----------------------------------------------------
# 本地调试：可以传空 event，会用默认值
# -----------------------------------------------------
# if __name__ == "__main__":
    # 测试：自定义 event
    # test_event = {
    #     "codes": ['002001', '002003', '002004', '002006', '002007', '002008', '002009', '002010', '002011', '002012', '002014', '002015', '002016', '002017', '002019', '002020', '002021', '002022', '002023', '002025'],
    #     "factor_name": "net_profit_growth_rate"
    # }

    # result = handler(test_event, {})
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    # import os
    # import oss2

    # ---------------------
    # 配置区
    # ---------------------
    # auth = oss2.Auth('你的AccessKeyId', '你的AccessKeySecret')
    # bucket = oss2.Bucket(auth, 'https://oss-cn-hangzhou.aliyuncs.com', '你的Bucket名字')

    # 本地要上传的文件夹
    # local_dir = r"/workspace/alphalens-reloaded/index_data"
    # # OSS 目标目录（相当于前缀）
    # oss_prefix = "index/"

    # ---------------------
    # 递归上传函数
    # ---------------------
    # def upload_folder(local_dir, oss_prefix):
        
    #     for root, dirs, files in os.walk(local_dir):
    #         for filename in files:
    #             local_path = os.path.join(root, filename)
                
    #             # 计算 OSS 路径（保持目录结构）
    #             relative_path = os.path.relpath(local_path, local_dir)
    #             oss_key = os.path.join(oss_prefix, relative_path).replace("\\", "/")

    #             print(f"上传 {local_path} -> {oss_key}")
    #             bucket.put_object_from_file(oss_key, local_path)

    # # 执行上传
    # upload_folder(local_dir, oss_prefix)
    # print("✅ 上传完成！")
    # get_index_stocks("000002")
    

def get_trading_dates(
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    as_str: bool = False
) -> List[dt.date] | List[str]:
    """
    获取 [start, end] 区间内的所有 A 股交易日
    
    参数
    ----
    start, end : str | date | datetime
        开始和结束日期，可以传 '2025-08-25' / '20250825' / dt.date / dt.datetime
    as_str : bool
        是否返回字符串格式（YYYYMMDD）。默认 False，返回 dt.date。
    
    返回
    ----
    List[date] 或 List[str]
    """
    # 1. 转换成 date
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
    
    # 2. 逐日循环
    trading_days = []
    if CALENDAR_AVAILABLE and calendar is not None:
        is_workday = calendar.is_workday  # type: ignore[attr-defined]
    else:
        is_workday = lambda d: d.weekday() < 5  # noqa: E731

    current = start_date
    while current <= end_date:
        if is_workday(current):
            trading_days.append(current)
        current += dt.timedelta(days=1)
    
    # 3. 输出格式
    if as_str:
        return [d.strftime("%Y%m%d") for d in trading_days]
    return trading_days


"""
snapshot_bt_adapter.py
在 load_new_stocks 的外面再包一层，生成 Backtrader 可用的 PandasData 对象。
"""



# ------------------------------------------------------------------
# 1. 把快照宽表（index=date, columns=code, values=今开）转成长表 OHLCV
# ------------------------------------------------------------------
# def _wide_to_ohlcv(wide: pd.DataFrame) -> pd.DataFrame:
#     """
#     wide : 由 load_new_stocks 返回的 DataFrame
#            index 为 date, columns 为股票代码, values 为今开
#     return: 长表 DataFrame[date, asset, open, high, low, close, volume]
#              其中 open/high/low/close 都用今开填充，volume 用 0
#     """
#     # 拉成长表
#     long = (
#         wide.stack()
#         .rename("close")
#         .reset_index()
#         .rename(columns={"level_1": "asset"})
#     )

#     # 复制 OHLC
#     long["open"]  = long["close"]
#     long["high"]  = long["close"]
#     long["low"]   = long["close"]
#     long["volume"] = 0.0          # 没有成交量
#     return long[["date", "asset", "open", "high", "low", "close", "volume"]]
def _wide_to_ohlcv(wide: pd.DataFrame) -> pd.DataFrame:
    """
    将宽表或收盘后 CSV 快照转换成长表 OHLCV。
    
    支持两种输入：
    1) 宽表：index=date, columns=股票代码, values=今开
    2) CSV快照：列包括 '代码','今开','最高','最低','最新价','成交量' 等
    
    return: DataFrame[date, asset, open, high, low, close, volume]
    """
    df = wide.copy()
    snapshot_cols = {"代码", "今开", "最高", "最低", "最新价", "成交量"}

    if snapshot_cols.issubset(df.columns):
        # 处理单日 CSV 快照
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

    # 处理宽表：index 为日期，列为股票代码，值为价格
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
    从 OSS 目录 /stock_zh_a_spot_em/ 拉取 start~end 区间内的所有快照，
    返回 DataFrame，保留原始字段（'代码','今开','最新价','最高','最低','成交量'等）
    """
    start_date = _normalize_date_arg(start, default=dt.date(2000, 1, 1), as_date=True)
    end_date = _normalize_date_arg(end, default=dt.date.today(), as_date=True)

    file_map = _collect_files(start_date, end_date)
    if not file_map:
        return pd.DataFrame()   # 无数据

    normalized_codes = _normalize_code_arg(codes)

    frames = []
    for file_date, key in sorted(file_map.items()):
        local_path = f"/tmp/{key.split('/')[-1]}"
        bucket.get_object_to_file(key, local_path)

        df = pd.read_csv(local_path, dtype={"代码": str})

        # 只过滤指定股票
        if normalized_codes:
            df = df[df["代码"].isin(normalized_codes)]

        df["date"] = pd.to_datetime(file_date)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    
    # 返回原始长表，不 pivot 成宽表
    # 让 _wide_to_ohlcv 根据列名处理 CSV 或宽表
    return df_all


# ------------------------------------------------------------------
# 2. 生成 Backtrader PandasData 对象
# ------------------------------------------------------------------
def load_bt_stocks(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> Dict[str, bt.feeds.PandasData]:
    """
    一次加载多只股票的 Backtrader 数据，返回字典 {code: PandasData}。
    规则：
      - 如果整只股票没有数据，跳过
      - 如果 close 列有任意 NaN，整只股票跳过，并打印警告
    """
    if isinstance(codes, str):
        codes = [codes]

    # 1) 读取 OSS CSV 长表
    wide = load_bt_oss_stocks(codes=codes, start=start, end=end)
    if wide.empty:
        print("没有任何股票历史行情数据")
        return {}

    # 2) 转 OHLCV
    ohlcv = _wide_to_ohlcv(wide)

    feeds: Dict[str, bt.feeds.PandasData] = {}

    normalized_codes = _normalize_code_arg(codes, allow_none=False) or []
    for code in normalized_codes:
        sub = ohlcv[ohlcv["asset"] == code].copy()
        if sub.empty:
            print(f"跳过股票 {code}, 没有历史行情数据")
            continue

        # 检查 close 是否有 NaN
        if sub["close"].isna().any():
            print(f"跳过股票 {code}, close 列存在 NaN：")
            print(sub[sub["close"].isna()])
            continue

        # 设置 index 并排序
        sub.set_index("date", inplace=True)
        sub.sort_index(inplace=True)

        # 转 PandasData
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



# ------------------------------------------------------------------
# 4. 使用示例
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     feeds = snapshot_to_bt_feeds(codes=["000001", "600519"], start="2024-01-01")
#     cerebro = bt.Cerebro()
#     for code, data in feeds.items():
#         cerebro.adddata(data)
#     # 接下来继续你的策略、运行回测...


def get_index_daily(
    index_symbol: str,
    start: Union[str, dt_date, dt_datetime],
    end: Union[str, dt_date, dt_datetime],
) -> pd.Series:
    """
    从 OSS 获取指数行情，并计算指定区间的净值序列 (benchmark NAV)。

    参数
    ----
    index_symbol : str
        指数代码，如 '000300'，在 OSS 中对应文件前缀 stock_zh_index_daily/{index_symbol}_*.csv
    start : str | datetime.date | datetime.datetime
        起始日期
    end : str | datetime.date | datetime.datetime
        结束日期

    返回
    ----
    pd.Series
        指数净值序列（起始点归一化为 1），index 为日期
    """
    start, end = pd.to_datetime(start), pd.to_datetime(end)

    prefix = f"stock_zh_index_daily/{index_symbol}"
    dfs = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        if obj.key.endswith(".csv"):
            content = bucket.get_object(obj.key).read()
            df = pd.read_csv(io.BytesIO(content))
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"未找到指数 {index_symbol} 的行情数据")

    # 合并并转换日期
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values("date")

    # 裁剪区间
    mask = (all_df["date"] >= start) & (all_df["date"] <= end)
    df = all_df.loc[mask].copy()

    if df.empty:
        raise ValueError(f"{index_symbol} 在 {start}~{end} 区间没有数据")

    # 计算归一化净值
    df["nav"] = df["close"] / df["close"].iloc[0]

    return df.set_index("date")["nav"]


def load_bt_pricing(
    codes: Union[str, List[str]] = None,
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    """
    基于 load_bt_stocks 生成 alphalens 需要的 pricing DataFrame。
    返回：
        pd.DataFrame
        - index  : DatetimeIndex，日期
        - columns: 股票代码（str）
        - values : 收盘价
    """
    # 1. 调用你已有的函数，拿到 {code: PandasData}
    feeds = load_bt_stocks(codes=codes, start=start, end=end)

    if not feeds:
        # 没有任何有效股票
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    # 2. 逐个把 PandasData 里的 DataFrame 拿出来拼到一张宽表
    frames = []
    for code, data in feeds.items():
        df = data.params.dataname.copy()
        df = df[["close"]].rename(columns={"close": code})
        frames.append(df)

    # 3. 横向合并
    pricing = pd.concat(frames, axis=1)

    # 4. 统一索引类型并排序
    if not isinstance(pricing.index, pd.DatetimeIndex):
        pricing.index = pd.to_datetime(pricing.index)
    pricing = pricing.sort_index()

    return pricing


MAPPING_FILE = "all_a_stocks.csv"    # 股票代码-名称映射 CSV


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

INDUSTRY_CSV_PATH = os.path.join(os.path.dirname(__file__), "config", "industry_category.csv")
_industry_df = None


def get_industry_category(codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[str, dict, None]:
    """
    获取股票行业板块。支持多种股票代码输入格式。
    单个代码返回一个字符串，多代码返回 dict。
    没找到行业返回 None。
    """
    global _industry_df
    if _industry_df is None:
        _industry_df = pd.read_csv(INDUSTRY_CSV_PATH)
        _industry_df["code"] = _industry_df["code"].str.strip().str.upper()
        _industry_df.set_index("code", inplace=True)
    
    def query(code):
        suffix_code = _ensure_exchange_suffix(code)
        return _industry_df["category"].get(suffix_code, None)

    # 统一规范：返回中的股票代码一律为6位数字字符串
    normalized = _normalize_code_arg(codes, allow_none=False)

    if isinstance(codes, (str, int)):
        return query(normalized[0])
    else:
        result = {}
        for code6 in normalized:
            result[code6] = query(code6)
        return result


# 概念板块（一个代码可对应多个概念）
CONCEPT_CSV_PATH = os.path.join(os.path.dirname(__file__), "config", "concept_category.csv")
_concept_df = None


def get_concept_categories(codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[List[str], Dict[str, List[str]]]:
    """
    获取股票概念板块（一个股票可能对应多个概念）。

    - 入参支持多种股票代码格式（6位、带前缀 sh/sz/bj、带后缀 .XSHG/.XSHE/.XBJ）。
    - 单个代码返回 List[str]（概念列表，按文件顺序去重）。
    - 多个代码返回 Dict[str, List[str]]，字典 key 统一为 6 位数字字符串。
    - 若某代码无概念，返回空列表。
    """
    global _concept_df
    if _concept_df is None:
        df = pd.read_csv(CONCEPT_CSV_PATH)
        # 规范列名与数据形态
        df.columns = [c.strip().lower() for c in df.columns]
        # 仅保留必要列
        if not {"code", "category"}.issubset(df.columns):
            raise ValueError("concept_category.csv 需包含 'code','category' 列")
        df["code"] = df["code"].astype(str).str.strip().str.upper()
        df["category"] = df["category"].astype(str).str.strip()
        _concept_df = df

    normalized = _normalize_code_arg(codes, allow_none=False)

    def lookup(norm_code: str) -> List[str]:
        suffix_code = _ensure_exchange_suffix(norm_code)
        rows = _concept_df[_concept_df["code"] == suffix_code]["category"].tolist()
        # 去重，保顺序
        seen = set()
        result: List[str] = []
        for item in rows:
            if item not in seen and item:
                seen.add(item)
                result.append(item)
        return result

    if isinstance(codes, (str, int)):
        return lookup(normalized[0])
    else:
        out: Dict[str, List[str]] = {}
        for norm_code in normalized:
            out[norm_code] = lookup(norm_code)
        return out