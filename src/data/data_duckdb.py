"""
DuckDB 数据源模块

提供与 data.py 相同的接口，但数据从 DuckDB 数据库读取
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Iterable, Literal, Any
from pathlib import Path
import datetime as dt
from datetime import date, datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import duckdb
except ImportError:
    print("警告: 未安装 duckdb，请运行 'pip install duckdb' 安装")
    duckdb = None

from .duckdb_storage import DuckDBStorage


class DuckDBDataSource:
    """
    DuckDB 数据源类

    提供与 data.py 相同的接口，从 DuckDB 数据库读取数据
    """

    def __init__(self, db_path: str = './data/ohlcv_data.duckdb'):
        """
        初始化 DuckDB 数据源

        Args:
            db_path: DuckDB 数据库文件路径
        """
        self.db_path = db_path
        self._storage = DuckDBStorage(db_path)
        self._con = None

    def _get_connection(self):
        """获取数据库连接"""
        if self._con is None and duckdb is not None:
            self._con = duckdb.connect(self.db_path)
        return self._con

    def _normalize_date_arg(self, date_arg, default=None, as_date=False):
        """标准化日期参数"""
        if date_arg is None:
            result = default or pd.Timestamp.now()
        elif isinstance(date_arg, str):
            result = pd.to_datetime(date_arg)
        elif isinstance(date_arg, (date, datetime)):
            result = pd.Timestamp(date_arg)
        else:
            result = pd.Timestamp(date_arg)

        return result.date() if as_date else result

    def _normalize_code_arg(self, codes, allow_none=True):
        """标准化股票代码参数"""
        if codes is None:
            return None if allow_none else []

        if isinstance(codes, str):
            codes = [codes]
        elif not isinstance(codes, (list, tuple)):
            codes = [str(codes)]

        # 标准化为6位数字字符串
        normalized = []
        for code in codes:
            code_str = str(code).strip()
            # 移除交易所后缀
            code_str = code_str.replace('.XSHG', '').replace('.XSHE', '').replace('.XBJ', '')
            # 移除交易所前缀
            if code_str.startswith(('sh', 'sz', 'bj')):
                code_str = code_str[2:]
            # 补齐6位
            code_str = code_str.zfill(6)
            normalized.append(code_str)

        return normalized

    def _should_fetch_from_original(self, end_date: Union[str, dt.date, dt.datetime, None] = None, days_threshold: int = 3) -> bool:
        """
        判断是否应该从原始数据源获取数据
        
        Args:
            end_date: 结束日期
            days_threshold: 天数阈值，如果end_date在今天之前的days_threshold天内，则从原始数据源获取
            
        Returns:
            bool: True表示应该从原始数据源获取，False表示可以从DuckDB获取
        """
        if end_date is None:
            return True
            
        # 标准化日期
        end_ts = self._normalize_date_arg(end_date, as_date=True)
        today = dt.date.today()
        
        # 计算天数差
        days_diff = (today - end_ts).days
        
        # 如果end_date是今天或最近几天的数据，应该从原始数据源获取
        return days_diff <= days_threshold

    def load_oss_stocks(self, codes: Union[str, List[str]] = None,
                       start: str = None, end: str = None) -> pd.DataFrame:
        """
        从 DuckDB 加载股票收盘价数据

        优先从 DuckDB 获取，如果没有数据则从原始数据源获取并更新到 DuckDB
        """
        # 标准化参数
        start_date = self._normalize_date_arg(start, default=pd.Timestamp('2000-01-01'), as_date=True)
        end_date = self._normalize_date_arg(end, default=pd.Timestamp.now(), as_date=True)
        normalized_codes = self._normalize_code_arg(codes)

        # 从 DuckDB 查询数据
        df = self._storage.load_ohlcv_data(
            stock_codes=normalized_codes,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None
        )

        # 检查数据是否完整：如果请求的结束日期比库中最新的日期更新，就需要更新
        data_is_complete = True
        if not df.empty:
            # 检查库中最新的日期
            latest_date_in_db = pd.to_datetime(df['date']).max().date()
            # 如果请求的结束日期比库中最新的日期更新，数据不完整
            if end_date > latest_date_in_db:
                data_is_complete = False
                print(f"DuckDB 数据不完整，库中最新日期: {latest_date_in_db}, 请求结束日期: {end_date}")

        if df.empty or not data_is_complete:
            # DuckDB 中没有数据或数据不完整，从原始数据源获取并导入
            print(f"DuckDB 中没有完整OHLCV数据，正在从原始数据源获取: {normalized_codes}")
            try:
                from . import data
                original_df = data.load_oss_stocks(codes, start, end)
                
                if not original_df.empty:
                    # 导入到DuckDB
                    self._storage.save_ohlcv_data(original_df, normalized_codes)
                    return original_df
                    
            except Exception as e:
                print(f"从原始数据源获取并导入OHLCV数据失败: {e}")
                
            return pd.DataFrame(dtype=float)

        # 转换为宽表格式 (index=date, columns=stock_codes, values=close)
        prices = (
            df
            .drop_duplicates(subset=["date", "stock_code"], keep="last")
            .pivot(index="date", columns="stock_code", values="close")
            .sort_index()
        )

        return prices

    def load_oss_complex_stocks(self, codes: Union[str, List[str]] = None,
                               start: str = None, end: str = None,
                               fields: Union[str, List[str]] = "close") -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        从 DuckDB 加载复杂股票数据，支持多字段

        优先从 DuckDB 获取，如果没有数据则从原始数据源获取并更新到 DuckDB
        """
        # 标准化参数
        start_date = self._normalize_date_arg(start, default=pd.Timestamp('2000-01-01'), as_date=True)
        end_date = self._normalize_date_arg(end, default=pd.Timestamp.now(), as_date=True)
        normalized_codes = self._normalize_code_arg(codes)

        # 从 DuckDB 查询数据
        df = self._storage.load_ohlcv_data(
            stock_codes=normalized_codes,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None
        )

        # 检查数据是否完整：如果请求的结束日期比库中最新的日期更新，就需要更新
        data_is_complete = True
        if not df.empty:
            # 检查库中最新的日期
            latest_date_in_db = pd.to_datetime(df['date']).max().date()
            # 如果请求的结束日期比库中最新的日期更新，数据不完整
            if end_date > latest_date_in_db:
                data_is_complete = False
                print(f"DuckDB 数据不完整，库中最新日期: {latest_date_in_db}, 请求结束日期: {end_date}")

        if df.empty or not data_is_complete:
            # DuckDB 中没有数据或数据不完整，从原始数据源获取并导入
            print(f"DuckDB 中没有完整复杂OHLCV数据，正在从原始数据源获取: {normalized_codes}")
            try:
                from . import data
                original_result = data.load_oss_complex_stocks(codes, start, end, fields)
                
                # 处理不同类型的返回值
                if isinstance(original_result, pd.DataFrame) and not original_result.empty:
                    # 单字段情况：宽表格式，需要转换为长表
                    if isinstance(original_result.index, pd.DatetimeIndex):
                        # 转换为长表格式
                        long_df = original_result.stack().reset_index()
                        # 根据 fields 参数设置正确的列名
                        if isinstance(fields, str):
                            long_df.columns = ['date', 'stock_code', fields]
                        else:
                            # 默认使用 'close'
                            long_df.columns = ['date', 'stock_code', 'close']
                        
                        # 为所有期望的 OHLCV 字段设置值
                        field_value_col = fields if isinstance(fields, str) and fields in long_df.columns else long_df.columns[2]
                        
                        # 设置 OHLCV 字段
                        long_df['open'] = long_df[field_value_col]
                        long_df['high'] = long_df[field_value_col]
                        long_df['low'] = long_df[field_value_col]
                        long_df['close'] = long_df[field_value_col]
                        long_df['volume'] = 0.0
                        
                        # 设置额外的字段（如果不存在则设为默认值）
                        if 'amount' not in long_df.columns:
                            long_df['amount'] = 0.0
                        if 'outstanding_share' not in long_df.columns:
                            long_df['outstanding_share'] = 0.0
                        if 'turnover' not in long_df.columns:
                            long_df['turnover'] = 0.0
                        
                        # 确保列的顺序正确
                        expected_columns = ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover']
                        long_df = long_df[expected_columns]
                        
                        self._storage.save_ohlcv_data(long_df, normalized_codes)
                    return original_result
                    
                elif isinstance(original_result, dict) and original_result:
                    # 多字段情况：字典格式，需要转换为长表并导入
                    # 从字典中提取所有字段的数据
                    all_fields = list(original_result.keys())
                    if all_fields:
                        # 获取第一个字段的DataFrame来确定结构
                        first_field = all_fields[0]
                        first_df = original_result[first_field]
                        
                        if isinstance(first_df.index, pd.DatetimeIndex) and not first_df.empty:
                            # 创建一个新的DataFrame来存储所有数据
                            all_data = []
                            
                            # 获取所有唯一的日期和股票代码组合
                            dates = first_df.index
                            stock_codes = first_df.columns
                            
                            for date in dates:
                                for stock_code in stock_codes:
                                    row_data = {'date': date, 'stock_code': stock_code}
                                    
                                    # 为每个字段添加值
                                    for field in all_fields:
                                        field_df = original_result[field]
                                        if isinstance(field_df.index, pd.DatetimeIndex):
                                            try:
                                                value = field_df.loc[date, stock_code]
                                                if pd.notna(value):
                                                    row_data[field] = value
                                                else:
                                                    row_data[field] = 0.0 if field in ['volume', 'amount', 'outstanding_share', 'turnover'] else None
                                            except KeyError:
                                                row_data[field] = 0.0 if field in ['volume', 'amount', 'outstanding_share', 'turnover'] else None
                                        else:
                                            row_data[field] = 0.0 if field in ['volume', 'amount', 'outstanding_share', 'turnover'] else None
                                    
                                    # 只在至少有一个非空值时才添加行
                                    if any(pd.notna(v) and v != 0.0 for k, v in row_data.items() if k not in ['date', 'stock_code']):
                                        all_data.append(row_data)
                            
                            if all_data:
                                long_df = pd.DataFrame(all_data)
                                
                                # 重命名列以匹配DuckDB存储格式
                                column_mapping = {
                                    'open': 'open',
                                    'high': 'high', 
                                    'low': 'low',
                                    'close': 'close',
                                    'volume': 'volume',
                                    'amount': 'amount',
                                    'outstanding_share': 'outstanding_share',
                                    'turnover': 'turnover'
                                }
                                
                                # 只保留OHLCV相关的列
                                keep_columns = ['date', 'stock_code'] + [col for col in long_df.columns 
                                                                       if col in column_mapping]
                                long_df = long_df[keep_columns]
                                
                                # 重命名列
                                long_df = long_df.rename(columns={col: column_mapping.get(col, col) 
                                                                for col in long_df.columns 
                                                                if col in column_mapping})
                                
                                # 确保所有期望的列都存在
                                expected_columns = ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover']
                                for col in expected_columns:
                                    if col not in long_df.columns:
                                        if col in ['amount', 'outstanding_share', 'turnover']:
                                            long_df[col] = 0.0
                                        elif col == 'volume':
                                            long_df[col] = 0.0
                                        else:
                                            # 对于 OHLC 字段，如果不存在，使用 close 的值
                                            long_df[col] = long_df['close'] if 'close' in long_df.columns else 0.0
                                
                                # 确保列的顺序正确
                                long_df = long_df[expected_columns]
                                
                                self._storage.save_ohlcv_data(long_df, normalized_codes)
                    
                    return original_result
                    
            except Exception as e:
                print(f"从原始数据源获取并导入复杂OHLCV数据失败: {e}")
                
            if isinstance(fields, str) and fields.lower() == "all":
                return {}
            if isinstance(fields, list):
                return {}
            return pd.DataFrame(dtype=float)

        # 根据 fields 返回
        if isinstance(fields, str) and fields.lower() == "all":
            # 全部字段转宽表
            value_cols = [c for c in df.columns if c not in ["date", "stock_code"]]
            result = {col: df.pivot(index="date", columns="stock_code", values=col).sort_index()
                     for col in value_cols if col in df.columns}
            # 重命名列索引名为 'asset' 以保持与 data.py 接口一致
            result = {col: df_result.rename_axis(columns='asset') for col, df_result in result.items()}
            return result

        elif isinstance(fields, str):
            # 单个字段
            if fields not in df.columns:
                return pd.DataFrame(dtype=float)
            result = df.pivot(index="date", columns="stock_code", values=fields).sort_index()
            # 重命名列索引名为 'asset' 以保持与 data.py 接口一致
            result = result.rename_axis(columns='asset')
            return result

        elif isinstance(fields, list):
            # 多个字段 -> dict
            result = {}
            for col in fields:
                if col in df.columns:
                    df_result = df.pivot(index="date", columns="stock_code", values=col).sort_index()
                    # 重命名列索引名为 'asset' 以保持与 data.py 接口一致
                    df_result = df_result.rename_axis(columns='asset')
                    result[col] = df_result
            return result

        else:
            raise ValueError("fields 必须是 'close' / 'all' / [字段列表]")

    def get_index_stocks(self, index_symbol: str, date: Optional[Union[str, date, datetime]] = None) -> List[str]:
        """
        获取指数成分股列表

        从原始 data.py 获取指数成分股数据
        """
        try:
            # 导入原始 data 模块
            from . import data
            return data.get_index_stocks(index_symbol, date)
        except Exception as e:
            print(f"从原始数据源获取指数成分股失败 {index_symbol}: {e}")
            return []

    def load_bt_oss_stocks(self, codes: Union[str, List[str]] = None,
                          start: str = None, end: str = None) -> pd.DataFrame:
        """
        从 DuckDB 加载快照数据（用于 Backtrader）

        优先从 DuckDB 获取，如果没有数据则从原始数据源获取并更新到 DuckDB
        返回长表格式 DataFrame，包含原始快照字段，与原始 data.py 保持一致
        """
        # 标准化参数
        start_date = self._normalize_date_arg(start, default=pd.Timestamp('2000-01-01'), as_date=True)
        end_date = self._normalize_date_arg(end, default=pd.Timestamp.now(), as_date=True)
        normalized_codes = self._normalize_code_arg(codes)

        # 从 DuckDB 查询数据
        df = self._storage.load_ohlcv_data(
            stock_codes=normalized_codes,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None
        )

        # 检查数据是否完整：如果请求的结束日期比库中最新的日期更新，就需要更新
        data_is_complete = True
        if not df.empty:
            # 检查库中最新的日期
            latest_date_in_db = pd.to_datetime(df['date']).max().date()
            # 如果请求的结束日期比库中最新的日期更新，数据不完整
            if end_date > latest_date_in_db:
                data_is_complete = False
                print(f"DuckDB 数据不完整，库中最新日期: {latest_date_in_db}, 请求结束日期: {end_date}")

        if df.empty or not data_is_complete:
            # DuckDB 中没有数据或数据不完整，从原始数据源获取并导入
            print(f"DuckDB 中没有完整快照数据，正在从原始数据源获取: {normalized_codes}")
            try:
                from . import data
                original_df = data.load_bt_oss_stocks(codes, start, end)
                
                if not original_df.empty:
                    # 将原始快照数据转换为OHLCV格式并导入DuckDB
                    # 假设原始快照数据包含OHLCV字段
                    if all(col in original_df.columns for col in ['今开', '最高', '最低', '最新价', '成交量']):
                        ohlcv_df = original_df[['date', '代码', '今开', '最高', '最低', '最新价', '成交量']].copy()
                        ohlcv_df.columns = ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume']
                        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
                        self._storage.save_ohlcv_data(ohlcv_df, normalized_codes)
                    return original_df
                    
            except Exception as e:
                print(f"从原始数据源获取并导入快照数据失败: {e}")
                
            return pd.DataFrame()

        # 转换为原始快照格式（模拟完整的快照字段）
        # 原始格式包含：'代码','今开','最高','最低','最新价','成交量','date'等字段
        snapshot_data = []
        for _, row in df.iterrows():
            snapshot_row = {
                '代码': row['stock_code'],
                '今开': row['open'],
                '最高': row['high'],
                '最低': row['low'],
                '最新价': row['close'],
                '成交量': row['volume'],
                'date': row['date'],
                # 添加其他原始快照字段（用相同值填充）
                '涨跌幅': 0.0,  # 暂时设为0
                '涨跌额': 0.0,  # 暂时设为0
                '振幅': 0.0,    # 暂时设为0
                '换手率': 0.0,  # 暂时设为0
                '市盈率-动态': 0.0,  # 暂时设为0
                '市净率': 0.0,  # 暂时设为0
                '总市值': 0.0,  # 暂时设为0
                '流通市值': 0.0,  # 暂时设为0
            }
            snapshot_data.append(snapshot_row)

        result_df = pd.DataFrame(snapshot_data)
        return result_df

    def get_industry_category(self, codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[str, dict, None]:
        """
        获取股票行业分类

        由于 DuckDB 中没有行业分类数据，这里返回 None 或空字典
        实际使用时需要从配置文件或其他数据源获取
        """
        print("警告: DuckDB 数据源不支持获取行业分类数据")
        normalized = self._normalize_code_arg(codes, allow_none=False)

        if isinstance(codes, (str, int)):
            return None
        else:
            return {code: None for code in normalized}

    def get_concept_categories(self, codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[List[str], Dict[str, List[str]]]:
        """
        获取股票概念分类

        由于 DuckDB 中没有概念分类数据，这里返回空列表
        实际使用时需要从配置文件或其他数据源获取
        """
        print("警告: DuckDB 数据源不支持获取概念分类数据")
        normalized = self._normalize_code_arg(codes, allow_none=False)

        if isinstance(codes, (str, int)):
            return []
        else:
            return {code: [] for code in normalized}

    def get_table_info(self) -> Dict[str, Any]:
        """获取数据库表信息"""
        return self._storage.get_table_info('ohlcv_data')

    def _load_financial_data_from_duckdb(self, data_type: str, stock_code: str = None,
                                        date: Union[str, dt.date, dt.datetime, None] = None,
                                        report_type: str = None, fields: List[str] = None) -> pd.DataFrame:
        """
        从 DuckDB 加载财务数据

        Args:
            data_type: 数据类型 ('balance', 'income', 'cashflow', 'valuation', 'history_fundamentals')
            stock_code: 股票代码
            date: 日期
            report_type: 报告类型
            fields: 字段列表（用于历史财务数据）

        Returns:
            pd.DataFrame: 财务数据
        """
        try:
            con = self._get_connection()
            if con is None:
                return pd.DataFrame()

            # 根据数据类型确定表名
            table_name = f"{data_type}_data"
            
            # 构建查询条件
            conditions = []
            if stock_code:
                conditions.append(f"stock_code = '{stock_code}'")
            if report_type and data_type in ['balance', 'income', 'cashflow']:
                conditions.append(f"report_type = '{report_type}'")
            if date:
                date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                # 根据数据类型使用不同的日期字段
                if data_type == 'valuation':
                    conditions.append(f"日期 >= '{date_str}'")
                else:
                    # 财务报表数据使用报告日
                    conditions.append(f"报告日 >= '{date_str}'")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # 查询数据
            query = f"SELECT * FROM {table_name} WHERE {where_clause}"
            df = con.execute(query).fetchdf()

            if df.empty:
                return pd.DataFrame()

            # 移除元数据列
            metadata_cols = ['stock_code', 'report_type', 'data_type', 'fields', 'interval']
            data_cols = [col for col in df.columns if col not in metadata_cols]
            result_df = df[data_cols].copy()

            # 设置正确的索引
            if data_type == 'valuation':
                if '日期' in result_df.columns:
                    result_df['日期'] = pd.to_datetime(result_df['日期'])
                    result_df = result_df.set_index('日期').sort_index()
            else:
                # 财务报表数据使用报告日作为索引
                if '报告日' in result_df.columns:
                    result_df['报告日'] = pd.to_datetime(result_df['报告日'])
                    result_df = result_df.set_index('报告日').sort_index()

            return result_df

        except Exception as e:
            print(f"从 DuckDB 查询财务数据失败 {data_type}: {e}")
            return pd.DataFrame()

    def get_balance(self, code: str,
                   date: Union[str, dt.date, dt.datetime, None] = None,
                   *,
                   report_type: str = "合并期末") -> pd.DataFrame:
        """
        获取资产负债表数据

        优先从 DuckDB 获取，如果没有数据则导入后获取
        """
        # 标准化股票代码
        normalized_code = self._normalize_code_arg(code, allow_none=False)[0]

        # 先尝试从 DuckDB 获取
        balance_df = self._load_financial_data_from_duckdb(
            'balance', normalized_code, date, report_type
        )

        if not balance_df.empty:
            return balance_df

        # DuckDB 中没有数据，导入数据
        print(f"DuckDB 中没有资产负债表数据，正在导入: {code}")
        try:
            # 导入原始 data 模块
            from . import data
            # 调用导入函数
            from .import_data_to_duckdb import import_balance_data_to_duckdb
            success = import_balance_data_to_duckdb(code, self.db_path, date, report_type)

            if success:
                # 重新从 DuckDB 获取
                balance_df = self._load_financial_data_from_duckdb(
                    'balance', normalized_code, date, report_type
                )
                if not balance_df.empty:
                    return balance_df

        except Exception as e:
            print(f"导入资产负债表数据失败 {code}: {e}")

        # 如果导入失败，返回原始数据
        try:
            from . import data
            return data.get_balance(code, date, report_type=report_type)
        except Exception as e:
            print(f"从原始数据源获取资产负债表数据失败 {code}: {e}")
            return pd.DataFrame()

    def get_income(self, code: str,
                  date: Union[str, dt.date, dt.datetime, None] = None,
                  *,
                  report_type: str = "合并期末") -> pd.DataFrame:
        """
        获取利润表数据

        优先从 DuckDB 获取，如果没有数据则导入后获取
        """
        # 标准化股票代码
        normalized_code = self._normalize_code_arg(code, allow_none=False)[0]

        # 先尝试从 DuckDB 获取
        income_df = self._load_financial_data_from_duckdb(
            'income', normalized_code, date, report_type
        )

        if not income_df.empty:
            return income_df

        # DuckDB 中没有数据，导入数据
        print(f"DuckDB 中没有利润表数据，正在导入: {code}")
        try:
            # 调用导入函数
            from .import_data_to_duckdb import import_income_data_to_duckdb
            success = import_income_data_to_duckdb(code, self.db_path, date, report_type)

            if success:
                # 重新从 DuckDB 获取
                income_df = self._load_financial_data_from_duckdb(
                    'income', normalized_code, date, report_type
                )
                if not income_df.empty:
                    return income_df

        except Exception as e:
            print(f"导入利润表数据失败 {code}: {e}")

        # 如果导入失败，返回原始数据
        try:
            from . import data
            return data.get_income(code, date, report_type=report_type)
        except Exception as e:
            print(f"从原始数据源获取利润表数据失败 {code}: {e}")
            return pd.DataFrame()

    def get_cashflow(self, code: str,
                    date: Union[str, dt.date, dt.datetime, None] = None,
                    *,
                    report_type: str = "合并期末") -> pd.DataFrame:
        """
        获取现金流量表数据

        优先从 DuckDB 获取，如果没有数据则导入后获取
        """
        # 标准化股票代码
        normalized_code = self._normalize_code_arg(code, allow_none=False)[0]

        # 先尝试从 DuckDB 获取
        cashflow_df = self._load_financial_data_from_duckdb(
            'cashflow', normalized_code, date, report_type
        )

        if not cashflow_df.empty:
            return cashflow_df

        # DuckDB 中没有数据，导入数据
        print(f"DuckDB 中没有现金流量表数据，正在导入: {code}")
        try:
            # 调用导入函数
            from .import_data_to_duckdb import import_cashflow_data_to_duckdb
            success = import_cashflow_data_to_duckdb(code, self.db_path, date, report_type)

            if success:
                # 重新从 DuckDB 获取
                cashflow_df = self._load_financial_data_from_duckdb(
                    'cashflow', normalized_code, date, report_type
                )
                if not cashflow_df.empty:
                    return cashflow_df

        except Exception as e:
            print(f"导入现金流量表数据失败 {code}: {e}")

        # 如果导入失败，返回原始数据
        try:
            from . import data
            return data.get_cashflow(code, date, report_type=report_type)
        except Exception as e:
            print(f"从原始数据源获取现金流量表数据失败 {code}: {e}")
            return pd.DataFrame()

    def get_valuation(self, code: str,
                     date: Union[str, dt.date, dt.datetime, None] = None) -> pd.DataFrame:
        """
        获取估值数据

        优先从 DuckDB 获取，如果没有数据则导入后获取
        """
        # 标准化股票代码
        normalized_code = self._normalize_code_arg(code, allow_none=False)[0]

        # 先尝试从 DuckDB 获取
        valuation_df = self._load_financial_data_from_duckdb(
            'valuation', normalized_code, date
        )

        if not valuation_df.empty:
            return valuation_df

        # DuckDB 中没有数据，导入数据
        print(f"DuckDB 中没有估值数据，正在导入: {code}")
        try:
            # 调用导入函数
            from .import_data_to_duckdb import import_valuation_data_to_duckdb
            success = import_valuation_data_to_duckdb(code, self.db_path, date)

            if success:
                # 重新从 DuckDB 获取
                valuation_df = self._load_financial_data_from_duckdb(
                    'valuation', normalized_code, date
                )
                if not valuation_df.empty:
                    return valuation_df

        except Exception as e:
            print(f"导入估值数据失败 {code}: {e}")

        # 如果导入失败，返回原始数据
        try:
            from . import data
            return data.get_valuation(code, date)
        except Exception as e:
            print(f"从原始数据源获取估值数据失败 {code}: {e}")
            return pd.DataFrame()

    def get_history_fundamentals(self, security: Union[str, List[str]],
                               fields: List[str],
                               watch_date: Union[str, dt.date, dt.datetime, None] = None,
                               stat_date: Union[str, None] = None,
                               count: int = 1,
                               interval: str = "1q",
                               report_type: str = "合并期末") -> pd.DataFrame:
        """
        获取历史财务数据

        优先从 DuckDB 获取，如果没有数据则导入后获取
        """
        # TODO: 尝试从 DuckDB 获取历史财务数据
        # 目前历史财务数据还没有导入到 DuckDB，所以直接从原始数据源获取

        try:
            # 导入原始 data 模块
            from . import data
            return data.get_history_fundamentals(security, fields, watch_date, stat_date, count, interval, report_type)
        except Exception as e:
            print(f"从原始数据源获取历史财务数据失败: {e}")
            return pd.DataFrame()

    def get_trading_dates(self, start: Union[str, dt.date, dt.datetime],
                         end: Union[str, dt.date, dt.datetime],
                         as_str: bool = False) -> Union[List[dt.date], List[str]]:
        """
        获取交易日历

        从原始 data.py 获取交易日历数据
        """
        try:
            # 导入原始 data 模块
            from . import data
            return data.get_trading_dates(start, end, as_str)
        except Exception as e:
            print(f"从原始数据源获取交易日历失败: {e}")
            return [] if not as_str else []

    def get_index_daily(self, index_symbol: str,
                       start: Union[str, dt.date, dt.datetime],
                       end: Union[str, dt.date, dt.datetime]) -> pd.Series:
        """
        获取指数每日数据

        优先从 DuckDB 获取，如果没有数据则从原始 data.py 获取
        """
        # TODO: 尝试从 DuckDB 获取指数数据
        # 目前指数数据还没有导入到 DuckDB，所以直接从原始数据源获取

        try:
            # 导入原始 data 模块
            from . import data
            return data.get_index_daily(index_symbol, start, end)
        except Exception as e:
            print(f"从原始数据源获取指数数据失败 {index_symbol}: {e}")
            return pd.Series(dtype=float)

    def load_new_stocks(self, codes: Union[str, List[str]] = None,
                       start: str = None, end: str = None) -> pd.DataFrame:
        """
        加载新股数据

        优先从 DuckDB 获取，如果没有数据则从原始 data.py 获取
        """
        # TODO: 尝试从 DuckDB 获取新股数据
        # 目前新股数据还没有导入到 DuckDB，所以直接从原始数据源获取

        try:
            # 导入原始 data 模块
            from . import data
            return data.load_new_stocks(codes, start, end)
        except Exception as e:
            print(f"从原始数据源获取新股数据失败: {e}")
            return pd.DataFrame()

    def load_bt_stocks(self, codes: Union[str, List[str]] = None,
                      start: str = None, end: str = None) -> Dict[str, Any]:
        """
        加载 Backtrader 股票数据

        优先从 DuckDB 获取，如果没有数据则从原始 data.py 获取
        返回 {code: PandasData} 字典
        """
        # 导入必要的模块
        try:
            import backtrader as bt
        except ImportError:
            print("警告: 未安装 backtrader，无法创建 PandasData 对象")
            return {}

        if isinstance(codes, str):
            codes = [codes]

        # 1) 读取长表数据
        wide = self.load_bt_oss_stocks(codes=codes, start=start, end=end)
        if wide.empty:
            print("没有任何股票历史行情数据")
            return {}

        # 2) 转 OHLCV 格式（复用原始 data.py 的转换逻辑）
        # 模拟原始 data.py 中的 _wide_to_ohlcv 函数逻辑
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
        else:
            # 处理宽表（这里不应该到达，因为我们返回的是长表）
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

            ohlcv = long[["date", "asset", "open", "high", "low", "close", "volume"]]

        # 3) 为每只股票创建 PandasData 对象
        feeds = {}

        normalized_codes = self._normalize_code_arg(codes, allow_none=False) or []
        for code in normalized_codes:
            sub = ohlcv[ohlcv["asset"] == code].copy()
            if sub.empty:
                print(f"跳过股票 {code}, 没有历史行情数据")
                continue

            # 检查 close 是否有 NaN
            if sub["close"].isna().any():
                print(f"跳过股票 {code}, close 列存在 NaN")
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

    def load_bt_pricing(self, codes: Union[str, List[str]] = None,
                       start: str = None, end: str = None) -> pd.DataFrame:
        """
        加载定价数据

        优先从 DuckDB 获取，如果没有数据则从原始 data.py 获取
        """
        # TODO: 尝试从 DuckDB 获取快照数据
        # 目前快照数据还没有导入到 DuckDB，所以直接从原始数据源获取

        try:
            # 导入原始 data 模块
            from . import data
            return data.load_bt_pricing(codes, start, end)
        except Exception as e:
            print(f"从原始数据源获取快照数据失败: {e}")
            return pd.DataFrame()

    def close(self):
        """关闭数据库连接"""
        if self._con:
            self._con.close()
            self._con = None


# 全局实例
_duckdb_data_source = None

def get_duckdb_data_source(db_path: str = './data/ohlcv_data.duckdb') -> DuckDBDataSource:
    """获取 DuckDB 数据源实例"""
    global _duckdb_data_source
    if _duckdb_data_source is None or _duckdb_data_source.db_path != db_path:
        _duckdb_data_source = DuckDBDataSource(db_path)
    return _duckdb_data_source


# 便捷函数 - 与 data.py 接口保持一致
def load_oss_stocks(codes: Union[str, List[str]] = None,
                   start: str = None, end: str = None) -> pd.DataFrame:
    """从 DuckDB 加载股票收盘价数据"""
    return get_duckdb_data_source().load_oss_stocks(codes, start, end)

def load_oss_complex_stocks(codes: Union[str, List[str]] = None,
                           start: str = None, end: str = None,
                           fields: Union[str, List[str]] = "close") -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """从 DuckDB 加载复杂股票数据"""
    return get_duckdb_data_source().load_oss_complex_stocks(codes, start, end, fields)

def get_index_stocks(index_symbol: str, date: Optional[Union[str, date, datetime]] = None) -> List[str]:
    """获取指数成分股列表"""
    return get_duckdb_data_source().get_index_stocks(index_symbol, date)

def load_bt_oss_stocks(codes: Union[str, List[str]] = None,
                      start: str = None, end: str = None) -> pd.DataFrame:
    """从 DuckDB 加载快照数据"""
    return get_duckdb_data_source().load_bt_oss_stocks(codes, start, end)

def get_industry_category(codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[str, dict, None]:
    """获取行业分类"""
    return get_duckdb_data_source().get_industry_category(codes)

def get_concept_categories(codes: Union[str, int, Iterable[Union[str, int]]]) -> Union[List[str], Dict[str, List[str]]]:
    """获取概念分类"""
    return get_duckdb_data_source().get_concept_categories(codes)


# 财务数据相关函数
def get_balance(code: str,
               date: Union[str, dt.date, dt.datetime, None] = None,
               *,
               report_type: str = "合并期末") -> pd.DataFrame:
    """获取资产负债表数据"""
    return get_duckdb_data_source().get_balance(code, date, report_type=report_type)

def get_income(code: str,
              date: Union[str, dt.date, dt.datetime, None] = None,
              *,
              report_type: str = "合并期末") -> pd.DataFrame:
    """获取利润表数据"""
    return get_duckdb_data_source().get_income(code, date, report_type=report_type)

def get_cashflow(code: str,
                date: Union[str, dt.date, dt.datetime, None] = None,
                *,
                report_type: str = "合并期末") -> pd.DataFrame:
    """获取现金流量表数据"""
    return get_duckdb_data_source().get_cashflow(code, date, report_type=report_type)

def get_valuation(code: str,
                 date: Union[str, dt.date, dt.datetime, None] = None) -> pd.DataFrame:
    """获取估值数据"""
    return get_duckdb_data_source().get_valuation(code, date)

def get_history_fundamentals(security: Union[str, List[str]],
                           fields: List[str],
                           watch_date: Union[str, dt.date, dt.datetime, None] = None,
                           stat_date: Union[str, None] = None,
                           count: int = 1,
                           interval: str = "1q",
                           report_type: str = "合并期末") -> pd.DataFrame:
    """获取历史财务数据"""
    return get_duckdb_data_source().get_history_fundamentals(
        security, fields, watch_date, stat_date, count, interval, report_type
    )

def get_trading_dates(start: Union[str, dt.date, dt.datetime],
                     end: Union[str, dt.date, dt.datetime],
                     as_str: bool = False) -> Union[List[dt.date], List[str]]:
    """获取交易日历"""
    return get_duckdb_data_source().get_trading_dates(start, end, as_str)

def get_index_daily(index_symbol: str,
                   start: Union[str, dt.date, dt.datetime],
                   end: Union[str, dt.date, dt.datetime]) -> pd.Series:
    """获取指数每日数据"""
    return get_duckdb_data_source().get_index_daily(index_symbol, start, end)

def load_new_stocks(codes: Union[str, List[str]] = None,
                   start: str = None, end: str = None) -> pd.DataFrame:
    """加载新股数据"""
    return get_duckdb_data_source().load_new_stocks(codes, start, end)

def load_bt_stocks(codes: Union[str, List[str]] = None,
                  start: str = None, end: str = None) -> Dict[str, Any]:
    """加载 Backtrader 股票数据"""
    return get_duckdb_data_source().load_bt_stocks(codes, start, end)

def load_bt_pricing(codes: Union[str, List[str]] = None,
                   start: str = None, end: str = None) -> pd.DataFrame:
    """加载定价数据"""
    return get_duckdb_data_source().load_bt_pricing(codes, start, end)


# 测试函数
def test_duckdb_data_source():
    """测试 DuckDB 数据源功能"""
    print("=== 测试 DuckDB 数据源 ===\n")

    # 创建数据源实例
    data_source = get_duckdb_data_source('./data/test_ohlcv.duckdb')

    # 获取表信息
    print("1. 数据库表信息:")
    info = data_source.get_table_info()
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("  无表信息")
    print()

    # 测试加载数据
    print("2. 测试 load_oss_stocks:")
    df = data_source.load_oss_stocks(['000001', '000002'], '2024-01-01', '2024-01-05')
    print(f"  加载到 {len(df)} 行数据")
    if not df.empty:
        print("  数据预览:")
        print(df.head())
    print()

    print("3. 测试 load_oss_complex_stocks (多字段):")
    complex_data = data_source.load_oss_complex_stocks(
        ['000001'], '2024-01-01', '2024-01-05',
        fields=['open', 'close']
    )
    if isinstance(complex_data, dict):
        print(f"  加载到 {len(complex_data)} 个字段")
        for field, field_df in complex_data.items():
            print(f"  {field}: {field_df.shape}")
    else:
        print(f"  加载到 {complex_data.shape} 数据")
    print()

    print("4. 测试便捷函数:")
    df2 = load_oss_stocks(['000001'], '2024-01-01', '2024-01-05')
    print(f"  便捷函数加载到 {len(df2)} 行数据")
    print()

    print("=== 测试完成 ===")


if __name__ == '__main__':
    test_duckdb_data_source()