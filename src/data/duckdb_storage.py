"""
DuckDB 存储模块

提供 DuckDB 数据库的基本操作功能
"""

import os
import pandas as pd
import duckdb
from typing import List, Optional, Dict, Any


class DuckDBStorage:
    """
    DuckDB 存储类

    提供基本的 DuckDB 数据库操作功能
    """

    def __init__(self, db_path: str):
        """
        初始化 DuckDB 存储

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._con = None

    def _get_connection(self):
        """获取数据库连接"""
        if self._con is None:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._con = duckdb.connect(self.db_path)
        return self._con

    def _save_dataframe_to_duckdb(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        将 DataFrame 保存到 DuckDB 表中

        Args:
            df: 要保存的 DataFrame
            table_name: 表名

        Returns:
            bool: 保存是否成功
        """
        if df.empty:
            return False

        try:
            con = self._get_connection()

            # 如果表不存在，先创建表
            if not self._table_exists(table_name):
                # 根据 DataFrame 的列创建表
                columns_def = []
                for col in df.columns:
                    if col == 'date':
                        columns_def.append(f'"{col}" DATE')
                    elif col == 'stock_code':
                        columns_def.append(f'"{col}" VARCHAR')
                    elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                        columns_def.append(f'"{col}" BIGINT')
                    elif df[col].dtype in ['float64', 'float32']:
                        columns_def.append(f'"{col}" DOUBLE')
                    else:
                        columns_def.append(f'"{col}" VARCHAR')

                create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns_def)})"
                con.execute(create_sql)

            # 插入数据
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

            return True

        except Exception as e:
            print(f"保存数据到 DuckDB 失败: {e}")
            return False

    def _table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            con = self._get_connection()
            result = con.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchall()
            return len(result) > 0
        except:
            # DuckDB 使用不同的系统表
            try:
                result = con.execute(f"SELECT table_name FROM information_schema.tables WHERE table_name='{table_name}'").fetchall()
                return len(result) > 0
            except:
                return False

    def load_ohlcv_data(self, stock_codes: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从 DuckDB 加载 OHLCV 数据

        Args:
            stock_codes: 股票代码列表，为 None 则加载所有
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'

        Returns:
            pd.DataFrame: OHLCV 数据
        """
        try:
            con = self._get_connection()

            # 检查表是否存在
            if not self._table_exists('ohlcv_data'):
                return pd.DataFrame()

            # 构建查询
            query = "SELECT * FROM ohlcv_data WHERE 1=1"

            if stock_codes:
                codes_str = "', '".join(stock_codes)
                query += f" AND stock_code IN ('{codes_str}')"

            if start_date:
                query += f" AND date >= '{start_date}'"

            if end_date:
                query += f" AND date <= '{end_date}'"

            query += " ORDER BY date, stock_code"

            # 执行查询
            result = con.execute(query).fetchdf()

            return result

        except Exception as e:
            print(f"从 DuckDB 加载数据失败: {e}")
            return pd.DataFrame()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        获取表信息

        Args:
            table_name: 表名

        Returns:
            Dict[str, Any]: 表信息
        """
        try:
            con = self._get_connection()

            if not self._table_exists(table_name):
                return {}

            # 获取行数
            count_result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchall()
            row_count = count_result[0][0] if count_result else 0

            # 获取列信息
            columns_result = con.execute(f"DESCRIBE {table_name}").fetchdf()
            columns = columns_result['column_name'].tolist() if not columns_result.empty else []

            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': columns,
                'column_count': len(columns)
            }

        except Exception as e:
            print(f"获取表信息失败: {e}")
            return {}

    def close(self):
        """关闭数据库连接"""
        if self._con:
            self._con.close()
            self._con = None

    def save_ohlcv_data(self, df: pd.DataFrame, stock_codes: Optional[List[str]] = None) -> bool:
        """
        保存 OHLCV 数据到 DuckDB

        Args:
            df: 包含 OHLCV 数据的 DataFrame
            stock_codes: 股票代码列表，用于过滤

        Returns:
            bool: 保存是否成功
        """
        if df.empty:
            return False

        try:
            # 确保数据格式正确
            required_cols = ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"数据格式不正确，缺少必需列: {required_cols}")
                return False

            # 过滤指定的股票代码
            if stock_codes:
                df = df[df['stock_code'].isin(stock_codes)].copy()

            if df.empty:
                return False

            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date']).dt.date

            # 在保存新数据之前，删除相同股票代码和日期范围的旧数据
            con = self._get_connection()
            
            if stock_codes and self._table_exists('ohlcv_data'):
                # 获取要保存的日期范围
                min_date = df['date'].min()
                max_date = df['date'].max()
                
                # 删除重叠的数据
                delete_sql = f"""
                DELETE FROM ohlcv_data 
                WHERE stock_code IN ({','.join(['?' for _ in stock_codes])}) 
                AND date >= ? AND date <= ?
                """
                con.execute(delete_sql, stock_codes + [min_date, max_date])

            # 保存到 DuckDB
            return self._save_dataframe_to_duckdb(df, 'ohlcv_data')

        except Exception as e:
            print(f"保存 OHLCV 数据到 DuckDB 失败: {e}")
            return False