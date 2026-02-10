#!/usr/bin/env python3
"""
从 data.py 导入数据到 DuckDB

这个脚本从 data.py 获取最近的股票数据并        # 分批处理股票
        for batch_start in range(0, total_stocks, batch_size):
            batch_end = min(batch_start + batch_size, total_stocks)
            batch_stocks = sample_stocks[batch_start:batch_end]

            print(f"📦 处理批次 {batch_start//batch_size + 1}/{(total_stocks + batch_size - 1)//batch_size}: 股票 {batch_start+1}-{batch_end}")

            # 批量获取数据
            try:
                batch_ohlcv_data = data.load_oss_complex_stocks(
                    codes=batch_stocks,
                    start=start_date,
                    end=end_date,
                    fields='all'
                )

                if not batch_ohlcv_data or (isinstance(batch_ohlcv_data, dict) and not batch_ohlcv_data):
                    print(f"  ⚠️  批次 {batch_start//batch_size + 1} 未获取到数据，跳过")
                    continue

                # 处理每只股票
                for stock_code in batch_stocks:uckDB 数据库中，
确保 DuckDB 有数据可用进行测试和开发。
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Union, List
import datetime as dt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import data
from src.data.duckdb_storage import DuckDBStorage

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_balance_data_to_duckdb(code: str,
                                  db_path: str = './data/ohlcv_data.duckdb',
                                  date: Union[str, dt.date, dt.datetime, None] = None,
                                  report_type: str = "合并期末") -> bool:
    """
    导入资产负债表数据到 DuckDB

    Args:
        code: 股票代码
        db_path: DuckDB 数据库路径
        date: 日期
        report_type: 报告类型

    Returns:
        bool: 是否成功
    """
    print(f"📊 导入资产负债表数据: {code}")

    try:
        # 从原始数据源获取数据
        balance_df = data.get_balance(code, date, report_type=report_type)

        if balance_df.empty:
            print(f"  ⚠️  未获取到资产负债表数据: {code}")
            return False

        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 添加元数据列
        balance_df = balance_df.copy()
        balance_df['stock_code'] = code
        balance_df['report_type'] = report_type
        balance_df['data_type'] = 'balance'

        # 重新排列列，把元数据列放在前面
        metadata_cols = ['stock_code', 'report_type', 'data_type']
        other_cols = [col for col in balance_df.columns if col not in metadata_cols]
        balance_df = balance_df[metadata_cols + other_cols]

        # 保存到 DuckDB - 使用专门的财务数据表
        success = storage._save_dataframe_to_duckdb(balance_df, 'balance_data')

        if success:
            print(f"  ✅ 资产负债表数据导入成功: {code} ({len(balance_df)} 条记录)")
            return True
        else:
            print(f"  ❌ 资产负债表数据保存失败: {code}")
            return False

    except Exception as e:
        print(f"  ❌ 导入资产负债表数据失败 {code}: {e}")
        return False


def import_income_data_to_duckdb(code: str,
                                db_path: str = './data/ohlcv_data.duckdb',
                                date: Union[str, dt.date, dt.datetime, None] = None,
                                report_type: str = "合并期末") -> bool:
    """
    导入利润表数据到 DuckDB

    Args:
        code: 股票代码
        db_path: DuckDB 数据库路径
        date: 日期
        report_type: 报告类型

    Returns:
        bool: 是否成功
    """
    print(f"📊 导入利润表数据: {code}")

    try:
        # 从原始数据源获取数据
        income_df = data.get_income(code, date, report_type=report_type)

        if income_df.empty:
            print(f"  ⚠️  未获取到利润表数据: {code}")
            return False

        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 添加元数据列
        income_df = income_df.copy()
        income_df['stock_code'] = code
        income_df['report_type'] = report_type
        income_df['data_type'] = 'income'

        # 重新排列列，把元数据列放在前面
        metadata_cols = ['stock_code', 'report_type', 'data_type']
        other_cols = [col for col in income_df.columns if col not in metadata_cols]
        income_df = income_df[metadata_cols + other_cols]

        # 保存到 DuckDB - 使用专门的利润表
        success = storage._save_dataframe_to_duckdb(income_df, 'income_data')

        if success:
            print(f"  ✅ 利润表数据导入成功: {code} ({len(income_df)} 条记录)")
            return True
        else:
            print(f"  ❌ 利润表数据保存失败: {code}")
            return False

    except Exception as e:
        print(f"  ❌ 导入利润表数据失败 {code}: {e}")
        return False


def import_cashflow_data_to_duckdb(code: str,
                                  db_path: str = './data/ohlcv_data.duckdb',
                                  date: Union[str, dt.date, dt.datetime, None] = None,
                                  report_type: str = "合并期末") -> bool:
    """
    导入现金流量表数据到 DuckDB

    Args:
        code: 股票代码
        db_path: DuckDB 数据库路径
        date: 日期
        report_type: 报告类型

    Returns:
        bool: 是否成功
    """
    print(f"📊 导入现金流量表数据: {code}")

    try:
        # 从原始数据源获取数据
        cashflow_df = data.get_cashflow(code, date, report_type=report_type)

        if cashflow_df.empty:
            print(f"  ⚠️  未获取到现金流量表数据: {code}")
            return False

        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 添加元数据列
        cashflow_df = cashflow_df.copy()
        cashflow_df['stock_code'] = code
        cashflow_df['report_type'] = report_type
        cashflow_df['data_type'] = 'cashflow'

        # 重新排列列，把元数据列放在前面
        metadata_cols = ['stock_code', 'report_type', 'data_type']
        other_cols = [col for col in cashflow_df.columns if col not in metadata_cols]
        cashflow_df = cashflow_df[metadata_cols + other_cols]

        # 保存到 DuckDB - 使用专门的现金流量表
        success = storage._save_dataframe_to_duckdb(cashflow_df, 'cashflow_data')

        if success:
            print(f"  ✅ 现金流量表数据导入成功: {code} ({len(cashflow_df)} 条记录)")
            return True
        else:
            print(f"  ❌ 现金流量表数据保存失败: {code}")
            return False

    except Exception as e:
        print(f"  ❌ 导入现金流量表数据失败 {code}: {e}")
        return False


def import_valuation_data_to_duckdb(code: str,
                                   db_path: str = './data/ohlcv_data.duckdb',
                                   date: Union[str, dt.date, dt.datetime, None] = None) -> bool:
    """
    导入估值数据到 DuckDB

    Args:
        code: 股票代码
        db_path: DuckDB 数据库路径
        date: 日期

    Returns:
        bool: 是否成功
    """
    print(f"📊 导入估值数据: {code}")

    try:
        # 从原始数据源获取数据
        valuation_df = data.get_valuation(code, date)

        if valuation_df.empty:
            print(f"  ⚠️  未获取到估值数据: {code}")
            return False

        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 添加元数据列
        valuation_df = valuation_df.copy()
        valuation_df['stock_code'] = code
        valuation_df['data_type'] = 'valuation'

        # 重新排列列，把元数据列放在前面
        metadata_cols = ['stock_code', 'data_type']
        other_cols = [col for col in valuation_df.columns if col not in metadata_cols]
        valuation_df = valuation_df[metadata_cols + other_cols]

        # 保存到 DuckDB - 使用专门的估值表
        success = storage._save_dataframe_to_duckdb(valuation_df, 'valuation_data')

        if success:
            print(f"  ✅ 估值数据导入成功: {code} ({len(valuation_df)} 条记录)")
            return True
        else:
            print(f"  ❌ 估值数据保存失败: {code}")
            return False

    except Exception as e:
        print(f"  ❌ 导入估值数据失败 {code}: {e}")
        return False


def import_history_fundamentals_to_duckdb(security: Union[str, List[str]],
                                         fields: List[str],
                                         db_path: str = './data/ohlcv_data.duckdb',
                                         watch_date: Union[str, dt.date, dt.datetime, None] = None,
                                         stat_date: Union[str, None] = None,
                                         count: int = 1,
                                         interval: str = "1q",
                                         report_type: str = "合并期末") -> bool:
    """
    导入历史财务数据到 DuckDB

    Args:
        security: 股票代码
        fields: 字段列表
        db_path: DuckDB 数据库路径
        watch_date: 观察日期
        stat_date: 统计日期
        count: 数量
        interval: 间隔
        report_type: 报告类型

    Returns:
        bool: 是否成功
    """
    print(f"📊 导入历史财务数据: {security}")

    try:
        # 从原始数据源获取数据
        fundamentals_df = data.get_history_fundamentals(
            security, fields, watch_date, stat_date, count, interval, report_type
        )

        if fundamentals_df.empty:
            print(f"  ⚠️  未获取到历史财务数据: {security}")
            return False

        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 添加元数据列
        fundamentals_df = fundamentals_df.copy()
        fundamentals_df['data_type'] = 'history_fundamentals'
        fundamentals_df['fields'] = str(fields)
        fundamentals_df['interval'] = interval
        fundamentals_df['report_type'] = report_type

        # 保存到 DuckDB
        success = storage._save_dataframe_to_duckdb(fundamentals_df, 'financial_data')

        if success:
            print(f"  ✅ 历史财务数据导入成功: {security} ({len(fundamentals_df)} 条记录)")
            return True
        else:
            print(f"  ❌ 历史财务数据保存失败: {security}")
            return False

    except Exception as e:
        print(f"  ❌ 导入历史财务数据失败 {security}: {e}")
        return False


def import_recent_data_to_duckdb(db_path: str = './data/ohlcv_data.duckdb',
                                days_back: int = 30,
                                sample_stocks: list = None,
                                force_update: bool = False,
                                batch_size: int = 50):
    """
    从 data.py 导入最近的数据到 DuckDB

    Args:
        db_path: DuckDB 数据库路径
        days_back: 获取多少天的数据
        sample_stocks: 要获取的股票列表，如果为 None 则使用默认列表
        force_update: 是否强制更新已存在的股票数据
        batch_size: 批量处理的大小，用于优化性能
    """
    print("=== 从 data.py 导入数据到 DuckDB ===\n")

    # 默认股票列表（使用所有small指数股票）
    if sample_stocks is None:
        try:
            sample_stocks = data.get_index_stocks("small")  # 获取所有小盘股
            logger.info(f"使用 small 指数股票列表，共 {len(sample_stocks)} 只股票")
        except Exception as e:
            logger.warning(f"获取 small 指数股票列表失败: {e}，使用示例股票")
            sample_stocks = ['000001', '000002', '600000', '600036', '000858']

    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"📅 日期范围: {start_date} ~ {end_date}")
    print(f"📊 股票数量: {len(sample_stocks)}")
    print(f"🏦 目标数据库: {db_path}")
    print(f"🔄 强制更新: {force_update}")
    print(f"📦 批量大小: {batch_size}")
    print()

    try:
        # 创建存储实例
        storage = DuckDBStorage(db_path)

        total_stocks = len(sample_stocks)
        success_count = 0
        total_records = 0
        skipped_count = 0

        print(f"📊 总共需要处理 {total_stocks} 只股票")
        print()

        # 逐个股票处理
        for i, stock_code in enumerate(sample_stocks, 1):
            print(f"� [{i}/{total_stocks}] 处理股票 {stock_code}...")

            try:
                # 检查是否已有该股票的数据
                if not force_update:
                    existing_data = storage.load_ohlcv_data(
                        stock_codes=[stock_code],
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not existing_data.empty:
                        existing_records = len(existing_data)
                        print(f"  ⏭️  股票 {stock_code} 已存在 {existing_records} 条记录，跳过")
                        skipped_count += 1
                        continue

                # 从 data.py 获取单个股票的数据
                ohlcv_data = data.load_oss_complex_stocks(
                    codes=[stock_code],  # 单个股票
                    start=start_date,
                    end=end_date,
                    fields='all'  # 获取所有字段
                )

                if not ohlcv_data or (isinstance(ohlcv_data, dict) and not ohlcv_data):
                    print(f"  ⚠️  股票 {stock_code} 未获取到数据，跳过")
                    continue

                # 转换数据格式：从宽表转换为长表
                long_data = convert_wide_to_long_single_stock(ohlcv_data, stock_code)

                if long_data.empty:
                    print(f"  ⚠️  股票 {stock_code} 转换后数据为空，跳过")
                    continue

                # 数据验证
                if not validate_ohlcv_data(long_data):
                    print(f"  ❌ 股票 {stock_code} 数据验证失败，跳过")
                    continue

                # 保存到 DuckDB
                success = storage._save_dataframe_to_duckdb(long_data, 'ohlcv_data')

                if success:
                    records_added = len(long_data)
                    total_records += records_added
                    success_count += 1
                    print(f"  ✅ 股票 {stock_code} 导入成功，添加 {records_added} 条记录")
                else:
                    print(f"  ❌ 股票 {stock_code} 保存失败")

            except Exception as e:
                print(f"  ❌ 股票 {stock_code} 处理出错: {e}")
                continue

        print()
        print(f"📊 处理完成: {success_count}/{total_stocks} 股票成功导入")
        print(f"⏭️  跳过 {skipped_count} 只已有数据的股票")
        print(f"📈 总共添加了 {total_records} 条记录")

        if success_count > 0 or total_records > 0:
            print("✅ 数据导入成功！")

            # 显示数据库信息
            info = storage.get_table_info('ohlcv_data')
            if info:
                print("\n📊 数据库表信息:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

            return True
        else:
            print("❌ 没有成功导入任何新股票数据")
            return False

    except Exception as e:
        print(f"❌ 导入过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """
    验证 OHLCV 数据的基本完整性

    Args:
        df: 要验证的 DataFrame

    Returns:
        bool: 数据是否有效
    """
    if df.empty:
        return False

    # 检查必需的列
    required_columns = ['date', 'stock_code', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"缺少必需的列: {missing_columns}")
        return False

    # 检查是否有有效的 OHLC 数据
    ohlc_cols = ['open', 'high', 'low', 'close']
    valid_rows = 0
    for col in ohlc_cols:
        if col in df.columns:
            valid_values = df[col].notna().sum()
            valid_rows = max(valid_rows, valid_values)

    if valid_rows == 0:
        logger.warning("没有有效的 OHLC 数据")
        return False

    # 检查数据合理性（high >= low, close 在 high 和 low 之间等）
    if 'high' in df.columns and 'low' in df.columns:
        invalid_high_low = (df['high'] < df['low']).sum()
        if invalid_high_low > 0:
            logger.warning(f"发现 {invalid_high_low} 条 high < low 的无效数据")

    return True


def convert_wide_to_long_single_stock(ohlcv_dict: dict, stock_code: str) -> pd.DataFrame:
    """
    将单个股票的 data.load_oss_complex_stocks 返回的宽表字典转换为长表 DataFrame

    Args:
        ohlcv_dict: {字段名: DataFrame} 格式的数据
        stock_code: 股票代码

    Returns:
        pd.DataFrame: 长表格式的 DataFrame
    """
    if not ohlcv_dict:
        return pd.DataFrame()

    # 获取所有日期（从任意一个 field 的 DataFrame）
    first_field = next(iter(ohlcv_dict.keys()))
    dates = ohlcv_dict[first_field].index

    all_data = []

    # 对于每个日期，构建一行记录
    for date in dates:
        row_data = {'date': date, 'stock_code': stock_code}

        # 从各个字段的 DataFrame 中提取该日期、该股票的数据
        for field, field_df in ohlcv_dict.items():
            if stock_code in field_df.columns and date in field_df.index:
                value = field_df.loc[date, stock_code]
                # 只保存非 NaN 值
                if pd.notna(value):
                    row_data[field] = value

        # 只有在有至少一个 OHLCV 值的情况下才添加该行
        if len(row_data) > 2:  # date 和 stock_code 加上至少一个数据字段
            all_data.append(row_data)

    if not all_data:
        return pd.DataFrame()

    # 转换为 DataFrame
    result_df = pd.DataFrame(all_data)
    result_df['date'] = pd.to_datetime(result_df['date'])

    # 确保所有数值列都是 float 类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover']
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    # 排序
    result_df = result_df.sort_values(['date', 'stock_code']).reset_index(drop=True)

    return result_df


def convert_wide_to_long(ohlcv_dict: dict, stock_codes: list) -> pd.DataFrame:
    """
    将 data.load_oss_complex_stocks 返回的宽表字典转换为长表 DataFrame

    Args:
        ohlcv_dict: {字段名: DataFrame} 格式的数据
        stock_codes: 股票代码列表

    Returns:
        pd.DataFrame: 长表格式的 DataFrame
    """
    if not ohlcv_dict:
        return pd.DataFrame()

    # 获取所有日期（从任意一个 field 的 DataFrame）
    first_field = next(iter(ohlcv_dict.keys()))
    dates = ohlcv_dict[first_field].index

    all_data = []

    # 对于每个日期和股票，构建一行记录
    for date in dates:
        for stock_code in stock_codes:
            row_data = {'date': date, 'stock_code': stock_code}

            # 从各个字段的 DataFrame 中提取该日期、该股票的数据
            for field, field_df in ohlcv_dict.items():
                if stock_code in field_df.columns and date in field_df.index:
                    value = field_df.loc[date, stock_code]
                    # 只保存非 NaN 值
                    if pd.notna(value):
                        row_data[field] = value

            # 只有在有至少一个 OHLCV 值的情况下才添加该行
            if len(row_data) > 2:  # date 和 stock_code 加上至少一个数据字段
                all_data.append(row_data)

    if not all_data:
        return pd.DataFrame()

    # 转换为 DataFrame
    result_df = pd.DataFrame(all_data)
    result_df['date'] = pd.to_datetime(result_df['date'])

    # 确保所有数值列都是 float 类型
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'turnover']
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    # 排序
    result_df = result_df.sort_values(['date', 'stock_code']).reset_index(drop=True)

    return result_df


def clean_duplicate_data(db_path: str = './data/ohlcv_data.duckdb'):
    """
    清理数据库中的重复数据

    Args:
        db_path: DuckDB 数据库路径
    """
    print("🧹 清理数据库中的重复数据...")

    try:
        storage = DuckDBStorage(db_path)

        # 获取当前数据量
        info_before = storage.get_table_info('ohlcv_data')
        if not info_before:
            print("❌ 数据库表不存在")
            return False

        print(f"清理前: {info_before['row_count']} 条记录")

        # 使用 DuckDB 直接清理重复数据
        con = storage._get_connection()

        # 创建临时表存储去重后的数据
        con.execute("""
            CREATE TEMP TABLE temp_ohlcv AS
            SELECT DISTINCT * FROM ohlcv_data
            ORDER BY date, stock_code
        """)

        # 删除原表并重命名临时表
        con.execute("DROP TABLE ohlcv_data")
        con.execute("CREATE TABLE ohlcv_data AS SELECT * FROM temp_ohlcv")
        con.execute("DROP TABLE temp_ohlcv")

        # 获取清理后的数据量
        info_after = storage.get_table_info('ohlcv_data')
        removed_count = info_before['row_count'] - info_after['row_count']

        print(f"清理后: {info_after['row_count']} 条记录")
        print(f"移除了 {removed_count} 条重复记录")

        return True

    except Exception as e:
        logger.error(f"清理重复数据失败: {e}")
        print(f"❌ 清理失败: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='从 data.py 导入数据到 DuckDB')
    parser.add_argument('--db-path', default='./data/ohlcv_data.duckdb',
                       help='DuckDB 数据库路径')
    parser.add_argument('--days', type=int, default=365,
                       help='获取多少天的数据')
    parser.add_argument('--stocks', nargs='+',
                       help='指定股票代码，不指定则使用默认列表')
    parser.add_argument('--force', action='store_true',
                       help='强制更新已存在的股票数据')
    parser.add_argument('--clean', action='store_true',
                       help='只执行数据清理操作，不导入新数据')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='批量处理大小，用于优化性能')

    args = parser.parse_args()

    if args.clean:
        # 只执行清理操作
        success = clean_duplicate_data(db_path=args.db_path)
        if success:
            print("\n🧹 数据清理完成！")
        else:
            print("\n❌ 数据清理失败！")
            sys.exit(1)
        return

    success = import_recent_data_to_duckdb(
        db_path=args.db_path,
        days_back=args.days,
        sample_stocks=args.stocks,
        force_update=args.force,
        batch_size=args.batch_size
    )

    if success:
        print("\n🎉 数据导入完成！现在可以使用 DuckDB 数据源了。")
        print(f"数据库路径: {args.db_path}")
    else:
        print("\n❌ 数据导入失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()