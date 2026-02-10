#!/usr/bin/env python3
"""
DuckDB 数据每日更新脚本

此脚本用于每天自动更新最新的股票数据到 DuckDB 数据库中。
通常在交易日结束后运行，更新前一天的交易数据。
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import data
from src.data.duckdb_storage import DuckDBStorage

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_daily_data(db_path: str = './data/ohlcv_data.duckdb',
                     days_to_update: int = 1,
                     batch_size: int = 100):
    """
    更新最近几天的股票数据到 DuckDB

    Args:
        db_path: DuckDB 数据库路径
        days_to_update: 更新多少天的最新数据，默认1天（昨天）
        batch_size: 批量处理大小
    """
    print("=== DuckDB 数据每日更新 ===\n")

    # 计算要更新的日期范围
    # 默认更新昨天的数据（因为今天可能还在交易中）
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_to_update)).strftime('%Y-%m-%d')

    print(f"📅 更新日期范围: {start_date} ~ {end_date}")
    print(f"🏦 目标数据库: {db_path}")
    print(f"📦 批量大小: {batch_size}")
    print()

    try:
        # 创建存储实例
        storage = DuckDBStorage(db_path)

        # 获取数据库中已有的所有股票代码
        existing_stocks_df = storage._get_connection().execute("""
            SELECT DISTINCT stock_code FROM ohlcv_data ORDER BY stock_code
        """).fetchall()

        if not existing_stocks_df:
            print("❌ 数据库中没有股票数据，请先运行完整导入")
            return False

        existing_stocks = [row[0] for row in existing_stocks_df]
        total_stocks = len(existing_stocks)

        print(f"📊 数据库中有 {total_stocks} 只股票需要更新")
        print()

        success_count = 0
        total_records = 0
        updated_count = 0

        # 分批处理股票
        for batch_start in range(0, total_stocks, batch_size):
            batch_end = min(batch_start + batch_size, total_stocks)
            batch_stocks = existing_stocks[batch_start:batch_end]

            print(f"📦 处理批次 {batch_start//batch_size + 1}/{(total_stocks + batch_size - 1)//batch_size}: 股票 {batch_start+1}-{batch_end}")

            # 尝试批量获取数据
            batch_ohlcv_data = None
            try:
                batch_ohlcv_data = data.load_oss_complex_stocks(
                    codes=batch_stocks,
                    start=start_date,
                    end=end_date,
                    fields='all'
                )

                if not batch_ohlcv_data or (isinstance(batch_ohlcv_data, dict) and not batch_ohlcv_data):
                    batch_ohlcv_data = None

            except Exception as e:
                print(f"  ⚠️  批次 {batch_start//batch_size + 1} 批量获取出错: {e}，将逐个获取")
                batch_ohlcv_data = None

            # 处理每只股票
            for stock_code in batch_stocks:
                try:
                    stock_ohlcv_data = {}

                    if batch_ohlcv_data is not None:
                        # 从批量数据中提取单个股票的数据
                        if isinstance(batch_ohlcv_data, dict):
                            for field, field_df in batch_ohlcv_data.items():
                                if stock_code in field_df.columns:
                                    # 创建单股票的DataFrame，保持列名为股票代码
                                    stock_df = field_df[[stock_code]].copy()
                                    # 列名保持为股票代码，稍后在转换函数中处理
                                    stock_ohlcv_data[field] = stock_df
                        print(f"  📊 股票 {stock_code} 从批量数据中提取")
                    else:
                        # 逐个获取股票数据
                        print(f"  🔄 股票 {stock_code} 逐个获取")
                        try:
                            single_stock_data = data.load_oss_complex_stocks(
                                codes=[stock_code],
                                start=start_date,
                                end=end_date,
                                fields='all'
                            )
                            if single_stock_data and isinstance(single_stock_data, dict):
                                stock_ohlcv_data = single_stock_data
                                print(f"     ✅ 逐个获取成功")
                            else:
                                print(f"     ❌ 逐个获取返回空数据")
                        except Exception as single_e:
                            print(f"     ❌ 逐个获取出错: {single_e}")
                            continue

                    if not stock_ohlcv_data:
                        print(f"  ⚠️  股票 {stock_code} 未获取到数据，跳过")
                        continue

                    # 转换数据格式：从宽表转换为长表
                    long_data = convert_wide_to_long_single_stock(stock_ohlcv_data, stock_code)

                    if long_data.empty:
                        print(f"  ⚠️  股票 {stock_code} 转换后数据为空，跳过")
                        continue

                    # 数据验证
                    if not validate_ohlcv_data(long_data):
                        print(f"  ❌ 股票 {stock_code} 数据验证失败，跳过")
                        continue

                    # 检查是否已有该日期范围的数据，如果有则更新
                    existing_data = storage.load_ohlcv_data(
                        stock_codes=[stock_code],
                        start_date=start_date,
                        end_date=end_date
                    )

                    if not existing_data.empty:
                        # 有重复数据，需要先删除再插入
                        conn = storage._get_connection()
                        conn.execute("""
                            DELETE FROM ohlcv_data
                            WHERE stock_code = ? AND date >= ? AND date <= ?
                        """, [stock_code, start_date, end_date])
                        print(f"  🔄 股票 {stock_code} 更新了 {len(existing_data)} 条记录")
                    else:
                        print(f"  ➕ 股票 {stock_code} 添加了 {len(long_data)} 条记录")

                    # 保存到 DuckDB
                    success = storage._save_dataframe_to_duckdb(long_data, 'ohlcv_data')

                    if success:
                        records_added = len(long_data)
                        total_records += records_added
                        success_count += 1
                        if existing_data.empty:
                            updated_count += 1
                    else:
                        print(f"  ❌ 股票 {stock_code} 保存失败")

                except Exception as e:
                    print(f"  ❌ 股票 {stock_code} 处理出错: {e}")
                    continue

        print()
        print(f"📊 更新完成: {success_count}/{total_stocks} 股票成功处理")
        print(f"📈 总共处理了 {total_records} 条记录")
        print(f"🔄 更新了 {updated_count} 只股票")

        if success_count > 0:
            print("✅ 数据更新成功！")

            # 显示数据库信息
            info = storage.get_table_info('ohlcv_data')
            if info:
                print("\n📊 数据库表信息:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

            return True
        else:
            print("❌ 没有成功更新任何股票数据")
            return False

    except Exception as e:
        print(f"❌ 更新过程中出错: {e}")
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


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='DuckDB 数据每日更新')
    parser.add_argument('--db-path', default='./data/ohlcv_data.duckdb',
                       help='DuckDB 数据库路径')
    parser.add_argument('--days', type=int, default=1,
                       help='更新多少天的最新数据')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='批量处理大小')

    args = parser.parse_args()

    success = update_daily_data(
        db_path=args.db_path,
        days_to_update=args.days,
        batch_size=args.batch_size
    )

    if success:
        print("\n🎉 数据更新完成！")
        print(f"数据库路径: {args.db_path}")
    else:
        print("\n❌ 数据更新失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()