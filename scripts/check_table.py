import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data import data_duckdb
import duckdb

def check_table_structure():
    """检查 DuckDB 表结构"""

    print("=== 检查 DuckDB 表结构 ===\n")

    # 创建数据源实例
    data_source = data_duckdb.get_duckdb_data_source('./data/test_ohlcv.duckdb')

    # 直接连接到 DuckDB 获取所有表信息
    con = duckdb.connect('./data/test_ohlcv.duckdb')

    # 获取所有表
    tables = con.execute("SHOW TABLES").fetchall()
    print("数据库中的所有表:")
    for table in tables:
        table_name = table[0]
        print(f"  - {table_name}")

        # 获取表的详细信息
        try:
            # 获取行数
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"    行数: {row_count}")

            # 获取列信息
            columns = con.execute(f"DESCRIBE {table_name}").fetchall()
            print(f"    列数: {len(columns)}")
            print("    列信息:")
            for col in columns:
                col_name, col_type = col[0], col[1]
                print(f"      - {col_name}: {col_type}")

        except Exception as e:
            print(f"    获取表 {table_name} 详细信息失败: {e}")

        print()  # 空行分隔

    con.close()

    print("=== 检查完成 ===")

if __name__ == '__main__':
    check_table_structure()