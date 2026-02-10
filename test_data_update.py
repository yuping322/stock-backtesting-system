import os
import sys
import pandas as pd
from datetime import datetime, date

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data import data_duckdb

def test_data_update_logic():
    """测试数据更新逻辑：当库中有旧数据时，是否会正确更新"""

    print("=== 测试数据更新逻辑 ===\n")

    # 创建数据源实例
    data_source = data_duckdb.get_duckdb_data_source('./data/test_ohlcv.duckdb')

    # 测试股票代码
    test_codes = ['000001']
    test_start = '2024-01-01'

    # 第一次查询：应该从原始数据源获取并缓存
    print("1. 第一次查询（应该从原始数据源获取）:")
    end_date_1 = '2024-01-03'  # 较早的日期
    result1 = data_source.load_oss_complex_stocks(
        codes=test_codes,
        start=test_start,
        end=end_date_1,
        fields="close"
    )
    print(f"   获取到数据形状: {result1.shape if isinstance(result1, pd.DataFrame) else 'N/A'}")

    # 第二次查询：同样的日期范围，应该从缓存获取
    print("\n2. 第二次查询（同样的日期范围，应该从缓存获取）:")
    result2 = data_source.load_oss_complex_stocks(
        codes=test_codes,
        start=test_start,
        end=end_date_1,
        fields="close"
    )
    print(f"   获取到数据形状: {result2.shape if isinstance(result2, pd.DataFrame) else 'N/A'}")

    # 第三次查询：更新的结束日期，应该触发数据更新
    print("\n3. 第三次查询（更新的结束日期，应该触发数据更新）:")
    end_date_2 = '2024-01-05'  # 更新的日期
    result3 = data_source.load_oss_complex_stocks(
        codes=test_codes,
        start=test_start,
        end=end_date_2,
        fields="close"
    )
    print(f"   获取到数据形状: {result3.shape if isinstance(result3, pd.DataFrame) else 'N/A'}")

    # 第四次查询：同样的更新日期，应该从缓存获取
    print("\n4. 第四次查询（同样的更新日期，应该从缓存获取）:")
    result4 = data_source.load_oss_complex_stocks(
        codes=test_codes,
        start=test_start,
        end=end_date_2,
        fields="close"
    )
    print(f"   获取到数据形状: {result4.shape if isinstance(result4, pd.DataFrame) else 'N/A'}")

    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_data_update_logic()