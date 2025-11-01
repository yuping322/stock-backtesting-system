#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出数据示例脚本

这是一个简单的示例，展示如何使用 export_data.py 导出价格和因子数据
"""

import os
import sys
import re
from typing import List

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from export_data import export_formatted_csv
import data


# 固定日期区间（来自 scripts/run_all_factors.sh）
FIXED_START_DATE = "2025-07-26"
FIXED_END_DATE = "2025-10-24"


def get_all_factors_and_stocks(start_date: str, end_date: str) -> tuple[List[str], List[str]]:
    """读取该期间内的全量因子列和股票列表（从因子数据本身推断）"""
    df = data.read_factor_data(
        codes=None,
        start_date=start_date,
        end_date=end_date,
        factors=None,
        base_path="uploads"
    )
    if df.empty:
        raise ValueError("该日期区间未读取到因子数据")
    factors = [c for c in df.columns]
    stocks = df.index.get_level_values('code').unique().tolist()
    # 统一为6位代码（去掉后缀），export内部会做规范化，这里尽量简化
    stocks = [s.replace('.XSHG', '').replace('.XSHE', '') for s in stocks]
    return factors, stocks


def example_export_use_script_dates_and_all():
    """
    使用 scripts/run_all_factors.sh 的日期，
    自动读取该区间内的全量因子、全量股票，
    按自定义格式导出（formatted_data.csv）。
    """
    # 使用固定日期区间
    start_date, end_date = FIXED_START_DATE, FIXED_END_DATE

    # 全量因子 + 全量股票（从因子数据自身获取）
    factors, stocks = get_all_factors_and_stocks(start_date, end_date)

    output_dir = './exported_data_all'
    print("="*60)
    print("示例: 使用指定日期 + 全量因子/股票 导出")
    print("="*60)

    export_formatted_csv(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        factors=factors,            # 全量因子
        industry_default='Unknown'  # 可调
    )

    print(f"\n✓ 数据已导出到: {output_dir}")
    return output_dir


def example_custom_date_range():
    """自定义日期范围示例（使用格式化导出）"""
    stocks = ['000001']
    start_date = '2024-01-01'
    end_date = '2024-03-31'
    output_dir = './exported_data_custom'

    print("\n" + "="*60)
    print("示例2: 自定义日期范围（formatted_data.csv）")
    print("="*60)

    export_formatted_csv(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        try_load_extra_factors=False,
        industry_default='Unknown'
    )

    print(f"\n✓ 数据已导出到: {output_dir}")
    return output_dir


if __name__ == '__main__':
    example_export_use_script_dates_and_all()
    print("\n" + "="*60)
    print("示例运行完成！")
    print("="*60)
    print("\n命令行等价用法（直接生成同样格式）:")
    print("  python factor/export_data.py --stocks <全量股票> --factors <全量因子> --mode custom")

