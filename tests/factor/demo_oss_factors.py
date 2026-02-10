#!/usr/bin/env python
"""
OSS 因子生成 - 快速演示

演示如何从 OSS 存储中加载预先计算的因子数据
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.factor.generator.oss import generate_oss_factors


def demo_oss_factor_from_file():
    """演示 1: 从指定的因子文件加载"""
    print("\n" + "="*60)
    print("演示 1: 从指定的因子文件加载 OSS 因子")
    print("="*60)
    
    # 假设你有一个预先计算好的因子文件
    # 文件格式: date,code,ALPHA158_001,ALPHA158_002,...
    
    try:
        result = generate_oss_factors(
            factor_names=['ALPHA158_001', 'ALPHA158_002'],
            stock_codes=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            factor_file='./data/alpha158_factors.csv'  # 替换为你的实际文件路径
        )
        print(f"✅ 成功! 因子文件: {result['factor_file']}")
    except FileNotFoundError as e:
        print(f"⚠️  演示需要因子文件: {e}")
        print("   请确保因子文件存在或修改文件路径")
    except Exception as e:
        print(f"❌ 错误: {e}")


def demo_oss_factor_from_directory():
    """演示 2: 从目录中自动查找因子文件"""
    print("\n" + "="*60)
    print("演示 2: 从目录中自动查找因子文件")
    print("="*60)
    
    try:
        result = generate_oss_factors(
            factor_names=['ALPHA158_001', 'ALPHA158_002'],
            stock_codes=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            factor_dir='./oss_factors'  # 生成器会在此目录中查找包含 ALPHA158_001 等的 CSV 文件
        )
        print(f"✅ 成功! 因子文件: {result['factor_file']}")
    except FileNotFoundError as e:
        print(f"⚠️  演示需要因子目录: {e}")
        print("   请创建目录并添加因子文件，或修改目录路径")
    except Exception as e:
        print(f"❌ 错误: {e}")


def demo_factor_file_format():
    """演示 3: 显示因子文件的期望格式"""
    print("\n" + "="*60)
    print("演示 3: OSS 因子文件的期望格式")
    print("="*60)
    
    print("""
OSS 因子文件应该是 CSV 格式，包含以下列：
- date: 交易日期 (YYYY-MM-DD 格式)
- code: 股票代码 (6位数字或带交易所后缀)
- 因子列: ALPHA158_001, ALPHA158_002, 等等

示例内容：
┌─────────────┬────────┬──────────────┬──────────────┐
│ date        │ code   │ ALPHA158_001 │ ALPHA158_002 │
├─────────────┼────────┼──────────────┼──────────────┤
│ 2024-01-15  │ 000001 │ 0.00123      │ 0.00456      │
│ 2024-01-16  │ 000001 │ 0.00234      │ 0.00567      │
│ 2024-01-17  │ 000001 │ 0.00345      │ 0.00678      │
│ 2024-01-15  │ 000002 │ 0.00789      │ 0.00890      │
│ 2024-01-16  │ 000002 │ 0.00890      │ 0.00901      │
│ 2024-01-17  │ 000002 │ 0.00901      │ 0.00912      │
└─────────────┴────────┴──────────────┴──────────────┘

创建示例文件：
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-15', periods=100)
    codes = ['000001', '000002', '000003']
    
    data = []
    for date in dates:
        for code in codes:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'code': code,
                'ALPHA158_001': np.random.rand(),
                'ALPHA158_002': np.random.rand(),
            })
    
    df = pd.DataFrame(data)
    df.to_csv('alpha158_factors.csv', index=False)
    """)


def main():
    print("\n" + "="*60)
    print("OSS 因子生成 - 快速演示")
    print("="*60)
    
    print("""
OSS 因子是用户自己计算并存储的因子数据。
新的 OSS 因子生成器支持从以下位置加载因子：

1. 直接指定因子文件路径 (factor_file 参数)
2. 从因子目录自动查找 (factor_dir 参数)

生成器会自动：
- 加载 CSV 文件中的数据
- 按股票和日期范围筛选
- 整理成统一的输出格式
""")
    
    # 显示文件格式说明
    demo_factor_file_format()
    
    # 演示 1: 从文件加载
    demo_oss_factor_from_file()
    
    # 演示 2: 从目录加载
    demo_oss_factor_from_directory()
    
    print("\n" + "="*60)
    print("快速开始:")
    print("="*60)
    print("""
方式 1: 使用你已有的因子文件
    from src.factor.generator import generate_oss_factors
    
    result = generate_oss_factors(
        factor_names=['ALPHA158_001', 'ALPHA158_002'],
        stock_codes=['000001', '000002'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        factor_file='./your_alpha_factors.csv'
    )

方式 2: 生成器自动查找因子文件
    result = generate_oss_factors(
        factor_names=['ALPHA158_001'],
        stock_codes=['000001'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        factor_dir='./oss_factors'
    )

查看生成结果:
    print(result['factor_file'])      # 因子文件路径
    print(result['metadata_file'])    # 元信息文件
    print(result['readme_file'])      # 说明文档
    """)


if __name__ == "__main__":
    main()
