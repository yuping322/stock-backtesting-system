#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 因子生成器演示脚本

展示如何使用新的 Qlib 集成系统生成和加载因子
"""

import sys
import os
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.factor.generator.qlib import generate_qlib_factors
from src.factor.generator.file import generate_file_factors


def demo_qlib_generation():
    """演示 Qlib 因子生成"""
    print("\n" + "="*70)
    print("演示 1: 生成 Qlib Alpha158 因子")
    print("="*70)
    
    print("\n使用 Qlib 生成器生成 Alpha158 因子...")
    
    df = generate_qlib_factors(
        stock_codes=['000001', '000002'],
        start_date='2024-09-01',
        end_date='2024-09-10',
        factor_set='Alpha158',
        output_file='./data/qlib_demo_alpha158.csv'
    )
    
    if not df.empty:
        print(f"\n✅ 生成成功!")
        print(f"  数据形状: {df.shape}")
        print(f"  列数: {len(df.columns)} (date + stock_code + {len(df.columns) - 2} 个因子)")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  股票数: {df['stock_code'].nunique()}")
        print(f"\n样本数据:")
        print(df.head(5))
        
        return df
    else:
        print("❌ 生成失败")
        return None


def demo_file_loading(csv_path):
    """演示 File 加载器"""
    print("\n" + "="*70)
    print("演示 2: 用 File 加载器加载 CSV")
    print("="*70)
    
    print(f"\n用 File 加载器加载 {csv_path}...")
    
    result = generate_file_factors(
        factor_file_paths={
            'qlib_alpha158': csv_path
        },
        stock_codes=['000001'],  # 只加载第一只股票
        start_date='2024-09-02',
        end_date='2024-09-05'
    )
    
    if result:
        print(f"\n✅ 加载成功!")
        print(f"  输出目录: {os.path.dirname(result['factor_file'])}")
        print(f"  因子文件: {os.path.basename(result['factor_file'])}")
        
        # 读取输出文件并显示
        df_output = pd.read_csv(result['factor_file'])
        print(f"\n输出数据形状: {df_output.shape}")
        print(f"  行数: {len(df_output)} (1 只股票 × 4 天)")
        print(f"  列数: {len(df_output.columns)} (date + stock_code + {len(df_output.columns) - 2} 个因子)")
        print(f"\n样本输出:")
        print(df_output.head())
        
        return result
    else:
        print("❌ 加载失败")
        return None


def demo_multi_source_loading():
    """演示多源加载"""
    print("\n" + "="*70)
    print("演示 3: 多源因子加载")
    print("="*70)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建示例 CSV 文件
        custom_csv = os.path.join(tmpdir, 'custom_factors.csv')
        custom_data = pd.DataFrame({
            'date': ['2024-09-02', '2024-09-03', '2024-09-04', '2024-09-05'],
            'code': ['000001', '000001', '000001', '000001'],
            'my_custom_factor': [1.1, 1.2, 1.3, 1.4]
        })
        custom_data.to_csv(custom_csv, index=False)
        
        print(f"\n创建了两个数据源:")
        print(f"  1. Qlib Alpha158 因子: ./data/qlib_demo_alpha158.csv")
        print(f"  2. 自定义因子: {custom_csv}")
        
        print(f"\n将两个数据源合并加载...")
        
        result = generate_file_factors(
            factor_file_paths={
                'qlib_alpha158': './data/qlib_demo_alpha158.csv',
                'custom_factors': custom_csv
            },
            stock_codes=['000001'],
            start_date='2024-09-02',
            end_date='2024-09-05'
        )
        
        if result:
            print(f"\n✅ 多源加载成功!")
            
            df_output = pd.read_csv(result['factor_file'])
            print(f"  合并后数据形状: {df_output.shape}")
            print(f"  因子数: {len(df_output.columns) - 2}")
            
            # 显示因子列
            factor_cols = [col for col in df_output.columns if col not in ['date', 'stock_code']]
            print(f"\n因子列表 (前 20 个):")
            for i, col in enumerate(sorted(factor_cols)[:20], 1):
                print(f"  {i:3d}. {col}")
            print(f"  ... (共 {len(factor_cols)} 个因子)")
            
            return result
        else:
            print("❌ 多源加载失败")
            return None


def main():
    """主演示程序"""
    print("="*70)
    print("Qlib 因子生成器集成演示")
    print("="*70)
    
    # 确保输出目录存在
    os.makedirs('./data', exist_ok=True)
    
    # 演示 1: 生成 Qlib 因子
    df_generated = demo_qlib_generation()
    if df_generated is None:
        print("\n演示终止: 第一步失败")
        return
    
    # 演示 2: 加载 CSV
    csv_path = './data/qlib_demo_alpha158.csv'
    result_loaded = demo_file_loading(csv_path)
    if result_loaded is None:
        print("\n演示终止: 第二步失败")
        return
    
    # 演示 3: 多源加载（可选）
    try:
        result_multi = demo_multi_source_loading()
    except Exception as e:
        print(f"\n⚠️  多源加载演示失败: {e}")
    
    # 总结
    print("\n" + "="*70)
    print("演示完成!")
    print("="*70)
    print("""
关键要点:
✓ Qlib 生成器生成了 158 个 Alpha 因子
✓ File 加载器能够自动识别和加载 Qlib CSV 格式
✓ 支持数据过滤 (股票、日期)
✓ 支持多源因子的合并加载

文件输出:
- ./data/qlib_demo_alpha158.csv - Qlib 生成的因子文件
- ./data/factor_tasks/task_*/factors_*.csv - File 加载器输出

更多信息请参考:
- docs/QLIB_GENERATOR_GUIDE.md
- src/factor/generator/qlib.py
- tests/test_qlib_integration.py
""")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
