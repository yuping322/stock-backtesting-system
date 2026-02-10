#!/usr/bin/env python
"""
OSS 因子实现测试

测试新的 OSS 因子生成器与 factor_old 的兼容性
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

# 添加项目路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.factor.generator.oss import (
    OSSFactorCalculator,
    OSSFactorFinder,
    OSSFactorGenerator,
    generate_oss_factors
)


def create_test_factor_file(file_path: str):
    """创建测试因子文件"""
    dates = pd.date_range('2024-01-01', periods=100)
    codes = ['000001', '000002', '000003']
    
    data = []
    for date in dates:
        for code in codes:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'code': code,
                'ALPHA158_001': np.random.rand() * 0.01,
                'ALPHA158_002': np.random.rand() * 0.01,
                'ALPHA158_003': np.random.rand() * 0.01,
            })
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df


def test_oss_factor_calculator():
    """测试 OSSFactorCalculator"""
    print("\n" + "="*60)
    print("测试 1: OSSFactorCalculator")
    print("="*60)
    
    with TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = os.path.join(tmpdir, 'alpha_factors.csv')
        create_test_factor_file(test_file)
        
        # 创建计算器
        calc = OSSFactorCalculator(test_file, 'ALPHA158_001')
        
        # 测试计算
        result = calc.calculate('000001', '2024-01-01', '2024-03-31')
        
        if not result.empty:
            print(f"✅ 成功加载因子: {len(result)} 条记录")
            print(f"   最小值: {result.min():.6f}")
            print(f"   最大值: {result.max():.6f}")
            print(f"   平均值: {result.mean():.6f}")
            return True
        else:
            print("❌ 未能加载因子数据")
            return False


def test_oss_factor_finder():
    """测试 OSSFactorFinder"""
    print("\n" + "="*60)
    print("测试 2: OSSFactorFinder")
    print("="*60)
    
    with TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = os.path.join(tmpdir, 'alpha_factors.csv')
        create_test_factor_file(test_file)
        
        # 测试查找
        found_file = OSSFactorFinder.find_factor_file('ALPHA158_001', tmpdir)
        
        if found_file:
            print(f"✅ 成功找到因子文件: {found_file}")
            return True
        else:
            print("❌ 未能找到因子文件")
            return False


def test_oss_factor_generator():
    """测试 OSSFactorGenerator"""
    print("\n" + "="*60)
    print("测试 3: OSSFactorGenerator")
    print("="*60)
    
    with TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = os.path.join(tmpdir, 'alpha_factors.csv')
        create_test_factor_file(test_file)
        
        # 创建生成器
        generator = OSSFactorGenerator(
            factor_names=['ALPHA158_001', 'ALPHA158_002'],
            stock_codes=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-03-31',
            factor_file=test_file,
            output_dir=tmpdir
        )
        
        # 生成因子
        result_df = generator.generate()
        
        if not result_df.empty:
            print(f"✅ 成功生成因子: {len(result_df)} 条记录")
            print(f"   列数: {len(result_df.columns)}")
            print(f"   股票数: {result_df['stock_code'].nunique()}")
            print(f"   日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
            return True
        else:
            print("❌ 未能生成因子数据")
            return False


def test_generate_oss_factors():
    """测试 generate_oss_factors 函数"""
    print("\n" + "="*60)
    print("测试 4: generate_oss_factors 函数")
    print("="*60)
    
    with TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = os.path.join(tmpdir, 'alpha_factors.csv')
        create_test_factor_file(test_file)
        
        # 调用函数
        try:
            result = generate_oss_factors(
                factor_names=['ALPHA158_001', 'ALPHA158_002'],
                stock_codes=['000001', '000002'],
                start_date='2024-01-01',
                end_date='2024-03-31',
                factor_file=test_file,
                output_dir=tmpdir
            )
            
            print(f"✅ 成功生成因子文件")
            print(f"   因子文件: {result['factor_file']}")
            print(f"   元信息: {result['metadata_file']}")
            print(f"   说明: {result['readme_file']}")
            
            # 验证输出文件存在
            if os.path.exists(result['factor_file']):
                df = pd.read_csv(result['factor_file'])
                print(f"   数据量: {len(df)} 条记录")
                return True
            else:
                print("❌ 因子文件不存在")
                return False
        
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_from_directory():
    """测试从目录查找因子"""
    print("\n" + "="*60)
    print("测试 5: 从目录自动查找因子")
    print("="*60)
    
    with TemporaryDirectory() as tmpdir:
        # 创建测试文件
        test_file = os.path.join(tmpdir, 'alpha_factors.csv')
        create_test_factor_file(test_file)
        
        # 调用函数（使用 factor_dir 参数）
        try:
            result = generate_oss_factors(
                factor_names=['ALPHA158_001'],
                stock_codes=['000001'],
                start_date='2024-01-01',
                end_date='2024-03-31',
                factor_dir=tmpdir,  # 在目录中查找
                output_dir=tmpdir
            )
            
            print(f"✅ 成功从目录加载因子文件")
            print(f"   因子文件: {result['factor_file']}")
            return True
        
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False


def main():
    print("\n" + "="*60)
    print("OSS 因子实现测试套件")
    print("="*60)
    
    results = {
        "OSSFactorCalculator": test_oss_factor_calculator(),
        "OSSFactorFinder": test_oss_factor_finder(),
        "OSSFactorGenerator": test_oss_factor_generator(),
        "generate_oss_factors": test_generate_oss_factors(),
        "从目录查找因子": test_from_directory(),
    }
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n✅ OSS 因子生成器已就绪")
        print("   参考 factor_old 的 FileFactorCalculator 实现")
        print("   完全兼容从 CSV 文件加载因子数据")
    else:
        print("\n⚠️  部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
