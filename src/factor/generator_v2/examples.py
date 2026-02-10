"""
V2 生成器使用示例

这个文件展示了如何使用新的 V2 生成器框架
"""

import logging
from typing import List
import pandas as pd

from src.factor.generator_v2 import (
    BuiltinFactorGenerator,
    create_factor_calculator,
    DataQualityChecker,
    FactorCalculationError,
    PartialResultError,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_single_calculator():
    """
    示例 1: 使用单个计算器计算因子
    """
    print("\n" + "="*60)
    print("示例 1: 使用单个计算器")
    print("="*60)
    
    # 创建计算器
    calculator = create_factor_calculator('VOL10')
    
    # 计算因子
    result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
    
    print(f"✓ 成功计算 VOL10 因子")
    print(f"  数据点数: {len(result)}")
    print(f"  日期范围: {result.index.min()} ~ {result.index.max()}")
    print(f"  值范围: {result.min():.4f} ~ {result.max():.4f}")


def example_2_multiple_calculators():
    """
    示例 2: 使用多个计算器为单只股票计算多个因子
    """
    print("\n" + "="*60)
    print("示例 2: 多个计算器")
    print("="*60)
    
    factor_names = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
    results = {}
    
    for factor_name in factor_names:
        try:
            calculator = create_factor_calculator(factor_name)
            result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
            results[factor_name] = result
            print(f"✓ {factor_name}: {len(result)} 条数据")
        except FactorCalculationError as e:
            print(f"✗ {factor_name}: {e.reason}")
    
    # 合并结果
    if results:
        merged = pd.DataFrame(results)
        print(f"\n✓ 成功计算 {len(results)} 个因子")
        print(f"  合并数据形状: {merged.shape}")


def example_3_builtin_generator():
    """
    示例 3: 使用内置因子生成器
    """
    print("\n" + "="*60)
    print("示例 3: 内置因子生成器")
    print("="*60)
    
    # 创建生成器
    generator = BuiltinFactorGenerator(
        stock_codes=['000001', '000002', '000858'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        factor_names=['VOL10', 'RSI_14', 'MA_20'],
        output_dir='./data/factor_tasks'
    )
    
    # 生成因子
    try:
        df = generator.generate()
        
        print(f"\n✓ 因子生成成功!")
        print(f"  数据点数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  列名: {list(df.columns)}")
    
    except PartialResultError as e:
        print(f"\n⚠️  部分因子生成失败")
        print(f"  成功: {e.successful_count}")
        print(f"  失败: {e.failed_count}")
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")


def example_4_quality_check():
    """
    示例 4: 质量检查
    """
    print("\n" + "="*60)
    print("示例 4: 数据质量检查")
    print("="*60)
    
    # 创建示例数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    data = {
        'date': dates,
        'stock_code': '000001',
        'VOL10': range(len(dates)),
        'RSI_14': [50 + i % 50 for i in range(len(dates))],
        'MA_20': [100 + i % 100 for i in range(len(dates))],
    }
    df = pd.DataFrame(data)
    
    # 执行质量检查
    factor_cols = ['VOL10', 'RSI_14', 'MA_20']
    check_result = DataQualityChecker.check_factor_output(df, factor_cols)
    
    # 打印结果
    DataQualityChecker.print_check_result(check_result, verbose=True)


def example_5_error_handling():
    """
    示例 5: 错误处理
    """
    print("\n" + "="*60)
    print("示例 5: 错误处理")
    print("="*60)
    
    from src.factor.generator_v2 import (
        DataNotAvailableError,
        FactorCalculationError,
        FactorValidationError,
    )
    
    # 尝试计算不存在的因子
    try:
        calculator = create_factor_calculator('UNKNOWN_FACTOR')
    except Exception as e:
        print(f"✓ 捕获异常: {type(e).__name__}")
        print(f"  信息: {e}")
    
    # 尝试为不存在的股票计算因子
    try:
        calculator = create_factor_calculator('VOL10')
        result = calculator.calculate('999999', '2024-01-01', '2024-12-31')
    except DataNotAvailableError as e:
        print(f"✓ 捕获 DataNotAvailableError")
        print(f"  股票: {e.stock_code}")
        print(f"  原因: {e.reason}")


def run_all_examples():
    """
    运行所有示例
    """
    print("\n" + "="*80)
    print("V2 生成器使用示例")
    print("="*80)
    
    try:
        example_1_single_calculator()
    except Exception as e:
        logger.error(f"示例 1 失败: {e}")
    
    try:
        example_2_multiple_calculators()
    except Exception as e:
        logger.error(f"示例 2 失败: {e}")
    
    try:
        example_3_builtin_generator()
    except Exception as e:
        logger.error(f"示例 3 失败: {e}")
    
    try:
        example_4_quality_check()
    except Exception as e:
        logger.error(f"示例 4 失败: {e}")
    
    try:
        example_5_error_handling()
    except Exception as e:
        logger.error(f"示例 5 失败: {e}")
    
    print("\n" + "="*80)
    print("所有示例完成")
    print("="*80)


if __name__ == '__main__':
    run_all_examples()
