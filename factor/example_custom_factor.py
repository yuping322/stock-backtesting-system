"""
自定义因子使用示例

展示如何使用自定义因子计算接口
"""

import sys
import os
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor.factor import parse_args, CFG, FactorTester
from factor.factor_calculator import create_factor_calculator


def example_1_builtin_factor():
    """示例 1: 使用内置因子"""
    print("=" * 60)
    print("示例 1: 使用内置因子")
    print("=" * 60)
    
    # 创建配置
    args = parse_args()
    args.factors = ['VOL10', 'RSI_14']  # 使用内置因子
    cfg = CFG(args)
    
    # 创建内置因子计算器
    custom_factors = {
        'VOL10': create_factor_calculator(factor_name='VOL10'),
        'RSI_14': create_factor_calculator(factor_name='RSI_14'),
    }
    
    # 运行检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


def example_2_ohlcv_factor():
    """示例 2: 使用自定义 OHLCV 因子函数"""
    print("=" * 60)
    print("示例 2: 使用自定义 OHLCV 因子函数")
    print("=" * 60)
    
    # 定义自定义因子函数（接受 OHLCV DataFrame）
    def my_custom_factor(ohlcv):
        """
        自定义因子：当前收盘价 / 20日均价
        
        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            
        Returns:
            pd.Series: 因子值
        """
        close = ohlcv['close']
        ma20 = close.rolling(20).mean()
        return close / ma20
    
    # 创建配置
    args = parse_args()
    args.factors = ['MA_RATIO']  # 使用自定义因子
    cfg = CFG(args)
    
    # 创建自定义因子计算器
    custom_factors = {
        'MA_RATIO': create_factor_calculator(factor_func=my_custom_factor),
    }
    
    # 运行检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


def example_3_full_custom_factor():
    """示例 3: 完全自定义的因子计算函数"""
    print("=" * 60)
    print("示例 3: 完全自定义的因子计算函数")
    print("=" * 60)
    
    def fully_custom_factor(stock_code, start_date, end_date):
        """
        完全自定义的因子计算函数
        
        可以从任意数据源读取数据，返回因子值
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.Series: 因子值，索引为日期
        """
        # 这里可以自己实现数据读取逻辑
        # 例如：从文件、数据库、API 等读取数据
        
        # 示例：返回一个简单的常数因子
        dates = pd.date_range(start_date, end_date, freq='D')
        factor_values = pd.Series([1.0] * len(dates), index=dates)
        
        return factor_values
    
    # 创建配置
    args = parse_args()
    args.factors = ['CONSTANT_FACTOR']
    cfg = CFG(args)
    
    # 创建完全自定义的因子计算器
    custom_factors = {
        'CONSTANT_FACTOR': create_factor_calculator(factor_func=fully_custom_factor),
    }
    
    # 运行检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


def example_4_multi_factors():
    """示例 4: 组合使用多种因子"""
    print("=" * 60)
    print("示例 4: 组合使用多种因子")
    print("=" * 60)
    
    # 定义多个自定义因子
    def price_momentum(ohlcv):
        """价格动量因子"""
        return ohlcv['close'].pct_change(10)
    
    def volume_trend(ohlcv):
        """成交量趋势因子"""
        return ohlcv['volume'] / ohlcv['volume'].rolling(20).mean()
    
    # 创建配置
    args = parse_args()
    args.factors = ['PRICE_MOMENTUM', 'VOLUME_TREND', 'VOL10']
    cfg = CFG(args)
    
    # 创建多个因子计算器
    custom_factors = {
        'PRICE_MOMENTUM': create_factor_calculator(factor_func=price_momentum),
        'VOLUME_TREND': create_factor_calculator(factor_func=volume_trend),
        'VOL10': create_factor_calculator(factor_name='VOL10'),  # 使用内置因子
    }
    
    # 运行检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


def example_5_load_from_file():
    """示例 5: 从文件加载因子"""
    print("=" * 60)
    print("示例 5: 从文件加载因子")
    print("=" * 60)
    
    # 假设有一个因子文件
    file_path = 'data/sample_factors.csv'
    
    # 创建配置
    args = parse_args()
    args.factors = ['MY_FACTOR']  # 文件中存在的因子列名
    cfg = CFG(args)
    
    # 从文件创建因子计算器
    custom_factors = {
        'MY_FACTOR': create_factor_calculator(
            file_path=file_path,
            factor_name='MY_FACTOR'
        ),
    }
    
    # 运行检验
    tester = FactorTester(cfg, custom_factors=custom_factors)
    tester.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自定义因子示例')
    parser.add_argument('--example', type=int, default=1, 
                       help='示例编号 (1-5)')
    args = parser.parse_args()
    
    examples = {
        1: example_1_builtin_factor,
        2: example_2_ohlcv_factor,
        3: example_3_full_custom_factor,
        4: example_4_multi_factors,
        5: example_5_load_from_file,
    }
    
    if args.example in examples:
        examples[args.example]()
    else:
        print(f"未知示例编号: {args.example}")
        print("可用示例: 1-5")
        print("\n使用方法:")
        print("  python example_custom_factor.py --example 1  # 内置因子")
        print("  python example_custom_factor.py --example 2  # OHLCV 因子")
        print("  python example_custom_factor.py --example 3  # 完全自定义")
        print("  python example_custom_factor.py --example 4  # 多因子组合")
        print("  python example_custom_factor.py --example 5  # 从文件加载")
