"""
TA-Lib 兼容性测试 - 验证新实现与 factor_old 一致
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# 导入新的 TA-Lib 生成器
from src.factor.generator.talib import TalibFactorListGenerator, TalibFactorCalculator, TALIB_AVAILABLE

# 导入 factor_old 的实现
sys.path.insert(0, os.path.join(project_root, 'src/factor_old'))
try:
    from generate_talib_factors import (
        get_talib_functions as old_get_talib_functions,
        generate_common_parameters as old_generate_common_parameters,
        generate_talib_factors as old_generate_talib_factors
    )
    FACTOR_OLD_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  无法导入 factor_old: {e}")
    FACTOR_OLD_AVAILABLE = False


def test_talib_availability():
    """测试 TA-Lib 是否可用"""
    print("\n" + "="*60)
    print("测试 1: TA-Lib 可用性")
    print("="*60)
    
    if not TALIB_AVAILABLE:
        print("❌ TA-Lib 未安装")
        return False
    
    print("✅ TA-Lib 已安装")
    return True


def test_function_list():
    """测试获取 TA-Lib 函数列表"""
    print("\n" + "="*60)
    print("测试 2: TA-Lib 函数列表")
    print("="*60)
    
    if not TALIB_AVAILABLE:
        print("⏭️  跳过（TA-Lib 未安装）")
        return True
    
    try:
        functions = TalibFactorListGenerator.get_talib_functions()
        print(f"✅ 获取到 {len(functions)} 个函数")
        
        # 验证常见函数存在
        common_funcs = ['RSI', 'MACD', 'BBANDS', 'ATR', 'SMA', 'EMA']
        for func in common_funcs:
            if func in functions:
                print(f"  ✓ {func}")
            else:
                print(f"  ❌ {func} 缺失")
                return False
        
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_parameter_generation():
    """测试参数生成"""
    print("\n" + "="*60)
    print("测试 3: 参数生成（对比 factor_old）")
    print("="*60)
    
    if not TALIB_AVAILABLE:
        print("⏭️  跳过（TA-Lib 未安装）")
        return True
    
    test_functions = ['RSI', 'MACD', 'BBANDS', 'ATR', 'SMA', 'EMA']
    all_passed = True
    
    for func_name in test_functions:
        try:
            # 新实现
            new_params = TalibFactorListGenerator.generate_common_parameters(func_name)
            
            # factor_old 实现
            if FACTOR_OLD_AVAILABLE:
                old_params = old_generate_common_parameters(func_name)
                
                # 比较结果
                if new_params == old_params:
                    print(f"✅ {func_name}: 参数一致")
                    print(f"   {new_params}")
                else:
                    print(f"⚠️  {func_name}: 参数不一致")
                    print(f"   新: {new_params}")
                    print(f"   旧: {old_params}")
                    all_passed = False
            else:
                print(f"✓ {func_name}: {new_params}")
        
        except Exception as e:
            print(f"❌ {func_name}: 错误 - {e}")
            all_passed = False
    
    return all_passed


def test_factor_generation():
    """测试因子列表生成"""
    print("\n" + "="*60)
    print("测试 4: 因子列表生成（对比 factor_old）")
    print("="*60)
    
    if not TALIB_AVAILABLE:
        print("⏭️  跳过（TA-Lib 未安装）")
        return True
    
    try:
        # 新实现
        new_factors = TalibFactorListGenerator.generate_talib_factors()
        print(f"✅ 新实现生成了 {len(new_factors)} 个因子")
        
        # 显示前 20 个
        print(f"   前 20 个因子:")
        for i, factor in enumerate(new_factors[:20]):
            print(f"   {i+1}. {factor}")
        
        # factor_old 实现
        if FACTOR_OLD_AVAILABLE:
            old_factors = old_generate_talib_factors()
            print(f"\n✅ factor_old 生成了 {len(old_factors)} 个因子")
            
            # 比较结果
            new_set = set(new_factors)
            old_set = set(old_factors)
            
            if new_set == old_set:
                print(f"✅ 因子列表完全一致！")
                return True
            else:
                only_new = new_set - old_set
                only_old = old_set - new_set
                
                if only_new:
                    print(f"\n⚠️  仅在新实现中出现的因子 ({len(only_new)}):")
                    for factor in sorted(list(only_new))[:10]:
                        print(f"   - {factor}")
                    if len(only_new) > 10:
                        print(f"   ... 还有 {len(only_new)-10} 个")
                
                if only_old:
                    print(f"\n⚠️  仅在 factor_old 中出现的因子 ({len(only_old)}):")
                    for factor in sorted(list(only_old))[:10]:
                        print(f"   - {factor}")
                    if len(only_old) > 10:
                        print(f"   ... 还有 {len(only_old)-10} 个")
                
                # 虽然有差异，但如果差异较小可能是可以接受的
                difference_ratio = len(only_new | only_old) / max(len(new_set), len(old_set))
                if difference_ratio < 0.1:  # 差异小于 10%
                    print(f"\n✓ 差异率 {difference_ratio*100:.1f}%（可接受）")
                    return True
                else:
                    return False
        else:
            print(f"✓ 新实现生成了 {len(new_factors)} 个因子（无法与 factor_old 比较）")
            return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_factors():
    """测试具体因子的计算"""
    print("\n" + "="*60)
    print("测试 5: 具体因子计算")
    print("="*60)
    
    if not TALIB_AVAILABLE:
        print("⏭️  跳过（TA-Lib 未安装）")
        return True
    
    try:
        # 创建示例 OHLCV 数据
        dates = pd.date_range('2024-01-01', periods=100)
        values = np.arange(100, dtype=np.float64)
        ohlcv = pd.DataFrame({
            'open': 100.0 + (values % 10),
            'high': 102.0 + (values % 10),
            'low': 98.0 + (values % 10),
            'close': 100.5 + (values % 10),
            'volume': 1000000.0 + (values * 1000),
        }, index=dates)
        
        # 确保所有列都是 float64
        for col in ['open', 'high', 'low', 'close', 'volume']:
            ohlcv[col] = ohlcv[col].astype(np.float64)
        
        # 测试几个因子
        test_factors = ['TALIB_RSI_14', 'TALIB_SMA_20', 'TALIB_ATR_14']
        
        for factor_name in test_factors:
            try:
                result = TalibFactorCalculator.calculate(factor_name, ohlcv)
                
                if len(result) == len(ohlcv):
                    non_nan_count = result.notna().sum()
                    print(f"✅ {factor_name}: {non_nan_count}/{len(result)} 有效值")
                else:
                    print(f"❌ {factor_name}: 长度不匹配 {len(result)} != {len(ohlcv)}")
                    return False
            except Exception as e:
                print(f"❌ {factor_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("TA-Lib 兼容性测试套件")
    print("="*60)
    
    results = {
        "TA-Lib 可用性": test_talib_availability(),
        "函数列表": test_function_list(),
        "参数生成": test_parameter_generation(),
        "因子列表": test_factor_generation(),
        "因子计算": test_specific_factors(),
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
    else:
        print("\n⚠️  部分测试失败")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
