#!/usr/bin/env python
"""
TA-Lib 快速验证脚本

用途: 快速验证 TA-Lib 因子生成是否正常工作

使用方式:
    python verify_talib.py
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)

try:
    from src.factor.generator.talib import (
        TalibFactorListGenerator,
        TalibFactorCalculator,
        TALIB_AVAILABLE
    )
    print("✅ 成功导入 TA-Lib 生成器模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def main():
    print("\n" + "="*60)
    print("TA-Lib 因子生成 - 快速验证")
    print("="*60)
    
    # 1. 检查 TA-Lib 是否可用
    print("\n1️⃣  检查 TA-Lib 库")
    if not TALIB_AVAILABLE:
        print("❌ TA-Lib 未安装")
        print("   请运行: pip install TA-Lib")
        return 1
    print("✅ TA-Lib 已安装")
    
    # 2. 获取可用函数
    print("\n2️⃣  扫描 TA-Lib 函数")
    try:
        functions = TalibFactorListGenerator.get_talib_functions()
        print(f"✅ 发现 {len(functions)} 个函数")
        print(f"   示例: {', '.join(functions[:5])}")
    except Exception as e:
        print(f"❌ 扫描失败: {e}")
        return 1
    
    # 3. 生成因子列表
    print("\n3️⃣  生成因子列表")
    try:
        factors = TalibFactorListGenerator.generate_talib_factors()
        print(f"✅ 生成了 {len(factors)} 个因子")
        print(f"   示例因子:")
        for i, factor in enumerate(factors[:10]):
            print(f"   {i+1:2d}. {factor}")
        print(f"   ...")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return 1
    
    # 4. 测试参数生成
    print("\n4️⃣  测试参数生成（示例指标）")
    test_indicators = {
        'RSI': 'RSI 相对强弱指标',
        'MACD': 'MACD 指数平滑异同移动平均线',
        'BBANDS': 'BBANDS 布林带',
        'ATR': 'ATR 真实波幅',
        'SMA': 'SMA 简单移动平均线',
    }
    
    for indicator, description in test_indicators.items():
        try:
            params = TalibFactorListGenerator.generate_common_parameters(indicator)
            param_str = ', '.join([str(p) for p in params])
            print(f"✅ {indicator:8s} ({description}): {param_str}")
        except Exception as e:
            print(f"⚠️  {indicator:8s}: {e}")
    
    # 5. 总结
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"✅ TA-Lib 因子生成模块正常工作！")
    print(f"\n   可用因子总数: {len(factors)}")
    print(f"   支持的函数: {len(functions)}")
    print(f"\n   快速开始:")
    print(f"   >>> from src.factor.generator import generate_talib_factors")
    print(f"   >>> result = generate_talib_factors(")
    print(f"   ...     stock_codes=['000001'],")
    print(f"   ...     start_date='2024-01-01',")
    print(f"   ...     end_date='2024-12-31'")
    print(f"   ... )")
    print(f"   >>> print(result['factor_file'])")
    print(f"\n   详见: docs/TALIB_IMPLEMENTATION.md")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
