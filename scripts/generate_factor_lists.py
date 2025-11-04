#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成Alpha158和Alpha360因子列表文件
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL

def get_alpha158_factors():
    """获取Alpha158因子名称列表（从实际因子文件读取）"""
    import pandas as pd
    import glob
    
    # 优先从实际因子文件中读取
    factor_files = glob.glob('factors/Alpha158_*.csv')
    if factor_files:
        # 使用最新的因子文件
        factor_file = sorted(factor_files)[-1]
        df = pd.read_csv(factor_file, nrows=1)
        factors = sorted([col for col in df.columns if col not in ['date', 'code']])
        print(f"  从实际因子文件读取: {factor_file} ({len(factors)} 个因子)")
        return factors
    
    # 如果没有因子文件，使用标准Alpha158配置（不包含扩展配置）
    # 注意：标准Alpha158 handler默认配置，不包含CLOSE和VOLUME历史窗口
    # 使用最小配置来获取标准158个因子
    conf = {
        'kbar': {},
        'price': {'windows': [0], 'feature': ['OPEN', 'HIGH', 'LOW', 'VWAP']},  # 只使用窗口0，不包含CLOSE
        'volume': {},  # 不包含volume窗口
        'rolling': {'windows': [5, 10, 20, 30, 60], 'include': None, 'exclude': []},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    print(f"  从Alpha158DL配置获取: {len(names)} 个因子")
    return names

def get_alpha360_factors():
    """获取Alpha360因子名称列表（从实际因子文件读取）"""
    import pandas as pd
    import glob
    
    # 优先从实际因子文件中读取
    factor_files = glob.glob('factors/Alpha360*.csv')
    if factor_files:
        # 使用最新的因子文件
        factor_file = sorted(factor_files)[-1]
        df = pd.read_csv(factor_file, nrows=1)
        factors = sorted([col for col in df.columns if col not in ['date', 'code']])
        print(f"  从实际因子文件读取: {factor_file} ({len(factors)} 个因子)")
        return factors
    
    # 如果没有因子文件，给出明确警告
    print(f"  ⚠️  警告：未找到Alpha360因子文件")
    print(f"      Alpha360DL.get_feature_config()生成的因子列表可能包含扩展配置的因子")
    print(f"      （如CLOSE0-59, VWAP0-59, VOLUME0-59等），但实际Alpha360 handler可能不包含这些")
    print(f"      请先生成Alpha360因子文件：")
    print(f"      python factor/generate_qlib_factors.py --factor-set Alpha360 --stock-pool small --start 2025-07-26 --end 2025-11-04 --output factors")
    print(f"      然后再运行此脚本更新因子列表")
    print(f"")
    print(f"      现在将使用Alpha360DL配置（可能不准确）")
    fields, names = Alpha360DL.get_feature_config()
    print(f"  从Alpha360DL配置获取: {len(names)} 个因子（可能不准确）")
    return names

def main():
    """主函数"""
    print("生成Alpha158和Alpha360因子列表...")
    
    # 获取因子列表
    factors158 = get_alpha158_factors()
    factors360 = get_alpha360_factors()
    
    print(f"Alpha158因子数量: {len(factors158)}")
    print(f"Alpha360因子数量: {len(factors360)}")
    
    # 生成Alpha158因子文件
    output_file_158 = 'alpha158_factors.txt'
    with open(output_file_158, 'w', encoding='utf-8') as f:
        # 写入Alpha158因子
        f.write("# Alpha158因子列表 ({}个因子)\n".format(len(factors158)))
        f.write("# ============================================\n")
        for factor in factors158:
            f.write(f"{factor}\n")
    
    # 生成Alpha360因子文件
    output_file_360 = 'alpha360_factors.txt'
    with open(output_file_360, 'w', encoding='utf-8') as f:
        # 写入Alpha360因子
        f.write("# Alpha360因子列表 ({}个因子)\n".format(len(factors360)))
        f.write("# ============================================\n")
        for factor in factors360:
            f.write(f"{factor}\n")
    
    print(f"\n✅ 因子列表已生成:")
    print(f"   Alpha158: {output_file_158} ({len(factors158)} 个因子)")
    print(f"   Alpha360: {output_file_360} ({len(factors360)} 个因子)")
    print(f"   总计: {len(factors158) + len(factors360)} 个因子")

if __name__ == '__main__':
    main()

