#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版Alpha158因子测试脚本
只测试2个因子，使用因子文件中的数据
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from main_factor import main, parse_main_args
import argparse

def main_simple():
    """简化的主函数"""
    print("=" * 80)
    print("Alpha158因子测试（简化版 - 2个因子）")
    print("=" * 80)
    print()
    
    # 因子文件路径
    factor_file = 'factors/Alpha158_20250805_20251103.csv'
    
    # 从因子文件中提取日期范围
    print("📅 读取因子文件信息...")
    df = pd.read_csv(factor_file)
    start_date = df['date'].min()
    end_date = df['date'].max()
    print(f"✅ 日期范围: {start_date} 到 {end_date}")
    
    # 测试因子
    factors = ['KMID', 'KLEN']
    print(f"✅ 测试因子: {', '.join(factors)}")
    print()
    
    # 输出目录
    output_dir = 'results/alpha158_simple'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📊 输出目录: {output_dir}")
    print()
    
    # 构建命令行参数
    sys.argv = [
        'main_factor.py',
        '--factors', 'KMID', 'KLEN',
        '--start', start_date,
        '--end', end_date,
        '--factor-dir', 'factors',
        '--stock-pool', '000300',
        '--quantiles', '5',
        '--periods', '5', '10',
        '--roll-win', '20',
        '--monitor-csv', f'{output_dir}/monitor.csv',
        '--output-dir', output_dir,
    ]
    
    print("🚀 开始运行因子检验...")
    print("=" * 80)
    print()
    
    try:
        # 运行主函数
        main()
        
        print()
        print("=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        print()
        print(f"📊 查看结果:")
        print(f"  cat {output_dir}/monitor.csv")
        print(f"  ls {output_dir}/*/README.md")
        
        return 0
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main_simple())

