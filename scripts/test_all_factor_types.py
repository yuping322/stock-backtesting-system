#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有因子类型的因子检验
包括：
1. 内置函数计算的因子（VOL10, single_day_VPT_12）
2. 从OSS读取的因子
3. 从文件加载的Alpha因子（Alpha158/Alpha360）
4. 自定义临时生成的因子

通过调用 main_factor.py 来执行因子检验
"""
import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# 注意：自定义因子需要通过其他方式实现，这里先使用内置因子和文件因子

def main():
    """主函数"""
    print("=" * 80)
    print("测试所有因子类型的因子检验")
    print("=" * 80)
    
    # 1. 查找Alpha因子文件（从文件加载）
    factor_dir = 'factors'
    alpha_factor_file = None
    alpha_factors = []
    
    if os.path.exists(factor_dir):
        alpha_files = [f for f in os.listdir(factor_dir) if f.startswith('Alpha') and f.endswith('.csv')]
        if alpha_files:
            # 使用最新的Alpha因子文件
            alpha_factor_file = os.path.join(factor_dir, sorted(alpha_files)[-1])
            print(f"\n📁 找到Alpha因子文件: {alpha_factor_file}")
            
            # 读取因子名称
            df_sample = pd.read_csv(alpha_factor_file, nrows=1)
            available_factors = [col for col in df_sample.columns if col not in ['date', 'code']]
            # 选择前2个因子
            alpha_factors = available_factors[:2]
            print(f"📊 使用Alpha因子: {alpha_factors}")
    
    # 2. 准备所有因子
    all_factors = []
    custom_factors = {}
    
    # 2.1 内置函数计算的因子（2个）
    builtin_factors = ['VOL10', 'single_day_VPT_12']
    all_factors.extend(builtin_factors)
    print(f"\n📊 内置函数因子: {builtin_factors}")
    
    # 2.2 从OSS读取的因子（2个）- 需要知道OSS中有哪些因子
    # 这里假设OSS中有一些因子，如果不存在会跳过
    oss_factors = ['OPEN0', 'HIGH0']  # 尝试从OSS读取，如果不存在会跳过
    all_factors.extend(oss_factors)
    print(f"📊 OSS因子（尝试）: {oss_factors}")
    
    # 2.3 从文件加载的Alpha因子（2个）
    if alpha_factors:
        all_factors.extend(alpha_factors)
        print(f"📊 Alpha文件因子: {alpha_factors}")
    
    # 2.4 自定义临时生成的因子（2个）
    # 注意：自定义因子需要通过自定义因子文件或OHLCV因子实现
    # 这里先使用内置因子代替，或者可以通过自定义因子文件实现
    # all_factors.extend(['volume_ma20', 'price_momentum'])
    # print(f"📊 自定义因子: ['volume_ma20', 'price_momentum']")
    # 暂时跳过自定义因子，因为需要额外的实现方式
    
    # 3. 确定日期范围
    # 如果使用文件因子，从文件读取日期范围；否则使用最近3个月
    if alpha_factor_file:
        try:
            df_dates = pd.read_csv(alpha_factor_file, usecols=['date'])
            dates = pd.to_datetime(df_dates['date'])
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            print(f"\n📅 使用Alpha因子文件日期范围: {start_date} 到 {end_date}")
        except Exception as e:
            print(f"⚠️  读取日期范围失败: {e}，使用默认值")
            from datetime import datetime, timedelta
            end_date_obj = datetime.now().date()
            start_date_obj = end_date_obj - timedelta(days=90)
            start_date = start_date_obj.strftime('%Y-%m-%d')
            end_date = end_date_obj.strftime('%Y-%m-%d')
    else:
        from datetime import datetime, timedelta
        end_date_obj = datetime.now().date()
        start_date_obj = end_date_obj - timedelta(days=90)
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')
        print(f"\n📅 使用默认日期范围: {start_date} 到 {end_date}")
    
    # 4. 准备输出目录
    output_dir = 'results/test_all_factor_types'
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 打印配置信息
    print("\n" + "=" * 80)
    print("因子检验配置")
    print("=" * 80)
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"股票池: small")
    print(f"因子列表 ({len(all_factors)} 个):")
    print(f"  内置函数因子: {builtin_factors}")
    print(f"  OSS因子: {oss_factors}")
    if alpha_factors:
        print(f"  Alpha文件因子: {alpha_factors}")
    print(f"分位数: 5")
    print(f"调仓周期: [5, 10]")
    print(f"滚动窗口: 20 天")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print()
    
    # 6. 构建main_factor.py的命令行参数
    main_factor_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main_factor.py')
    
    cmd = [
        sys.executable,
        main_factor_script,
        '--start', start_date,
        '--end', end_date,
        '--stock-pool', 'small',
        '--factors'] + all_factors + [
        '--quantiles', '5',
        '--periods', '5', '10',
        '--roll-win', '20',
        '--plot', 'false',
        '--output-dir', output_dir,
        '--monitor-csv', os.path.join(output_dir, 'monitor.csv')
    ]
    
    # 如果使用Alpha因子文件，添加factor-dir参数
    if alpha_factor_file:
        cmd.extend(['--factor-dir', factor_dir])
    
    # 7. 调用main_factor.py
    print("开始运行因子检验（通过main_factor.py）...\n")
    print(f"执行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 因子检验完成")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 因子检验失败: {e}")
        return None
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        return None
    
    # 8. 读取结果并生成汇总
    monitor_file = os.path.join(output_dir, 'monitor.csv')
    if os.path.exists(monitor_file):
        print(f"\n✅ 结果已保存到: {monitor_file}")
        
        # 读取结果进行统计
        try:
            df = pd.read_csv(monitor_file)
            print("\n" + "=" * 80)
            print("测试完成汇总")
            print("=" * 80)
            print(f"✅ 总记录数: {len(df)}")
            
            # 按因子类型分组统计
            factor_stats = {}
            for factor in df['factor'].unique():
                if factor in builtin_factors:
                    factor_type = '内置函数'
                elif factor in oss_factors:
                    factor_type = 'OSS'
                elif factor in alpha_factors:
                    factor_type = 'Alpha文件'
                else:
                    factor_type = '其他'
                
                if factor_type not in factor_stats:
                    factor_stats[factor_type] = []
                factor_stats[factor_type].append(factor)
            
            print("\n因子类型统计:")
            for factor_type, factors in factor_stats.items():
                unique_factors = list(set(factors))
                print(f"  {factor_type}: {len(unique_factors)} 个因子 - {unique_factors}")
            
            print(f"\n✅ 所有结果已保存到: {output_dir}")
            print("=" * 80)
        except Exception as e:
            print(f"⚠️  读取结果文件失败: {e}")
    else:
        print(f"⚠️  结果文件不存在: {monitor_file}")
    
    return None

if __name__ == '__main__':
    main()

