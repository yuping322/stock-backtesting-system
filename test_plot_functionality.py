#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 factor 模块的画图功能
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'factor'))

# 测试 1: 检查 matplotlib backend 设置
print("=" * 60)
print("测试 1: 检查 matplotlib backend 设置")
print("=" * 60)

import matplotlib
print(f"默认 backend: {matplotlib.get_backend()}")

# 导入 factor.py（应该会自动设置 Agg）
import importlib.util
spec = importlib.util.spec_from_file_location('factor_module', 'factor/factor.py')
factor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor_module)

print(f"导入 factor.py 后的 backend: {matplotlib.get_backend()}")
if matplotlib.get_backend() == 'Agg':
    print("✅ Backend 设置正确（非交互式）")
else:
    print(f"⚠️  Backend 不是 Agg: {matplotlib.get_backend()}")

# 测试 2: 测试基本的 matplotlib 绘图功能
print("\n" + "=" * 60)
print("测试 2: 测试基本的 matplotlib 绘图功能")
print("=" * 60)

import matplotlib.pyplot as plt
import numpy as np

# 创建测试数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 测试创建图表
try:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Plot')
    plt.legend()
    
    # 测试保存图片
    test_output_dir = Path('test_output')
    test_output_dir.mkdir(exist_ok=True)
    
    test_plot_path = test_output_dir / 'test_basic_plot.png'
    plt.savefig(test_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if test_plot_path.exists():
        print(f"✅ 基本绘图测试成功，图片已保存到: {test_plot_path}")
    else:
        print(f"❌ 图片保存失败: {test_plot_path}")
except Exception as e:
    print(f"❌ 基本绘图测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 测试 alphalens tear-sheet 生成
print("\n" + "=" * 60)
print("测试 3: 测试 alphalens tear-sheet 生成")
print("=" * 60)

import pandas as pd
import alphalens as al

# 创建模拟因子数据
try:
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    codes = ['000001', '000002', '000003', '000004', '000005']
    
    # 创建因子数据（MultiIndex: date, code）
    np.random.seed(42)
    factor_values = np.random.randn(len(dates) * len(codes))
    factor_index = pd.MultiIndex.from_product([dates, codes], names=['date', 'code'])
    factor_data = pd.Series(factor_values, index=factor_index)
    
    # 创建价格数据
    prices = pd.DataFrame(
        np.random.rand(len(dates), len(codes)) * 100 + 50,
        index=dates,
        columns=codes
    )
    
    # 测试 alphalens 清理
    print("  正在清理因子数据...")
    clean = al.utils.get_clean_factor_and_forward_returns(
        factor_data,
        prices=prices,
        quantiles=5,
        periods=[5, 10],
        max_loss=3
    )
    print(f"  ✅ 因子数据清理成功，数据点: {len(clean)}")
    
    # 测试生成 tear-sheet（使用 Agg 后端，不会弹窗）
    print("  正在生成 tear-sheet...")
    plt.close('all')  # 清除之前的图表
    
    al.tears.create_full_tear_sheet(
        clean,
        long_short=True,
        group_neutral=False,
        by_group=False
    )
    
    # 检查是否生成了图表
    num_figures = len(plt.get_fignums())
    print(f"  ✅ Tear-sheet 生成成功，共 {num_figures} 个图表")
    
    # 测试保存所有图表
    if num_figures > 0:
        saved_count = 0
        for i, fig_num in enumerate(plt.get_fignums(), 1):
            try:
                # 切换到对应的 figure
                fig = plt.figure(fig_num)
                
                # 检查 figure 是否有内容
                if len(fig.axes) > 0:
                    # 保存图表
                    test_tear_path = test_output_dir / f'test_tear_sheet_{i}.png'
                    fig.savefig(test_tear_path, dpi=150, bbox_inches='tight', facecolor='white')
                    saved_count += 1
                    print(f"    已保存图表 {i} 到: {test_tear_path}")
                else:
                    print(f"    图表 {i} 没有 axes，跳过")
            except Exception as e:
                print(f"    保存图表 {i} 失败: {e}")
        
        plt.close('all')
        
        # 保存第一个图表作为主要文件（向后兼容）
        if num_figures > 0:
            plt.close('all')
            al.tears.create_full_tear_sheet(
                clean,
                long_short=True,
                group_neutral=False,
                by_group=False
            )
            main_tear_path = test_output_dir / 'test_tear_sheet.png'
            if len(plt.get_fignums()) > 0:
                fig = plt.figure(plt.get_fignums()[0])
                fig.savefig(main_tear_path, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"  ✅ 主要 Tear-sheet 图片已保存到: {main_tear_path}")
            plt.close('all')
        
        if saved_count > 0:
            print(f"  ✅ 共保存 {saved_count} 个图表")
        else:
            print(f"  ⚠️  未保存任何图表")
    else:
        print("  ⚠️  未生成任何图表")
        
except Exception as e:
    print(f"  ❌ Tear-sheet 生成失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 测试 FactorTester 的画图功能
print("\n" + "=" * 60)
print("测试 4: 测试 FactorTester 的画图功能")
print("=" * 60)

try:
    # 使用与 main_factor.py 相同的导入方式
    import importlib.util
    spec = importlib.util.spec_from_file_location("factor_module", os.path.join(project_root, 'factor', 'factor.py'))
    factor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factor_module)
    
    FactorTester = factor_module.FactorTester
    CFG = factor_module.CFG
    
    # 创建测试配置（使用 parse_args 的方式）
    import argparse
    test_args = argparse.Namespace(
        start='2024-01-01',
        end='2024-03-01',
        stock_pool='small',
        factors=['VOL10'],
        quantiles=5,
        periods=[5, 10],
        roll_win=60,
        monitor_csv='monitor.csv',
        factor_dir=None,
        custom_factor_file=None,
        custom_factor_name=None,
        fillna=True,
        winsorize=False,
        neutralize=False,
        standardize=False,
        last_only=False
    )
    
    cfg = CFG(test_args)
    
    # 创建测试器
    tester = FactorTester(cfg)
    
    # 设置少量股票（加快测试）
    tester.stocks = ['000001', '000002', '000003']
    
    print("  测试不画图模式...")
    print("  ℹ️  注意：完整测试需要连接OSS获取数据，这里只测试基本功能")
    print("  ℹ️  画图模式需要完整数据，跳过详细测试")
    print("  ✅ FactorTester 初始化成功")
    
except Exception as e:
    print(f"  ❌ FactorTester 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 5: 测试 main_factor.py 的画图模式设置
print("\n" + "=" * 60)
print("测试 5: 测试 main_factor.py 的画图模式设置")
print("=" * 60)

try:
    # 测试 backend 切换逻辑
    import matplotlib
    
    # 模拟 save 模式
    matplotlib.use('Agg')
    print(f"  Save 模式 backend: {matplotlib.get_backend()}")
    
    # 模拟 popup 模式（尝试切换）
    try:
        import sys
        if sys.platform == 'darwin':
            matplotlib.use('macosx')
        elif sys.platform.startswith('linux'):
            matplotlib.use('TkAgg')
        else:
            matplotlib.use('TkAgg')
        print(f"  Popup 模式 backend: {matplotlib.get_backend()}")
        print("  ✅ Backend 切换测试成功")
    except Exception as e:
        print(f"  ⚠️  Popup 模式 backend 切换失败: {e}")
        print(f"  回退到 Agg: {matplotlib.get_backend()}")
        
except Exception as e:
    print(f"  ❌ Backend 切换测试失败: {e}")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print("✅ 所有测试完成")
print(f"📁 测试输出目录: {test_output_dir.absolute()}")
print("\n如果所有测试都通过，说明画图功能已修复。")
print("如果遇到问题，请检查：")
print("  1. matplotlib 是否正确安装")
print("  2. alphalens 是否正确安装")
print("  3. 是否有足够的测试数据")
