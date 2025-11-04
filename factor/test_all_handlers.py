#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有支持的Handler类型

运行:
    python factor/test_all_handlers.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 测试配置
TEST_CODES = ['000001', '000002', '600000']
TEST_DAYS = 30
OUTPUT_DIR = './factors_test'

# 所有要测试的handler类型（只测试实际支持的Handler类型）
# 注意：Alpha158DL和Alpha360DL是DataLoader，使用方式不同，暂不支持
HANDLER_TYPES = [
    'Alpha158',
    'Alpha360',
    'Alpha158vwap',
    'Alpha360vwap',
]


def test_handler(handler_type: str, start_date: str, end_date: str) -> bool:
    """测试单个handler类型"""
    print(f'\n{"=" * 80}')
    print(f'测试 {handler_type}')
    print('=' * 80)
    
    # 构建命令
    cmd = [
        'python', 'factor/generate_qlib_factors.py',
        '--factor-set', handler_type,
        '--codes'] + TEST_CODES + [
        '--start', start_date,
        '--end', end_date,
        '--output', OUTPUT_DIR,
        '--rebuild'
    ]
    
    print(f'命令: {" ".join(cmd)}')
    print()
    
    try:
        # 运行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 输出结果
        if result.stdout:
            # 只显示关键信息
            lines = result.stdout.split('\n')
            important_lines = [
                l for l in lines
                if any(keyword in l for keyword in [
                    '[FACTOR]', '[SAVE]', '[ERROR]', '✓', '✗', 
                    'Extracted', 'Factors saved', '失败', '成功'
                ])
            ]
            if important_lines:
                print('\n'.join(important_lines[-20:]))  # 显示最后20行关键信息
        
        if result.stderr:
            print(f'\nSTDERR:')
            print(result.stderr[:500])  # 只显示前500个字符
        
        if result.returncode == 0:
            print(f'\n✓ {handler_type} 测试成功')
            return True
        else:
            print(f'\n✗ {handler_type} 测试失败 (exit code: {result.returncode})')
            return False
            
    except subprocess.TimeoutExpired:
        print(f'\n✗ {handler_type} 测试超时')
        return False
    except Exception as e:
        print(f'\n✗ {handler_type} 测试出错: {e}')
        return False


def main():
    """主函数"""
    print('=' * 80)
    print('测试所有Handler类型')
    print('=' * 80)
    
    # 计算日期范围
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=TEST_DAYS)
    
    print(f'测试日期范围: {start_date} 到 {end_date}')
    print(f'测试股票: {", ".join(TEST_CODES)}')
    print(f'输出目录: {OUTPUT_DIR}')
    print()
    
    # 测试结果
    results = {}
    
    # 逐个测试
    for handler_type in HANDLER_TYPES:
        success = test_handler(
            handler_type=handler_type,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        results[handler_type] = success
        print()
    
    # 汇总结果
    print('=' * 80)
    print('测试结果汇总')
    print('=' * 80)
    
    for handler_type, success in results.items():
        status = '✓ 成功' if success else '✗ 失败'
        print(f'{handler_type:20s}: {status}')
    
    # 统计
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print()
    print(f'总计: {total} 个测试')
    print(f'成功: {passed} 个')
    print(f'失败: {failed} 个')
    
    if failed == 0:
        print('\n✓ 所有测试通过！')
        return 0
    else:
        print(f'\n✗ {failed} 个测试失败')
        return 1


if __name__ == '__main__':
    sys.exit(main())

