"""
基于所有因子类型的完整分析工作流

这个脚本展示了如何：
1. 生成多种类型的因子数据（TALIB、Alpha158、Alpha360等）
2. 进行大规模因子测试
3. 分析测试结果并筛选优秀因子
4. 进行深入分析和参数调优
5. 生成可用于 factor_workflow 的数据
6. 运行完整的量化投资流程

支持的因子类型：
- TALIB因子：技术指标因子（216个）
- Alpha158因子：量化因子（158个）
- Alpha360因子：扩展量化因子（360个）
- 其他因子：从 available_factors.txt 加载
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# 因子类型定义
FACTOR_TYPES = {
    'talib': {
        'name': 'TALIB因子',
        'file': '../talib_factors.txt',
        'description': '技术分析指标因子'
    },
    'alpha158': {
        'name': 'Alpha158因子',
        'file': '../alpha158_factors.txt',
        'description': '量化投资158个因子'
    },
    'alpha360': {
        'name': 'Alpha360因子',
        'file': '../alpha360_factors.txt',
        'description': '量化投资360个因子'
    },
    'available': {
        'name': '可用因子',
        'file': '../available_factors.txt',
        'description': '其他可用因子'
    }
}


def load_factor_list(factor_type: str) -> List[str]:
    """加载指定类型的因子列表"""
    if factor_type not in FACTOR_TYPES:
        print(f"错误：未知的因子类型 {factor_type}")
        return []

    factor_info = FACTOR_TYPES[factor_type]
    factor_file = Path(factor_info['file'])

    if not factor_file.exists():
        print(f"警告：因子文件不存在 {factor_file}")
        return []

    try:
        with open(factor_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 过滤注释行和空行
        factors = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                factors.append(line)

        print(f"✓ 加载 {factor_info['name']}: {len(factors)} 个因子")
        return factors

    except Exception as e:
        print(f"❌ 加载因子文件失败 {factor_file}: {e}")
        return []


def load_all_factors() -> Dict[str, List[str]]:
    """加载所有类型的因子"""
    all_factors = {}
    for factor_type in FACTOR_TYPES.keys():
        factors = load_factor_list(factor_type)
        if factors:
            all_factors[factor_type] = factors
    return all_factors


def generate_factor_data(factor_types: List[str] = None,
                        start_date: str = None,
                        end_date: str = None,
                        max_factors_per_type: int = 50) -> bool:
    """生成指定类型因子的数据"""
    print("\n=== 生成因子数据 ===")

    if factor_types is None:
        factor_types = list(FACTOR_TYPES.keys())

    if start_date is None or end_date is None:
        # 默认使用最近3个月的数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"因子类型: {factor_types}")

    success_count = 0

    for factor_type in factor_types:
        if factor_type not in FACTOR_TYPES:
            print(f"⚠️  跳过未知因子类型: {factor_type}")
            continue

        print(f"\n处理 {FACTOR_TYPES[factor_type]['name']}...")

        # 加载因子列表
        factors = load_factor_list(factor_type)
        if not factors:
            continue

        # 限制因子数量以提高性能
        if len(factors) > max_factors_per_type:
            factors = factors[:max_factors_per_type]
            print(f"  限制为前 {max_factors_per_type} 个因子")

        # 确定生成脚本
        if factor_type == 'talib':
            script_name = '../generate_talib_factors.py'
        else:
            script_name = '../generate_factors.py'

        script_path = Path(script_name)
        if not script_path.exists():
            print(f"  ❌ 生成脚本不存在: {script_path}")
            continue

        # 构建命令
        cmd = [
            sys.executable, str(script_path),
            '--start', start_date,
            '--end', end_date,
            '--factors'
        ] + factors + [
            '--output-dir', f'../factor_data_{factor_type}',
            '--stock-pool', '000300',  # 使用沪深300成分股
            '--max-stocks', '200'  # 限制股票数量
        ]

        print(f"  执行命令: {' '.join(cmd[:5])} ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print(f"  ✓ {FACTOR_TYPES[factor_type]['name']} 生成成功")
                success_count += 1
            else:
                print(f"  ❌ {FACTOR_TYPES[factor_type]['name']} 生成失败")
                print(f"    错误: {result.stderr[:200]}...")
        except Exception as e:
            print(f"  ❌ 执行失败: {e}")

    print(f"\n因子数据生成完成: {success_count}/{len(factor_types)} 种类型成功")
    return success_count > 0


def run_factor_tests(factor_types: List[str] = None,
                    start_date: str = None,
                    end_date: str = None,
                    max_factors_per_type: int = 20) -> Dict[str, str]:
    """运行大规模因子测试"""
    print("\n=== 运行大规模因子测试 ===")

    if factor_types is None:
        factor_types = list(FACTOR_TYPES.keys())

    if start_date is None or end_date is None:
        # 默认使用最近3个月的数据
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"因子类型: {factor_types}")

    test_results = {}

    for factor_type in factor_types:
        if factor_type not in FACTOR_TYPES:
            print(f"⚠️  跳过未知因子类型: {factor_type}")
            continue

        print(f"\n测试 {FACTOR_TYPES[factor_type]['name']}...")

        # 加载因子列表
        factors = load_factor_list(factor_type)
        if not factors:
            continue

        # 限制因子数量以加快测试
        if len(factors) > max_factors_per_type:
            factors = factors[:max_factors_per_type]
            print(f"  限制为前 {max_factors_per_type} 个因子")

        # 确定测试脚本
        if factor_type == 'talib':
            script_name = '../scripts/run_all_talib_factors.sh'
        elif factor_type in ['alpha158', 'alpha360']:
            script_name = '../scripts/test_alpha158_factors.sh'  # 可以通用
        else:
            script_name = '../scripts/test_all_factor_types.py'

        script_path = Path(script_name)
        if not script_path.exists():
            print(f"  ❌ 测试脚本不存在: {script_path}")
            continue

        # 构建命令
        if script_name.endswith('.sh'):
            cmd = ['bash', str(script_path)]
        else:
            cmd = [
                sys.executable, str(script_path),
                '--start', start_date,
                '--end', end_date
            ]

        print(f"  执行命令: {' '.join(cmd[:3])} ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print(f"  ✓ {FACTOR_TYPES[factor_type]['name']} 测试成功")

                # 查找结果目录
                result_dir = find_latest_test_result(factor_type)
                if result_dir:
                    test_results[factor_type] = result_dir
                    print(f"    结果目录: {result_dir}")
                else:
                    print("    ⚠️  未找到结果目录")
            else:
                print(f"  ❌ {FACTOR_TYPES[factor_type]['name']} 测试失败")
                print(f"    错误: {result.stderr[:200]}...")
        except Exception as e:
            print(f"  ❌ 执行失败: {e}")

    print(f"\n因子测试完成: {len(test_results)}/{len(factor_types)} 种类型成功")
    return test_results


def find_latest_test_result(factor_type: str) -> Optional[str]:
    """查找最新的测试结果目录"""
    results_dir = Path("../results")

    if not results_dir.exists():
        return None

    # 根据因子类型查找对应的结果目录
    pattern = f"factor_test_{factor_type}_*"
    matching_dirs = list(results_dir.glob(pattern))

    if matching_dirs:
        # 返回最新的目录
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        return str(latest_dir)

    # 如果没找到，查找通用模式
    all_test_dirs = list(results_dir.glob("factor_test_*"))
    if all_test_dirs:
        latest_dir = max(all_test_dirs, key=lambda x: x.stat().st_mtime)
        return str(latest_dir)

    return None


def get_all_factors_and_stocks(start_date: str, end_date: str) -> tuple[list[str], list[str]]:
    """读取该期间内的全量因子列和股票列表（从因子数据本身推断）"""
    from factor.export_data import data
    df = data.read_factor_data(
        codes=None,
        start_date=start_date,
        end_date=end_date,
        factors=None,
        base_path="uploads"
    )
    if df.empty:
        raise ValueError("该日期区间未读取到因子数据")
    factors = [c for c in df.columns]
    stocks = df.index.get_level_values('code').unique().tolist()
    # 统一为6位代码（去掉后缀），export内部会做规范化，这里尽量简化
    stocks = [s.replace('.XSHG', '').replace('.XSHE', '') for s in stocks]
    return factors, stocks


def run_workflow_pipeline():
    """运行完整的 factor_workflow 流程"""
    print("\n=== 运行 factor_workflow 流程 ===")

    # 检查必要的数据文件是否存在
    required_files = [
        "../exported_data_all/features_panel.pkl",
        "../exported_data_all/label_panel.pkl",
        "../exported_data_all/meta_series.pkl"
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("缺少必要的数据文件:")
        for f in missing_files:
            print(f"  - {f}")
        print("请先运行数据转换步骤")
        return

    workflow_dir = Path(".")
    if not workflow_dir.exists():
        print(f"错误：当前目录不是有效的 factor_workflow 目录")
        return

    # 已经在 factor_workflow 目录下，直接运行
    # 运行各个步骤
    steps = [
        ([sys.executable, 'workflow_main.py'], "训练模型"),
        ([sys.executable, 'backtest_evaluation.py'], "评估和回测"),
        ([sys.executable, 'export_scores.py'], "导出预测权重")
    ]

    for cmd, desc in steps:
        print(f"\n执行: {desc}")
        print(' '.join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {desc} 完成")
        else:
            print(f"✗ {desc} 失败")
            print("错误输出:", result.stderr)
            break


def analyze_all_test_results(test_results: Dict[str, str]) -> Dict[str, List[str]]:
    """分析所有因子类型的测试结果"""
    print("\n=== 分析所有因子测试结果 ===")

    all_good_factors = {}

    for factor_type, results_dir in test_results.items():
        print(f"\n分析 {FACTOR_TYPES[factor_type]['name']} 结果...")

        summary_file = Path(results_dir) / "summary.csv"
        if not summary_file.exists():
            print(f"  ❌ 找不到汇总文件 {summary_file}")
            continue

        try:
            # 读取汇总结果
            summary = pd.read_csv(summary_file)

            print(f"  总测试组合数: {len(summary)}")
            print(f"  因子数量: {summary['factor_name'].nunique()}")

            # 筛选优秀因子
            good_factors = summary[
                (summary['level'] == '优秀') &
                (summary['status_flag'] != '🔴 dead')
            ].copy()

            good_factor_list = sorted(good_factors['factor_name'].unique())
            all_good_factors[factor_type] = good_factor_list

            print(f"  ✓ 优秀因子数量: {len(good_factor_list)}")

            if good_factor_list:
                print(f"  优秀因子: {good_factor_list[:5]}..." if len(good_factor_list) > 5 else f"  优秀因子: {good_factor_list}")

        except Exception as e:
            print(f"  ❌ 分析失败: {e}")

    return all_good_factors


def create_unified_workflow_data(all_good_factors: Dict[str, List[str]]) -> bool:
    """创建统一的 factor_workflow 数据"""
    print("\n=== 创建统一的 factor_workflow 数据 ===")

    # 合并所有优秀因子
    all_factors = []
    for factor_list in all_good_factors.values():
        all_factors.extend(factor_list)

    if not all_factors:
        print("❌ 没有找到任何优秀因子")
        return False

    print(f"总优秀因子数量: {len(all_factors)}")
    print(f"因子类型分布: { {k: len(v) for k, v in all_good_factors.items()} }")

    # 使用最近3个月的数据范围
    from factor.export_data import get_last_3_months
    start_date, end_date = get_last_3_months()

    # 获取股票列表
    factors, stocks = get_all_factors_and_stocks(start_date, end_date)

    # 只保留优秀因子
    factors = [f for f in factors if f in all_factors]

    if not factors:
        print("⚠️  优秀因子在指定日期范围内无数据，使用示例因子")
        factors = all_factors[:10]  # 使用前10个优秀因子作为示例

    print(f"实际使用的因子数量: {len(factors)}")

    # 生成 formatted_data.csv
    print("\n步骤1: 生成 formatted_data.csv")
    from factor.export_data import export_formatted_csv

    csv_file = export_formatted_csv(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        output_dir='../exported_data_all',
        factors=factors,
        industry_default='Unknown'
    )

    print(f"✓ formatted_data.csv 生成完成: {csv_file}")

    # 转换数据格式
    print("\n步骤2: 转换数据格式")
    cmd = [
        sys.executable, 'convert_sample.py',
        '--input-csv', csv_file,
        '--limit-stocks', '100'  # 稍微增加股票数量
    ]

    print("执行命令:")
    print(' '.join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 数据转换完成")
            return True
        else:
            print("✗ 数据转换失败")
            print("错误输出:", result.stderr)
            return False
    except Exception as e:
        print(f"执行失败: {e}")
        return False


def main():
    """主函数 - 支持所有因子类型的完整流程"""
    print("基于所有因子类型的完整量化投资工作流")
    print("=" * 60)

    # 步骤1：生成因子数据
    print("\n步骤1: 生成因子数据")
    data_success = generate_factor_data(max_factors_per_type=30)  # 每个类型生成30个因子

    if not data_success:
        print("⚠️  因子数据生成部分失败，继续使用现有数据")

    # 步骤2：运行大规模因子测试
    print("\n步骤2: 运行大规模因子测试")
    test_results = run_factor_tests(max_factors_per_type=20)  # 每个类型测试20个因子

    if not test_results:
        print("❌ 没有成功的因子测试结果")
        return

    # 步骤3：分析测试结果
    print("\n步骤3: 分析测试结果")
    all_good_factors = analyze_all_test_results(test_results)

    if not all_good_factors:
        print("❌ 没有找到优秀因子")
        return

    # 步骤4：创建工作流数据
    print("\n步骤4: 创建工作流数据")
    workflow_success = create_unified_workflow_data(all_good_factors)

    if not workflow_success:
        print("❌ 工作流数据创建失败")
        return

    # 步骤5：运行量化投资流程
    print("\n步骤5: 运行量化投资流程")
    run_workflow_pipeline()

    print("\n" + "=" * 60)
    print("完整量化投资流程完成！")
    print("=" * 60)
    print("\n输出文件:")
    print("- ../results/good_*_factors.txt: 各类型优秀因子列表")
    print("- ../exported_data_all/formatted_data.csv: 宽表格式数据")
    print("- ../exported_data_all/features_panel.pkl: 因子数据")
    print("- ../exported_data_all/label_panel.pkl: 标签数据")
    print("- ../exported_data_all/meta_series.pkl: 元数据")
    print("- ../results/factor_workflow/scores.csv: 最终预测权重")


if __name__ == "__main__":
    main()