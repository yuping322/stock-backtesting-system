#!/usr/bin/env python3
"""
TALIB因子数据质量检查程序

检查生成的数据是否有问题，包括：
- 文件格式正确性
- 数据类型检查
- 缺失值检查
- 日期范围验证
- 股票代码格式检查
- 因子值异常值检测
- 数据一致性检查
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

def check_file_format(file_path):
    """检查文件格式是否正确"""
    try:
        # 读取前几行检查格式
        df_sample = pd.read_csv(file_path, nrows=5)

        # 检查必需的列
        required_columns = ['date', 'code', 'factor_value']
        missing_columns = [col for col in required_columns if col not in df_sample.columns]

        if missing_columns:
            return False, f"缺少必需列: {missing_columns}"

        # 检查列的数据类型
        if not pd.api.types.is_string_dtype(df_sample['date']):
            return False, "date列不是字符串类型"

        # 检查code列：可以是字符串或整数（股票代码）
        if not (pd.api.types.is_string_dtype(df_sample['code']) or pd.api.types.is_integer_dtype(df_sample['code'])):
            return False, "code列不是字符串或整数类型"

        if not pd.api.types.is_numeric_dtype(df_sample['factor_value']):
            return False, "factor_value列不是数值类型"

        return True, "文件格式正确"

    except Exception as e:
        return False, f"文件读取失败: {str(e)}"

def check_data_quality(file_path, factor_name):
    """检查单个文件的数据质量"""
    issues = []

    try:
        # 读取数据
        df = pd.read_csv(file_path)

        # 1. 检查记录数量
        expected_records = 80915  # 根据汇总报告
        if len(df) != expected_records:
            issues.append(f"记录数量异常: 期望{expected_records}, 实际{len(df)}")

        # 2. 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.any():
            for col, count in missing_values.items():
                if count > 0:
                    issues.append(f"{col}列有{count}个缺失值")

        # 3. 检查日期格式和范围
        try:
            df['date'] = pd.to_datetime(df['date'])
            date_range = df['date'].agg(['min', 'max'])
            expected_min = pd.Timestamp('2025-01-01')
            expected_max = pd.Timestamp('2025-11-23')

            if date_range['min'] < expected_min or date_range['max'] > expected_max:
                issues.append(f"日期范围异常: {date_range['min']} ~ {date_range['max']}")

        except Exception as e:
            issues.append(f"日期格式错误: {str(e)}")

        # 4. 检查股票代码格式
        invalid_codes = []
        for code in df['code'].unique():
            # 如果是整数，转换为6位字符串格式
            if isinstance(code, (int, np.integer)):
                code_str = f"{int(code):06d}"
            else:
                code_str = str(code).strip()
            
            # 检查是否为6位数字
            if not (len(code_str) == 6 and code_str.isdigit()):
                invalid_codes.append(code_str)

        if len(invalid_codes) > 10:  # 只显示前10个
            issues.append(f"股票代码格式异常: {len(invalid_codes)}个无效代码 (示例: {invalid_codes[:5]})")
        elif invalid_codes:
            issues.append(f"股票代码格式异常: {invalid_codes}")

        # 5. 检查因子值异常
        factor_values = df['factor_value']

        # 检查NaN和Inf
        nan_count = factor_values.isna().sum()
        inf_count = np.isinf(factor_values).sum()

        if nan_count > 0:
            issues.append(f"因子值包含{nan_count}个NaN值")

        if inf_count > 0:
            issues.append(f"因子值包含{inf_count}个Inf值")

        # 检查异常大值（根据因子类型设置阈值）
        if 'RSI' in factor_name:
            # RSI应该在0-100之间
            outliers = ((factor_values < -10) | (factor_values > 110)).sum()
            if outliers > 0:
                issues.append(f"RSI值异常: {outliers}个值超出合理范围(-10, 110)")
        elif 'MACD' in factor_name:
            # MACD通常不会超过股价太多，这里设置一个宽松的阈值
            outliers = (abs(factor_values) > 1000).sum()
            if outliers > 0:
                issues.append(f"MACD值异常: {outliers}个绝对值超过1000")
        elif any(x in factor_name for x in ['ROC', 'MOM', 'CMO']):
            # 变化率指标，允许较大范围但检查极端值
            outliers = (abs(factor_values) > 10000).sum()
            if outliers > 0:
                issues.append(f"变化率值异常: {outliers}个绝对值超过10000")
        else:
            # 其他指标检查极端值
            outliers = (abs(factor_values) > 1000000).sum()
            if outliers > 0:
                issues.append(f"因子值异常: {outliers}个绝对值超过1000000")

        # 6. 检查数据分布
        if len(factor_values) > 0:
            stats = factor_values.describe()
            # 检查是否所有值都相同（可能是计算错误）
            if stats['std'] == 0 and len(factor_values) > 1:
                issues.append("所有因子值都相同，可能存在计算错误")

        return issues

    except Exception as e:
        return [f"数据质量检查失败: {str(e)}"]

def check_data_consistency(data_dir):
    """检查数据一致性"""
    issues = []

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        return ["未找到任何CSV文件"]

    # 检查文件数量
    if len(csv_files) != 216:  # 根据汇总报告
        issues.append(f"文件数量异常: 期望216个文件, 实际{len(csv_files)}个")

    # 检查每个文件的记录数
    file_record_counts = {}
    for file_path in csv_files:
        try:
            # 使用wc -l快速计数（比pandas快很多）
            import subprocess
            result = subprocess.run(['wc', '-l', file_path],
                                  capture_output=True, text=True)
            line_count = int(result.stdout.split()[0])
            # 减去标题行
            record_count = line_count - 1
            file_record_counts[os.path.basename(file_path)] = record_count
        except Exception as e:
            issues.append(f"无法统计文件记录数 {os.path.basename(file_path)}: {str(e)}")
            continue

    # 检查记录数是否一致
    if file_record_counts:
        record_counts = list(file_record_counts.values())
        unique_counts = set(record_counts)

        if len(unique_counts) > 1:
            # 找出异常的文件
            expected_count = max(set(record_counts), key=record_counts.count)  # 最常见的数量
            abnormal_files = [f for f, c in file_record_counts.items() if c != expected_count]

            if len(abnormal_files) <= 5:  # 只显示前5个
                issues.append(f"记录数不一致: 大多数文件有{expected_count}条记录, 异常文件: {abnormal_files}")
            else:
                issues.append(f"记录数不一致: {len(abnormal_files)}个文件记录数异常 (期望{expected_count}条)")

    return issues, file_record_counts

def generate_quality_report(data_dir, output_file):
    """生成数据质量报告"""
    print("开始数据质量检查...")
    print(f"检查目录: {data_dir}")

    all_issues = []
    file_results = {}

    # 1. 检查数据一致性
    print("\n1. 检查数据一致性...")
    consistency_issues, record_counts = check_data_consistency(data_dir)
    all_issues.extend(consistency_issues)

    if consistency_issues:
        print("   ❌ 发现一致性问题:")
        for issue in consistency_issues:
            print(f"      {issue}")
    else:
        print("   ✅ 数据一致性检查通过")

    # 2. 随机抽样检查文件质量
    print("\n2. 检查文件质量 (随机抽样)...")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print("   ❌ 未找到CSV文件")
        return

    # 随机选择一些文件进行详细检查
    import random
    sample_size = min(10, len(csv_files))  # 检查10个或全部文件
    sample_files = random.sample(csv_files, sample_size)

    print(f"   随机选择{sample_size}个文件进行详细检查...")

    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        print(f"   检查文件: {file_name}")

        # 检查文件格式
        format_ok, format_msg = check_file_format(file_path)
        if not format_ok:
            all_issues.append(f"{file_name}: {format_msg}")
            print(f"      ❌ 格式错误: {format_msg}")
            continue
        else:
            print("      ✅ 文件格式正确")

        # 检查数据质量
        quality_issues = check_data_quality(file_path, file_name)
        if quality_issues:
            for issue in quality_issues:
                all_issues.append(f"{file_name}: {issue}")
            print(f"      ❌ 发现{len(quality_issues)}个问题")
            for issue in quality_issues[:3]:  # 只显示前3个问题
                print(f"         {issue}")
        else:
            print("      ✅ 数据质量检查通过")

        file_results[file_name] = {
            'format_ok': format_ok,
            'quality_issues': quality_issues
        }

    # 3. 生成报告
    print(f"\n3. 生成质量报告: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TALIB因子数据质量检查报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检查目录: {data_dir}\n")
        f.write(f"总文件数: {len(csv_files)}\n")
        f.write(f"抽样检查数: {sample_size}\n\n")

        f.write("数据一致性检查:\n")
        if consistency_issues:
            f.write("❌ 发现问题:\n")
            for issue in consistency_issues:
                f.write(f"   {issue}\n")
        else:
            f.write("✅ 通过\n")
        f.write("\n")

        f.write("详细检查结果:\n")
        total_files_checked = len(file_results)
        files_with_issues = sum(1 for r in file_results.values() if r['quality_issues'])

        f.write(f"检查文件数: {total_files_checked}\n")
        f.write(f"有问题文件数: {files_with_issues}\n")
        f.write(f"正常文件数: {total_files_checked - files_with_issues}\n\n")

        if all_issues:
            f.write("所有发现的问题:\n")
            for i, issue in enumerate(all_issues, 1):
                f.write(f"{i:3d}. {issue}\n")
        else:
            f.write("✅ 未发现任何问题！\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("检查完成\n")

    print(f"✅ 质量报告已保存: {output_file}")

    # 总结
    print("\n" + "=" * 50)
    print("检查总结:")
    print(f"总文件数: {len(csv_files)}")
    print(f"抽样检查: {sample_size}")
    print(f"发现问题: {len(all_issues)}")
    if all_issues:
        print("主要问题:")
        for issue in all_issues[:5]:  # 显示前5个问题
            print(f"  - {issue}")
        if len(all_issues) > 5:
            print(f"  ... 还有{len(all_issues) - 5}个问题")
    else:
        print("✅ 所有检查通过！数据质量良好。")
    print("=" * 50)

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python check_talib_data_quality.py <数据目录>")
        print("示例: python check_talib_data_quality.py data/talib_factors_2025")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在: {data_dir}")
        sys.exit(1)

    # 生成报告文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"talib_data_quality_report_{timestamp}.txt"

    # 执行检查
    generate_quality_report(data_dir, report_file)

if __name__ == "__main__":
    main()