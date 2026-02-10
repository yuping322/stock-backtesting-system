"""
因子分析层

包含：
- FactorAnalyzer: 因子分析类
- FactorAnalysisCore: 核心分析方法
- export_analysis_report(): 导出分析报告
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import time
from typing import List, Optional, Dict, Any


# 提前将本地 src/factor 目录加入路径，优先加载自带的 alphalens
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_DIR = os.path.dirname(CURRENT_DIR)
SRC_ROOT = os.path.dirname(FACTOR_DIR)
REPO_ROOT = os.path.dirname(SRC_ROOT)

for path in (FACTOR_DIR, SRC_ROOT, REPO_ROOT):
    if path and path not in sys.path:
        sys.path.insert(0, path)

import alphalens as al
import data

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

from .core import FactorAnalyzer, FactorAnalysisCore


def export_analysis_report(analyzer: FactorAnalyzer, output_dir: str, append: bool = False):
    """
    导出分析报告

    Args:
        analyzer: FactorAnalyzer 实例
        output_dir: 输出目录
        append: 是否追加模式（用于增量分析）
    """
    import os
    import pandas as pd
    from pathlib import Path
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not analyzer.analysis_results:
        print("⚠️  没有分析结果可导出")
        return
    
    # 准备汇总数据
    summary_data = []
    
    for result in analyzer.analysis_results:
        if isinstance(result, dict):
            # 兼容旧格式
            row = {
                '因子名称': result.get('factor_name', ''),
                '调仓周期': result.get('period', ''),
                '等级': result.get('level', ''),
                'IC均值': result.get('scores', {}).get('IC均值', [np.nan, False])[0],
                'IR比率': result.get('scores', {}).get('IR比率', [np.nan, False])[0],
                '多空年化': result.get('scores', {}).get('多空年化', [np.nan, False])[0],
                '单调性': result.get('scores', {}).get('单调性', [np.nan, False])[0],
                'Top换手率': result.get('scores', {}).get('Top换手率', [np.nan, False])[0],
                '状态': getattr(result.get('rolling_monitor'), 'status_flag', '') if result.get('rolling_monitor') else '',
                'Top换手率数值': result.get('top_turnover', np.nan)
            }
        else:
            # 新格式 FactorTestResult
            row = {
                '因子名称': result.factor_name,
                '调仓周期': result.period,
                '等级': result.level,
                'IC均值': result.scores.get('IC均值', [np.nan, False])[0],
                'IR比率': result.scores.get('IR比率', [np.nan, False])[0],
                '多空年化': result.scores.get('多空年化', [np.nan, False])[0],
                '单调性': result.scores.get('单调性', [np.nan, False])[0],
                'Top换手率': result.scores.get('Top换手率', [np.nan, False])[0],
                '状态': result.status_flag,
                'Top换手率数值': result.top_turnover
            }
        summary_data.append(row)
    
    # 创建汇总表
    summary_df = pd.DataFrame(summary_data)
    
    # 导出到CSV
    output_file = output_path / 'factor_analysis_summary.csv'
    
    if append and output_file.exists():
        # 追加模式：读取现有文件，合并新结果，避免重复
        existing_df = pd.read_csv(output_file)
        # 合并，去重（基于因子名称和调仓周期）
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['因子名称', '调仓周期'], keep='last')
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.4f')
    else:
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig', float_format='%.4f')
    
    print(f"✓ 因子分析汇总表已导出: {output_file}")
    print(f"  共 {len(summary_df)} 条记录")
    
    # 打印简要统计
    if not summary_df.empty:
        print("\n📊 汇总统计:")
        print(f"  因子数量: {summary_df['因子名称'].nunique()}")
        print(f"  优秀因子: {len(summary_df[summary_df['等级'] == '优秀'])}")
        print(f"  良好因子: {len(summary_df[summary_df['等级'] == '良好'])}")
        print(f"  一般因子: {len(summary_df[summary_df['等级'] == '一般'])}")
        
        # 计算平均指标
        numeric_cols = ['IC均值', 'IR比率', '多空年化', '单调性']
        print("\n📈 平均指标:")
        for col in numeric_cols:
            if col in summary_df.columns:
                mean_val = summary_df[col].mean()
                print(f"  {col}: {mean_val:.4f}")
