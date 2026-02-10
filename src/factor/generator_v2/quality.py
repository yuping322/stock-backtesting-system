"""
数据质量检查 - V2 版本中的质量保证机制

该模块提供了一个完整的数据质量检查框架，确保生成的因子数据符合质量标准。

检查项:
    1. 列的完整性检查
    2. 数据类型检查
    3. NaN 值检查
    4. 每日股票数一致性检查
    5. 因子值极端异常检查
    6. 日期连续性检查
    7. 股票代码标准化检查
"""

import logging
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    数据质量检查器
    
    对因子输出进行多维度的质量检查。
    """
    
    @staticmethod
    def check_factor_output(
        df: pd.DataFrame,
        factor_names: List[str],
        allow_warnings: bool = True
    ) -> Dict[str, Any]:
        """
        检查因子输出的质量
        
        Args:
            df: 因子数据 DataFrame
            factor_names: 因子列名列表
            allow_warnings: 是否允许警告级别的问题（如果否，警告会变为错误）
        
        Returns:
            检查结果字典:
            {
                'passed': bool,                    # 是否通过检查
                'issues': [
                    {'level': 'error|warning', 'message': '...'},
                    ...
                ],
                'summary': {
                    'total_issues': int,
                    'errors': int,
                    'warnings': int
                }
            }
        """
        issues = []
        
        # 检查 1: 必要的列
        issues.extend(DataQualityChecker._check_required_columns(df, factor_names))
        
        # 检查 2: 数据类型
        issues.extend(DataQualityChecker._check_data_types(df, factor_names))
        
        # 检查 3: NaN 值
        issues.extend(DataQualityChecker._check_nan_values(df, factor_names))
        
        # 检查 4: 每日股票数一致性
        issues.extend(DataQualityChecker._check_daily_consistency(df))
        
        # 检查 5: 极端异常值
        issues.extend(DataQualityChecker._check_extreme_values(df, factor_names))
        
        # 检查 6: 日期连续性
        issues.extend(DataQualityChecker._check_date_continuity(df))
        
        # 检查 7: 股票代码标准化
        issues.extend(DataQualityChecker._check_stock_codes(df))
        
        # 根据 allow_warnings 调整
        if not allow_warnings:
            for issue in issues:
                if issue['level'] == 'warning':
                    issue['level'] = 'error'
        
        # 统计
        passed = not any(issue['level'] == 'error' for issue in issues)
        summary = {
            'total_issues': len(issues),
            'errors': sum(1 for i in issues if i['level'] == 'error'),
            'warnings': sum(1 for i in issues if i['level'] == 'warning')
        }
        
        return {
            'passed': passed,
            'issues': issues,
            'summary': summary
        }
    
    @staticmethod
    def _check_required_columns(df: pd.DataFrame, factor_names: List[str]) -> List[Dict]:
        """检查必要的列是否存在"""
        issues = []
        
        required_cols = {'date', 'stock_code'} | set(factor_names)
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            issues.append({
                'level': 'error',
                'message': f"缺少必要的列: {missing_cols}"
            })
        
        return issues
    
    @staticmethod
    def _check_data_types(df: pd.DataFrame, factor_names: List[str]) -> List[Dict]:
        """检查数据类型是否正确"""
        issues = []
        
        # 检查 date 列
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                issues.append({
                    'level': 'warning',
                    'message': "date 列不是 datetime 类型，应该转换为 datetime"
                })
        
        # 检查 factor 列
        for col in factor_names:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append({
                        'level': 'error',
                        'message': f"因子列 {col} 不是数值类型"
                    })
        
        return issues
    
    @staticmethod
    def _check_nan_values(df: pd.DataFrame, factor_names: List[str]) -> List[Dict]:
        """检查 NaN 值的比例"""
        issues = []
        
        for col in factor_names:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                total = len(df)
                nan_ratio = nan_count / total if total > 0 else 0
                
                if nan_ratio > 0.7:
                    issues.append({
                        'level': 'error',
                        'message': f"因子 {col} NaN 比例过高: {nan_ratio:.1%} ({nan_count}/{total})"
                    })
                elif nan_ratio > 0.2:
                    issues.append({
                        'level': 'warning',
                        'message': f"因子 {col} 有较多 NaN 值: {nan_ratio:.1%} ({nan_count}/{total})"
                    })
        
        return issues
    
    @staticmethod
    def _check_daily_consistency(df: pd.DataFrame) -> List[Dict]:
        """检查每日股票数的一致性"""
        issues = []
        
        if 'date' not in df.columns or 'stock_code' not in df.columns:
            return issues
        
        daily_counts = df.groupby('date')['stock_code'].count()
        
        if daily_counts.empty:
            return issues
        
        count_std = daily_counts.std()
        count_mean = daily_counts.mean()
        
        if count_mean == 0:
            return issues
        
        cv = count_std / count_mean  # 变异系数
        
        if cv > 0.5:
            issues.append({
                'level': 'warning',
                'message': f"每日股票数波动较大 (CV={cv:.2f}): " \
                           f"min={daily_counts.min()}, max={daily_counts.max()}, mean={count_mean:.1f}"
            })
        
        return issues
    
    @staticmethod
    def _check_extreme_values(df: pd.DataFrame, factor_names: List[str]) -> List[Dict]:
        """检查极端异常值"""
        issues = []
        
        for col in factor_names:
            if col not in df.columns:
                continue
            
            # 过滤 NaN 值
            data = df[col].dropna()
            if data.empty:
                continue
            
            # 计算四分位数
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            # 计算异常值边界（3 倍 IQR）
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_ratio = outliers / len(data) if len(data) > 0 else 0
            
            if outlier_ratio > 0.1:
                issues.append({
                    'level': 'warning',
                    'message': f"因子 {col} 有 {outlier_ratio:.1%} 的异常值 "
                               f"(范围: {lower_bound:.4f} ~ {upper_bound:.4f})"
                })
        
        return issues
    
    @staticmethod
    def _check_date_continuity(df: pd.DataFrame) -> List[Dict]:
        """检查日期序列的连续性"""
        issues = []
        
        if 'date' not in df.columns:
            return issues
        
        dates = sorted(df['date'].unique())
        if len(dates) < 2:
            return issues
        
        dates_series = pd.Series(dates)
        date_diffs = dates_series.diff().dt.days
        
        # 移除第一个 NaT
        date_diffs = date_diffs[1:]
        
        if date_diffs.empty:
            return issues
        
        max_gap = date_diffs.max()
        
        # 允许 5 天的间隔（正常工作日间隔 + 周末）
        if max_gap > 5:
            issues.append({
                'level': 'warning',
                'message': f"日期序列不连续，最大间隔 {max_gap} 天"
            })
        
        return issues
    
    @staticmethod
    def _check_stock_codes(df: pd.DataFrame) -> List[Dict]:
        """检查股票代码是否标准化"""
        issues = []
        
        if 'stock_code' not in df.columns:
            return issues
        
        codes = df['stock_code']
        
        # 检查长度
        invalid_codes = codes[codes.astype(str).str.len() != 6].nunique()
        if invalid_codes > 0:
            issues.append({
                'level': 'error',
                'message': f"发现 {invalid_codes} 个非标准股票代码（应该是 6 位数字）"
            })
        
        # 检查是否都是数字
        try:
            codes.astype(int)
        except (ValueError, TypeError):
            non_numeric = codes[~codes.astype(str).str.isdigit()].nunique()
            if non_numeric > 0:
                issues.append({
                    'level': 'error',
                    'message': f"发现 {non_numeric} 个非数字股票代码"
                })
        
        return issues
    
    @staticmethod
    def print_check_result(result: Dict[str, Any], verbose: bool = True) -> None:
        """
        打印检查结果
        
        Args:
            result: 检查结果字典
            verbose: 是否打印详细信息
        """
        summary = result['summary']
        print(f"\n数据质量检查结果:")
        print(f"{'='*50}")
        print(f"总问题数: {summary['total_issues']}")
        print(f"错误数:   {summary['errors']}")
        print(f"警告数:   {summary['warnings']}")
        print(f"状态:     {'✅ 通过' if result['passed'] else '❌ 未通过'}")
        print(f"{'='*50}")
        
        if verbose and result['issues']:
            print(f"\n问题详情:")
            for i, issue in enumerate(result['issues'], 1):
                level_icon = "❌" if issue['level'] == 'error' else "⚠️ "
                print(f"{i}. {level_icon} [{issue['level'].upper()}] {issue['message']}")
