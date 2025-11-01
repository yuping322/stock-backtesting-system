"""
测试 main_factor.py 的功能
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMainFactor(unittest.TestCase):
    """测试 main_factor.py"""
    
    def test_factor_test_result(self):
        """测试 FactorTestResult 类"""
        from factor.factor import FactorTestResult
        
        result = FactorTestResult()
        result.factor_name = 'TEST_FACTOR'
        result.period = 5
        result.level = '优秀'
        result.status_flag = '🟢 alive'
        result.scores = {'IC均值': (0.08, True), 'IR比率': (1.5, True)}
        
        # 测试 to_dict
        result_dict = result.to_dict()
        self.assertEqual(result_dict['factor_name'], 'TEST_FACTOR')
        self.assertEqual(result_dict['period'], 5)
        self.assertEqual(result_dict['level'], '优秀')
    
    def test_save_results(self):
        """测试结果保存功能（使用 generate_summary_report 替代）"""
        from main_factor import generate_summary_report
        from factor.factor import FactorTestResult
        
        # 创建测试结果
        result1 = FactorTestResult()
        result1.factor_name = 'VOL10'
        result1.period = 5
        result1.level = '优秀'
        result1.status_flag = '🟢 alive'
        result1.scores = {'IC均值': (0.08, True)}
        result1.ic_series = pd.Series([0.1, 0.15, 0.12], index=pd.date_range('2024-01-01', periods=3))
        result1.ret_series = pd.Series([0.01, -0.02, 0.03], index=pd.date_range('2024-01-01', periods=3))
        
        result2 = FactorTestResult()
        result2.factor_name = 'VOL10'
        result2.period = 10
        result2.level = '良好'
        result2.status_flag = '🟡 warning'
        result2.scores = {'IC均值': (0.06, False)}
        
        test_results = [result1, result2]
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造最小 cfg/args
            from types import SimpleNamespace
            cfg = SimpleNamespace(
                START='2024-01-01',
                END='2024-12-31',
                STOCK_POOL='HS300',
                QUANTILES=10,
                PERIODS=[5, 10],
                ROLL_WIN=60,
                FACTORS=['VOL10']
            )
            args = SimpleNamespace()
            # 生成汇总报告
            generate_summary_report(tmpdir, test_results, cfg, args)
            
            # 检查文件是否存在
            summary_file = Path(tmpdir) / 'summary.csv'
            self.assertTrue(summary_file.exists())
            
            # 检查内容
            summary_df = pd.read_csv(summary_file)
            self.assertEqual(len(summary_df), 2)
            self.assertIn('factor_name', summary_df.columns)
            self.assertIn('period', summary_df.columns)
            self.assertIn('level', summary_df.columns)
    
    def test_output_dir_setup(self):
        """测试输出目录设置"""
        from main_factor import setup_output_dir
        
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = setup_output_dir(tmpdir)
            self.assertTrue(os.path.exists(output_dir))
            
            # 测试自动生成目录
            output_dir2 = setup_output_dir(None)
            self.assertTrue(os.path.exists(output_dir2))


if __name__ == '__main__':
    unittest.main()

