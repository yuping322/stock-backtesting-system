"""
因子模块测试

测试 factor_calculator、factor.py 和相关功能
"""

import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactorCalculator(unittest.TestCase):
    """测试因子计算器"""

    def setUp(self):
        """设置测试数据"""
        # 创建示例 OHLCV 数据
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        self.ohlcv_data = pd.DataFrame({
            'open': np.random.rand(30) * 100,
            'high': np.random.rand(30) * 100 + 10,
            'low': np.random.rand(30) * 100 - 10,
            'close': np.random.rand(30) * 100,
            'volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)

    def test_builtin_factor_calculator(self):
        """测试内置因子计算器"""
        from factor.factor_calculator import BuiltinFactorCalculator

        # 测试 VOL10 因子
        calc = BuiltinFactorCalculator('VOL10')
        self.assertEqual(calc.factor_name, 'VOL10')
        self.assertIsNotNone(calc.factor_func)

        # 测试不存在的因子
        with self.assertRaises(ValueError):
            BuiltinFactorCalculator('INVALID_FACTOR')

    def test_ohlcv_factor_calculator(self):
        """测试 OHLCV 因子计算器"""
        from factor.factor_calculator import OHLCVFactorCalculator

        def my_factor(ohlcv):
            return ohlcv['close'] / ohlcv['close'].rolling(5).mean()

        calc = OHLCVFactorCalculator(my_factor)
        result = calc.factor_func(self.ohlcv_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 30)

    def test_file_factor_calculator(self):
        """测试文件因子计算器"""
        from factor.factor_calculator import FileFactorCalculator

        # 创建临时因子文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,code,MY_FACTOR\n')
            f.write('2024-01-01,000001,1.23\n')
            f.write('2024-01-02,000001,1.25\n')
            f.write('2024-01-01,000002,2.34\n')
            f.write('2024-01-02,000002,2.36\n')
            temp_file = f.name

        try:
            calc = FileFactorCalculator(temp_file, 'MY_FACTOR')
            
            # 测试计算
            result = calc.calculate('000001', '2024-01-01', '2024-01-02')
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), 2)
            
            # 测试不存在的股票
            result_empty = calc.calculate('999999', '2024-01-01', '2024-01-02')
            self.assertEqual(len(result_empty), 0)
            
        finally:
            os.unlink(temp_file)

    def test_custom_factor_calculator(self):
        """测试自定义因子计算器"""
        from factor.factor_calculator import CustomFactorCalculator

        def custom_calc(code, start, end):
            dates = pd.date_range(start, end, freq='D')
            return pd.Series([1.0] * len(dates), index=dates)

        calc = CustomFactorCalculator(custom_calc)
        result = calc.calculate('000001', '2024-01-01', '2024-01-05')
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)

    def test_create_factor_calculator(self):
        """测试工厂函数"""
        from factor.factor_calculator import create_factor_calculator

        # 测试内置因子
        calc1 = create_factor_calculator(factor_name='VOL10')
        self.assertIsNotNone(calc1)

        # 测试 OHLCV 函数
        def my_factor(ohlcv):
            return ohlcv['close'].pct_change()
        
        calc2 = create_factor_calculator(factor_func=my_factor)
        self.assertIsNotNone(calc2)

        # 测试自定义函数
        def custom_calc(code, start, end):
            return pd.Series([1.0], index=pd.date_range(start, periods=1))
        
        calc3 = create_factor_calculator(factor_func=custom_calc)
        self.assertIsNotNone(calc3)

        # 测试从文件加载
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,code,FACTOR\n')
            f.write('2024-01-01,000001,1.0\n')
            temp_file = f.name

        try:
            calc4 = create_factor_calculator(file_path=temp_file, factor_name='FACTOR')
            self.assertIsNotNone(calc4)
        finally:
            os.unlink(temp_file)

        # 测试错误情况
        with self.assertRaises(ValueError):
            create_factor_calculator()  # 无参数


class TestDataIntegration(unittest.TestCase):
    """测试与 data.py 的集成"""

    def test_load_factor_from_file(self):
        """测试从文件加载因子"""
        import data

        # 创建临时因子文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,code,MY_FACTOR\n')
            f.write('2024-01-01,000001,1.23\n')
            f.write('2024-01-02,000001,1.25\n')
            f.write('2024-01-03,000001,1.27\n')
            temp_file = f.name

        try:
            # 测试 factor_for_al 的新功能
            result = data.factor_for_al(
                codes=['000001'],
                start_date='2024-01-01',
                end_date='2024-01-03',
                factor_name='MY_FACTOR',
                file_path=temp_file
            )
            
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), 3)
            self.assertIn('date', result.index.names)
            self.assertIn('asset', result.index.names)
            
        finally:
            os.unlink(temp_file)

    def test_load_factor_from_file_with_normalized_codes(self):
        """测试文件加载时的代码标准化"""
        import data

        # 创建临时因子文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,code,MY_FACTOR\n')
            f.write('2024-01-01,000001.XSHG,1.23\n')
            f.write('2024-01-02,000001.XSHG,1.25\n')
            temp_file = f.name

        try:
            result = data.factor_for_al(
                codes=['000001'],
                start_date='2024-01-01',
                end_date='2024-01-02',
                factor_name='MY_FACTOR',
                file_path=temp_file
            )
            
            # 应该能匹配到数据（自动处理后缀）
            self.assertGreater(len(result), 0)
            
        finally:
            os.unlink(temp_file)


class TestFactorParser(unittest.TestCase):
    """测试因子模块的命令行解析"""

    def test_parse_args_default(self):
        """测试默认参数解析"""
        from factor.factor import parse_args
        import sys

        # 保存原始 sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ['factor.py']
            args = parse_args()
            
            self.assertEqual(args.start, '2024-09-25')
            self.assertEqual(args.end, '2025-10-14')
            self.assertEqual(args.stock_pool, '000510.XSHG')
            self.assertEqual(args.quantiles, 10)
            self.assertEqual(args.roll_win, 60)
            
        finally:
            sys.argv = original_argv

    def test_parse_args_custom(self):
        """测试自定义参数解析"""
        from factor.factor import parse_args
        import sys

        original_argv = sys.argv
        try:
            sys.argv = [
                'factor.py',
                '--start', '2024-01-01',
                '--end', '2024-12-31',
                '--stock-pool', 'stock',
                '--factors', 'VOL10', 'RSI_14',
                '--quantiles', '5',
                '--periods', '5', '10', '15',
                '--roll-win', '30'
            ]
            args = parse_args()
            
            self.assertEqual(args.start, '2024-01-01')
            self.assertEqual(args.end, '2024-12-31')
            self.assertEqual(args.stock_pool, 'stock')
            self.assertEqual(args.factors, ['VOL10', 'RSI_14'])
            self.assertEqual(args.quantiles, 5)
            self.assertEqual(args.periods, [5, 10, 15])
            self.assertEqual(args.roll_win, 30)
            
        finally:
            sys.argv = original_argv


class TestCFG(unittest.TestCase):
    """测试配置类"""

    def test_cfg_initialization(self):
        """测试配置初始化"""
        from factor.factor import CFG, parse_args
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['factor.py', '--start', '2024-01-01', '--end', '2024-12-31']
            args = parse_args()
            cfg = CFG(args)
            
            self.assertEqual(cfg.START, '2024-01-01')
            self.assertEqual(cfg.END, '2024-12-31')
            # 兼容字符串或 pandas 日频对象
            try:
                import pandas as pd
                if isinstance(cfg.FREQ, str):
                    self.assertEqual(cfg.FREQ, 'daily')
                else:
                    self.assertTrue(isinstance(cfg.FREQ, pd.offsets.Day))
            except Exception:
                # 最低保障：转成字符串比较
                self.assertIn(str(cfg.FREQ).lower(), ['daily', 'day', 'd'])
            self.assertIsInstance(cfg.CLEAN, dict)
            
        finally:
            sys.argv = original_argv


class TestFactorHelperFunctions(unittest.TestCase):
    """测试辅助函数"""

    def test_rolling_monitor(self):
        """测试滚动监控函数"""
        from factor.factor import rolling_monitor, CFG, parse_args
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['factor.py']
            args = parse_args()
            cfg = CFG(args)
            
            # 创建示例数据
            ic_series = pd.Series([0.1, 0.15, 0.12, 0.18, 0.14])
            tb_ret_series = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
            
            result = rolling_monitor('TEST_FACTOR', ic_series, tb_ret_series, 5, cfg)
            
            self.assertIsNotNone(result)
            self.assertIn('roll_ic', result)
            self.assertIn('roll_ir', result)
            
        finally:
            sys.argv = original_argv


class TestFactorTester(unittest.TestCase):
    """测试 FactorTester 类"""

    def test_factor_tester_initialization(self):
        """测试 FactorTester 初始化"""
        from factor.factor import FactorTester, CFG, parse_args
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['factor.py']
            args = parse_args()
            cfg = CFG(args)
            
            tester = FactorTester(cfg)
            self.assertEqual(tester.cfg, cfg)
            self.assertEqual(tester.stocks, [])
            self.assertEqual(tester.custom_factors, {})
            
        finally:
            sys.argv = original_argv

    def test_factor_tester_with_custom_factors(self):
        """测试带自定义因子的 FactorTester"""
        from factor.factor import FactorTester, CFG, parse_args
        from factor.factor_calculator import create_factor_calculator
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['factor.py']
            args = parse_args()
            cfg = CFG(args)
            
            custom_factors = {
                'TEST_FACTOR': create_factor_calculator(factor_name='VOL10')
            }
            
            tester = FactorTester(cfg, custom_factors=custom_factors)
            self.assertEqual(len(tester.custom_factors), 1)
            self.assertIn('TEST_FACTOR', tester.custom_factors)
            
        finally:
            sys.argv = original_argv


class TestBuiltinFactors(unittest.TestCase):
    """测试内置因子"""

    def test_builtin_factors_list(self):
        """测试内置因子列表"""
        from factor.factor_calculator import BuiltinFactorCalculator

        # 检查内置因子列表
        builtin_factors = BuiltinFactorCalculator.BUILTIN_FACTORS
        
        self.assertIn('VOL10', builtin_factors)
        self.assertIn('VOL20', builtin_factors)
        self.assertIn('VPT_12', builtin_factors)
        self.assertIn('RSI_14', builtin_factors)
        self.assertIn('MA_5', builtin_factors)
        self.assertIn('MA_10', builtin_factors)
        self.assertIn('MA_20', builtin_factors)
        self.assertIn('VOLUME_RATIO', builtin_factors)
        self.assertIn('PRICE_CHANGE', builtin_factors)
        self.assertIn('HIGH_LOW_RATIO', builtin_factors)

    def test_builtin_factor_functions(self):
        """测试内置因子函数"""
        from factor.factor_calculator import BuiltinFactorCalculator

        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        ohlcv = pd.DataFrame({
            'open': np.arange(30),
            'high': np.arange(30) + 1,
            'low': np.arange(30) - 1,
            'close': np.arange(30),
            'volume': np.arange(30) * 1000
        }, index=dates)

        # 测试 VOL10
        vol10_func = BuiltinFactorCalculator.BUILTIN_FACTORS['VOL10']
        result = vol10_func(ohlcv)
        self.assertIsInstance(result, pd.Series)

        # 测试 MA_5
        ma5_func = BuiltinFactorCalculator.BUILTIN_FACTORS['MA_5']
        result = ma5_func(ohlcv)
        self.assertIsInstance(result, pd.Series)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()

