"""
data.py 接口测试脚本
测试所有公开接口的功能
"""

import sys
import os
import traceback
from datetime import datetime, date, timedelta
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入data模块
try:
    from data import (
        # 工具函数
        _normalize_date_arg,
        _normalize_code_arg,
        _ensure_exchange_prefix,
        _ensure_exchange_suffix,
        # OSS数据
        load_new_stocks,
        load_oss_stocks,
        read_factor_data,
        # 已移除的旧接口（不再导入）
        # 因子分析
        factor_for_al,
        # 财务报表
        get_balance,
        get_income,
        get_cashflow,
        get_valuation,
        get_history_fundamentals,
        # 指数
        get_index_stocks,
        get_index_daily,
        # Backtrader
        load_bt_stocks,
        load_bt_pricing,
        # 交易日历
        get_trading_dates,
        # 映射
        code2name,
        # 数据结构
        DateRange,
    )
    print("✅ 成功导入data模块")
except Exception as e:
    print(f"❌ 导入data模块失败: {e}")
    sys.exit(1)


class TestResult:
    """测试结果记录"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
    
    def add_pass(self, name: str):
        self.passed.append(name)
        print(f"✅ {name}")
    
    def add_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"❌ {name}: {error}")
    
    def add_skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        print(f"⏭️  {name}: {reason}")
    
    def summary(self):
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"✅ 通过: {len(self.passed)}")
        print(f"❌ 失败: {len(self.failed)}")
        print(f"⏭️  跳过: {len(self.skipped)}")
        
        if self.failed:
            print("\n失败的测试:")
            for name, error in self.failed:
                print(f"  - {name}: {error}")
        
        if self.skipped:
            print("\n跳过的测试:")
            for name, reason in self.skipped:
                print(f"  - {name}: {reason}")


# 全局测试结果
test_result = TestResult()


def test_normalize_date_arg():
    """测试日期规范化"""
    try:
        # 测试字符串
        result = _normalize_date_arg("2024-01-01")
        assert isinstance(result, pd.Timestamp)
        
        # 测试date对象
        result = _normalize_date_arg(date(2024, 1, 1))
        assert isinstance(result, pd.Timestamp)
        
        # 测试None
        result = _normalize_date_arg(None, default="2024-01-01")
        assert isinstance(result, pd.Timestamp)
        
        test_result.add_pass("test_normalize_date_arg")
    except Exception as e:
        test_result.add_fail("test_normalize_date_arg", str(e))


def test_normalize_code_arg():
    """测试股票代码规范化"""
    try:
        # 测试单个代码
        result = _normalize_code_arg("000001")
        assert result == ["000001"]
        
        # 测试列表
        result = _normalize_code_arg(["000001", "600000"])
        assert "000001" in result and "600000" in result
        
        # 测试带前缀
        result = _normalize_code_arg("sh600000")
        assert result == ["600000"]
        
        # 测试None
        result = _normalize_code_arg(None)
        assert result is None
        
        test_result.add_pass("test_normalize_code_arg")
    except Exception as e:
        test_result.add_fail("test_normalize_code_arg", str(e))


def test_ensure_exchange_prefix():
    """测试交易所前缀"""
    try:
        # 测试上证
        result = _ensure_exchange_prefix("600000")
        assert result == "sh600000"
        
        # 测试深证
        result = _ensure_exchange_prefix("000001")
        assert result == "sz000001"
        
        # 测试创业板
        result = _ensure_exchange_prefix("300001")
        assert result == "sz300001"
        
        test_result.add_pass("test_ensure_exchange_prefix")
    except Exception as e:
        test_result.add_fail("test_ensure_exchange_prefix", str(e))


def test_ensure_exchange_suffix():
    """测试交易所后缀"""
    try:
        # 测试上证
        result = _ensure_exchange_suffix("600000")
        assert result == "600000.XSHG"
        
        # 测试深证
        result = _ensure_exchange_suffix("000001")
        assert result == "000001.XSHE"
        
        test_result.add_pass("test_ensure_exchange_suffix")
    except Exception as e:
        test_result.add_fail("test_ensure_exchange_suffix", str(e))


def test_code2name():
    """测试代码到名称映射"""
    try:
        assert isinstance(code2name, dict)
        assert len(code2name) > 0
        
        # 测试获取名称
        if "000001" in code2name:
            name = code2name["000001"]
            assert isinstance(name, str)
        
        test_result.add_pass("test_code2name")
    except Exception as e:
        test_result.add_fail("test_code2name", str(e))


def test_get_trading_dates():
    """测试交易日历"""
    try:
        start = "2024-01-01"
        end = "2024-01-05"
        
        # 测试返回date对象
        dates = get_trading_dates(start, end, as_str=False)
        assert isinstance(dates, list)
        assert len(dates) > 0
        
        # 测试返回字符串
        dates_str = get_trading_dates(start, end, as_str=True)
        assert isinstance(dates_str, list)
        assert all(isinstance(d, str) for d in dates_str)
        
        test_result.add_pass("test_get_trading_dates")
    except Exception as e:
        test_result.add_fail("test_get_trading_dates", str(e))


def test_date_range():
    """测试DateRange类"""
    try:
        date_range = DateRange(
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31")
        )
        
        # 创建测试数据
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", "2024-01-31", freq="D"),
            "value": range(31)
        })
        
        # 测试apply方法
        filtered = date_range.apply(df)
        assert len(filtered) <= len(df)
        
        test_result.add_pass("test_date_range")
    except Exception as e:
        test_result.add_fail("test_date_range", str(e))


def test_load_prices_with_oss():
    """使用 OSS 接口加载行情"""
    try:
        codes = ["000001"]
        start = "2024-01-01"
        end = "2024-01-10"
        df = load_oss_stocks(codes, start=start, end=end)
        assert isinstance(df, pd.DataFrame)
        test_result.add_pass("test_load_prices_with_oss")
    except Exception as e:
        test_result.add_fail("test_load_prices_with_oss", str(e))


def test_get_index_stocks():
    """测试获取指数成分股"""
    try:
        # 测试沪深300
        stocks = get_index_stocks("000300")
        
        assert isinstance(stocks, list)
        assert len(stocks) > 0
        
        # 测试带日期
        stocks = get_index_stocks("000300", "2024-01-01")
        assert isinstance(stocks, list)
        
        test_result.add_pass("test_get_index_stocks")
    except Exception as e:
        test_result.add_fail("test_get_index_stocks", str(e))


def test_get_index_daily():
    """测试获取指数日线"""
    try:
        start = "2024-01-01"
        end = "2024-01-10"
        
        nav = get_index_daily("000300", start, end)
        
        assert isinstance(nav, pd.Series)
        assert isinstance(nav.index, pd.DatetimeIndex)
        assert len(nav) > 0
        
        test_result.add_pass("test_get_index_daily")
    except Exception as e:
        test_result.add_fail("test_get_index_daily", str(e))


def test_get_valuation():
    """测试获取估值数据"""
    try:
        code = "000001"
        df = get_valuation(code)
        
        assert isinstance(df, pd.DataFrame)
        
        test_result.add_pass("test_get_valuation")
    except Exception as e:
        test_result.add_fail("test_get_valuation", str(e))


def test_read_factor_data():
    """测试读取因子数据"""
    try:
        # 注意：需要OSS中有实际数据，否则会跳过
        test_result.add_skip("test_read_factor_data", "需要OSS实际数据")
    except Exception as e:
        test_result.add_fail("test_read_factor_data", str(e))


def test_factor_for_al():
    """测试Alphalens因子格式"""
    try:
        # 注意：需要OSS中有实际数据
        test_result.add_skip("test_factor_for_al", "需要OSS实际数据")
    except Exception as e:
        test_result.add_fail("test_factor_for_al", str(e))


def test_get_balance():
    """测试获取资产负债表"""
    try:
        code = "000001"
        df = get_balance(code)
        
        assert isinstance(df, pd.DataFrame)
        
        test_result.add_pass("test_get_balance")
    except Exception as e:
        test_result.add_fail("test_get_balance", str(e))


def test_get_income():
    """测试获取利润表"""
    try:
        code = "000001"
        df = get_income(code)
        
        assert isinstance(df, pd.DataFrame)
        
        test_result.add_pass("test_get_income")
    except Exception as e:
        test_result.add_fail("test_get_income", str(e))


def test_get_cashflow():
    """测试获取现金流量表"""
    try:
        code = "000001"
        df = get_cashflow(code)
        
        assert isinstance(df, pd.DataFrame)
        
        test_result.add_pass("test_get_cashflow")
    except Exception as e:
        test_result.add_fail("test_get_cashflow", str(e))


def test_get_history_fundamentals():
    """测试批量获取财务数据"""
    try:
        codes = ["000001"]
        fields = ["balance.total_assets", "income.net_profit"]
        
        df = get_history_fundamentals(
            security=codes,
            fields=fields,
            count=1
        )
        
        assert isinstance(df, pd.DataFrame)
        
        test_result.add_pass("test_get_history_fundamentals")
    except Exception as e:
        test_result.add_fail("test_get_history_fundamentals", str(e))


def test_load_bt_stocks():
    """测试Backtrader数据加载"""
    try:
        codes = ["000001"]
        start = "2024-01-01"
        end = "2024-01-10"
        
        feeds = load_bt_stocks(codes=codes, start=start, end=end)
        
        assert isinstance(feeds, dict)
        
        test_result.add_pass("test_load_bt_stocks")
    except Exception as e:
        test_result.add_fail("test_load_bt_stocks", str(e))


def test_load_bt_pricing():
    """测试Alphalens价格数据"""
    try:
        codes = ["000001"]
        start = "2024-01-01"
        end = "2024-01-10"
        
        pricing = load_bt_pricing(codes=codes, start=start, end=end)
        
        assert isinstance(pricing, pd.DataFrame)
        
        test_result.add_pass("test_load_bt_pricing")
    except Exception as e:
        test_result.add_fail("test_load_bt_pricing", str(e))


def test_load_oss_stocks():
    """测试OSS日线数据加载"""
    try:
        # 需要OSS实际数据
        test_result.add_skip("test_load_oss_stocks", "需要OSS实际数据")
    except Exception as e:
        test_result.add_fail("test_load_oss_stocks", str(e))


def test_load_new_stocks():
    """测试OSS快照数据加载"""
    try:
        # 需要OSS实际数据
        test_result.add_skip("test_load_new_stocks", "需要OSS实际数据")
    except Exception as e:
        test_result.add_fail("test_load_new_stocks", str(e))


def main():
    """运行所有测试"""
    print("="*60)
    print("data.py 接口测试")
    print("="*60)
    print()
    
    # 运行所有测试
    tests = [
        test_normalize_date_arg,
        test_normalize_code_arg,
        test_ensure_exchange_prefix,
        test_ensure_exchange_suffix,
        test_code2name,
        test_get_trading_dates,
        test_date_range,
        test_load_modelscope_stocks,
        test_get_index_stocks,
        test_get_index_daily,
        test_get_valuation,
        test_get_balance,
        test_get_income,
        test_get_cashflow,
        test_get_history_fundamentals,
        test_load_bt_stocks,
        test_load_bt_pricing,
        test_read_factor_data,
        test_factor_for_al,
        test_load_oss_stocks,
        test_load_new_stocks,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 执行异常: {e}")
            traceback.print_exc()
    
    # 输出总结
    test_result.summary()


if __name__ == "__main__":
    main()

