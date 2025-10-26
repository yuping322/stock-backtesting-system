import io
import os
import sys
import types
import tempfile
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd


if "akshare" not in sys.modules:
    sys.modules["akshare"] = types.ModuleType("akshare")

if "alphalens" not in sys.modules:
    alphalens_module = types.ModuleType("alphalens")
    sys.modules["alphalens"] = alphalens_module
    utils_module = types.ModuleType("alphalens.utils")
    utils_module.get_clean_factor_and_forward_returns = lambda *a, **k: pd.DataFrame()
    sys.modules["alphalens.utils"] = utils_module
    performance_module = types.ModuleType("alphalens.performance")
    performance_module.mean_information_coefficient = lambda *a, **k: pd.Series(dtype=float)
    performance_module.factor_returns = lambda *a, **k: pd.Series(dtype=float)
    performance_module.mean_return_by_quantile = lambda *a, **k: (pd.DataFrame(), pd.DataFrame())
    sys.modules["alphalens.performance"] = performance_module

if "oss2" not in sys.modules:
    class _DummyAuth:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyBucket:
        def __init__(self, *args, **kwargs):
            pass

    def _dummy_iterator(*args, **kwargs):
        return iter(())

    exceptions_module = types.ModuleType("oss2.exceptions")
    exceptions_module.NoSuchKey = KeyError
    oss2_module = types.ModuleType("oss2")
    oss2_module.Auth = _DummyAuth
    oss2_module.Bucket = _DummyBucket
    oss2_module.ObjectIterator = _dummy_iterator
    oss2_module.exceptions = exceptions_module
    sys.modules["oss2"] = oss2_module
    sys.modules["oss2.exceptions"] = exceptions_module

if "chinese_calendar" not in sys.modules:
    calendar_module = types.ModuleType("chinese_calendar")
    calendar_module.is_workday = lambda d: d.weekday() < 5
    sys.modules["chinese_calendar"] = calendar_module

import data


class BucketStub:
    def __init__(self, file_contents=None, object_contents=None):
        self.file_contents = file_contents or {}
        self.object_contents = object_contents or {}

    def get_object_to_file(self, key, path):
        if key not in self.file_contents:
            raise KeyError(f"missing file content for {key}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.file_contents[key])

    def get_object(self, key):
        if key not in self.object_contents:
            raise KeyError(f"missing object content for {key}")
        return io.BytesIO(self.object_contents[key].encode("utf-8"))

    def object_exists(self, key):
        return key in self.object_contents

    def put_object(self, key, buf):
        if hasattr(buf, "read"):
            content = buf.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
        elif isinstance(buf, bytes):
            content = buf.decode("utf-8")
        else:
            content = str(buf)
        self.object_contents[key] = content


class TestDataModule(unittest.TestCase):
    def test_normalize_codes(self):
        codes = ["000001", "600000", "300750.XSHE", "sh000002"]
        normalized = data._normalize_codes(codes)
        self.assertEqual(normalized[0], "000001.XSHE")
        self.assertEqual(normalized[1], "600000.XSHG")
        self.assertEqual(normalized[2], "300750.XSHE")
        self.assertEqual(normalized[3], "sh000002")

    def test_add_prefix(self):
        self.assertEqual(data._add_prefix("000001"), "sz000001")
        self.assertEqual(data._add_prefix("600000"), "sh600000")
        self.assertEqual(data._add_prefix("430001"), "bj430001")

    def test_parse_date(self):
        ts = data._parse_date("2020-01-02")
        self.assertEqual(ts, pd.Timestamp("2020-01-02"))
        date_obj = date(2020, 1, 3)
        self.assertEqual(data._parse_date(date_obj), pd.Timestamp("2020-01-03"))
        dt_obj = datetime(2020, 1, 4, 10, 0)
        self.assertEqual(data._parse_date(dt_obj), pd.Timestamp("2020-01-04 10:00"))
        with self.assertRaises(TypeError):
            data._parse_date(123)

    def test_wide_to_ohlcv(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "代码": ["000001", "000002"],
            "今开": [10, 20],
            "最高": [11, 21],
            "最低": [9, 19],
            "最新价": [10.5, 20.5],
            "成交量": [1000, 2000],
        })
        ohlcv = data._wide_to_ohlcv(df)
        self.assertEqual(sorted(ohlcv.columns), ["asset", "close", "date", "high", "low", "open", "volume"])
        self.assertAlmostEqual(ohlcv["close"].iloc[0], 10.5)

    def test_load_new_stocks(self):
        sample_csv = "代码,今开\n000001,10\n000002,20\n"
        bucket_stub = BucketStub(file_contents={"key1": sample_csv})
        mapping = {date(2020, 1, 2): "key1"}
        with patch.object(data, "bucket", bucket_stub), \
             patch.object(data, "_collect_files", return_value=mapping):
            df = data.load_new_stocks(["000001"], start="2020-01-01", end="2020-01-05")
        self.assertEqual(df.loc[pd.Timestamp("2020-01-02"), "000001"], 10)

    def test_load_oss_stocks(self):
        csv_text = "日期,close\n2020-01-01,10\n2020-01-02,11\n"
        bucket_stub = BucketStub(object_contents={
            "hangqing/daily_data/sz000001.csv": csv_text
        })
        with patch.object(data, "bucket", bucket_stub):
            prices = data.load_oss_stocks(["000001"], start="2020-01-01", end="2020-01-03")
        self.assertEqual(prices.loc[pd.Timestamp("2020-01-01"), "000001"], 10)
        self.assertEqual(prices.loc[pd.Timestamp("2020-01-02"), "000001"], 11)

    def test_load_modelscope_stocks(self):
        csv_text = "date,close\n2020-01-01,10\n2020-01-02,11\n"
        response = SimpleNamespace(content=csv_text.encode("utf-8"), raise_for_status=lambda: None)
        with patch("data.requests.get", return_value=response):
            prices = data.load_modelscope_stocks(["000001"], start="2020-01-01", end="2020-01-03")
        self.assertIn("000001", prices.columns)
        self.assertEqual(prices.loc[pd.Timestamp("2020-01-02"), "000001"], 11)

    def test_load_modelscope_complex_stocks(self):
        csv_text = "date,open,high,low,close\n2020-01-01,9,11,8,10\n2020-01-02,10,12,9,11\n"
        response = SimpleNamespace(content=csv_text.encode("utf-8"), raise_for_status=lambda: None)
        with patch("data.requests.get", return_value=response):
            close_df = data.load_modelscope_complex_stocks(["000001"], start="2020-01-01", end="2020-01-02")
            all_fields = data.load_modelscope_complex_stocks(["000001"], fields="all")
        self.assertEqual(close_df.loc[pd.Timestamp("2020-01-01"), "000001"], 10)
        self.assertIn("open", all_fields)
        expected_open = pd.DataFrame({"000001": [9, 10]}, index=close_df.index)
        self.assertTrue(all_fields["open"].equals(expected_open))

    def test_get_fin_df(self):
        csv_text = "报告日,类型,资产总计\n2020-03-31,合并期末,100\n2020-06-30,合并期末,110\n"
        bucket_stub = BucketStub(object_contents={
            "jukuan/stock_financial_report_sina/sz000001.csv": csv_text
        })
        with patch.object(data, "bucket", bucket_stub):
            df = data._get_fin_df("000001", "2020-04-01", "合并期末", table="balance")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "资产总计"], "100")

    def test_read_factor_data(self):
        csv1 = "code,f1,f2\n000001.XSHE,1,10\n000002.XSHE,2,20\n"
        csv2 = "code,f1,f2\n000001.XSHE,3,30\n000002.XSHE,4,40\n"
        bucket_stub = BucketStub(object_contents={
            "uploads/2020/factors_20200101_all.csv": csv1,
            "uploads/2020/factors_20200102_all.csv": csv2,
        })
        with patch.object(data, "bucket", bucket_stub):
            df = data.read_factor_data(["000001"], "2020-01-01", "2020-01-02", factors=["f1"])
        self.assertEqual(df.shape, (2, 1))
        self.assertEqual(df.iloc[0, 0], 1)
        self.assertTrue((df.index.get_level_values("code") == "000001.XSHE").all())

    def test_read_factor_data_loal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            year_dir = os.path.join(tmpdir, "2020")
            os.makedirs(year_dir)
            file_path = os.path.join(year_dir, "factors_20200101_all.csv")
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write("code,f1\n000001.XSHE,1\n")
            df = data.read_factor_data_loal(["000001"], "2020-01-01", "2020-01-01", factors=["f1"], base_path=tmpdir)
        self.assertEqual(df.iloc[0, 0], 1)

    def test_factor_for_al(self):
        index = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")], ["000001.XSHE"]], names=["date", "code"])
        df = pd.DataFrame({"score": [1.5]}, index=index)
        with patch("data.read_factor_data", return_value=df):
            series = data.factor_for_al(["000001"], "2020-01-01", "2020-01-01", "score")
        self.assertEqual(series.index.names, ["date", "asset"])
        self.assertEqual(series.index[0][1], "000001")

    def test_get_index_stocks(self):
        mock_df = pd.DataFrame({
            "code": ["000001", "000002"],
            "in_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
        })
        with patch("data._load_index_df", return_value=mock_df):
            codes = data.get_index_stocks("000300", "2020-01-02")
        self.assertEqual(codes, ["000001"])

    def test_load_index_df(self):
        csv_text = "品种代码,指数纳入日期\n000001,2020-01-01\n"
        bucket_stub = BucketStub(object_contents={
            "index/000300_sample.csv": csv_text
        })
        iterator = iter([SimpleNamespace(key="index/000300_sample.csv")])
        with patch.object(data, "bucket", bucket_stub), \
             patch("data.oss2.ObjectIterator", return_value=iterator):
            df = data._load_index_df("000300")
        self.assertEqual(df.loc[0, "code"], "000001")

    def test_financial_statements(self):
        balance_csv = "报告日,类型,资产总计\n2020-03-31,合并期末,100\n2020-06-30,合并期末,110\n"
        income_csv = "报告日,类型,净利润\n2020-03-31,合并期末,10\n2020-06-30,合并期末,11\n"
        cashflow_csv = "报告日,类型,经营活动产生的现金流量净额\n2020-03-31,合并期末,5\n2020-06-30,合并期末,6\n"
        bucket_stub = BucketStub(object_contents={
            "jukuan/stock_financial_report_sina/sz000001.csv": balance_csv,
            "jukuan/stock_financial_report_sina_lirun/sz000001.csv": income_csv,
            "jukuan/stock_financial_report_sina_xianjinliu/sz000001.csv": cashflow_csv,
        })
        with patch.object(data, "bucket", bucket_stub):
            balance = data.get_balance("000001", date="2020-06-30")
            income = data.get_income("000001", date="2020-06-30")
            cashflow = data.get_cashflow("000001", date="2020-06-30")
        self.assertEqual(balance.loc[0, "资产总计"], "110")
        self.assertEqual(income.loc[0, "净利润"], "11")
        self.assertEqual(cashflow.loc[0, "经营活动产生的现金流量净额"], "6")

    def test_get_valuation(self):
        csv_text = "日期,close\n2020-01-01,10\n2020-01-02,11\n"
        bucket_stub = BucketStub(object_contents={
            "hangqing/daily_data/sz000001.csv": csv_text
        })
        with patch.object(data, "bucket", bucket_stub):
            df = data.get_valuation("000001", date="2020-01-02")
        self.assertEqual(df.loc[0, "日期"], "2020-01-02")

    def test_save_result(self):
        bucket_stub = BucketStub(object_contents={})
        with patch.object(data, "bucket", bucket_stub):
            data.save_result(bucket_stub, "2020-01-02", {
                "factor_name": "test",
                "IC_mean": 0.1,
            })
        self.assertIn("daily_metrics/daily_metrics.csv", bucket_stub.object_contents)
        saved = pd.read_csv(io.StringIO(bucket_stub.object_contents["daily_metrics/daily_metrics.csv"]))
        self.assertIn("test", saved["factor_name"].values)

    def test_save_result_append(self):
        existing = "trade_date,factor_name,IC_mean\n2020-01-01,test,0.1\n"
        bucket_stub = BucketStub(object_contents={"daily_metrics/daily_metrics.csv": existing})
        with patch.object(data, "bucket", bucket_stub):
            data.save_result(bucket_stub, "2020-01-02", {
                "factor_name": "test",
                "IC_mean": 0.2,
            })
        saved = pd.read_csv(io.StringIO(bucket_stub.object_contents["daily_metrics/daily_metrics.csv"]))
        self.assertEqual(saved.iloc[-1]["IC_mean"], 0.2)

    def test_get_history_fundamentals(self):
        balance_df = pd.DataFrame({
            "报告日": pd.to_datetime(["2020-03-31"]),
            "资产总计": [100],
        })
        income_df = pd.DataFrame({
            "报告日": pd.to_datetime(["2020-03-31"]),
            "净利润": [10],
        })
        cashflow_df = pd.DataFrame({
            "报告日": pd.to_datetime(["2020-03-31"]),
            "经营活动产生的现金流量净额": [5],
        })
        with patch("data.get_balance", side_effect=lambda *a, **k: balance_df.copy()), \
             patch("data.get_income", side_effect=lambda *a, **k: income_df.copy()), \
             patch("data.get_cashflow", side_effect=lambda *a, **k: cashflow_df.copy()):
            result = data.get_history_fundamentals(
                security="000001",
                fields=[
                    "balance.total_assets",
                    "income.net_profit",
                    "cashflow.net_cash_operating",
                ],
                stat_date="2020q1",
                count=1,
            )
        self.assertEqual(result.loc[("000001", "2020-03-31"), "balance.total_assets"], 100)

    def test_get_trading_dates(self):
        with patch("data.calendar.is_workday", side_effect=lambda d: d.weekday() < 5):
            days = data.get_trading_dates("2024-05-01", "2024-05-05")
        self.assertEqual(days, [pd.Timestamp("2024-05-01").date(), pd.Timestamp("2024-05-02").date(), pd.Timestamp("2024-05-03").date()])

    def test_load_bt_oss_stocks(self):
        csv_text = "代码,今开,最高,最低,最新价,成交量\n000001,10,11,9,10.5,1000\n"
        bucket_stub = BucketStub(file_contents={"key1": csv_text})
        mapping = {date(2020, 1, 2): "key1"}
        with patch.object(data, "bucket", bucket_stub), \
             patch.object(data, "_collect_files", return_value=mapping):
            df = data.load_bt_oss_stocks(["000001"], start="2020-01-01", end="2020-01-03")
        self.assertIn("最新价", df.columns)
        self.assertEqual(len(df), 1)

    def test_load_bt_stocks(self):
        sample_df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "代码": ["000001", "000001"],
            "今开": [10, 11],
            "最高": [11, 12],
            "最低": [9, 10],
            "最新价": [10.5, 11.5],
            "成交量": [1000, 1200],
        })
        with patch("data.load_bt_oss_stocks", return_value=sample_df):
            feeds = data.load_bt_stocks(["000001"], start="2020-01-01", end="2020-01-02")
        self.assertIn("000001", feeds)
        dataname = feeds["000001"].params.dataname
        self.assertTrue({"open", "high", "low", "close", "volume"}.issubset(dataname.columns))

    def test_get_index_daily(self):
        csv_text = "date,close\n2020-01-01,100\n2020-01-02,102\n"
        bucket_stub = BucketStub(object_contents={
            "stock_zh_index_daily/000300_part.csv": csv_text
        })
        objects = [SimpleNamespace(key="stock_zh_index_daily/000300_part.csv")]
        with patch.object(data, "bucket", bucket_stub), \
             patch("data.oss2.ObjectIterator", return_value=iter(objects)):
            nav = data.get_index_daily("000300", "2020-01-01", "2020-01-02")
        self.assertAlmostEqual(nav.iloc[0], 1.0)
        self.assertAlmostEqual(nav.iloc[-1], 1.02)

    def test_load_bt_pricing(self):
        index = pd.to_datetime(["2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"open": [10, 11], "high": [11, 12], "low": [9, 10], "close": [10.5, 11.5], "volume": [1000, 1200]}, index=index)
        feed = SimpleNamespace(params=SimpleNamespace(dataname=df))
        with patch("data.load_bt_stocks", return_value={"000001": feed}):
            pricing = data.load_bt_pricing(["000001"], start="2020-01-01", end="2020-01-02")
        self.assertAlmostEqual(pricing.loc[pd.Timestamp("2020-01-02"), "000001"], 11.5)

    def test_load_code2name(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as tmp:
            tmp.write("code,name\n1,Alpha\n2,Beta\n")
            temp_path = tmp.name
        try:
            with patch.object(data, "MAPPING_FILE", temp_path):
                mapping = data.load_code2name()
            self.assertEqual(mapping["000001"], "Alpha")
        finally:
            os.remove(temp_path)

    def test_get_default_date(self):
        ctx = SimpleNamespace(current_dt=pd.Timestamp("2020-01-02"))
        data.context = ctx
        self.assertEqual(data._get_default_date(), ctx.current_dt)
        if hasattr(data, "context"):
            delattr(data, "context")
        default = data._get_default_date()
        self.assertIsInstance(default, datetime)

    def test_handler(self):
        multi_index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2020-01-01", "2020-01-02"]), ["000001"]], names=["date", "asset"]
        )
        factor_series = pd.Series([1.0, 1.2], index=multi_index)
        pricing_df = pd.DataFrame({"000001": [10, 10.5]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))

        with patch("data.factor_for_al", return_value=factor_series), \
             patch("data.load_modelscope_stocks", return_value=pricing_df), \
             patch("data.get_clean_factor_and_forward_returns", return_value=pd.DataFrame({"ic": [0.1]})), \
             patch("data.mean_information_coefficient", return_value=pd.Series([0.1])), \
             patch("data.factor_returns", return_value=pd.Series([0.01])), \
             patch("data.mean_return_by_quantile", return_value=(pd.DataFrame({0: [0.01, 0.02]}), None)), \
             patch("data.save_result") as mock_save:
            result = data.handler({"codes": ["000001"], "factor_name": "ic"}, None)

        self.assertIn("IC_mean", result)
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
