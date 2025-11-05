"""A minimal fake replacement for akshare to exercise DataProvider behaviors in tests.

This fake module exposes functions used by data_provider with configurable behaviors
via function attributes. Tests can set these attributes to simulate different upstream
responses (raise exceptions, return empty DataFrames, return frames with missing
columns, etc.).
"""
from types import SimpleNamespace
import pandas as pd

def _make_df(cols, rows=None):
    rows = rows or []
    return pd.DataFrame(rows, columns=cols)

# Default behaviors (can be reassigned in tests)
def stock_info_a_code_name():
    # Return a dataframe with 'code' column
    return _make_df(['code','name'], [["600000","PingAn"],["000001","PCIC"]])

def index_stock_cons(symbol: str):
    # Return different frames depending on symbol
    if symbol == "000300":
        return _make_df(['代码','名称'], [["600000","PingAn"],["600519","Kweichow"]])
    if symbol == "000852":
        return _make_df(['代码','名称'], [["000001","PCIC"]])
    return _make_df(['代码'], [])

def stock_board_industry_name_ths():
    # two boards
    return _make_df(['名称'], [["板块A"],["板块B"]])

def stock_board_industry_cons_ths(symbol: str):
    if symbol == "板块A":
        return _make_df(['代码','名称'], [["600000","PingAn"]])
    if symbol == "板块B":
        # intentionally missing '代码' to simulate edge case
        return _make_df(['名称'], [["NoCode"]])
    return _make_df(['代码','名称'], [])

def stock_zh_a_spot_em():
    # Provide a mix: one complete, one missing columns, one ST, one limit-up
    df1 = _make_df(['代码','名称','最新价','总市值','流通市值','成交量','成交额','换手率'],
                   [["600000","PingAn",10.5,1000000000,500000000,10000,2000000,0.5]])
    df2 = _make_df(['代码','名称','最新价'], [["000001","PCIC",1.5]])
    df3 = _make_df(['代码','名称','最新价','总市值','流通市值','成交量','成交额','换手率'],
                   [["600519","*ST Kweichow",1500.0,2000000000,1000000000,5000,7500000,0.3]])  # ST stock
    df4 = _make_df(['代码','名称','最新价','总市值','流通市值','成交量','成交额','换手率'],
                   [["000858","Wuliangye",200.0,3000000000,1500000000,0,0,0.0]])  # limit-up simulation (volume=0, suspended)
    df5 = _make_df(['代码','名称','最新价','总市值','流通市值','成交量','成交额','换手率'],
                   [["000002","Vanke",5.0,1000000000,500000000,1000,5000000,0.5]])  # normal stock
    return pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

def stock_zh_a_spot():
    # fallback with minimal columns
    return _make_df(['代码','名称','最新价'], [["600519","Kweichow",1500.0]])

def stock_zh_a_hist(symbol: str, start_date: str, end_date: str, adjust: str=""):
    # return a one-row historical frame for common codes, empty otherwise
    if symbol.lower().endswith("600000"):
        return _make_df(['日期','开盘','收盘','最高','最低','成交量','成交额'], [[start_date,10.0,10.5,11.0,9.8,1000,10000]])
    if symbol.lower().endswith("000001"):
        # simulate missing some fields
        return _make_df(['日期','开盘','收盘'], [[start_date,1.4,1.5]])
    return _make_df(['日期','开盘','收盘','最高','最低','成交量','成交额'], [])

def stock_zh_index_daily(symbol: str):
    # pretend index has daily bars
    return _make_df(['日期','开盘','收盘','成交量','成交额'], [["20250101",3000,3050,100000,500000]])

# Export as module-level attributes
__all__ = [
    'stock_info_a_code_name', 'index_stock_cons', 'stock_board_industry_name_ths',
    'stock_board_industry_cons_ths', 'stock_zh_a_spot_em', 'stock_zh_a_spot',
    'stock_zh_a_hist', 'stock_zh_index_daily'
]
