#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 TA-Lib 支持的所有因子列表
类似于 alpha158_factors.txt 的格式
"""

import talib
import inspect
import numpy as np
from typing import List, Dict, Any


def get_talib_functions() -> List[str]:
    """获取所有 TA-Lib 指标函数名称，过滤掉不需要的函数"""
    functions = []
    for attr in dir(talib):
        if not attr.startswith('_'):
            obj = getattr(talib, attr)
            if callable(obj):
                # 过滤掉不需要的函数
                if not attr.startswith('stream_') and \
                   not attr.startswith('wrapped_') and \
                   not attr.startswith('chain') and \
                   not attr.startswith('get_') and \
                   not attr.startswith('set_') and \
                   attr not in ['wraps']:
                    functions.append(attr)
    return sorted(functions)


def get_function_signature(func_name: str) -> Dict[str, Any]:
    """获取函数签名信息"""
    func = getattr(talib, func_name)
    try:
        sig = inspect.signature(func)
        params = {}
        for name, param in sig.parameters.items():
            if name != 'price':  # 跳过 price 参数（通常是第一个参数）
                if param.default != inspect.Parameter.empty:
                    params[name] = param.default
                else:
                    params[name] = None
        return params
    except:
        return {}


def generate_common_parameters(func_name: str) -> List[List[int]]:
    """为指标函数生成常见的参数组合"""
    # 常见的周期参数
    common_periods = [5, 10, 14, 20, 21, 26, 30, 50, 60]

    # 特殊指标的参数组合
    special_params = {
        'SMA': [[p] for p in common_periods],
        'EMA': [[p] for p in common_periods],
        'SMA': [[p] for p in common_periods],
        'WMA': [[p] for p in common_periods],
        'DEMA': [[p] for p in common_periods],
        'TEMA': [[p] for p in common_periods],
        'TRIMA': [[p] for p in common_periods],
        'KAMA': [[p] for p in common_periods],
        'MAMA': [[0.5, 0.05]],  # 默认参数
        'T3': [[p, 0.7] for p in common_periods],  # period, vfactor

        # 动量指标
        'RSI': [[p] for p in [6, 14, 21]],
        'STOCHRSI': [[p, 14, 3, 3] for p in [14]],  # timeperiod, fastk_period, fastd_period, fastd_matype
        'MOM': [[p] for p in common_periods],
        'ROC': [[p] for p in common_periods],
        'ROCP': [[p] for p in common_periods],
        'ROCR': [[p] for p in common_periods],
        'ROCR100': [[p] for p in common_periods],
        'TRIX': [[p] for p in common_periods],
        'WILLR': [[p] for p in [14, 21]],
        'CCI': [[p] for p in [14, 20]],
        'CMO': [[p] for p in common_periods],
        'PPO': [[12, 26, 0]],  # fastperiod, slowperiod, matype
        'APO': [[12, 26, 0]],  # fastperiod, slowperiod, matype

        # 波动率指标
        'ATR': [[p] for p in [14, 21]],
        'NATR': [[p] for p in [14, 21]],
        'TRANGE': [[]],  # 无参数
        'ADX': [[p] for p in [14, 21]],
        'ADXR': [[p] for p in [14, 21]],
        'DX': [[p] for p in [14, 21]],
        'PLUS_DI': [[p] for p in [14, 21]],
        'PLUS_DM': [[p] for p in [14, 21]],
        'MINUS_DI': [[p] for p in [14, 21]],
        'MINUS_DM': [[p] for p in [14, 21]],

        # 成交量指标
        'AD': [[]],  # 无参数
        'ADOSC': [[3, 10]],  # fastperiod, slowperiod
        'OBV': [[]],  # 无参数
        'MFI': [[p] for p in [14, 21]],

        # MACD 相关
        'MACD': [[12, 26, 9]],  # fastperiod, slowperiod, signalperiod
        'MACDEXT': [[12, 26, 9, 0, 0, 0]],  # fastperiod, slowperiod, signalperiod, fastmatype, slowmatype, signalmatype
        'MACDFIX': [[9]],  # signalperiod
        'STOCH': [[14, 3, 3]],  # fastk_period, slowk_period, slowd_period
        'STOCHF': [[14, 3]],  # fastk_period, fastd_period

        # 布林带
        'BBANDS': [[p, 2, 2] for p in [5, 10, 20, 21]],  # timeperiod, nbdevup, nbdevdn

        # AROON 指标
        'AROON': [[p] for p in [14, 21]],
        'AROONOSC': [[p] for p in [14, 21]],

        # 其他技术指标
        'HT_DCPERIOD': [[]],  # 无参数
        'HT_DCPHASE': [[]],  # 无参数
        'HT_PHASOR': [[]],  # 无参数
        'HT_SINE': [[]],  # 无参数
        'HT_TRENDMODE': [[]],  # 无参数
        'HT_DCPERIOD': [[]],  # 无参数
        'HT_DCPHASE': [[]],  # 无参数
        'HT_PHASOR': [[]],  # 无参数
        'HT_SINE': [[]],  # 无参数
        'HT_TRENDMODE': [[]],  # 无参数

        # 价格变换
        'AVGPRICE': [[]],  # 无参数
        'MEDPRICE': [[]],  # 无参数
        'TYPPRICE': [[]],  # 无参数
        'WCLPRICE': [[]],  # 无参数

        # 数学函数（跳过）
        'CEIL': [[]],
        'FLOOR': [[]],
        'COS': [[]],
        'SIN': [[]],
        'TAN': [[]],
        'ACOS': [[]],
        'ASIN': [[]],
        'ATAN': [[]],
        'COSH': [[]],
        'SINH': [[]],
        'TANH': [[]],
        'EXP': [[]],
        'LN': [[]],
        'LOG10': [[]],
        'SQRT': [[]],
        'DIV': [[]],
        'ADD': [[]],
        'SUB': [[]],
        'MULT': [[]],
        'MAX': [[]],
        'MIN': [[]],
        'MAXINDEX': [[]],
        'MININDEX': [[]],
        'SUM': [[]],

        # 其他跳过
        'LINEARREG': [[]],
        'LINEARREG_ANGLE': [[]],
        'LINEARREG_INTERCEPT': [[]],
        'LINEARREG_SLOPE': [[]],
        'TSF': [[]],
        'VAR': [[]],
        'STDDEV': [[]],
        'CORREL': [[]],
        'BETA': [[]],
        'COVAR': [[]],
    }

    # 检查是否是特殊指标
    if func_name in special_params:
        return special_params[func_name]

    # 默认参数：尝试单周期参数
    try:
        sig = get_function_signature(func_name)
        if sig:
            # 如果有 timeperiod 参数，使用常见周期
            if 'timeperiod' in sig:
                return [[p] for p in common_periods[:3]]  # 只取前3个避免太多
            # 如果有 period 参数
            elif 'period' in sig:
                return [[p] for p in common_periods[:3]]
            # 如果有 fastperiod 参数（MACD类）
            elif 'fastperiod' in sig:
                return [[12, 26, 9]]
        else:
            # 无参数函数
            return [[]]
    except:
        return [[]]


def generate_talib_factors() -> List[str]:
    """生成所有 TA-Lib 因子名称"""
    functions = get_talib_functions()
    factors = []

    # 完全跳过的函数（数学函数、K线形态等）
    skip_functions = {
        # 数学函数
        'CEIL', 'FLOOR', 'COS', 'SIN', 'TAN', 'ACOS', 'ASIN', 'ATAN',
        'COSH', 'SINH', 'TANH', 'EXP', 'LN', 'LOG10', 'SQRT',
        'DIV', 'ADD', 'SUB', 'MULT', 'MAX', 'MIN', 'MAXINDEX', 'MININDEX', 'SUM',

        # 其他不需要的
        'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
        'TSF', 'VAR', 'STDDEV', 'CORREL', 'BETA', 'COVAR',

        # K线形态识别（太复杂，跳过）
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
        'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
        'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
        'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE',
        'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS',
        'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
        'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
        'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD',
        'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN',
        'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE',
        'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
        'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',

        # 其他复杂或不需要的函数
        'BOP', 'HT_TRENDLINE', 'MAVP', 'SAR', 'SAREXT', 'ULTOSC',
    }

    for func_name in functions:
        if func_name in skip_functions:
            continue

        try:
            param_combinations = generate_common_parameters(func_name)

            for params in param_combinations:
                if params:
                    # 有参数的情况
                    param_str = '_'.join(str(p) for p in params)
                    factor_name = f"TALIB_{func_name}_{param_str}"
                else:
                    # 无参数的情况
                    factor_name = f"TALIB_{func_name}"

                factors.append(factor_name)

        except Exception as e:
            # 静默跳过有问题的函数
            continue

    return sorted(list(set(factors)))  # 去重并排序


def save_talib_factors(filename: str = 'talib_factors.txt'):
    """保存 TA-Lib 因子列表到文件"""
    factors = generate_talib_factors()

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# TA-Lib 技术指标因子列表\n")
        f.write(f"# 生成时间: {np.datetime64('now').astype(str)[:19]}\n")
        f.write(f"# 总计因子数量: {len(factors)}\n")
        f.write("# ============================================\n")
        f.write("\n")

        for factor in factors:
            f.write(f"{factor}\n")

    print(f"✅ 已生成 {len(factors)} 个 TA-Lib 因子，保存到 {filename}")
    return factors


def main():
    """主函数"""
    print("🚀 开始生成 TA-Lib 因子列表...")
    factors = save_talib_factors()
    print(f"📊 共生成 {len(factors)} 个因子")
    print("\n前10个因子示例:")
    for factor in factors[:10]:
        print(f"  {factor}")
    print("\n后10个因子示例:")
    for factor in factors[-10:]:
        print(f"  {factor}")


if __name__ == "__main__":
    main()