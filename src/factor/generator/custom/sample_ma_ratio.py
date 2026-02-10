"""示例自定义因子模块

将此文件放到 src/factor/generator/custom/ 目录，下次运行时会自动加载。
因子函数可以直接调用 OHLCV DataFrame，也可以实现完全自定义的 (code, start, end) 策略。
"""

import pandas as pd


def ma_ratio(ohlcv: pd.DataFrame) -> pd.Series:
    """20 日均线比率因子"""
    close = ohlcv['close']
    ma20 = close.rolling(window=20).mean()
    return close / ma20


CUSTOM_FACTORS = {
    'CUSTOM_MA_RATIO': ma_ratio,
}
