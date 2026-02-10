"""
因子生成层 (File 已移除)

包含所有因子生成函数：
- generate_builtin_factors()
- generate_talib_factors()
- generate_oss_factors()
- generate_qlib_158_factors() 等 4 个 Qlib 变体
"""

from .builtin import generate_builtin_factors
from .talib import generate_talib_factors
from .oss import generate_oss_factors
from .qlib import (
    generate_qlib_158_factors,
    generate_qlib_360_factors,
    generate_qlib_158vwap_factors,
    generate_qlib_360vwap_factors
)

__all__ = [
    'generate_builtin_factors',
    'generate_talib_factors',
    'generate_oss_factors',
    'generate_qlib_158_factors',
    'generate_qlib_360_factors',
    'generate_qlib_158vwap_factors',
    'generate_qlib_360vwap_factors',
]
