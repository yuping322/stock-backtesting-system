"""Registry for repository-specific custom factors"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable, Dict

CUSTOM_FACTORS: Dict[str, Callable] = {}
"""因子名称 -> 计算函数 (DataFrame 或三参数函数) 的映射"""

_LOADED = False


def load_custom_factors() -> Dict[str, Callable]:
    """Load all custom factor modules under this package."""
    global _LOADED
    if _LOADED:
        return CUSTOM_FACTORS

    package = __name__
    package_path = Path(__file__).parent
    if not package_path.exists():
        return CUSTOM_FACTORS

    for finder, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
        if module_name.startswith('_'):
            continue
        try:
            module = importlib.import_module(f"{package}.{module_name}")
            factor_mapping = getattr(module, 'CUSTOM_FACTORS', None)
            if isinstance(factor_mapping, dict):
                for factor_name, factor_func in factor_mapping.items():
                    if callable(factor_func):
                        CUSTOM_FACTORS[factor_name] = factor_func
        except Exception:
            continue

    _LOADED = True
    return CUSTOM_FACTORS
