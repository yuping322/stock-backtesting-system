"""
因子生成模块 V2 - 改进版本

该模块提供了一个更清晰、更可维护的因子生成框架。

核心特性:
- 统一的计算器接口
- 体系化的异常处理
- 数据质量检查
- 清晰的职责分工

导出的类:
    - FactorCalculator: 因子计算器基类
    - BuiltinFactorCalculator: 内置因子计算器
    - TalibFactorCalculator: Talib 因子计算器
    - CustomFunctionCalculator: 自定义因子计算器
    - FileFactorCalculator: 文件加载因子计算器
    - FactorGenerator: 因子生成器基类
    - BuiltinFactorGenerator: 内置因子生成器
    - create_factor_calculator: 计算器工厂函数
    - DataQualityChecker: 数据质量检查器

使用示例:
    from src.factor.generator_v2 import create_factor_calculator, BuiltinFactorGenerator
    
    # 方式 1: 使用计算器
    calculator = create_factor_calculator('VOL10')
    result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
    
    # 方式 2: 使用生成器
    generator = BuiltinFactorGenerator(
        stock_codes=['000001', '000002'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        factor_names=['VOL10', 'RSI_14']
    )
    df = generator.generate()
"""

from .exceptions import (
    FactorGenerationException,
    DataNotAvailableError,
    FactorCalculationError,
    FactorValidationError,
    PartialResultError,
)

from .calculator import (
    FactorCalculator,
    BuiltinFactorCalculator,
    TalibFactorCalculator,
    CustomFunctionCalculator,
    FileFactorCalculator,
    create_factor_calculator,
)

from .generator import (
    FactorGenerator,
    BuiltinFactorGenerator,
)

from .quality import DataQualityChecker

__version__ = '2.0.0'

__all__ = [
    # 异常类
    'FactorGenerationException',
    'DataNotAvailableError',
    'FactorCalculationError',
    'FactorValidationError',
    'PartialResultError',
    # 计算器
    'FactorCalculator',
    'BuiltinFactorCalculator',
    'TalibFactorCalculator',
    'CustomFunctionCalculator',
    'FileFactorCalculator',
    'create_factor_calculator',
    # 生成器
    'FactorGenerator',
    'BuiltinFactorGenerator',
    # 质量检查
    'DataQualityChecker',
]
