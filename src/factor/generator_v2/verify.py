"""
Generator V2 验证脚本

检查所有模块是否能正确导入和使用
"""

import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_imports():
    """验证所有模块导入"""
    logger.info("="*60)
    logger.info("验证 Generator V2 模块导入")
    logger.info("="*60)
    
    tests = []
    
    # 测试 1: 异常模块
    try:
        from src.factor.generator_v2.exceptions import (
            FactorGenerationException,
            DataNotAvailableError,
            FactorCalculationError,
            FactorValidationError,
            PartialResultError,
        )
        logger.info("✅ 异常模块导入成功 (5 个异常类)")
        tests.append(('exceptions', True, None))
    except Exception as e:
        logger.error(f"❌ 异常模块导入失败: {e}")
        tests.append(('exceptions', False, str(e)))
    
    # 测试 2: 计算器模块
    try:
        from src.factor.generator_v2.calculator import (
            FactorCalculator,
            BuiltinFactorCalculator,
            TalibFactorCalculator,
            CustomFunctionCalculator,
            FileFactorCalculator,
            create_factor_calculator,
        )
        logger.info("✅ 计算器模块导入成功 (5 个计算器 + 1 个工厂函数)")
        tests.append(('calculator', True, None))
    except Exception as e:
        logger.error(f"❌ 计算器模块导入失败: {e}")
        tests.append(('calculator', False, str(e)))
    
    # 测试 3: 生成器模块
    try:
        from src.factor.generator_v2.generator import (
            FactorGenerator,
            BuiltinFactorGenerator,
        )
        logger.info("✅ 生成器模块导入成功 (2 个生成器)")
        tests.append(('generator', True, None))
    except Exception as e:
        logger.error(f"❌ 生成器模块导入失败: {e}")
        tests.append(('generator', False, str(e)))
    
    # 测试 4: 质量检查模块
    try:
        from src.factor.generator_v2.quality import DataQualityChecker
        logger.info("✅ 质量检查模块导入成功 (1 个检查器)")
        tests.append(('quality', True, None))
    except Exception as e:
        logger.error(f"❌ 质量检查模块导入失败: {e}")
        tests.append(('quality', False, str(e)))
    
    # 测试 5: 工具模块
    try:
        from src.factor.generator_v2.utils import (
            DataLoader,
            DataProcessor,
            ConfigManager,
            ProgressTracker,
        )
        logger.info("✅ 工具模块导入成功 (4 个工具类)")
        tests.append(('utils', True, None))
    except Exception as e:
        logger.error(f"❌ 工具模块导入失败: {e}")
        tests.append(('utils', False, str(e)))
    
    # 测试 6: 包级导入
    try:
        from src.factor.generator_v2 import (
            FactorGenerationException,
            DataNotAvailableError,
            FactorCalculationError,
            FactorValidationError,
            PartialResultError,
            FactorCalculator,
            BuiltinFactorCalculator,
            TalibFactorCalculator,
            CustomFunctionCalculator,
            FileFactorCalculator,
            create_factor_calculator,
            DataQualityChecker,
            FactorGenerator,
            BuiltinFactorGenerator,
        )
        logger.info("✅ 包级导入成功 (13 个公共 API)")
        tests.append(('package', True, None))
    except Exception as e:
        logger.error(f"❌ 包级导入失败: {e}")
        tests.append(('package', False, str(e)))
    
    # 测试 7: 版本检查
    try:
        from src.factor.generator_v2 import __version__
        logger.info(f"✅ 版本信息: {__version__}")
        tests.append(('version', True, None))
    except:
        logger.warning("⚠️  版本信息不可用（非严重错误）")
        tests.append(('version', False, 'not available'))
    
    return tests

def verify_functionality():
    """验证基本功能"""
    logger.info("\n" + "="*60)
    logger.info("验证基本功能")
    logger.info("="*60)
    
    tests = []
    
    # 测试 1: 创建计算器
    try:
        from src.factor.generator_v2 import create_factor_calculator
        
        calc = create_factor_calculator('VOL10')
        assert calc is not None
        logger.info("✅ 计算器创建成功")
        tests.append(('create_calculator', True, None))
    except Exception as e:
        logger.error(f"❌ 计算器创建失败: {e}")
        tests.append(('create_calculator', False, str(e)))
    
    # 测试 2: 创建生成器
    try:
        from src.factor.generator_v2 import BuiltinFactorGenerator
        
        gen = BuiltinFactorGenerator(
            stock_codes=['000001'],
            start_date='2024-01-01',
            end_date='2024-01-31',
            factor_names=['VOL10']
        )
        assert gen is not None
        logger.info("✅ 生成器创建成功")
        tests.append(('create_generator', True, None))
    except Exception as e:
        logger.error(f"❌ 生成器创建失败: {e}")
        tests.append(('create_generator', False, str(e)))
    
    # 测试 3: 异常抛出
    try:
        from src.factor.generator_v2 import FactorValidationError
        
        try:
            raise FactorValidationError('TEST_FACTOR', 'test issue')
        except FactorValidationError as e:
            assert e.factor_name == 'TEST_FACTOR'
            assert 'test issue' in str(e)
            logger.info("✅ 异常处理成功")
            tests.append(('exception_handling', True, None))
    except Exception as e:
        logger.error(f"❌ 异常处理失败: {e}")
        tests.append(('exception_handling', False, str(e)))
    
    return tests

def print_summary(import_tests, func_tests):
    """打印总结"""
    logger.info("\n" + "="*60)
    logger.info("验证总结")
    logger.info("="*60)
    
    import_passed = sum(1 for _, passed, _ in import_tests if passed)
    func_passed = sum(1 for _, passed, _ in func_tests if passed)
    
    logger.info(f"\n导入测试: {import_passed}/{len(import_tests)} 通过")
    for name, passed, error in import_tests:
        status = "✅" if passed else "❌"
        msg = f"{status} {name}"
        if error:
            msg += f" ({error})"
        logger.info(f"  {msg}")
    
    logger.info(f"\n功能测试: {func_passed}/{len(func_tests)} 通过")
    for name, passed, error in func_tests:
        status = "✅" if passed else "❌"
        msg = f"{status} {name}"
        if error:
            msg += f" ({error})"
        logger.info(f"  {msg}")
    
    total_passed = import_passed + func_passed
    total_tests = len(import_tests) + len(func_tests)
    
    logger.info(f"\n总计: {total_passed}/{total_tests} 通过")
    
    if total_passed == total_tests:
        logger.info("\n🎉 所有验证通过！Generator V2 已准备就绪！")
        return 0
    else:
        logger.warning(f"\n⚠️  {total_tests - total_passed} 个验证失败")
        return 1

def main():
    """主函数"""
    logger.info("Generator V2 验证脚本")
    logger.info("Python 版本: %s" % sys.version.split()[0])
    
    # 运行验证
    import_tests = verify_imports()
    func_tests = verify_functionality()
    
    # 打印总结
    exit_code = print_summary(import_tests, func_tests)
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
