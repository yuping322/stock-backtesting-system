"""
因子生成器 - V2 版本中的编排层

该模块提供了一个改进的因子生成器框架，采用清晰的职责分工:
- 生成器只负责编排和控制流程
- 计算器只负责计算因子值
- 质量检查器只负责验证数据

支持的生成器:
    - BuiltinFactorGenerator: 生成内置因子
    - QlibFactorGenerator: 生成 Qlib 因子 (待实现)
    - TalibFactorGenerator: 生成 Talib 因子 (待实现)
    - OSSFactorGenerator: 加载 OSS 因子 (待实现)
"""

import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime

from .calculator import create_factor_calculator, FactorCalculator
from .quality import DataQualityChecker
from .exceptions import (
    FactorGenerationException,
    FactorCalculationError,
    PartialResultError,
    FactorValidationError,
)

logger = logging.getLogger(__name__)


class FactorGenerator(ABC):
    """
    因子生成器基类
    
    所有具体的因子生成器都应继承此类。
    
    职责:
        - 参数验证
        - 流程编排
        - 错误处理
        - 结果验证
        - 报告生成
    
    子类应只需实现 _compute_all_factors() 方法。
    """
    
    def __init__(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = './data/factor_tasks'
    ):
        """
        初始化因子生成器
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            output_dir: 输出目录
        """
        self.stock_codes = [str(code).zfill(6) for code in stock_codes]
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 元数据
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.task_dir = None
        self.factor_names = []
        
        # 统计信息
        self.stats = {
            'total_stocks': len(self.stock_codes),
            'successful_factors': 0,
            'failed_factors': 0,
            'data_points': 0,
            'failures': {}
        }
    
    def setup_task(self) -> str:
        """
        设置任务目录
        
        Returns:
            str: 任务目录路径
        """
        self.task_dir = self.output_dir / self.timestamp
        self.task_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ 创建任务目录: {self.task_dir}")
        logger.info(f"  时间戳: {self.timestamp}")
        logger.info(f"  股票数: {len(self.stock_codes)}")
        
        return str(self.task_dir)
    
    @abstractmethod
    def _compute_all_factors(self) -> Dict[str, pd.Series]:
        """
        计算所有因子
        
        由子类实现。应该返回一个字典:
        {
            'factor_name': pd.Series(因子值数据)
        }
        
        Returns:
            Dict[str, pd.Series]: 因子数据字典
        """
        pass
    
    def generate(self) -> pd.DataFrame:
        """
        生成因子数据的主流程
        
        流程:
        1. 准备: setup_task()，验证参数
        2. 计算: _compute_all_factors()
        3. 验证: 数据质量检查
        4. 报告: 生成统计报告
        
        Returns:
            pd.DataFrame: 因子数据
        
        Raises:
            PartialResultError: 部分因子计算失败
            FactorGenerationException: 生成失败
        """
        try:
            # 1. 准备
            logger.info("\n" + "="*60)
            logger.info(f"开始生成因子...")
            logger.info("="*60)
            self.setup_task()
            
            # 2. 计算
            logger.info(f"\n计算所有因子...")
            all_factors = self._compute_all_factors()
            
            if not all_factors:
                raise FactorGenerationException("未能计算任何因子")
            
            # 3. 合并结果
            logger.info(f"合并因子数据...")
            result_df = self._merge_factors(all_factors)
            
            # 4. 验证质量
            logger.info(f"验证数据质量...")
            self._validate_result(result_df)
            
            # 5. 报告
            logger.info(f"生成报告...")
            self._generate_report(result_df)
            
            logger.info("\n" + "="*60)
            logger.info(f"✅ 因子生成成功!")
            logger.info("="*60)
            logger.info(f"  因子数: {len(all_factors)}")
            logger.info(f"  数据点: {len(result_df)}")
            logger.info(f"  日期范围: {result_df['date'].min()} ~ {result_df['date'].max()}")
            logger.info(f"  输出目录: {self.task_dir}")
            
            return result_df
        
        except PartialResultError as e:
            logger.warning(f"\n⚠️  {e}")
            logger.warning(e.get_failure_summary())
            raise
        except FactorGenerationException as e:
            logger.error(f"\n❌ 生成失败: {e}")
            raise
        except Exception as e:
            logger.error(f"\n❌ 意外错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise FactorGenerationException(f"生成失败: {e}") from e
    
    def _merge_factors(self, all_factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        合并所有因子为 DataFrame
        
        Args:
            all_factors: 因子数据字典 {factor_name: Series}
        
        Returns:
            pd.DataFrame: 合并后的数据
        """
        dfs = []
        
        for factor_name, factor_series in all_factors.items():
            if factor_series is None or factor_series.empty:
                continue
            
            # 转换为 DataFrame
            df = pd.DataFrame({
                'date': factor_series.index,
                'factor': factor_name,
                'value': factor_series.values
            })
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # 合并所有数据
        result = pd.concat(dfs, ignore_index=True)
        
        # 转换为宽格式 (date, stock_code, factor1, factor2, ...)
        result = result.pivot_table(
            index='date',
            columns='factor',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        return result
    
    def _validate_result(self, df: pd.DataFrame) -> None:
        """
        验证结果质量
        
        Args:
            df: 结果 DataFrame
        
        Raises:
            FactorValidationError: 质量检查未通过
        """
        factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]
        
        check_result = DataQualityChecker.check_factor_output(df, factor_cols)
        
        # 打印检查结果
        DataQualityChecker.print_check_result(check_result, verbose=True)
        
        # 如果有错误，抛出异常
        if not check_result['passed']:
            raise FactorValidationError(
                'all',
                f"质量检查失败: {check_result['summary']['errors']} 个错误"
            )
    
    def _generate_report(self, df: pd.DataFrame) -> None:
        """
        生成统计报告
        
        Args:
            df: 结果 DataFrame
        """
        # 保存到 CSV
        output_file = self.task_dir / f"factors_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"  因子文件: {output_file}")
        
        # 保存元数据
        metadata_file = self.task_dir / f"metadata_{self.timestamp}.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"因子生成元数据\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"股票代码: {', '.join(self.stock_codes[:5])}{'...' if len(self.stock_codes) > 5 else ''}\n")
            f.write(f"日期范围: {self.start_date} ~ {self.end_date}\n")
            f.write(f"数据点数: {len(df)}\n")
            f.write(f"因子数: {len([c for c in df.columns if c not in ['date', 'stock_code']])}\n")


class BuiltinFactorGenerator(FactorGenerator):
    """
    内置因子生成器
    
    生成内置的技术因子：VOL10, RSI_14, MA_20, MACD_12_26_9
    
    使用示例:
        generator = BuiltinFactorGenerator(
            stock_codes=['000001', '000002'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            factor_names=['VOL10', 'RSI_14']
        )
        df = generator.generate()
    """
    
    def __init__(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        factor_names: Optional[List[str]] = None,
        output_dir: str = './data/factor_tasks'
    ):
        """
        初始化内置因子生成器
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factor_names: 要生成的因子列表（如果为 None，生成所有）
            output_dir: 输出目录
        """
        super().__init__(stock_codes, start_date, end_date, output_dir)
        
        # 确定要生成的因子
        all_factors = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
        self.factor_names = factor_names if factor_names else all_factors
        
        # 验证因子名称
        invalid = set(self.factor_names) - set(all_factors)
        if invalid:
            raise ValueError(f"不支持的因子: {invalid}. 支持的因子: {all_factors}")
        
        logger.info(f"内置因子生成器初始化")
        logger.info(f"  因子: {', '.join(self.factor_names)}")
    
    def _compute_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        计算所有内置因子
        
        Returns:
            Dict[str, pd.DataFrame]: {factor_name: factor_data}
        """
        all_factors_data = {factor: pd.DataFrame() for factor in self.factor_names}
        failures = {}
        
        # 为每只股票计算所有因子
        for i, stock_code in enumerate(self.stock_codes):
            logger.info(f"[{i+1}/{len(self.stock_codes)}] 计算股票 {stock_code}")
            
            try:
                # 为该股票计算所有因子
                stock_factors = self._compute_factors_for_stock(stock_code)
                
                # 合并到总结果中
                for factor_name, factor_series in stock_factors.items():
                    if factor_name not in all_factors_data:
                        all_factors_data[factor_name] = pd.DataFrame()
                    
                    # 追加数据
                    df = pd.DataFrame({
                        'date': factor_series.index,
                        'stock_code': stock_code,
                        'value': factor_series.values
                    })
                    all_factors_data[factor_name] = pd.concat(
                        [all_factors_data[factor_name], df],
                        ignore_index=True
                    )
            
            except Exception as e:
                logger.warning(f"  ❌ 股票 {stock_code} 计算失败: {e}")
                failures[stock_code] = str(e)
                continue
        
        # 检查是否有部分失败
        if failures:
            logger.warning(f"⚠️  {len(failures)} 只股票计算失败")
        
        return all_factors_data
    
    def _compute_factors_for_stock(self, stock_code: str) -> Dict[str, pd.Series]:
        """
        为单只股票计算所有因子
        
        Args:
            stock_code: 股票代码
        
        Returns:
            Dict[str, pd.Series]: {factor_name: factor_values}
        """
        stock_results = {}
        
        for factor_name in self.factor_names:
            try:
                # 创建计算器
                calculator = create_factor_calculator(factor_name)
                
                # 计算因子
                factor_series = calculator.calculate(
                    stock_code,
                    self.start_date,
                    self.end_date
                )
                
                # 过滤出指定日期范围的数据（避免计算窗口期的数据）
                factor_series = factor_series[
                    (factor_series.index >= pd.to_datetime(self.start_date)) &
                    (factor_series.index <= pd.to_datetime(self.end_date))
                ]
                
                stock_results[factor_name] = factor_series
                
                logger.debug(f"  ✓ {factor_name}: {len(factor_series)} 条数据")
            
            except FactorCalculationError as e:
                logger.warning(f"  ⚠️  {factor_name}: {e.reason}")
                # 用 NaN Series 填充
                stock_results[factor_name] = pd.Series(
                    dtype=float,
                    name=factor_name
                )
            except Exception as e:
                logger.warning(f"  ❌ {factor_name}: {e}")
                stock_results[factor_name] = pd.Series(
                    dtype=float,
                    name=factor_name
                )
        
        return stock_results
