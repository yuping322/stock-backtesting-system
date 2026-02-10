# 因子生成模块改进建议（详细）

**日期**: 2026-02-03  
**目标**: 为每个 P0/P1 问题提供具体的改进方案

---

## 目录

1. [P0 问题改进方案](#p0-问题改进方案)
2. [P1 问题改进方案](#p1-问题改进方案)
3. [代码示例](#代码示例)
4. [迁移计划](#迁移计划)

---

## P0 问题改进方案

### 问题 1: 计算器接口不一致

#### 当前状态
```
FactorCalculator (ABC)
├─ OHLCVFactorCalculator.calculate(stock_code, start_date, end_date)
├─ FileFactorCalculator.calculate(stock_code, start_date, end_date)
├─ CustomFactorCalculator.calculate(stock_code, start_date, end_date)
└─ BuiltinFactorCalculator.calculate(factor_name, ohlcv)  ❌ 签名不同
   └─ TalibFactorCalculator.calculate(factor_name, ohlcv)  ❌ 签名不同
```

#### 改进方案

**方案 A: 统一为 3 参数接口**（推荐）

```python
# calculator.py 中

from abc import ABC, abstractmethod
import pandas as pd

class FactorCalculator(ABC):
    """统一的因子计算器接口"""
    
    @abstractmethod
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """
        计算因子值
        
        Args:
            stock_code: 股票代码（如 '000001'）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        
        Returns:
            pd.Series: 因子值序列，索引为日期（DatetimeIndex）
        
        Raises:
            FactorCalculationError: 计算失败
            DataNotAvailableError: 数据不可用
        """
        pass


# 内置因子计算器
class BuiltinFactorCalculator(FactorCalculator):
    """内置因子计算器 - 统一接口"""
    
    SUPPORTED_FACTORS = ['VOL10', 'RSI_14', 'MA_20', 'MACD_12_26_9']
    
    def __init__(self, factor_name: str):
        if factor_name not in self.SUPPORTED_FACTORS:
            raise ValueError(f"不支持的因子: {factor_name}")
        self.factor_name = factor_name
        self._data_loader = None  # 注入依赖
    
    def set_data_loader(self, data_loader):
        """注入数据加载器（依赖注入）"""
        self._data_loader = data_loader
        return self
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算内置因子"""
        try:
            # 加载 OHLCV 数据
            ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
            if ohlcv.empty:
                raise DataNotAvailableError(f"无法加载 {stock_code} 的数据")
            
            # 使用私有方法计算
            if self.factor_name == 'VOL10':
                return self._calculate_vol10(ohlcv)
            elif self.factor_name == 'RSI_14':
                return self._calculate_rsi_14(ohlcv)
            # ... 其他因子
            else:
                raise ValueError(f"不支持的因子: {self.factor_name}")
        
        except Exception as e:
            raise FactorCalculationError(
                f"计算 {self.factor_name} 失败 ({stock_code}): {e}"
            ) from e
    
    def _load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载 OHLCV 数据"""
        if self._data_loader:
            return self._data_loader.load_ohlcv(stock_code, start_date, end_date)
        # 默认加载器
        from src.data import data
        return data.load_oss_complex_stocks([stock_code], start_date, end_date)
    
    @staticmethod
    def _calculate_vol10(ohlcv: pd.DataFrame) -> pd.Series:
        """计算 VOL10"""
        # 实现逻辑...
        pass
    
    # ... 其他计算方法


# Talib 计算器
class TalibFactorCalculator(FactorCalculator):
    """Talib 因子计算器 - 统一接口"""
    
    def __init__(self, factor_name: str, params: list = None):
        # 格式: 'TALIB_RSI_14', 'TALIB_SMA_20'
        if not factor_name.startswith('TALIB_'):
            raise ValueError(f"Talib 因子应以 'TALIB_' 开头: {factor_name}")
        
        self.factor_name = factor_name
        self.func_name, self.params = self._parse_factor_name(factor_name, params)
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算 Talib 因子"""
        try:
            # 1. 加载数据
            ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
            if ohlcv.empty:
                raise DataNotAvailableError(...)
            
            # 2. 调用 Talib
            result = self._call_talib(ohlcv)
            
            # 3. 转换结果为 Series
            return pd.Series(result, index=ohlcv.index)
        except Exception as e:
            raise FactorCalculationError(...) from e
    
    def _parse_factor_name(self, factor_name: str, params: list = None):
        """解析因子名称"""
        # 'TALIB_RSI_14' -> ('RSI', [14])
        pass
    
    def _call_talib(self, ohlcv: pd.DataFrame):
        """调用 Talib 函数"""
        import talib
        func = getattr(talib, self.func_name)
        # 根据 OHLCV 的哪些字段，调用对应的参数
        return func(...)
    
    def _load_ohlcv(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass


# 自定义函数计算器
class CustomFunctionCalculator(FactorCalculator):
    """自定义函数计算器 - 支持 2 种函数签名"""
    
    def __init__(self, factor_name: str, func: callable):
        self.factor_name = factor_name
        self.func = func
        self._func_type = self._detect_func_type(func)
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """计算自定义因子"""
        try:
            if self._func_type == 'ohlcv':
                # 函数签名: func(ohlcv: DataFrame) -> Series
                ohlcv = self._load_ohlcv(stock_code, start_date, end_date)
                return self.func(ohlcv)
            elif self._func_type == 'params':
                # 函数签名: func(stock_code, start_date, end_date) -> Series
                return self.func(stock_code, start_date, end_date)
            else:
                raise ValueError(f"不支持的函数类型")
        except Exception as e:
            raise FactorCalculationError(...) from e
    
    def _detect_func_type(self, func: callable) -> str:
        """检测函数签名类型"""
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if len(params) == 1:
            return 'ohlcv'  # 接收 DataFrame
        elif len(params) == 3:
            return 'params'  # 接收 (stock_code, start_date, end_date)
        else:
            raise ValueError(f"因子函数参数数量不对: {len(params)}")


# 文件加载器
class FileFactorCalculator(FactorCalculator):
    """从文件加载因子 - 统一接口"""
    
    def __init__(self, file_path: str, factor_name: str):
        self.file_path = file_path
        self.factor_name = factor_name
        self._data = None
    
    def calculate(self, stock_code: str, start_date: str, end_date: str) -> pd.Series:
        """从文件中获取因子值"""
        try:
            if self._data is None:
                self._load_file()
            
            # 过滤数据
            mask = (self._data['code'] == stock_code) & \
                   (self._data['date'] >= start_date) & \
                   (self._data['date'] <= end_date)
            
            result = self._data[mask].set_index('date')[self.factor_name]
            return pd.to_numeric(result)
        except Exception as e:
            raise FactorCalculationError(...) from e
    
    def _load_file(self):
        """加载文件"""
        self._data = pd.read_csv(self.file_path)
        # 验证必要的列
        required_cols = ['date', 'code', self.factor_name]
        if not all(col in self._data.columns for col in required_cols):
            raise ValueError(f"文件缺少必要的列: {required_cols}")


# 工厂函数
def create_factor_calculator(
    factor_name: str = None,
    factor_func: callable = None,
    file_path: str = None,
    params: list = None
) -> FactorCalculator:
    """
    创建因子计算器（统一入口）
    
    优先级:
    1. 文件加载 (file_path)
    2. 内置因子 (factor_name, 包括 TALIB_*)
    3. 自定义函数 (factor_func)
    """
    if file_path:
        if not factor_name:
            raise ValueError("使用 file_path 时必须提供 factor_name")
        return FileFactorCalculator(file_path, factor_name)
    
    if factor_name:
        if factor_name.startswith('TALIB_'):
            return TalibFactorCalculator(factor_name, params)
        elif factor_name in BuiltinFactorCalculator.SUPPORTED_FACTORS:
            return BuiltinFactorCalculator(factor_name)
        else:
            raise ValueError(f"未知的因子: {factor_name}")
    
    if factor_func:
        return CustomFunctionCalculator(factor_name or 'CUSTOM', factor_func)
    
    raise ValueError("必须提供 file_path、factor_name 或 factor_func 之一")
```

**优点**:
- ✅ 所有计算器实现相同的接口
- ✅ 通过依赖注入解耦（data_loader）
- ✅ 清晰的异常体系
- ✅ 易于单元测试

---

### 问题 2: 错误处理流程混乱

#### 当前问题
- 各生成器使用不同的异常处理策略
- 无法区分 **预期异常** vs **意外异常**
- 上游无法判断是"部分成功"还是"完全失败"

#### 改进方案

**定义异常体系**:

```python
# _base.py 中

class FactorGenerationException(Exception):
    """所有因子生成异常的基类"""
    pass


class DataNotAvailableError(FactorGenerationException):
    """数据不可用"""
    def __init__(self, stock_code: str, start_date: str, end_date: str):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(
            f"无法获取 {stock_code} 在 {start_date} ~ {end_date} 的数据"
        )


class FactorCalculationError(FactorGenerationException):
    """因子计算失败"""
    def __init__(self, factor_name: str, stock_code: str, reason: str):
        self.factor_name = factor_name
        self.stock_code = stock_code
        self.reason = reason
        super().__init__(
            f"计算因子 {factor_name} (股票 {stock_code}) 失败: {reason}"
        )


class FactorValidationError(FactorGenerationException):
    """因子验证失败（质量检查）"""
    def __init__(self, factor_name: str, issue: str):
        self.factor_name = factor_name
        self.issue = issue
        super().__init__(
            f"因子 {factor_name} 验证失败: {issue}"
        )


class PartialResultError(FactorGenerationException):
    """部分结果成功"""
    def __init__(self, successful: int, failed: int, failures: dict):
        self.successful = successful
        self.failed = failed
        self.failures = failures  # {factor_name: error_message}
        super().__init__(
            f"生成完成（{successful} 成功，{failed} 失败）"
        )


# 使用示例
class BuiltinFactorGenerator(FactorGenerator):
    def generate(self) -> pd.DataFrame:
        failures = {}
        successful_factors = []
        
        for factor_name in self.factor_names:
            try:
                # 生成因子...
                successful_factors.append(factor_name)
            except DataNotAvailableError as e:
                logger.warning(f"⚠️  {e}")
                failures[factor_name] = str(e)
            except FactorCalculationError as e:
                logger.error(f"❌ {e}")
                failures[factor_name] = str(e)
            except Exception as e:
                logger.error(f"❌ 意外错误: {e}")
                failures[factor_name] = f"意外错误: {e}"
        
        if not successful_factors:
            # 全部失败
            raise PartialResultError(0, len(failures), failures)
        
        if failures:
            # 部分失败
            raise PartialResultError(len(successful_factors), len(failures), failures)
        
        return result_df  # 全部成功
```

**在调用端处理异常**:

```python
# all_in_one.py 中

def run_all_generators():
    results = {}
    
    generators = [
        ('builtin', builtin.generate_builtin_factors),
        ('qlib', qlib.generate_qlib_factors),
        ('talib', talib.generate_talib_factors),
        ('oss', oss.generate_oss_factors),
    ]
    
    for gen_name, gen_func in generators:
        try:
            result = gen_func(...)
            results[gen_name] = {
                'status': 'success',
                'data': result,
                'failures': {}
            }
        except PartialResultError as e:
            # 部分成功
            results[gen_name] = {
                'status': 'partial',
                'data': e,
                'failures': e.failures,
                'successful': e.successful
            }
        except FactorGenerationException as e:
            # 完全失败（预期错误）
            results[gen_name] = {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
        except Exception as e:
            # 意外错误
            results[gen_name] = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
    
    # 生成报告
    generate_report(results)
```

---

### 问题 3: 数据质量检查缺失

#### 改进方案

```python
# _base.py 中

class DataQualityChecker:
    """数据质量检查器"""
    
    @staticmethod
    def check_factor_output(df: pd.DataFrame, factor_names: list) -> dict:
        """
        检查因子输出的质量
        
        Returns:
            {
                'passed': bool,
                'issues': [
                    {'level': 'error|warning', 'message': '...'},
                    ...
                ]
            }
        """
        issues = []
        
        # 检查 1: 必要的列
        required_cols = ['date', 'stock_code'] + factor_names
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append({
                'level': 'error',
                'message': f"缺少列: {missing_cols}"
            })
        
        # 检查 2: 数据类型
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                issues.append({
                    'level': 'warning',
                    'message': "date 列不是 datetime 类型"
                })
        
        # 检查 3: NaN 值比例
        for col in factor_names:
            if col in df.columns:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio > 0.5:
                    issues.append({
                        'level': 'error',
                        'message': f"因子 {col} NaN 比例过高: {nan_ratio:.1%}"
                    })
                elif nan_ratio > 0.1:
                    issues.append({
                        'level': 'warning',
                        'message': f"因子 {col} 有 {nan_ratio:.1%} 的 NaN 值"
                    })
        
        # 检查 4: 每日股票数的一致性
        if 'date' in df.columns and 'stock_code' in df.columns:
            daily_counts = df.groupby('date')['stock_code'].count()
            count_std = daily_counts.std()
            count_mean = daily_counts.mean()
            cv = count_std / count_mean  # 变异系数
            
            if cv > 0.5:
                issues.append({
                    'level': 'warning',
                    'message': f"每日股票数波动较大 (CV={cv:.2f}): " \
                               f"min={daily_counts.min()}, max={daily_counts.max()}"
                })
        
        # 检查 5: 因子值的极端异常
        for col in factor_names:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_ratio = outliers / len(df)
                
                if outlier_ratio > 0.05:
                    issues.append({
                        'level': 'warning',
                        'message': f"因子 {col} 有 {outlier_ratio:.1%} 的异常值"
                    })
        
        # 检查 6: 日期连续性
        if 'date' in df.columns:
            dates = sorted(df['date'].unique())
            date_diffs = pd.Series(dates).diff().dt.days
            
            if date_diffs.max() > 5:  # 假设最多允许周末
                issues.append({
                    'level': 'warning',
                    'message': f"日期序列不连续，最大间隔 {date_diffs.max()} 天"
                })
        
        # 检查 7: 股票代码标准化
        if 'stock_code' in df.columns:
            invalid_codes = df[df['stock_code'].str.len() != 6].shape[0]
            if invalid_codes > 0:
                issues.append({
                    'level': 'error',
                    'message': f"发现 {invalid_codes} 个非标准股票代码"
                })
        
        return {
            'passed': not any(issue['level'] == 'error' for issue in issues),
            'issues': issues,
            'summary': {
                'total_issues': len(issues),
                'errors': sum(1 for i in issues if i['level'] == 'error'),
                'warnings': sum(1 for i in issues if i['level'] == 'warning')
            }
        }


# 在生成器中使用
class FactorGenerator(ABC):
    def generate(self) -> pd.DataFrame:
        df = self._do_generate()
        
        # 质量检查
        check_result = DataQualityChecker.check_factor_output(df, self.factor_names)
        
        if not check_result['passed']:
            # 抛出异常，让调用端决定如何处理
            raise FactorValidationError(
                factor_name='all',
                issue=f"质量检查失败: {check_result['summary']}"
            )
        
        # 打印警告
        for issue in check_result['issues']:
            if issue['level'] == 'warning':
                logger.warning(f"⚠️  {issue['message']}")
        
        return df
```

---

### 问题 4: 生成器职责不清

#### 改进方案

**新的职责划分**:

```
┌─────────────────────────────────────┐
│ FactorGenerator (Orchestrator)      │
│ 职责: 编排、验证、报告              │
│ 不负责: 计算、数据加载              │
└─────────────────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ FactorCalculator (Computing Engine) │
│ 职责: 计算单个因子                  │
│ 不负责: 数据加载、格式转换          │
└─────────────────────────────────────┘
           ▼
┌─────────────────────────────────────┐
│ DataLoader, Processor               │
│ 职责: 数据加载、转换、验证          │
│ 不负责: 业务逻辑、计算              │
└─────────────────────────────────────┘
```

**重构后的生成器**:

```python
class BuiltinFactorGenerator(FactorGenerator):
    """内置因子生成器"""
    
    def __init__(self, stock_codes: list, start_date: str, end_date: str,
                 factor_names: list = None, output_dir: str = './data'):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.factor_names = factor_names or ['VOL10', 'RSI_14', 'MA_20']
        self.output_dir = output_dir
        
        # 依赖注入
        self._calculator_factory = CalculatorFactory()
        self._data_loader = DataLoader()
        self._processor = DataFrameProcessor()
        self._quality_checker = DataQualityChecker()
    
    def generate(self) -> pd.DataFrame:
        """
        生成因子数据
        
        Flow:
        1. 准备阶段: setup_task(), validate_params()
        2. 计算阶段: 为每个股票计算所有因子
        3. 合并阶段: merge_factor_dataframes()
        4. 验证阶段: validate_output()
        5. 报告阶段: generate_report()
        """
        try:
            # 1. 准备
            self.setup_task()
            
            # 2. 计算
            all_stock_data = self._compute_all_factors()
            
            # 3. 合并
            result_df = self._merge_results(all_stock_data)
            
            # 4. 验证
            self._validate_results(result_df)
            
            # 5. 报告
            self._report_success(result_df)
            
            return result_df
        
        except Exception as e:
            self._report_failure(e)
            raise
    
    def _compute_all_factors(self) -> dict:
        """计算所有因子"""
        all_stock_data = {}
        
        for i, stock_code in enumerate(self.stock_codes):
            logger.info(f"[{i+1}/{len(self.stock_codes)}] 计算股票 {stock_code} 的所有因子")
            
            try:
                # 为该股票计算所有因子
                stock_factors = self._compute_factors_for_stock(stock_code)
                all_stock_data[stock_code] = stock_factors
            except Exception as e:
                logger.error(f"❌ 股票 {stock_code} 计算失败: {e}")
                # 继续处理下一只股票
                continue
        
        return all_stock_data
    
    def _compute_factors_for_stock(self, stock_code: str) -> pd.DataFrame:
        """为单只股票计算所有因子"""
        stock_results = {}
        
        for factor_name in self.factor_names:
            try:
                # 创建计算器
                calculator = self._calculator_factory.create(factor_name)
                
                # 计算因子
                factor_series = calculator.calculate(
                    stock_code, self.start_date, self.end_date
                )
                
                stock_results[factor_name] = factor_series
            except FactorCalculationError as e:
                logger.warning(f"  ⚠️  {factor_name}: {e.reason}")
                # 用 NaN 填充
                stock_results[factor_name] = pd.Series(dtype=float)
        
        return stock_results
    
    def _merge_results(self, all_stock_data: dict) -> pd.DataFrame:
        """合并所有股票的因子数据"""
        all_dfs = []
        
        for stock_code, stock_factors in all_stock_data.items():
            df = pd.DataFrame(stock_factors)
            df['stock_code'] = stock_code
            all_dfs.append(df)
        
        result = pd.concat(all_dfs, ignore_index=True)
        
        # 处理 DataFrame
        result = self._processor.format_dataframe(
            result,
            date_col='index',
            code_col='stock_code',
            factor_cols=self.factor_names
        )
        
        return result
    
    def _validate_results(self, df: pd.DataFrame):
        """验证结果质量"""
        check = self._quality_checker.check_factor_output(df, self.factor_names)
        
        if not check['passed']:
            raise FactorValidationError(
                factor_name='output',
                issue=f"质量检查失败: {check['summary']}"
            )
        
        # 打印警告
        for issue in check['issues']:
            if issue['level'] == 'warning':
                logger.warning(f"  ⚠️  {issue['message']}")
    
    def _report_success(self, df: pd.DataFrame):
        """报告成功"""
        logger.info(f"✅ 因子生成成功!")
        logger.info(f"  因子: {', '.join(self.factor_names)}")
        logger.info(f"  股票数: {df['stock_code'].nunique()}")
        logger.info(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        logger.info(f"  数据点: {len(df)}")
    
    def _report_failure(self, error: Exception):
        """报告失败"""
        logger.error(f"❌ 因子生成失败: {error}")
```

---

## P1 问题改进方案

### 问题 5: Talib 参数配置不可外部化

#### 改进方案

**创建配置文件** `config/talib_parameters.yaml`:

```yaml
# Talib 因子参数配置

# 移动平均线相关
SMA:
  periods: [5, 10, 20, 50, 200]

EMA:
  periods: [5, 10, 20, 50, 200]

# 动量指标
RSI:
  periods: [6, 14, 21]

STOCHRSI:
  - { timeperiod: 14, fastk_period: 14, fastd_period: 3, fastd_matype: 0 }

MOM:
  periods: [5, 10, 20]

# 波动率指标
ATR:
  periods: [14, 21]

BBANDS:
  - { timeperiod: 5, nbdevup: 2, nbdevdn: 2 }
  - { timeperiod: 20, nbdevup: 2, nbdevdn: 2 }

# MACD
MACD:
  - { fastperiod: 12, slowperiod: 26, signalperiod: 9 }

# ... 更多
```

**在生成器中使用**:

```python
# talib.py 中

import yaml
from pathlib import Path

class TalibParameterLoader:
    """Talib 参数加载器"""
    
    @staticmethod
    def load_parameters(config_file: str = None) -> dict:
        """加载 Talib 参数配置"""
        if config_file is None:
            # 默认配置文件位置
            config_file = Path(__file__).parent.parent.parent / 'config' / 'talib_parameters.yaml'
        
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}，使用默认参数")
            return TalibParameterLoader.get_default_parameters()
        
        with open(config_file) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_default_parameters() -> dict:
        """获取默认参数"""
        return {
            'SMA': {'periods': [5, 10, 20, 50, 200]},
            'RSI': {'periods': [6, 14, 21]},
            # ...
        }


class TalibFactorGenerator(FactorGenerator):
    """改进后的 Talib 因子生成器"""
    
    def __init__(self, stock_codes: list, start_date: str, end_date: str,
                 output_dir: str = './data', config_file: str = None):
        # ...
        self.parameters = TalibParameterLoader.load_parameters(config_file)
    
    def _generate_talib_factors(self) -> list:
        """生成 Talib 因子列表"""
        factor_list = []
        
        for func_name, params_config in self.parameters.items():
            if isinstance(params_config, dict) and 'periods' in params_config:
                # 简单周期参数
                for period in params_config['periods']:
                    factor_list.append(f"TALIB_{func_name}_{period}")
            elif isinstance(params_config, list):
                # 复杂参数组合
                for i, params in enumerate(params_config):
                    param_str = '_'.join(str(v) for v in params.values())
                    factor_list.append(f"TALIB_{func_name}_{param_str}")
        
        return factor_list
```

---

### 问题 6: 缺少单元测试

#### 改进方案

**创建测试框架** `tests/factor/`:

```
tests/factor/
├─ __init__.py
├─ conftest.py              # pytest fixtures
├─ test_calculator.py       # 计算器测试
├─ test_builtin_generator.py
├─ test_qlib_generator.py
├─ test_talib_generator.py
├─ test_oss_generator.py
├─ test_data_quality.py
└─ fixtures/                # 测试数据
   ├─ sample_ohlcv.csv
   ├─ sample_factors.csv
   └─ ...
```

**测试示例** `test_calculator.py`:

```python
import pytest
import pandas as pd
from src.factor.generator.calculator import (
    BuiltinFactorCalculator,
    TalibFactorCalculator,
    create_factor_calculator
)


class TestBuiltinFactorCalculator:
    """测试内置因子计算器"""
    
    @pytest.fixture
    def calculator(self):
        """创建计算器"""
        return BuiltinFactorCalculator('VOL10')
    
    def test_calculate_vol10(self, calculator):
        """测试 VOL10 计算"""
        result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        assert result.dtype == float
        assert result.index.name == 'date'
    
    def test_unsupported_factor(self):
        """测试不支持的因子"""
        with pytest.raises(ValueError):
            BuiltinFactorCalculator('INVALID_FACTOR')
    
    def test_no_data_available(self, calculator):
        """测试数据不可用"""
        with pytest.raises(DataNotAvailableError):
            calculator.calculate('999999', '1900-01-01', '1900-12-31')


class TestTalibFactorCalculator:
    """测试 Talib 因子计算器"""
    
    def test_parse_factor_name(self):
        """测试因子名称解析"""
        calc = TalibFactorCalculator('TALIB_RSI_14')
        assert calc.func_name == 'RSI'
        assert calc.params == [14]
    
    def test_talib_not_installed(self):
        """测试 Talib 未安装"""
        with pytest.raises(ImportError):
            # Mock talib 不可用
            pass


class TestCreateFactorCalculator:
    """测试工厂函数"""
    
    def test_create_builtin_calculator(self):
        """测试创建内置计算器"""
        calc = create_factor_calculator('VOL10')
        assert isinstance(calc, BuiltinFactorCalculator)
    
    def test_create_talib_calculator(self):
        """测试创建 Talib 计算器"""
        calc = create_factor_calculator('TALIB_RSI_14')
        assert isinstance(calc, TalibFactorCalculator)
    
    def test_invalid_factor(self):
        """测试无效因子"""
        with pytest.raises(ValueError):
            create_factor_calculator('INVALID')
```

---

## 代码示例

### 示例 1: 使用改进后的 API

```python
from src.factor.generator import (
    create_factor_calculator,
    BuiltinFactorGenerator,
    TalibFactorGenerator,
    DataQualityChecker
)

# 方式 1: 使用计算器
calculator = create_factor_calculator('VOL10')
result = calculator.calculate('000001', '2024-01-01', '2024-12-31')
print(result)

# 方式 2: 使用生成器
generator = BuiltinFactorGenerator(
    stock_codes=['000001', '000002', '000003'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_names=['VOL10', 'RSI_14'],
    output_dir='./data/factors'
)

try:
    df = generator.generate()
    print(f"✅ 生成了 {len(df)} 条因子数据")
except PartialResultError as e:
    print(f"⚠️  部分成功: {e.successful} 成功，{e.failed} 失败")
    print(f"  失败因子: {e.failures}")
except FactorGenerationException as e:
    print(f"❌ 生成失败: {e}")
```

---

## 迁移计划

### 第 1 阶段: 接口统一（1-2 周）
1. 重构 `calculator.py` 中所有计算器的接口
2. 定义异常体系
3. 编写单元测试

### 第 2 阶段: 质量检查（1 周）
1. 实现 `DataQualityChecker` 类
2. 集成到各生成器中
3. 测试

### 第 3 阶段: 配置外部化（1 周）
1. 创建 `config/talib_parameters.yaml`
2. 实现 `TalibParameterLoader`
3. 更新 `TalibFactorGenerator`

### 第 4 阶段: 测试框架（1-2 周）
1. 创建测试文件结构
2. 编写测试用例
3. 达到 80% 覆盖率

### 第 5 阶段: 文档更新（1 周）
1. 更新 API 文档
2. 编写迁移指南
3. 添加使用示例

---

**总耗时**: 5-7 周，预期改进效果: 代码质量提升 50%+

