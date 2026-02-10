"""
异常定义 - V2 版本中的所有自定义异常

该模块定义了因子生成过程中可能遇到的所有异常类。
通过体系化的异常定义，使得错误处理更加清晰和可控。

异常层次:
    FactorGenerationException (基类)
    ├─ DataNotAvailableError      # 数据不可用
    ├─ FactorCalculationError      # 因子计算失败
    ├─ FactorValidationError       # 因子验证失败
    └─ PartialResultError          # 部分成功结果
"""


class FactorGenerationException(Exception):
    """
    所有因子生成异常的基类
    
    所有与因子生成相关的异常都应该继承这个基类。
    """
    pass


class DataNotAvailableError(FactorGenerationException):
    """
    数据不可用异常
    
    当无法为指定的股票和日期范围获取数据时抛出。
    这是一个预期的异常，表示数据源中没有相关数据。
    
    Attributes:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    def __init__(self, stock_code: str, start_date: str, end_date: str, reason: str = None):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        
        if reason:
            message = f"无法获取数据: {stock_code} ({start_date} ~ {end_date}) - {reason}"
        else:
            message = f"无法获取数据: {stock_code} ({start_date} ~ {end_date})"
        
        super().__init__(message)


class FactorCalculationError(FactorGenerationException):
    """
    因子计算失败异常
    
    当计算因子时出现错误时抛出。
    这包括因子函数异常、数据处理异常等。
    
    Attributes:
        factor_name: 因子名称
        stock_code: 股票代码
        reason: 失败原因
    """
    def __init__(self, factor_name: str, stock_code: str = None, reason: str = None):
        self.factor_name = factor_name
        self.stock_code = stock_code
        self.reason = reason
        
        if stock_code and reason:
            message = f"计算因子失败: {factor_name} ({stock_code}) - {reason}"
        elif stock_code:
            message = f"计算因子失败: {factor_name} ({stock_code})"
        elif reason:
            message = f"计算因子失败: {factor_name} - {reason}"
        else:
            message = f"计算因子失败: {factor_name}"
        
        super().__init__(message)


class FactorValidationError(FactorGenerationException):
    """
    因子验证失败异常
    
    当因子输出数据不满足质量检查时抛出。
    这包括 NaN 值过多、数据类型错误等。
    
    Attributes:
        factor_name: 因子名称
        issue: 问题描述
    """
    def __init__(self, factor_name: str, issue: str):
        self.factor_name = factor_name
        self.issue = issue
        message = f"因子验证失败: {factor_name} - {issue}"
        super().__init__(message)


class PartialResultError(FactorGenerationException):
    """
    部分结果异常
    
    当生成过程中有部分因子成功、部分失败时抛出。
    这不是完全的失败，而是部分成功的情况。
    
    Attributes:
        successful: 成功的个数
        failed: 失败的个数
        failures: 失败的详细信息 {name: error_message}
    """
    def __init__(self, successful: int, failed: int, failures: dict = None):
        self.successful = successful
        self.failed = failed
        self.failures = failures or {}
        
        message = f"部分成功: {successful} 成功，{failed} 失败"
        super().__init__(message)
    
    def get_failure_summary(self) -> str:
        """获取失败摘要"""
        if not self.failures:
            return ""
        
        lines = ["失败详情:"]
        for name, error in self.failures.items():
            lines.append(f"  - {name}: {error}")
        return "\n".join(lines)
