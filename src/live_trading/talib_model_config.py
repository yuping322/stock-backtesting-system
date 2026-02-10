#!/usr/bin/env python3
"""
TALIB模型配置
定义TALIB因子模型的配置参数和设置
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class TALIBModelConfig:
    """TALIB因子模型配置"""

    # 模型基本信息
    model_name: str = "talib_factor_model"
    version: str = "1.0.0"
    description: str = "基于TALIB技术指标的量化因子模型"

    # 数据路径
    model_results_dir: str = "debug_model_results"
    predictions_long_file: str = "predictions_long.pkl"
    predictions_short_file: str = "predictions_short.pkl"

    # 因子配置
    factors: List[str] = None

    # 策略配置
    available_strategies: List[str] = None
    default_strategy: str = "long"

    # 预测配置
    prediction_threshold: float = 0.0  # 预测分数的阈值
    max_positions: int = 50  # 最大持仓数量
    min_position_size: float = 0.001  # 最小仓位大小

    # 风险控制
    max_single_weight: float = 0.05  # 单个股票最大权重
    max_industry_weight: float = 0.25  # 单个行业最大权重
    risk_free_rate: float = 0.03  # 无风险利率

    # 交易成本
    transaction_cost_bps: float = 5.0  # 交易成本（基点）

    # 再平衡
    rebalance_frequency: str = "daily"  # 再平衡频率
    rebalance_threshold: float = 0.02  # 再平衡阈值

    def __post_init__(self):
        if self.factors is None:
            self.factors = [
                'TALIB_MACD_12_26_9',
                'TALIB_MACDEXT_12_26_9_0_0_0',
                'TALIB_MACDFIX_9',
                'TALIB_HT_DCPERIOD'
            ]

        if self.available_strategies is None:
            self.available_strategies = ['long', 'short']

    @property
    def predictions_long_path(self) -> Path:
        """Long预测文件完整路径"""
        return Path(self.model_results_dir) / self.predictions_long_file

    @property
    def predictions_short_path(self) -> Path:
        """Short预测文件完整路径"""
        return Path(self.model_results_dir) / self.predictions_short_file

    def get_strategy_config(self, strategy: str) -> Dict:
        """获取指定策略的配置"""
        if strategy not in self.available_strategies:
            raise ValueError(f"不支持的策略: {strategy}")

        base_config = {
            'strategy': strategy,
            'prediction_threshold': self.prediction_threshold,
            'max_positions': self.max_positions,
            'min_position_size': self.min_position_size,
            'max_single_weight': self.max_single_weight,
            'max_industry_weight': self.max_industry_weight,
            'transaction_cost_bps': self.transaction_cost_bps,
            'rebalance_frequency': self.rebalance_frequency,
            'rebalance_threshold': self.rebalance_threshold,
        }

        # 策略特定的配置
        if strategy == 'long':
            base_config.update({
                'direction': 'long_only',
                'min_score': 0.0,  # Long策略只选择正分数的股票
            })
        elif strategy == 'short':
            base_config.update({
                'direction': 'short_only',
                'max_score': 0.0,  # Short策略只选择负分数的股票
            })

        return base_config

    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必需的文件是否存在
            if not self.predictions_long_path.exists():
                print(f"警告: Long预测文件不存在: {self.predictions_long_path}")

            if not self.predictions_short_path.exists():
                print(f"警告: Short预测文件不存在: {self.predictions_short_path}")

            # 验证参数范围
            assert 0 <= self.prediction_threshold <= 1, "预测阈值必须在[0,1]范围内"
            assert self.max_positions > 0, "最大持仓数量必须大于0"
            assert 0 < self.min_position_size <= 1, "最小仓位大小必须在(0,1]范围内"
            assert 0 < self.max_single_weight <= 1, "单个股票最大权重必须在(0,1]范围内"
            assert 0 < self.max_industry_weight <= 1, "行业最大权重必须在(0,1]范围内"

            return True

        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 预定义配置实例
DEFAULT_TALIB_CONFIG = TALIBModelConfig()

# 保守型配置（较低风险）
CONSERVATIVE_CONFIG = TALIBModelConfig(
    max_positions=30,
    max_single_weight=0.03,
    max_industry_weight=0.15,
    prediction_threshold=0.1,
    rebalance_threshold=0.05,
    transaction_cost_bps=8.0
)

# 激进型配置（较高风险）
AGGRESSIVE_CONFIG = TALIBModelConfig(
    max_positions=80,
    max_single_weight=0.08,
    max_industry_weight=0.35,
    prediction_threshold=-0.05,
    rebalance_threshold=0.01,
    transaction_cost_bps=3.0
)

# 高频交易配置
HIGH_FREQUENCY_CONFIG = TALIBModelConfig(
    max_positions=100,
    max_single_weight=0.02,
    rebalance_frequency="hourly",
    rebalance_threshold=0.005,
    transaction_cost_bps=2.0
)

def get_config_by_risk_profile(risk_profile: str) -> TALIBModelConfig:
    """
    根据风险偏好获取配置

    Args:
        risk_profile: 风险偏好 ('conservative', 'moderate', 'aggressive', 'high_frequency')

    Returns:
        TALIBModelConfig: 对应的配置
    """
    configs = {
        'conservative': CONSERVATIVE_CONFIG,
        'moderate': DEFAULT_TALIB_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG,
        'high_frequency': HIGH_FREQUENCY_CONFIG,
    }

    if risk_profile not in configs:
        raise ValueError(f"不支持的风险偏好: {risk_profile}. 支持的选项: {list(configs.keys())}")

    return configs[risk_profile]

def create_custom_config(**kwargs) -> TALIBModelConfig:
    """
    创建自定义配置

    Args:
        **kwargs: 配置参数

    Returns:
        TALIBModelConfig: 自定义配置
    """
    # 从默认配置开始
    config_dict = DEFAULT_TALIB_CONFIG.__dict__.copy()

    # 更新自定义参数
    for key, value in kwargs.items():
        if key in config_dict:
            config_dict[key] = value
        else:
            raise ValueError(f"不支持的配置参数: {key}")

    # 创建新配置
    custom_config = TALIBModelConfig(**config_dict)
    return custom_config

if __name__ == "__main__":
    # 测试配置
    print("测试TALIB模型配置...")

    # 测试默认配置
    default_config = DEFAULT_TALIB_CONFIG
    print(f"默认配置验证: {default_config.validate_config()}")

    # 测试不同风险偏好
    for profile in ['conservative', 'moderate', 'aggressive', 'high_frequency']:
        config = get_config_by_risk_profile(profile)
        print(f"{profile}配置: max_positions={config.max_positions}, max_single_weight={config.max_single_weight}")

    # 测试自定义配置
    custom_config = create_custom_config(max_positions=40, max_single_weight=0.04)
    print(f"自定义配置: max_positions={custom_config.max_positions}, max_single_weight={custom_config.max_single_weight}")

    print("配置测试完成！")