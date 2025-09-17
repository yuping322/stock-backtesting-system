"""
配置文件 - 通用股票回测系统
包含所有可配置参数和基准指数定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os

# =========================
# 基准指数配置
# =========================
BENCHMARK_INDICES = {
    "sh000300": {
        "name": "沪深300",
        "description": "沪深300指数",
        "code": "sh000300"
    },
    "sh000001": {
        "name": "上证指数", 
        "description": "上海证券交易所综合指数",
        "code": "sh000001"
    },
    "sz399001": {
        "name": "深证成指",
        "description": "深圳证券交易所成份指数", 
        "code": "sz399001"
    },
    "sz399006": {
        "name": "创业板指",
        "description": "创业板指数",
        "code": "sz399006"
    },
    "sh000905": {
        "name": "中证500",
        "description": "中证500指数",
        "code": "sh000905"
    },
    "sh000016": {
        "name": "上证50",
        "description": "上证50指数",
        "code": "sh000016"
    }
}

# =========================
# 策略评价指标配置
# =========================
PERFORMANCE_METRICS = {
    "total_return": {
        "name": "总收益率",
        "description": "回测期间总收益率",
        "format": "{:.2%}",
        "higher_is_better": True
    },
    "annual_return": {
        "name": "年化收益率", 
        "description": "年化收益率",
        "format": "{:.2%}",
        "higher_is_better": True
    },
    "max_drawdown": {
        "name": "最大回撤",
        "description": "最大回撤",
        "format": "{:.2%}",
        "higher_is_better": False
    },
    "sharpe_ratio": {
        "name": "夏普比率",
        "description": "夏普比率",
        "format": "{:.3f}",
        "higher_is_better": True
    },
    "volatility": {
        "name": "波动率",
        "description": "收益率波动率",
        "format": "{:.2%}",
        "higher_is_better": False
    },
    "win_rate": {
        "name": "胜率",
        "description": "盈利交易占比",
        "format": "{:.2%}",
        "higher_is_better": True
    },
    "profit_factor": {
        "name": "盈亏比",
        "description": "总盈利/总亏损",
        "format": "{:.3f}",
        "higher_is_better": True
    },
    "trade_count": {
        "name": "交易次数",
        "description": "总交易次数",
        "format": "{:d}",
        "higher_is_better": None
    }
}

# =========================
# 策略参数配置
# =========================
STRATEGY_PARAMS = {
    "weighted_top_n": {
        "name": "加权TopN策略",
        "description": "按权重选择TopN股票",
        "parameters": {
            "hold_days": {
                "name": "持仓天数",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 30,
                "step": 1
            },
            "top_n_stocks": {
                "name": "股票数量",
                "type": "int", 
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1
            },
        }
    },
    "equal_weight": {
        "name": "等权重策略",
        "description": "等权重配置股票",
        "parameters": {
            "hold_days": {
                "name": "持仓天数",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 30,
                "step": 1
            },
            "top_n_stocks": {
                "name": "股票数量",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1
            }
        }
    },
    "momentum": {
        "name": "动量策略",
        "description": "基于动量因子选股",
        "parameters": {
            "hold_days": {
                "name": "持仓天数",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1
            },
            "top_n_stocks": {
                "name": "股票数量",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1
            },
            "momentum_period": {
                "name": "动量周期",
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 60,
                "step": 5
            }
        }
    }
}

# =========================
# 主配置类
# =========================
@dataclass
class SystemConfig:
    """系统主配置类"""
    
    # 基本参数
    initial_cash: float = 1_000_000
    commission_rate: float = 0.0002
    slippage_rate: float = 0.0
    
    # 数据参数
    data_dir: str = "./data"  # 固定数据目录
    result_cache_dir: str = "./results"
    
    # 回测参数
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # 显示参数
    show_plots: bool = True
    save_results: bool = True
    plot_size: tuple = (12, 6)
    
    # 缓存参数
    enable_cache: bool = True
    cache_expiry_days: int = 7
    
    # 日志参数
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "backtest.log"
    
    # 基准指数
    benchmark_index: str = "sh000300"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "initial_cash": self.initial_cash,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "data_dir": self.data_dir,
            "result_cache_dir": self.result_cache_dir,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark_index": self.benchmark_index,
            "show_plots": self.show_plots,
            "save_results": self.save_results,
            "plot_size": self.plot_size,
            "enable_cache": self.enable_cache,
            "cache_expiry_days": self.cache_expiry_days,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file": self.log_file
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemConfig':
        """从字典创建配置"""
        return cls(**data)
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'SystemConfig':
        """从文件加载配置"""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

# =========================
# 策略特定配置
# =========================
@dataclass
class StrategyConfig:
    """策略配置类"""
    
    strategy_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后验证参数"""
        if self.strategy_name not in STRATEGY_PARAMS:
            raise ValueError(f"不支持的策略: {self.strategy_name}")
        
        # 设置默认值
        param_specs = STRATEGY_PARAMS[self.strategy_name]["parameters"]
        for param_name, spec in param_specs.items():
            if param_name not in self.parameters:
                self.parameters[param_name] = spec["default"]
    
    def validate_parameters(self) -> bool:
        """验证参数有效性"""
        param_specs = STRATEGY_PARAMS[self.strategy_name]["parameters"]
        
        for param_name, value in self.parameters.items():
            if param_name not in param_specs:
                return False
            
            spec = param_specs[param_name]
            if spec["type"] == "int":
                if not (spec["min"] <= value <= spec["max"]):
                    return False
            elif spec["type"] == "float":
                if not (spec["min"] <= value <= spec["max"]):
                    return False
        
        return True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "parameters": self.parameters
        }

# =========================
# 全局配置实例
# =========================
DEFAULT_CONFIG = SystemConfig()

# 配置文件路径
CONFIG_FILE_PATH = "/mnt/workspace/btcode/config/system_config.json"

# 加载或创建配置
def load_system_config() -> SystemConfig:
    """加载系统配置"""
    try:
        return SystemConfig.load_from_file(CONFIG_FILE_PATH)
    except:
        config = SystemConfig()
        os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
        config.save_to_file(CONFIG_FILE_PATH)
        return config

def save_system_config(config: SystemConfig):
    """保存系统配置"""
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    config.save_to_file(CONFIG_FILE_PATH)

# =========================
# 辅助函数
# =========================
def get_benchmark_info(index_code: str) -> Optional[Dict]:
    """获取基准指数信息"""
    return BENCHMARK_INDICES.get(index_code)

def get_strategy_info(strategy_name: str) -> Optional[Dict]:
    """获取策略信息"""
    return STRATEGY_PARAMS.get(strategy_name)

def get_metric_info(metric_name: str) -> Optional[Dict]:
    """获取指标信息"""
    return PERFORMANCE_METRICS.get(metric_name)

def list_benchmark_indices() -> List[str]:
    """获取所有基准指数代码"""
    return list(BENCHMARK_INDICES.keys())

def list_strategies() -> List[str]:
    """获取所有策略名称"""
    return list(STRATEGY_PARAMS.keys())

def list_metrics() -> List[str]:
    """获取所有指标名称"""
    return list(PERFORMANCE_METRICS.keys())