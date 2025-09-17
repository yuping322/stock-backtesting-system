
"""
集成版通用股票回测系统
结合generic_backtesting_system.py的真实回测引擎和enhanced_final_system.py的丰富分析功能
"""

import os
import json
import pandas as pd
import numpy as np
import backtrader as bt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体和样式 - 使用标准英文字体，避免中文乱码问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = True  # 正确显示负号
sns.set_style("whitegrid")

# 导入原始模块
from backtrader_base_strategy import JQ2BTBaseStrategy, get_akshare_etf_data, get_akshare_stock_data, get_index_nav, analyze_performance
from data import get_trading_dates, load_bt_stocks, get_valuation, get_index_daily,code2name

# =========================
# 辅助函数
# =========================
def get_stock_display(code: str) -> str:
    """获取股票代码和名称的组合显示"""
    if code2name and code in code2name:
        return f"{code} - {code2name[code]}"
    return code

# 导入配置模块
try:
    from config import (
        SystemConfig, StrategyConfig, DEFAULT_CONFIG,
        BENCHMARK_INDICES, STRATEGY_PARAMS, PERFORMANCE_METRICS,
        get_benchmark_info, get_strategy_info, get_metric_info,
        list_benchmark_indices, list_strategies, list_metrics
    )
    CONFIG_AVAILABLE = True
except Exception as e:
    print(f"配置模块加载失败: {e}")
    CONFIG_AVAILABLE = False

# =========================
# 数据加载备用函数
# =========================
def load_bt_stocks_fallback(
    codes: list,
    start: str,
    end: str,
    cache_dir: str = 'stock_cache'
) -> dict:
    """
    备用数据加载函数 - 使用AkShare获取股票数据
    当主数据加载失败时作为fallback使用
    """
    feeds = {}
    
    print(f"使用AkShare备用数据加载: {len(codes)}只股票, {start} 到 {end}")
    
    for code in codes:
        try:
            # 尝试使用AkShare获取数据
            print(f"  加载 {code} 数据...")
            data_feed = get_akshare_stock_data(code, start, end, cache_dir=cache_dir)
            
            # get_akshare_stock_data 已经返回 Backtrader PandasData 对象
            feeds[code] = data_feed
            print(f"    ✅ {code} 数据加载成功")
            
        except Exception as e:
            print(f"    ❌ {code} 数据加载失败: {e}")
            continue
    
    if not feeds:
        print("警告: 备用数据加载也未获取到任何股票数据！")
        
    return feeds

# =========================
# 策略基类
# =========================
class BaseStrategy(JQ2BTBaseStrategy):
    """策略基类"""
    
    params = (
        ("config", None),  # 配置对象
        ("pred_df", None),  # 预测数据
    )
    
    def __init__(self):
        super().__init__()
        if self.p.config is None:
            raise ValueError("必须传入 config 参数")
        
        self.config = self.p.config
        self.pred_df = self._prepare_pred_df()
        self.holdings = {}  # {code: buy_date}
        self._daily_navs = []
        self._daily_holdings = []  # 每日持仓记录
        self._trade_history = []   # 交易历史
        
    def _prepare_pred_df(self) -> pd.DataFrame:
        """预处理预测数据"""
        if self.p.pred_df is None:
            return pd.DataFrame(columns=["date", "code", "weight"])
        
        if isinstance(self.p.pred_df, str):
            df = pd.read_json(self.p.pred_df)
        else:
            df = self.p.pred_df.copy()
        
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    
    def _find_data(self, code: str):
        """根据股票代码找到对应的data feed"""
        for data in self.datas:
            if data._name == code:
                return data
        return None
    
    def _record_trade(self, code: str, action: str, size: int, price: float, value: float):
        """记录交易"""
        self._trade_history.append({
            'date': self.current_dt.date(),
            'code': code,
            'action': action,
            'size': size,
            'price': price,
            'value': value,
            'portfolio_value': self.broker.getvalue()
        })
    
    def _record_holdings(self):
        """记录当前持仓"""
        current_holdings = []
        total_value = self.broker.getvalue()
        
        for code, buy_date in self.holdings.items():
            data = self._find_data(code)
            if data is not None:
                position = self.broker.getposition(data)
                if position.size > 0:
                    current_holdings.append({
                        'code': code,
                        'size': position.size,
                        'price': data.close[0],
                        'value': position.size * data.close[0],
                        'weight': (position.size * data.close[0]) / total_value if total_value > 0 else 0,
                        'buy_date': buy_date
                    })
        
        self._daily_holdings.append({
            'date': self.current_dt.date(),
            'holdings': current_holdings,
            'total_value': total_value,
            'cash': self.broker.getcash()
        })
    
    def execute_strategy(self):
        """执行具体策略逻辑，子类必须实现"""
        raise NotImplementedError("子类必须实现 execute_strategy 方法")
    
    def next(self):
        """主循环"""
        super().next()
        self.execute_strategy()
        self._daily_navs.append(self.broker.getvalue())
        self._record_holdings()  # 记录每日持仓
    
    def stop(self):
        """回测结束时的处理"""
        super().stop()
        if hasattr(self.config, 'save_results') and self.config.save_results:
            today_str = datetime.now().strftime("%Y%m%d")
            nav_df = pd.DataFrame({"nav": self._daily_navs})
            nav_df.to_csv(f"daily_nav_{today_str}.csv", index=False, encoding="utf-8-sig")
            print(f"保存每日 NAV 到 daily_nav_{today_str}.csv")

# =========================
# 具体策略实现
# =========================
class WeightedTopNStrategy(BaseStrategy):
    """加权TopN策略"""
    
    def execute_strategy(self):
        """执行加权TopN策略"""
        dt = self.current_dt.date()
        ts = pd.Timestamp(dt)
        
        # 获取当日选股
        today_df = self.pred_df[self.pred_df["date"] == ts]
        if today_df.empty:
            return
        
        # 参数验证
        hold_days = max(1, self.config.hold_days)  # 持有天数至少为1
        
        # 1. 卖出到期持仓
        to_sell = []
        for code, buy_date in list(self.holdings.items()):
            if (dt - buy_date).days >= hold_days:
                data = self._find_data(code)
                if data is not None:
                    position = self.broker.getposition(data)
                    if position.size > 0:
                        self.order_target_value(data, 0)
                        self._record_trade(code, 'SELL', -position.size, data.close[0], position.size * data.close[0])
                        stock_name = code2name.get(code, '') if code2name else ''
                        self.log(f"SELL {code} {stock_name} (到期)", log_type='trade')
                to_sell.append(code)
        
        for code in to_sell:
            del self.holdings[code]
        
        # 2. 重新按权重配置
        total_weight = today_df[self.config.weight_column].sum()
        if total_weight <= 0:
            return
        
        total_value = self.broker.getvalue()
        for _, row in today_df.iterrows():
            code = str(row.code).zfill(6)
            weight = getattr(row, self.config.weight_column, 1.0)
            
            data = self._find_data(code)
            if data is None:
                continue
            
            target_value = total_value * (weight / total_weight)
            current_position = self.broker.getposition(data)
            current_value = current_position.size * data.close[0] if current_position.size > 0 else 0
            
            if abs(target_value - current_value) / total_value > 0.01:  # 1%阈值
                size_change = int((target_value - current_value) / data.close[0])
                self.order_target_value(data, target_value)
                action = 'BUY' if size_change > 0 else 'SELL'
                self._record_trade(code, action, size_change, data.close[0], abs(size_change) * data.close[0])
                self.holdings[code] = dt
                stock_name = code2name.get(code, '') if code2name else ''
                self.log(f"SET {code} {stock_name} target_value={target_value:.2f}", log_type='trade')

class EqualWeightStrategy(BaseStrategy):
    """等权重策略"""
    
    def execute_strategy(self):
        """执行等权重策略"""
        dt = self.current_dt.date()
        ts = pd.Timestamp(dt)
        
        # 获取当日选股
        today_df = self.pred_df[self.pred_df["date"] == ts]
        if today_df.empty:
            return
        
        # 参数验证 - 确保top_n_stocks有效
        top_n_stocks = max(1, self.config.top_n_stocks)  # 至少选择1只股票
        
        # 限制股票数量
        if len(today_df) > top_n_stocks:
            today_df = today_df.head(top_n_stocks)
        
        # 1. 卖出不在当前选股中的持仓
        current_codes = set(today_df["code"].astype(str).str.zfill(6))
        to_sell = []
        for code in self.holdings.keys():
            if code not in current_codes:
                data = self._find_data(code)
                if data is not None:
                    position = self.broker.getposition(data)
                    if position.size > 0:
                        self.order_target_value(data, 0)
                        self._record_trade(code, 'SELL', -position.size, data.close[0], position.size * data.close[0])
                        stock_name = code2name.get(code, '') if code2name else ''
                        self.log(f"SELL {code} {stock_name} (调出)", log_type='trade')
                to_sell.append(code)
        
        for code in to_sell:
            del self.holdings[code]
        
        # 2. 等权重买入新股票
        total_value = self.broker.getvalue()
        n_stocks = len(today_df)
        if n_stocks > 0:
            weight_per_stock = 1.0 / n_stocks
            target_value_per_stock = total_value * weight_per_stock
            
            for _, row in today_df.iterrows():
                code = str(row.code).zfill(6)
                data = self._find_data(code)
                if data is None:
                    continue
                
                # 计算需要买入的数量
                current_position = self.broker.getposition(data)
                current_value = current_position.size * data.close[0] if current_position.size > 0 else 0
                
                if abs(target_value_per_stock - current_value) / total_value > 0.01:  # 1%阈值
                    size_change = int((target_value_per_stock - current_value) / data.close[0])
                    if size_change > 0:
                        self.order_target_value(data, target_value_per_stock)
                        self._record_trade(code, 'BUY', size_change, data.close[0], size_change * data.close[0])
                        self.holdings[code] = dt
                        stock_name = code2name.get(code, '') if code2name else ''
                        self.log(f"BUY {code} {stock_name} equal_weight={weight_per_stock:.3f}", log_type='trade')

class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def execute_strategy(self):
        """执行动量策略"""
        dt = self.current_dt.date()
        ts = pd.Timestamp(dt)
        
        # 获取当日选股
        today_df = self.pred_df[self.pred_df["date"] == ts]
        if today_df.empty:
            return
        
        # 参数验证
        top_n_stocks = max(1, self.config.top_n_stocks)  # 至少选择1只股票
        momentum_period = max(1, self.config.parameters.get("momentum_period", 20))  # 动量周期至少为1
        
        # 按动量排序并选择TopN
        if "momentum" in today_df.columns:
            today_df = today_df.sort_values("momentum", ascending=False)
        else:
            # 如果没有动量列，使用收益率作为替代
            if "return" in today_df.columns:
                today_df = today_df.sort_values("return", ascending=False)
            else:
                # 默认使用第一列数值列作为排序依据
                numeric_cols = today_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    today_df = today_df.sort_values(numeric_cols[0], ascending=False)
        
        # 限制股票数量
        if len(today_df) > top_n_stocks:
            today_df = today_df.head(top_n_stocks)
        
        # 1. 卖出不在当前选股中的持仓
        current_codes = set(today_df["code"].astype(str).str.zfill(6))
        to_sell = []
        for code in self.holdings.keys():
            if code not in current_codes:
                data = self._find_data(code)
                if data is not None:
                    position = self.broker.getposition(data)
                    if position.size > 0:
                        self.order_target_value(data, 0)
                        self._record_trade(code, 'SELL', -position.size, data.close[0], position.size * data.close[0])
                        stock_name = code2name.get(code, '') if code2name else ''
                        self.log(f"SELL {code} {stock_name} (动量下降)", log_type='trade')
                to_sell.append(code)
        
        for code in to_sell:
            del self.holdings[code]
        
        # 2. 等权重买入新高股票
        total_value = self.broker.getvalue()
        n_stocks = len(today_df)
        if n_stocks > 0:
            weight_per_stock = 1.0 / n_stocks
            target_value_per_stock = total_value * weight_per_stock
            
            for _, row in today_df.iterrows():
                code = str(row.code).zfill(6)
                data = self._find_data(code)
                if data is None:
                    continue
                
                # 计算需要买入的数量
                current_position = self.broker.getposition(data)
                current_value = current_position.size * data.close[0] if current_position.size > 0 else 0
                
                if abs(target_value_per_stock - current_value) / total_value > 0.01:  # 1%阈值
                    size_change = int((target_value_per_stock - current_value) / data.close[0])
                    if size_change > 0:
                        self.order_target_value(data, target_value_per_stock)
                        self._record_trade(code, 'BUY', size_change, data.close[0], size_change * data.close[0])
                        self.holdings[code] = dt
                        stock_name = code2name.get(code, '') if code2name else ''
                        self.log(f"BUY {code} {stock_name} momentum_weight={weight_per_stock:.3f}", log_type='trade')

# =========================
# 策略工厂
# =========================
class StrategyFactory:
    """策略工厂类"""
    
    _strategies = {
        "weighted_top_n": WeightedTopNStrategy,
        "equal_weight": EqualWeightStrategy,
        "momentum": MomentumStrategy,
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> type:
        """获取策略类"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"不支持的策略: {strategy_name}. 可选策略: {list(cls._strategies.keys())}")
        return cls._strategies[strategy_name]
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """获取所有支持的策略名称"""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """注册新策略"""
        cls._strategies[name] = strategy_class

# =========================
# 数据加载器
# =========================
class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_prediction_data(file_path: str, config) -> pd.DataFrame:
        """加载预测数据"""
        try:
            # 处理空数据情况
            if isinstance(file_path, pd.DataFrame):
                df = file_path.copy()
            else:
                df = pd.read_csv(file_path)
            
            # 处理空数据框
            if df.empty:
                print("警告：加载的数据为空DataFrame")
                # 返回包含必需列的空DataFrame
                return pd.DataFrame(columns=["date", "code", "weight"])
            
            # 验证必需列存在
            required_columns = ["date", "code"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据缺少必需列: {missing_columns}")
            
            # 标准化列名
            if "code" in df.columns and len(df) > 0:
                df["code"] = df["code"].astype(str).str.zfill(6)
            if "date" in df.columns and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            
            # 确保有权重列
            if hasattr(config, 'weight_column') and config.weight_column not in df.columns:
                df[config.weight_column] = 1.0
            elif 'weight' not in df.columns:
                df['weight'] = 1.0
            
            # 数据清理 - 移除无效数据
            df = df.dropna(subset=['date', 'code'])
            
            # 验证权重列数据
            if 'weight' in df.columns and len(df) > 0:
                # 处理负权重
                df['weight'] = df['weight'].clip(lower=0)
                # 处理无穷大值
                df['weight'] = df['weight'].replace([np.inf, -np.inf], 1.0)
                # 处理NaN值
                df['weight'] = df['weight'].fillna(1.0)
            
            return df
        except Exception as e:
            print(f"加载预测数据失败: {e}")
            # 返回包含必需列的空DataFrame
            return pd.DataFrame(columns=["date", "code", "weight"])
    
    @staticmethod
    def scan_prediction_files(data_dir: str) -> List[str]:
        """扫描预测文件"""
        if not os.path.exists(data_dir):
            return []
        
        return [f for f in os.listdir(data_dir) 
                if f.lower().endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))]

# =========================
# 增强版回测引擎
# =========================
class EnhancedBacktestEngine:
    """增强版回测引擎 - 集成真实回测和丰富分析"""
    
    def __init__(self, system_config: SystemConfig, strategy_config: StrategyConfig):
        self.system_config = system_config
        self.strategy_config = strategy_config
        self.results = {}
        self.trade_history = {}
        self.daily_holdings = {}
    
    def generate_cache_key(self, file_name: str, strategy_name: str, params: dict) -> str:
        """生成缓存键"""
        param_str = f"{file_name}_{strategy_name}_{str(sorted(params.items()))}"
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def get_strategy_params(self) -> Dict:
        """获取策略参数"""
        params = {
            'hold_days': self.strategy_config.parameters.get('hold_days', 2),
            'top_n_stocks': self.strategy_config.parameters.get('top_n_stocks', 10),
            'commission_rate': self.system_config.commission_rate,
            'weight_column': getattr(self.system_config, 'weight_column', 'weight')
        }
        
        # 添加策略特定参数
        for param_name, param_value in self.strategy_config.parameters.items():
            if param_name not in params:
                params[param_name] = param_value
        
        return params
    
    def calculate_detailed_metrics(self, strategy_nav: pd.Series, benchmark_nav: pd.Series, returns: np.ndarray) -> Dict:
        """计算详细的性能指标"""
        try:
            # 基本参数验证
            if len(strategy_nav) == 0 or len(benchmark_nav) == 0 or len(returns) == 0:
                print("警告：策略净值、基准净值或收益数据为空，返回空指标")
                return self._create_empty_metrics()
            
            # 验证输入数据的有效性
            if strategy_nav.isna().all() or benchmark_nav.isna().all():
                print("警告：策略净值或基准净值全部为NaN，返回空指标")
                return self._create_empty_metrics()
            
            if np.isnan(returns).all():
                print("警告：收益数据全部为NaN，返回空指标")
                return self._create_empty_metrics()
            
            # 确保长度一致性
            min_length = min(len(strategy_nav), len(benchmark_nav))
            if len(returns) != min_length - 1:  # returns应该比nav少一个（pct_change的结果）
                # 重新计算收益率以确保一致性
                try:
                    returns = strategy_nav.pct_change().dropna().values
                except Exception as e:
                    print(f"重新计算收益率失败: {e}，使用原数据")
                    # 如果重新计算失败，确保长度匹配
                    if len(returns) >= min_length:
                        returns = returns[:min_length-1]
            
            # 基本指标 - 修复总收益率计算
            # 检查strategy_nav是绝对金额还是累计收益率
            if strategy_nav.iloc[0] > 1000:  # 假设是绝对金额（如100万起始资金）
                # 从绝对金额计算收益率
                initial_capital = strategy_nav.iloc[0]
                final_capital = strategy_nav.iloc[-1]
                total_return = (final_capital - initial_capital) / initial_capital
            else:
                # 已经是累计收益率格式（从1.0开始）
                total_return = strategy_nav.iloc[-1] - 1
            
            # 修正年化收益率计算 - 使用实际的交易日数量
            actual_trading_days = len(strategy_nav) - 1  # 减1因为navs包含起始点
            if actual_trading_days > 0:
                annual_return = (1 + total_return) ** (252 / actual_trading_days) - 1
            else:
                annual_return = 0
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = annual_return / volatility if volatility > 0 and not np.isnan(volatility) and not np.isinf(volatility) else 0
            
            # 最大回撤
            running_max = strategy_nav.cummax()
            drawdown = (strategy_nav - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 恢复时间
            recovery_times = self.calculate_recovery_times(drawdown)
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
            
            # 基准相关指标 - 添加有效性检查
            try:
                benchmark_total_return = benchmark_nav.iloc[-1] - 1
                benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(benchmark_nav)) - 1 if len(benchmark_nav) > 0 else 0
            except (IndexError, ZeroDivisionError, OverflowError) as e:
                print(f"基准指标计算失败: {e}")
                benchmark_total_return = 0
                benchmark_annual_return = 0
            
            # Alpha和Beta - 确保长度匹配
            try:
                benchmark_returns = (benchmark_nav / benchmark_nav.iloc[0] - 1).values
            except Exception as e:
                print(f"基准收益率计算失败: {e}，使用零值")
                benchmark_returns = np.zeros(len(benchmark_nav))
            if len(benchmark_returns) != len(returns):
                min_len = min(len(returns), len(benchmark_returns))
                if min_len > 0:
                    alpha, beta = self.calculate_alpha_beta(returns[:min_len], benchmark_returns[:min_len])
                else:
                    alpha, beta = 0, 0
            else:
                alpha, beta = self.calculate_alpha_beta(returns, benchmark_returns)
            
            # 信息比率 - 确保长度匹配并添加错误处理
            try:
                if len(benchmark_returns) != len(returns):
                    min_len = min(len(returns), len(benchmark_returns))
                    if min_len > 0:
                        tracking_error = np.std(returns[:min_len] - benchmark_returns[:min_len]) * np.sqrt(252)
                    else:
                        tracking_error = 0
                else:
                    tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
                    
                # 确保跟踪误差是有限值
                if np.isnan(tracking_error) or np.isinf(tracking_error) or tracking_error <= 0:
                    information_ratio = 0
                else:
                    information_ratio = (annual_return - benchmark_annual_return) / tracking_error
                    # 确保信息比率是有限值
                    if np.isnan(information_ratio) or np.isinf(information_ratio):
                        information_ratio = 0
            except Exception as e:
                print(f"信息比率计算失败: {e}")
                tracking_error = 0
                information_ratio = 0
            
            # 月度胜率 - 使用正确的索引
            try:
                # 创建与收益率匹配的日期索引
                if len(strategy_nav) > 1:
                    returns_index = strategy_nav.index[1:]  # 跳过第一个日期（pct_change产生的NaN）
                    if len(returns_index) == len(returns):
                        returns_series = pd.Series(returns, index=returns_index)
                        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
                        monthly_win_rate = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
                    else:
                        # 如果索引不匹配，使用简单的月度计算
                        monthly_win_rate = (returns > 0).mean()
                        monthly_returns = pd.Series(dtype=float)
                else:
                    monthly_win_rate = (returns > 0).mean() if len(returns) > 0 else 0
                    monthly_returns = pd.Series(dtype=float)
                    
            except Exception as e:
                # 如果resample失败，使用简单计算
                monthly_win_rate = (returns > 0).mean() if len(returns) > 0 else 0
                monthly_returns = pd.Series(dtype=float)
            
            # 连续盈利/亏损
            consecutive_profits, consecutive_losses = self.calculate_consecutive_returns(returns)
            
            # VaR和CVaR - 添加错误处理
            try:
                var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
                # 确保var_95是有限值
                if np.isnan(var_95) or np.isinf(var_95):
                    var_95 = 0
                    cvar_95 = 0
                else:
                    # 计算CVaR时检查是否有足够的样本
                    tail_returns = returns[returns <= var_95]
                    if len(tail_returns) > 0:
                        cvar_95 = tail_returns.mean()
                    else:
                        cvar_95 = var_95
                    
                    # 确保cvar_95是有限值
                    if np.isnan(cvar_95) or np.isinf(cvar_95):
                        cvar_95 = var_95
            except Exception as e:
                print(f"VaR/CVaR计算失败: {e}")
                var_95 = 0
                cvar_95 = 0
            
            # 创建性能指标DataFrame - 修复除零错误和无穷大值
            # 计算胜率
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0
            
            # 计算盈利因子 - 防止除零和无穷大
            if (returns < 0).any() and len(returns[returns < 0]) > 0:
                positive_returns = returns[returns > 0]
                negative_returns = returns[returns < 0]
                if len(positive_returns) > 0 and len(negative_returns) > 0:
                    # 使用绝对值避免负值影响
                    profit_factor = abs(positive_returns.mean() / negative_returns.mean())
                else:
                    profit_factor = 999.0  # 只有盈利或只有亏损的情况
            else:
                profit_factor = 999.0  # 没有亏损交易的情况
            
            # 确保所有数值都是有限值
            def safe_float(value):
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return float(value)
            
            performance_data = {
                'total_return': safe_float(total_return),
                'annual_return': safe_float(annual_return),
                'max_drawdown': safe_float(max_drawdown),
                'sharpe_ratio': safe_float(sharpe_ratio),
                'volatility': safe_float(volatility),
                'win_rate': safe_float(win_rate),
                'profit_factor': safe_float(profit_factor),
                'trade_count': len(returns),
                'alpha': safe_float(alpha),
                'beta': safe_float(beta),
                'information_ratio': safe_float(information_ratio),
                'tracking_error': safe_float(tracking_error),
                'monthly_win_rate': safe_float(monthly_win_rate),
                'avg_recovery_time': safe_float(avg_recovery_time),
                'max_consecutive_profit': max(consecutive_profits) if consecutive_profits else 0,
                'max_consecutive_loss': max(consecutive_losses) if consecutive_losses else 0,
                'var_95': safe_float(var_95),
                'cvar_95': safe_float(cvar_95),
                'calmar_ratio': safe_float(annual_return / abs(max_drawdown)) if max_drawdown != 0 and not np.isnan(max_drawdown) and not np.isinf(max_drawdown) else 0
            }
            
            performance_df = pd.DataFrame(list(performance_data.items()), columns=['metric', 'value'])
            performance_df = performance_df.set_index('metric')
            
            detailed_metrics = {
                'drawdown_series': drawdown,
                'running_max': running_max,
                'monthly_returns': monthly_returns,
                'consecutive_profits': consecutive_profits,
                'consecutive_losses': consecutive_losses
            }
            
            return {
                'performance_df': performance_df,
                'detailed_metrics': detailed_metrics
            }
            
        except Exception as e:
            # 如果计算过程中出现任何错误，返回空指标
            print(f"计算详细指标时出错: {e}")
            return self._create_empty_metrics()
    
    def _create_empty_metrics(self) -> Dict:
        """创建空指标（当数据不足时）"""
        performance_data = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0,
            'alpha': 0.0,
            'beta': 0.0,
            'information_ratio': 0.0,
            'tracking_error': 0.0,
            'monthly_win_rate': 0.0,
            'avg_recovery_time': 0.0,
            'max_consecutive_profit': 0,
            'max_consecutive_loss': 0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'calmar_ratio': 0.0
        }
        
        performance_df = pd.DataFrame(list(performance_data.items()), columns=['metric', 'value'])
        performance_df = performance_df.set_index('metric')
        
        detailed_metrics = {
            'drawdown_series': pd.Series(dtype=float),
            'running_max': pd.Series(dtype=float),
            'monthly_returns': pd.Series(dtype=float),
            'consecutive_profits': [],
            'consecutive_losses': []
        }
        
        return {
            'performance_df': performance_df,
            'detailed_metrics': detailed_metrics
        }
    
    def calculate_recovery_times(self, drawdown: pd.Series) -> List[int]:
        """计算恢复时间"""
        recovery_times = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.01 and not in_drawdown:  # 开始回撤（1%以上）
                in_drawdown = True
                start_date = date
            elif dd >= 0 and in_drawdown:  # 回撤结束
                in_drawdown = False
                if start_date:
                    recovery_time = (date - start_date).days
                    recovery_times.append(recovery_time)
        
        return recovery_times
    
    def calculate_alpha_beta(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        # 简单线性回归计算Beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
        
        return alpha, beta
    
    def calculate_consecutive_returns(self, returns: np.ndarray) -> Tuple[List[int], List[int]]:
        """计算连续盈利/亏损次数"""
        profits = []
        losses = []
        current_profit = 0
        current_loss = 0
        
        for ret in returns:
            if ret > 0:
                current_profit += 1
                if current_loss > 0:
                    losses.append(current_loss)
                    current_loss = 0
            else:
                current_loss += 1
                if current_profit > 0:
                    profits.append(current_profit)
                    current_profit = 0
        
        # 添加最后的序列
        if current_profit > 0:
            profits.append(current_profit)
        if current_loss > 0:
            losses.append(current_loss)
        
        return profits, losses
    
    def generate_period_stats(self, strategy_nav: pd.Series, benchmark_nav: pd.Series, period: str) -> pd.DataFrame:
        """生成期间统计（月度/年度）"""
        try:
            # 确保索引是DatetimeIndex
            if not isinstance(strategy_nav.index, pd.DatetimeIndex):
                strategy_nav.index = pd.to_datetime(strategy_nav.index)
            if not isinstance(benchmark_nav.index, pd.DatetimeIndex):
                benchmark_nav.index = pd.to_datetime(benchmark_nav.index)
            
            # 确保两个Series的索引对齐
            if not strategy_nav.index.equals(benchmark_nav.index):
                # 重新索引以对齐数据
                common_index = strategy_nav.index.intersection(benchmark_nav.index)
                if len(common_index) == 0:
                    # 如果没有共同索引，返回空DataFrame
                    return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])
                
                strategy_nav = strategy_nav.reindex(common_index)
                benchmark_nav = benchmark_nav.reindex(common_index)
            
            # 检查数据是否足够进行统计
            if len(strategy_nav) < 2:
                return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])
            
            # 使用更健壮的期间计算方法
            try:
                # 方法1: 尝试使用resample（适用于较长的时间序列）
                if period == 'M':
                    freq = 'ME'  # 使用新的频率表示法
                elif period == 'Y':
                    freq = 'YE'
                else:
                    freq = period
                
                strategy_returns = strategy_nav.resample(freq).last().pct_change().dropna()
                benchmark_returns = benchmark_nav.resample(freq).last().pct_change().dropna()
                
                # 如果resample结果为空，尝试方法2
                if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
                    raise ValueError("Resample结果为空")
                    
            except Exception as resample_error:
                # 方法2: 使用手动分组（适用于短时间序列）
                if period == 'M':
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.to_period('M')).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.to_period('M')).last()
                elif period == 'Y':
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.to_period('Y')).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.to_period('Y')).last()
                else:
                    # 对于其他周期，直接使用简单分组
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.date).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.date).last()
                
                strategy_returns = strategy_groups.pct_change().dropna()
                benchmark_returns = benchmark_groups.pct_change().dropna()
            
            # 确保两个收益率序列对齐
            common_returns_index = strategy_returns.index.intersection(benchmark_returns.index)
            if len(common_returns_index) == 0:
                # 如果没有共同索引，尝试按位置对齐
                min_len = min(len(strategy_returns), len(benchmark_returns))
                if min_len > 0:
                    # 创建新的对齐索引
                    aligned_index = pd.date_range(
                        start=strategy_returns.index[0] if hasattr(strategy_returns.index[0], 'date') else strategy_nav.index[0],
                        periods=min_len,
                        freq='D'
                    )
                    strategy_returns = strategy_returns.iloc[:min_len]
                    strategy_returns.index = aligned_index
                    benchmark_returns = benchmark_returns.iloc[:min_len] 
                    benchmark_returns.index = aligned_index
                else:
                    return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])
            
            stats = pd.DataFrame({
                'Strategy_Return': strategy_returns,
                'Benchmark_Return': benchmark_returns,
                'Excess_Return': strategy_returns - benchmark_returns,
                'Win': strategy_returns > benchmark_returns
            })
            
            return stats
            
        except Exception as e:
            # 如果所有方法都失败，返回空DataFrame
            print(f"生成期间统计时出错: {e}")
            return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])
    
    def run_single_backtest(self, pred_df: pd.DataFrame, strategy_name: str, file_name: str = "") -> Dict:
        """运行单个回测，包含详细的持仓和交易记录"""
        try:
            # 输入验证
            if pred_df is None or pred_df.empty:
                raise ValueError("预测数据为空，无法进行回测")
            
            if 'code' not in pred_df.columns or 'date' not in pred_df.columns:
                raise ValueError("预测数据缺少必需的列：code 和 date")
            
            # 获取策略类
            strategy_class = StrategyFactory.get_strategy(strategy_name)
            
            # 提取股票代码并去重
            stock_codes = pred_df["code"].unique().tolist()
            if not stock_codes:
                raise ValueError("没有有效的股票代码")
            
            # 日期范围验证
            if pred_df["date"].min() is pd.NaT or pred_df["date"].max() is pd.NaT:
                raise ValueError("日期数据无效")
                
            start_date = pred_df["date"].min().strftime("%Y-%m-%d")
            end_date = pred_df["date"].max().strftime("%Y-%m-%d")
            
            # 验证日期范围合理性
            from datetime import datetime
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if start_dt >= end_dt:
                    raise ValueError("开始日期必须早于结束日期")
            except ValueError as ve:
                raise ValueError(f"日期格式错误: {ve}")
            
            print(f"开始回测: {strategy_name} 策略, 数据文件: {file_name}")
            print(f"股票数量: {len(stock_codes)}, 日期范围: {start_date} 到 {end_date}")
            
            # 初始化回测引擎
            cerebro = bt.Cerebro()
            cerebro.broker.setcash(self.system_config.initial_cash)
            cerebro.broker.setcommission(commission=self.system_config.commission_rate)
            cerebro.broker.set_slippage_perc(getattr(self.system_config, 'slippage_rate', 0.0))
            
            # 加载数据 - 先尝试主数据源，失败时使用AkShare备用数据源
            feeds = load_bt_stocks(stock_codes, start_date, end_date)
            
            # 如果主数据源没有获取到任何数据，使用AkShare备用数据源
            if not feeds:
                print("主数据源未获取到数据，尝试使用AkShare备用数据源...")
                feeds = load_bt_stocks_fallback(stock_codes, start_date, end_date)
            
            valid_stocks = []
            
            for code, data in feeds.items():
                cerebro.adddata(data, name=code)
                valid_stocks.append(code)
            
            # 过滤有效数据
            pred_df = pred_df[pred_df["code"].isin(valid_stocks)].copy()
            
            # 检查是否还有有效数据
            if pred_df.empty:
                raise ValueError("没有可用的股票数据，无法进行回测。请检查：1) 股票代码是否正确 2) 数据服务是否可用 3) 日期范围是否合适")
            
            if not valid_stocks:
                raise ValueError("没有成功加载任何股票数据，无法进行回测")
            
            print(f"成功加载 {len(valid_stocks)} 只股票数据: {valid_stocks}")
            
            # 创建策略配置对象
            from types import SimpleNamespace
            strategy_params = self.get_strategy_params()
            
            # 参数验证和清理
            if 'hold_days' in strategy_params:
                strategy_params['hold_days'] = max(1, int(strategy_params['hold_days']))
            if 'top_n_stocks' in strategy_params:
                strategy_params['top_n_stocks'] = max(1, int(strategy_params['top_n_stocks']))
            
            # 创建符合策略期望的配置结构
            config_obj = SimpleNamespace(
                parameters=strategy_params,  # 策略期望的是config.parameters结构
                hold_days=strategy_params.get('hold_days', 2),
                top_n_stocks=strategy_params.get('top_n_stocks', 10),
                commission_rate=strategy_params.get('commission_rate', 0.0002),
                weight_column=strategy_params.get('weight_column', 'weight')
            )
            
            # 添加其他参数到parameters中
            for key, value in strategy_params.items():
                if not hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            
            # 添加策略
            cerebro.addstrategy(
                strategy_class,
                config=config_obj,
                pred_df=pred_df
            )
            
            # 运行回测
            results = cerebro.run()
            if not results or len(results) == 0:
                raise ValueError("回测运行失败，没有产生结果")
                
            strat = results[0]
            final_value = cerebro.broker.getvalue()
            
            # 验证策略对象
            if not hasattr(strat, 'navs') or not strat.navs:
                raise ValueError("回测没有产生净值数据")
            
            # 构建结果 - 修复数据对齐问题
            try:
                # 获取策略运行期间的所有日期
                if hasattr(strat, 'datas') and len(strat.datas) > 0:
                    data = strat.datas[0]
                    length = data.buflen()
                    dates = []
                    for i in range(length):
                        try:
                            date = data.datetime.date(-i)
                            dates.append(date)
                        except (IndexError, AttributeError):
                            break
                    dates.reverse()
                else:
                    # 如果没有datas，使用navs的长度创建默认日期
                    dates = pd.date_range(start=start_date, periods=len(strat.navs), freq='B')
                    dates = [d.date() for d in dates]
                
                # 确保navs数据和日期长度匹配
                nav_length = len(strat.navs)
                date_length = len(dates)
                
                if nav_length != date_length:
                    print(f"警告: navs长度({nav_length})与日期长度({date_length})不匹配，进行对齐处理")
                    # 取较短的长度，确保数据一致性
                    min_length = min(nav_length, date_length)
                    navs_data = strat.navs[:min_length]
                    aligned_dates = dates[:min_length]
                else:
                    navs_data = strat.navs
                    aligned_dates = dates
                    
            except Exception as e:
                print(f"数据对齐处理失败: {e}，使用默认处理")
                # 使用navs的长度和默认日期序列
                navs_data = strat.navs
                aligned_dates = pd.date_range(start=start_date, periods=len(navs_data), freq='B')
                aligned_dates = [d.date() for d in aligned_dates]
            
            # 创建DatetimeIndex而不是普通Index
            date_index = pd.to_datetime(aligned_dates)
            strategy_nav = pd.Series(navs_data, index=date_index)
            
            # 获取基准数据
            benchmark_nav = get_index_daily(self.system_config.benchmark_index, start_date, end_date)
            benchmark_nav = benchmark_nav.reindex(strategy_nav.index).fillna(method='ffill')
            
            # 计算日收益率
            strategy_returns = strategy_nav.pct_change().dropna().values
            
            # 计算详细性能指标
            detailed_results = self.calculate_detailed_metrics(strategy_nav, benchmark_nav, strategy_returns)
            
            # 生成期间统计
            monthly_stats = self.generate_period_stats(strategy_nav, benchmark_nav, 'M')
            yearly_stats = self.generate_period_stats(strategy_nav, benchmark_nav, 'Y')
            
            # 获取交易历史和持仓记录
            trade_history = getattr(strat, '_trade_history', [])
            daily_holdings = getattr(strat, '_daily_holdings', [])
            
            final_result = {
                "strategy_nav": strategy_nav,
                "benchmark_nav": benchmark_nav,
                "performance": detailed_results['performance_df'],
                "detailed_metrics": detailed_results['detailed_metrics'],
                "final_value": final_value,
                "valid_stocks": len(valid_stocks),
                "dates": aligned_dates,
                "daily_holdings": daily_holdings,
                "trade_history": trade_history,
                "monthly_stats": monthly_stats,
                "yearly_stats": yearly_stats,
                "daily_returns": strategy_returns,
                "benchmark_returns": benchmark_nav.pct_change().dropna().values
            }
            
            return final_result
            
        except Exception as e:
            print(f"回测失败: {e}")
            # 提供更有用的错误信息
            error_msg = str(e)
            if "没有可用的股票数据" in error_msg:
                raise ValueError("回测失败：没有可用的股票数据。请检查：1) 股票代码是否正确；2) 数据服务是否可用；3) 日期范围是否合适")
            elif "没有成功加载任何股票数据" in error_msg:
                raise ValueError("回测失败：无法加载任何股票的历史数据。请检查网络连接和数据源")
            else:
                raise ValueError(f"回测执行失败: {error_msg}")

# =========================
# 增强版Streamlit应用
# =========================
def create_enhanced_streamlit_app():
    """创建增强版Streamlit应用"""
    
    st.set_page_config(
        page_title="集成版通用股票回测系统", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 集成版通用股票回测系统")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化配置
    if CONFIG_AVAILABLE:
        system_config = SystemConfig()
    else:
        system_config = type('MockConfig', (), {
            'data_dir': '/mnt/workspace/btcode/data',
            'initial_cash': 1000000,
            'commission_rate': 0.0002,
            'slippage_rate': 0.0,
            'benchmark_index': 'sh000300',
            'show_plots': True,
            'save_results': True,
            'plot_size': (12, 6),
            'enable_cache': True,
            'weight_column': 'weight'
        })()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 基本参数
        # 数据目录已隐藏，使用默认值
        data_dir = getattr(system_config, 'data_dir', '/mnt/workspace/btcode/data')
        # data_dir = st.text_input("数据目录", value=getattr(system_config, 'data_dir', '/mnt/workspace/btcode/data'))  # 已隐藏
        initial_cash = st.number_input("初始资金(元)", min_value=10000, value=int(system_config.initial_cash), step=10000)
        
        # 基准指数选择
        if CONFIG_AVAILABLE:
            benchmark_options = list_benchmark_indices()
            benchmark_names = [f"{code} - {BENCHMARK_INDICES[code]['name']}" for code in benchmark_options]
            selected_benchmark = st.selectbox("基准指数", benchmark_names)
            benchmark_index = selected_benchmark.split(" - ")[0]
        else:
            benchmark_options = ['sh000300', 'sh000001', 'sz399001']
            benchmark_index = st.selectbox("基准指数", benchmark_options)
        
        # 策略选择
        if CONFIG_AVAILABLE:
            strategy_options = list_strategies()
            strategy_names = [f"{strategy} - {STRATEGY_PARAMS[strategy]['name']}" for strategy in strategy_options]
            selected_strategy = st.selectbox("选择策略", strategy_names)
            strategy_name = selected_strategy.split(" - ")[0]
            
            # 获取策略参数配置
            strategy_info = get_strategy_info(strategy_name)
            strategy_params = {}
            
            if strategy_info:
                st.subheader("策略参数")
                for param_name, param_spec in strategy_info["parameters"].items():
                    if param_spec["type"] == "int":
                        strategy_params[param_name] = st.number_input(
                            param_spec["name"], 
                            min_value=param_spec["min"],
                            max_value=param_spec["max"],
                            value=param_spec["default"],
                            step=param_spec.get("step", 1)
                        )
                    elif param_spec["type"] == "float":
                        strategy_params[param_name] = st.number_input(
                            param_spec["name"],
                            min_value=param_spec["min"], 
                            max_value=param_spec["max"],
                            value=param_spec["default"],
                            format="%.4f"
                        )
                    elif param_spec["type"] == "str":
                        strategy_params[param_name] = st.text_input(
                            param_spec["name"],
                            value=param_spec["default"]
                        )
        else:
            strategy_name = st.selectbox("选择策略", ["weighted_top_n", "equal_weight", "momentum"])
            strategy_params = {
                "hold_days": st.number_input("持有天数", min_value=1, max_value=30, value=2),
                "top_n_stocks": st.number_input("股票数量", min_value=1, max_value=50, value=10)
            }
        
        # 高级参数
        with st.expander("高级参数"):
            commission_rate = st.number_input("手续费率", min_value=0.0, max_value=0.01, value=system_config.commission_rate, format="%.4f")
            slippage_rate = st.number_input("滑点率", min_value=0.0, max_value=0.01, value=getattr(system_config, 'slippage_rate', 0.0), format="%.4f")
            
            col1, col2 = st.columns(2)
            with col1:
                show_plots = st.checkbox("显示图表", value=system_config.show_plots)
            with col2:
                save_results = st.checkbox("保存结果", value=system_config.save_results)
        
        # 分析选项
        with st.expander("分析选项"):
            st.subheader("显示内容")
            show_daily_holdings = st.checkbox("显示每日持仓", value=True)
            show_trade_history = st.checkbox("显示交易历史", value=True)
            show_period_stats = st.checkbox("显示期间统计", value=True)
            show_risk_metrics = st.checkbox("显示风险指标", value=True)
            
            st.subheader("性能指标")
            if CONFIG_AVAILABLE:
                all_metrics = list_metrics()
                selected_metrics = st.multiselect(
                    "选择显示的指标",
                    all_metrics,
                    default=['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
                )
            else:
                selected_metrics = ["total_return", "annual_return", "max_drawdown", "sharpe_ratio"]
        
        # 文件选择
        available_files = DataLoader.scan_prediction_files(data_dir)
        selected_files = st.multiselect(
            "选择回测文件",
            available_files,
            default=available_files[:1] if available_files else []
        )
    
    # 主界面
    if st.sidebar.button("开始回测") and selected_files:
        
        # 创建系统配置
        system_config = SystemConfig(
            data_dir=data_dir,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            show_plots=show_plots,
            save_results=save_results
        )
        # 设置基准指数
        system_config.benchmark_index = benchmark_index
        
        # 创建策略配置
        strategy_config = StrategyConfig(
            strategy_name=strategy_name,
            parameters=strategy_params
        )
        
        # 创建回测引擎
        engine = EnhancedBacktestEngine(system_config, strategy_config)
        
        # 运行回测
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        for i, file_name in enumerate(selected_files):
            status_text.text(f"正在回测: {file_name} ({i+1}/{len(selected_files)})")
            
            try:
                file_path = os.path.join(data_dir, file_name)
                pred_df = DataLoader.load_prediction_data(file_path, system_config)
                result = engine.run_single_backtest(pred_df, strategy_name, file_name)
                results[file_name] = result
                
            except Exception as e:
                st.error(f"回测 {file_name} 失败: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(selected_files))
        
        status_text.text("回测完成！")
        
        # 展示结果
        if results:
            st.header("📊 综合分析结果")
            
            # 1. 结果摘要
            st.subheader("🎯 结果摘要")
            summary_data = []
            for file_name, result in results.items():
                final_value = result['final_value']
                return_rate = (final_value - system_config.initial_cash) / system_config.initial_cash
                
                # 处理性能指标
                perf_df = result['performance']
                annual_return = perf_df.loc['annual_return', 'value']
                max_drawdown = perf_df.loc['max_drawdown', 'value']
                sharpe_ratio = perf_df.loc['sharpe_ratio', 'value']
                
                summary_data.append({
                    '文件名': file_name,
                    '策略': strategy_name,
                    '初始资金': f"{system_config.initial_cash:,.0f}",
                    '最终资金': f"{final_value:,.0f}",
                    '总收益率': f"{return_rate:.2%}",
                    '年化收益率': f"{annual_return:.2%}",
                    '最大回撤': f"{max_drawdown:.2%}",
                    '夏普比率': f"{sharpe_ratio:.3f}",
                    '有效股票数': result['valid_stocks']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # 2. 详细结果展示
            tabs = st.tabs(list(results.keys()))
            
            for tab, (file_name, result) in zip(tabs, results.items()):
                with tab:
                    st.subheader(f"📈 详细分析: {file_name}")
                    
                    # 获取策略信息
                    if CONFIG_AVAILABLE:
                        strategy_info = get_strategy_info(strategy_name)
                        if strategy_info:
                            st.info(f"策略: {strategy_info['name']} - {strategy_info['description']}")
                    
                    # 创建子标签页
                    detail_tabs = st.tabs([
                        "📋 概览", 
                        "📊 净值曲线", 
                        "📈 收益分析", 
                        "⚠️ 风险分析",
                        "📅 期间统计",
                        "💼 持仓分析",
                        "🔄 交易记录"
                    ])
                    
                    # 概览标签
                    with detail_tabs[0]:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("初始资金", f"{system_config.initial_cash:,.0f}")
                        with col2:
                            final_value = result['final_value']
                            return_rate = (final_value - system_config.initial_cash) / system_config.initial_cash
                            st.metric("最终资金", f"{final_value:,.0f}", f"{return_rate:.2%}")
                        with col3:
                            st.metric("有效股票数", result['valid_stocks'])
                        with col4:
                            if CONFIG_AVAILABLE:
                                benchmark_info = get_benchmark_info(system_config.benchmark_index)
                                st.metric("基准指数", benchmark_info['name'] if benchmark_info else system_config.benchmark_index)
                            else:
                                st.metric("基准指数", system_config.benchmark_index)
                        
                        # 性能指标网格
                        st.subheader("关键性能指标")
                        perf_df = result['performance']
                        
                        if selected_metrics:
                            display_metrics = [m for m in selected_metrics if m in perf_df.index]
                            display_df = perf_df.loc[display_metrics]
                        else:
                            display_df = perf_df
                        
                        # 创建指标卡片
                        cols = st.columns(min(len(display_df), 4))
                        for i, metric_name in enumerate(display_df.index):
                            with cols[i % 4]:
                                if CONFIG_AVAILABLE:
                                    metric_info = get_metric_info(metric_name)
                                    if metric_info:
                                        value = display_df.loc[metric_name, 'value']
                                        formatted_value = metric_info["format"].format(value)
                                        st.metric(metric_info["name"], formatted_value, metric_info["description"])
                                else:
                                    value = display_df.loc[metric_name, 'value']
                                    st.metric(metric_name, f"{value:.3f}")
                    
                    # 净值曲线标签
                    with detail_tabs[1]:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 净值对比图
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
                            
                            strategy_nav = result['strategy_nav']
                            benchmark_nav = result['benchmark_nav']
                            
                            # 净值曲线
                            strategy_cumret = strategy_nav / strategy_nav.iloc[0]
                            benchmark_cumret = benchmark_nav / benchmark_nav.iloc[0]
                            
                            ax1.plot(strategy_cumret.index, strategy_cumret.values, label='Strategy NAV', linewidth=2, color='blue')
                            ax1.plot(benchmark_cumret.index, benchmark_cumret.values, label='Benchmark NAV', linestyle='--', alpha=0.8, color='red')
                            ax1.set_title(f'Net Asset Value Curve: {file_name}')
                            ax1.set_ylabel('Cumulative Return')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # 相对收益
                            relative_ret = strategy_cumret - benchmark_cumret
                            color = 'green' if relative_ret.iloc[-1] > 0 else 'red'
                            ax2.fill_between(relative_ret.index, relative_ret.values, alpha=0.3, color=color)
                            ax2.plot(relative_ret.index, relative_ret.values, color=color, linewidth=1)
                            ax2.set_title('Relative to Benchmark Return')
                            ax2.set_xlabel('Date')
                            ax2.set_ylabel('Relative Return')
                            ax2.grid(True, alpha=0.3)
                            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # 关键指标摘要
                            st.write("### 📊 关键指标")
                            metrics = result['performance']
                            
                            # 收益指标
                            st.write("**收益指标:**")
                            for metric in ['total_return', 'annual_return', 'alpha']:
                                if metric in metrics.index:
                                    value = metrics.loc[metric, 'value']
                                    if CONFIG_AVAILABLE:
                                        info = get_metric_info(metric)
                                        if info:
                                            st.write(f"{info['name']}: `{info['format'].format(value)}`")
                                    else:
                                        st.write(f"{metric}: `{value:.3f}`")
                            
                            # 风险指标
                            st.write("**风险指标:**")
                            for metric in ['max_drawdown', 'volatility', 'var_95']:
                                if metric in metrics.index:
                                    value = metrics.loc[metric, 'value']
                                    if CONFIG_AVAILABLE:
                                        info = get_metric_info(metric)
                                        if info:
                                            st.write(f"{info['name']}: `{info['format'].format(value)}`")
                                    else:
                                        st.write(f"{metric}: `{value:.3f}`")
                            
                            # 效率指标
                            st.write("**效率指标:**")
                            for metric in ['sharpe_ratio', 'information_ratio', 'calmar_ratio']:
                                if metric in metrics.index:
                                    value = metrics.loc[metric, 'value']
                                    if CONFIG_AVAILABLE:
                                        info = get_metric_info(metric)
                                        if info:
                                            st.write(f"{info['name']}: `{info['format'].format(value)}`")
                                    else:
                                        st.write(f"{metric}: `{value:.3f}`")
                    
                    # 收益分析标签
                    with detail_tabs[2]:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 日收益分布
                            st.write("### 📊 日收益分布")
                            daily_returns = result['daily_returns']
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # 收益分布直方图
                            ax1.hist(daily_returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
                            ax1.axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.4f}')
                            ax1.set_title('Daily Return Distribution')
                            ax1.set_xlabel('Return Rate')
                            ax1.set_ylabel('Frequency')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # 累计收益 - 修复长度不匹配问题
                            # daily_returns比strategy_nav.index少一个元素（pct_change移除了第一个NaN）
                            returns_index = strategy_nav.index[1:]  # 使用与daily_returns对应的索引
                            cumulative_returns = pd.Series((1 + daily_returns).cumprod(), index=returns_index)
                            ax2.plot(cumulative_returns.index, cumulative_returns.values, color='green', linewidth=2)
                            ax2.set_title('Cumulative Return')
                            ax2.set_xlabel('Date')
                            ax2.set_ylabel('Cumulative Return')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # 收益统计
                            st.write("### 📈 收益统计")
                            daily_returns = result['daily_returns']
                            
                            stats_data = {
                                '指标': ['日均收益', '日收益标准差', '正收益天数', '负收益天数', '胜率'],
                                '数值': [
                                    f"{daily_returns.mean():.4f}",
                                    f"{daily_returns.std():.4f}",
                                    f"{(daily_returns > 0).sum()}",
                                    f"{(daily_returns < 0).sum()}",
                                    f"{(daily_returns > 0).mean():.2%}"
                                ]
                            }
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # 月度收益
                            if 'monthly_stats' in result:
                                st.write("### 📅 月度收益")
                                monthly_stats = result['monthly_stats']
                                st.dataframe(monthly_stats.tail(6), use_container_width=True)  # 显示最近6个月
                    
                    # 风险分析标签
                    with detail_tabs[3]:
                        if show_risk_metrics:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 回撤分析
                                st.write("### ⚠️ 回撤分析")
                                strategy_nav = result['strategy_nav']
                                running_max = strategy_nav.cummax()
                                drawdown = (strategy_nav - running_max) / running_max
                                
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                                
                                # 净值和高点
                                ax1.plot(strategy_nav.index, strategy_nav.values, label='Strategy NAV', color='blue', linewidth=2)
                                ax1.plot(running_max.index, running_max.values, label='Historical High', color='green', linestyle='--', alpha=0.8)
                                ax1.set_title('Net Asset Value and Historical High')
                                ax1.set_ylabel('NAV')
                                ax1.legend()
                                ax1.grid(True, alpha=0.3)
                                
                                # 回撤
                                ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
                                ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
                                ax2.set_title('Drawdown Analysis')
                                ax2.set_xlabel('Date')
                                ax2.set_ylabel('Drawdown Magnitude')
                                ax2.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                # 风险指标
                                st.write("### 📊 风险指标")
                                metrics = result['performance']
                                
                                risk_metrics = ['max_drawdown', 'volatility', 'var_95', 'cvar_95', 'tracking_error']
                                risk_data = []
                                
                                for metric in risk_metrics:
                                    if metric in metrics.index:
                                        value = metrics.loc[metric, 'value']
                                        if CONFIG_AVAILABLE:
                                            info = get_metric_info(metric)
                                            if info:
                                                risk_data.append({
                                                    '风险指标': info['name'] if info else metric,
                                                    '数值': info['format'].format(value) if info else f"{value:.4f}",
                                                    '说明': info['description'] if info else ''
                                                })
                                
                                if risk_data:
                                    risk_df = pd.DataFrame(risk_data)
                                    st.dataframe(risk_df, use_container_width=True)
                    
                    # 期间统计标签
                    with detail_tabs[4]:
                        if show_period_stats and 'monthly_stats' in result and 'yearly_stats' in result:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("### 📅 月度统计")
                                monthly_stats = result['monthly_stats']
                                
                                # 月度收益热力图
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                # 创建月度收益数据
                                monthly_data = monthly_stats.copy()
                                monthly_data.index = pd.to_datetime(monthly_data.index)
                                monthly_data['年'] = monthly_data.index.year
                                monthly_data['月'] = monthly_data.index.month
                                
                                # 创建热力图数据
                                heatmap_data = monthly_data.pivot_table(
                                    values='Strategy_Return', 
                                    index='年', 
                                    columns='月',
                                    aggfunc='first'
                                )
                                
                                # 检查是否有足够的数据来创建热力图
                                if heatmap_data.empty or heatmap_data.isna().all().all():
                                    ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                                           ha='center', va='center', transform=ax.transAxes)
                                    ax.set_title('Monthly Return Heatmap (Insufficient Data)')
                                else:
                                    sns.heatmap(heatmap_data, annot=True, fmt='.2%', 
                                              cmap='RdYlGn', center=0, ax=ax)
                                    ax.set_title('Monthly Return Heatmap')
                                
                                ax.set_xlabel('Month')
                                ax.set_ylabel('Year')
                                
                                st.pyplot(fig)
                                plt.close()
                                
                                # 月度统计表
                                st.write("月度统计详情:")
                                st.dataframe(monthly_stats.tail(12), use_container_width=True)
                            
                            with col2:
                                st.write("### 📊 年度统计")
                                yearly_stats = result['yearly_stats']
                                
                                # 年度收益对比图
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                x = range(len(yearly_stats))
                                width = 0.35
                                
                                ax.bar([i - width/2 for i in x], yearly_stats['Strategy_Return'], 
                                       width, label='Strategy Return', color='blue', alpha=0.8)
                                ax.bar([i + width/2 for i in x], yearly_stats['Benchmark_Return'], 
                                       width, label='Benchmark Return', color='red', alpha=0.8)
                                
                                ax.set_xlabel('Year')
                                ax.set_ylabel('Return Rate')
                                ax.set_title('Annual Return Comparison')
                                ax.set_xticks(x)
                                ax.set_xticklabels([str(idx.year) for idx in yearly_stats.index])
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                                plt.close()
                                
                                # 年度统计表
                                st.write("年度统计详情:")
                                st.dataframe(yearly_stats, use_container_width=True)
                    
                    # 持仓分析标签
                    with detail_tabs[5]:
                        if show_daily_holdings and 'daily_holdings' in result:
                            st.write("### 💼 每日持仓分析")
                            daily_holdings = result['daily_holdings']
                            
                            # 显示所有日期的完整持仓数据
                            if daily_holdings:
                                # 创建完整的持仓表格，显示所有日期的数据
                                all_holdings_data = []
                                
                                for holdings_data in daily_holdings:
                                    date = holdings_data['date']
                                    holdings = holdings_data['holdings']
                                    
                                    for holding in holdings:
                                        stock_code = holding['code']
                                        stock_name = code2name.get(stock_code, '') if code2name else ''
                                        all_holdings_data.append({
                                            'Date': date,
                                            'Stock Code': stock_code,
                                            'Stock Name': stock_name,
                                            'Weight (%)': f"{holding['weight']:.2f}%",
                                            'Shares': f"{int(holding.get('shares', 0)):,}",
                                            'Value': f"¥{holding.get('value', 0):,.2f}",
                                            'Price': f"¥{holding.get('price', 0):.2f}"
                                        })
                                
                                # 创建完整的持仓DataFrame
                                complete_holdings_df = pd.DataFrame(all_holdings_data)
                                
                                # 按日期和权重排序
                                #complete_holdings_df = complete_holdings_df.sort_values(['Date', 'Weight (%)'], ascending=[True, False])
                                
                                st.write("**Complete Daily Holdings History:**")
                                
                                # 显示完整表格，支持分页
                                total_rows = len(complete_holdings_df)
                                rows_per_page = 50
                                
                                if total_rows > rows_per_page:
                                    # 分页显示
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        page = st.number_input(
                                            f"Page (1-{total_rows//rows_per_page + 1})", 
                                            min_value=1, 
                                            max_value=total_rows//rows_per_page + 1,
                                            value=1,
                                            key=f"holdings_page_{file_name}"
                                        )
                                    
                                    start_idx = (page - 1) * rows_per_page
                                    end_idx = min(start_idx + rows_per_page, total_rows)
                                    display_df = complete_holdings_df.iloc[start_idx:end_idx]
                                    
                                    st.info(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows} total holdings")
                                else:
                                    display_df = complete_holdings_df
                                
                                # 显示表格
                                st.dataframe(display_df, use_container_width=True)
                                
                                # 下载按钮
                                csv = complete_holdings_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Complete Holdings Data (CSV)",
                                    data=csv,
                                    file_name=f"{file_name}_complete_holdings.csv",
                                    mime="text/csv",
                                    key=f"download_holdings_{file_name}"
                                )
                                
                                # 持仓统计图表
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Position Count Over Time:**")
                                    # 每日持仓数量变化
                                    holding_counts = [len(h['holdings']) for h in daily_holdings]
                                    holding_dates = [h['date'] for h in daily_holdings]
                                    
                                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                                    ax1.plot(holding_dates, holding_counts, marker='o', linewidth=2, markersize=4, color='blue')
                                    ax1.set_title('Daily Position Count Changes')
                                    ax1.set_xlabel('Date')
                                    ax1.set_ylabel('Position Count')
                                    ax1.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    st.pyplot(fig1)
                                    plt.close(fig1)
                                
                                with col2:
                                    st.write("**Top Holdings by Average Weight:**")
                                    # 计算每只股票在整个期间的平均权重
                                    stock_weights = {}
                                    for holdings_data in daily_holdings:
                                        for holding in holdings_data['holdings']:
                                            code = holding['code']
                                            weight = holding['weight']
                                            if code not in stock_weights:
                                                stock_weights[code] = []
                                            stock_weights[code].append(weight)
                                    
                                    # 计算平均权重并排序
                                    avg_weights = [(code, np.mean(weights), len(weights)) for code, weights in stock_weights.items()]
                                    avg_weights.sort(key=lambda x: x[1], reverse=True)
                                    
                                    # 显示前10只股票
                                    top_stocks = avg_weights[:10]
                                    if top_stocks:
                                        top_df = pd.DataFrame(top_stocks, columns=['Stock Code', 'Avg Weight (%)', 'Days Held'])
                                        top_df['Stock Name'] = top_df['Stock Code'].apply(lambda x: code2name.get(x, '') if code2name else '')
                                        top_df['Avg Weight (%)'] = top_df['Avg Weight (%)'].apply(lambda x: f"{x:.2f}%")
                                        # 重新排列列顺序
                                        top_df = top_df[['Stock Code', 'Stock Name', 'Avg Weight (%)', 'Days Held']]
                                        st.dataframe(top_df, use_container_width=True)
                                
                                # 持仓权重分布热力图
                                if len(daily_holdings) > 0:
                                    st.write("**Position Weight Heatmap:**")
                                    
                                    # 创建权重矩阵数据
                                    all_dates = [h['date'] for h in daily_holdings]
                                    all_stocks = list(set(h['code'] for holdings_data in daily_holdings for h in holdings_data['holdings']))
                                    
                                    # 创建权重矩阵
                                    weight_matrix = pd.DataFrame(0.0, index=all_dates, columns=all_stocks)
                                    
                                    for holdings_data in daily_holdings:
                                        date = holdings_data['date']
                                        for holding in holdings_data['holdings']:
                                            weight_matrix.loc[date, holding['code']] = holding['weight']
                                    
                                    # 显示权重热力图
                                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                                    
                                    # 只显示有数据的日期（避免图表过长）
                                    if len(all_dates) > 20:
                                        # 每隔几天显示一次
                                        step = max(1, len(all_dates) // 15)
                                        display_dates = all_dates[::step]
                                        display_matrix = weight_matrix.loc[display_dates]
                                    else:
                                        display_matrix = weight_matrix
                                    
                                    if display_matrix.size > 0:
                                        sns.heatmap(display_matrix.T, annot=False, cmap='Blues', 
                                              cbar_kws={'label': 'Weight (%)'}, ax=ax2)
                                    else:
                                        st.warning("当前没有持仓数据，无法绘制热力图")

                                    # sns.heatmap(display_matrix.T, annot=False, cmap='Blues', 
                                    #           cbar_kws={'label': 'Weight (%)'}, ax=ax2)
                                    ax2.set_title('Position Weight Heatmap Over Time')
                                    ax2.set_xlabel('Date')
                                    ax2.set_ylabel('Stock Code')
                                    
                                    st.pyplot(fig2)
                                    plt.close(fig2)
                            
                            else:
                                st.info("No holdings data available")
                        
                        else:
                            st.info("No daily holdings data available")
                    
                    # 交易记录标签
                    with detail_tabs[6]:
                        if show_trade_history and 'trade_history' in result:
                            st.write("### 🔄 交易记录")
                            trade_history = result['trade_history']
                            
                            if trade_history:
                                trade_df = pd.DataFrame(trade_history)
                                
                                # 交易统计
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("总交易数", len(trade_df))
                                with col2:
                                    buy_trades = len(trade_df[trade_df['action'] == 'BUY'])
                                    st.metric("买入交易", buy_trades)
                                with col3:
                                    sell_trades = len(trade_df[trade_df['action'] == 'SELL'])
                                    st.metric("卖出交易", sell_trades)
                                
                                # 交易记录表
                                st.write("**交易详情:**")
                                
                                # 使用更稳定的分页方式 - 显示所有交易或使用数字分页
                                page_size = 100  # 大幅增加每页显示数量到100条
                                total_pages = (len(trade_df) + page_size - 1) // page_size
                                
                                if total_pages > 1:
                                    # 使用数字输入而不是选择框，避免页面刷新
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        page = st.number_input(
                                            f"Page (1-{total_pages})", 
                                            min_value=1, 
                                            max_value=total_pages,
                                            value=1,
                                            key=f"trade_page_{file_name}"
                                        )
                                    
                                    start_idx = (page - 1) * page_size
                                    end_idx = min(start_idx + page_size, len(trade_df))
                                    display_trades = trade_df.iloc[start_idx:end_idx]
                                    
                                    st.info(f"Showing trades {start_idx + 1} to {end_idx} of {len(trade_df)} total trades")
                                else:
                                    # 如果只有一页，显示所有交易
                                    display_trades = trade_df
                                    st.success(f"Showing all {len(trade_df)} trades")
                                
                                # 优化显示的列名和数据格式
                                display_cols = ['date', 'code', 'action', 'price', 'size', 'value', 'commission']
                                available_cols = [col for col in display_cols if col in display_trades.columns]
                                
                                if len(available_cols) > 0:
                                    # 创建更好的显示格式
                                    display_data = display_trades[available_cols].copy()
                                    
                                    # 添加股票名称列
                                    if 'code' in display_data.columns:
                                        display_data['stock_name'] = display_data['code'].apply(lambda x: code2name.get(x, '') if code2name else '')
                                    
                                    # 格式化数据
                                    if 'date' in display_data.columns:
                                        display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
                                    if 'price' in display_data.columns:
                                        display_data['price'] = display_data['price'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                                    if 'size' in display_data.columns:
                                        display_data['size'] = display_data['size'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
                                    if 'value' in display_data.columns:
                                        display_data['value'] = display_data['value'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
                                    if 'commission' in display_data.columns:
                                        display_data['commission'] = display_data['commission'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                                    
                                    # 重命名列名使其更友好
                                    column_names = {
                                        'date': 'Date',
                                        'code': 'Stock Code', 
                                        'stock_name': 'Stock Name',
                                        'action': 'Action',
                                        'price': 'Price',
                                        'size': 'Shares',
                                        'value': 'Trade Value',
                                        'commission': 'Commission'
                                    }
                                    display_data = display_data.rename(columns=column_names)
                                    
                                    # 重新排列列顺序，将股票名称放在代码后面
                                    col_order = ['Date', 'Stock Code', 'Stock Name', 'Action', 'Price', 'Shares', 'Trade Value', 'Commission']
                                    display_data = display_data[[col for col in col_order if col in display_data.columns]]
                                    
                                    st.dataframe(display_data, use_container_width=True)
                                else:
                                    st.dataframe(display_trades, use_container_width=True)
                                
                                # 提供下载功能
                                csv = trade_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Complete Trade History (CSV)",
                                    data=csv,
                                    file_name=f"{file_name}_complete_trades.csv",
                                    mime="text/csv",
                                    key=f"download_trades_{file_name}"
                                )
                                
                                # 交易时间分布
                                if 'date' in trade_df.columns:
                                    st.write("### 📅 交易时间分布")
                                    trade_df['date'] = pd.to_datetime(trade_df['date'])
                                    
                                    # 每日交易数量分布
                                    daily_trade_count = trade_df.groupby('date').size()
                                    
                                    fig1, ax1 = plt.subplots(figsize=(12, 4))
                                    daily_trade_count.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
                                    ax1.set_title('Daily Trade Count Distribution')
                                    ax1.set_xlabel('Date')
                                    ax1.set_ylabel('Trade Count')
                                    ax1.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    
                                    st.pyplot(fig1)
                                    plt.close(fig1)
                                    
                                    # 买入vs卖出对比
                                    st.write("**Buy vs Sell Analysis:**")
                                    buy_sell_daily = trade_df.groupby(['date', 'action']).size().unstack(fill_value=0)
                                    
                                    fig2, ax2 = plt.subplots(figsize=(12, 4))
                                    if 'BUY' in buy_sell_daily.columns:
                                        ax2.bar(buy_sell_daily.index, buy_sell_daily['BUY'], 
                                               label='BUY', alpha=0.7, color='green')
                                    if 'SELL' in buy_sell_daily.columns:
                                        ax2.bar(buy_sell_daily.index, buy_sell_daily['SELL'], 
                                               bottom=buy_sell_daily.get('BUY', 0), 
                                               label='SELL', alpha=0.7, color='red')
                                    
                                    ax2.set_title('Daily Buy vs Sell Trades')
                                    ax2.set_xlabel('Date')
                                    ax2.set_ylabel('Number of Trades')
                                    ax2.legend()
                                    ax2.grid(True, alpha=0.3)
                                    plt.xticks(rotation=45)
                                    
                                    st.pyplot(fig2)
                                    plt.close(fig2)
                                
                                # 交易价值分析
                                if 'value' in trade_df.columns:
                                    st.write("### 💰 交易价值分析")
                                    
                                    # 交易价值统计
                                    trade_stats = pd.DataFrame({
                                        'Metric': ['Total Trade Value', 'Average Trade Value', 'Largest Trade', 'Smallest Trade'],
                                        'Value': [
                                            f"¥{trade_df['value'].sum():,.2f}",
                                            f"¥{trade_df['value'].mean():,.2f}",
                                            f"¥{trade_df['value'].max():,.2f}",
                                            f"¥{trade_df['value'].min():,.2f}"
                                        ]
                                    })
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.dataframe(trade_stats, use_container_width=True)
                                    
                                    with col2:
                                        # 交易价值分布直方图
                                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                                        trade_df['value'].hist(bins=20, ax=ax3, alpha=0.7, color='orange')
                                        ax3.set_title('Trade Value Distribution')
                                        ax3.set_xlabel('Trade Value')
                                        ax3.set_ylabel('Frequency')
                                        ax3.grid(True, alpha=0.3)
                                        
                                        st.pyplot(fig3)
                                        plt.close(fig3)
                                
                                # 股票交易频率分析
                                if 'code' in trade_df.columns:
                                    st.write("### 📈 股票交易频率分析")
                                    
                                    stock_trade_count = trade_df.groupby('code').size().sort_values(ascending=False)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Most Traded Stocks:**")
                                        top_traded = stock_trade_count.head(10)
                                        top_traded_df = pd.DataFrame({
                                            'Stock Code': top_traded.index,
                                            'Stock Name': [code2name.get(code, '') if code2name else '' for code in top_traded.index],
                                            'Trade Count': top_traded.values
                                        })
                                        st.dataframe(top_traded_df, use_container_width=True)
                                    
                                    with col2:
                                        # 交易频率条形图
                                        fig4, ax4 = plt.subplots(figsize=(8, 5))
                                        top_traded.plot(kind='bar', ax=ax4, color='purple', alpha=0.7)
                                        ax4.set_title('Top 10 Most Traded Stocks')
                                        ax4.set_xlabel('Stock Code')
                                        ax4.set_ylabel('Trade Count')
                                        ax4.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        
                                        st.pyplot(fig4)
                                        plt.close(fig4)
                            else:
                                st.info("暂无交易记录")
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 🎯 系统功能
        1. **多策略支持**: 支持加权TopN、等权重、动量策略
        2. **多基准对比**: 支持多种基准指数对比
        3. **丰富的分析**: 包含风险分析、期间统计、持仓分析等
        4. **可视化展示**: 提供多种图表展示方式
        5. **真实回测**: 基于backtrader引擎的真实回测
        
        ### 📋 数据格式要求
        CSV文件必须包含以下列：
        - `date`: 日期 (YYYY-MM-DD格式)
        - `code`: 股票代码 (6位数字)
        - `weight`: 权重 (可选，用于加权策略)
        
        ### 🚀 使用步骤
        1. **准备数据**: 将CSV文件放入指定数据目录
        2. **选择文件**: 选择要回测的CSV文件
        3. **配置参数**: 设置初始资金、策略参数等
        4. **选择分析**: 选择要显示的分析内容
        5. **运行回测**: 点击"开始回测"按钮
        6. **查看结果**: 查看详细的回测结果和分析
        
        ### 📊 分析内容说明
        - **概览**: 基本信息和关键指标
        - **净值曲线**: 策略与基准的净值对比
        - **收益分析**: 日收益分布和统计
        - **风险分析**: 回撤分析和风险指标
        - **期间统计**: 月度/年度收益统计
        - **持仓分析**: 每日持仓详情和权重分布
        - **交易记录**: 详细的交易历史
        """)

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    create_enhanced_streamlit_app()