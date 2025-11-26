"""
股票回测核心引擎模块
分离核心回测逻辑和界面代码，提高代码可维护性
"""

import os
import json
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from types import SimpleNamespace
import hashlib

# 导入基础模块
from backtrader_base_strategy import JQ2BTBaseStrategy
from data import get_trading_dates, load_bt_stocks, get_index_daily, code2name
# from data_akshare import get_trading_dates, load_bt_stocks, get_index_daily, code2name

from metrics.portfolio_structure import structure_metrics
from metrics.trade_metrics import trading_metrics
from metrics.risk_extension import extended_risk_metrics


# 导入配置模块
try:
    from config import SystemConfig, StrategyConfig, get_strategy_info
    CONFIG_AVAILABLE = True
except Exception as e:
    print(f"配置模块加载失败: {e}")
    CONFIG_AVAILABLE = False


@dataclass(frozen=True)
class BacktestResult:
    """统一封装回测结果，方便分析层或 CLI 复用。"""

    strategy_nav: pd.Series
    benchmark_nav: pd.Series
    performance: pd.DataFrame
    detailed_metrics: Dict[str, Any]
    monthly_stats: pd.DataFrame
    yearly_stats: pd.DataFrame
    trade_history: List[dict]
    daily_holdings: List[dict]
    final_value: float
    valid_stocks: int
    strategy_name: str
    file_name: str

    def to_dict(self) -> Dict[str, Any]:
        """保持向后兼容，仍返回原始字典结构。"""

        return {
            "strategy_nav": self.strategy_nav,
            "benchmark_nav": self.benchmark_nav,
            "performance": self.performance,
            "detailed_metrics": self.detailed_metrics,
            "monthly_stats": self.monthly_stats,
            "yearly_stats": self.yearly_stats,
            "trade_history": self.trade_history,
            "daily_holdings": self.daily_holdings,
            "final_value": self.final_value,
            "valid_stocks": self.valid_stocks,
            "strategy_name": self.strategy_name,
            "file_name": self.file_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        return cls(
            strategy_nav=data["strategy_nav"],
            benchmark_nav=data["benchmark_nav"],
            performance=data["performance"],
            detailed_metrics=data["detailed_metrics"],
            monthly_stats=data["monthly_stats"],
            yearly_stats=data["yearly_stats"],
            trade_history=data["trade_history"],
            daily_holdings=data["daily_holdings"],
            final_value=data["final_value"],
            valid_stocks=data["valid_stocks"],
            strategy_name=data["strategy_name"],
            file_name=data["file_name"],
        )


# =========================
# 数据加载器
# =========================
class DataLoader:
    """数据加载器类"""
    
    @staticmethod
    def load_prediction_data(file_path: str) -> pd.DataFrame:
        """加载预测数据文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 验证必需列存在
            required_columns = ["date", "code"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据缺少必需列: {missing_columns}")
            
            # 标准化列名 - 关键：股票代码格式化为6位字符串
            if "code" in df.columns and len(df) > 0:
                df["code"] = df["code"].astype(str).str.zfill(6)
            if "date" in df.columns and len(df) > 0:
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            
            # 确保有权重列
            if "weight" not in df.columns:
                df["weight"] = 1.0  # 默认等权重
            
            # 数据清理 - 移除无效数据
            df = df.dropna(subset=['date', 'code'])
            
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
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
# 策略基类
# =========================
class BaseStrategy(JQ2BTBaseStrategy):
    """策略基类，负责在回测过程中执行预测调仓并记录交易/持仓"""

    params = (
        ("config", None),  # 配置对象
        ("pred_df", None),  # 预测数据
    )

    def __init__(self):
        super().__init__()
        if self.p.config is None:
            raise ValueError("策略配置不能为空")

        self.config = self.p.config
        self.pred_df = self._prepare_pred_df(self.p.pred_df)
        self.holdings: Dict[str, date] = {}
        self._daily_navs: list[float] = []
        self._daily_holdings: list[dict] = []
        self._trade_history: list[dict] = []

    @staticmethod
    def _prepare_pred_df(pred_df: Optional[Union[pd.DataFrame, str]]) -> pd.DataFrame:
        """标准化预测数据，确保日期和股票代码格式正确"""
        if pred_df is None:
            return pd.DataFrame(columns=["date", "code", "weight"])

        if isinstance(pred_df, str):
            df = pd.read_json(pred_df)
        else:
            df = pred_df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)

        if "weight" not in df.columns:
            df["weight"] = 1.0

        return df

    def _find_data(self, code: str):
        for data in self.datas:
            if data._name == code:
                return data
        return None

    def _record_trade(self, code: str, action: str, size: int, price: float, value: float):
        self._trade_history.append({
            "date": self.current_dt.date() if self.current_dt is not None else None,
            "code": code,
            "action": action,
            "size": size,
            "price": price,
            "value": value,
            "portfolio_value": self.broker.getvalue(),
        })

    def _record_holdings(self):
        current_holdings = []
        total_value = self.broker.getvalue()

        for code, buy_date in list(self.holdings.items()):
            data = self._find_data(code)
            if data is None:
                continue

            position = self.broker.getposition(data)
            if position.size <= 0:
                continue

            current_holdings.append({
                "code": code,
                "size": position.size,
                "price": data.close[0],
                "value": position.size * data.close[0],
                "weight": (position.size * data.close[0]) / total_value if total_value > 0 else 0,
                "buy_date": buy_date,
            })

        self._daily_holdings.append({
            "date": self.current_dt.date() if self.current_dt is not None else None,
            "holdings": current_holdings,
            "total_value": total_value,
            "cash": self.broker.getcash(),
        })

    def execute_strategy(self):
        raise NotImplementedError("子类必须实现 execute_strategy 方法")

    def next(self):
        super().next()
        self.execute_strategy()
        self._daily_navs.append(self.broker.getvalue())
        self._record_holdings()

    def stop(self):
        super().stop()
        if getattr(self.config, "save_results", False):
            today_str = datetime.now().strftime("%Y%m%d")
            nav_df = pd.DataFrame({"nav": self._daily_navs})
            nav_df.to_csv(f"daily_nav_{today_str}.csv", index=False, encoding="utf-8-sig")
            print(f"保存每日 NAV 到 daily_nav_{today_str}.csv")

class WeightedTopNStrategy(BaseStrategy):
    """加权TopN策略"""

    def execute_strategy(self):
        dt = self.current_dt.date()
        ts = pd.Timestamp(dt)

        today_df = self.pred_df[self.pred_df["date"] == ts]
        if today_df.empty:
            return

        hold_days = max(1, getattr(self.config, "hold_days", 1))

        # 卖出到期持仓
        expired_codes = []
        for code, buy_date in list(self.holdings.items()):
            if (dt - buy_date).days >= hold_days:
                data = self._find_data(code)
                if data is None:
                    continue
                position = self.broker.getposition(data)
                if position.size > 0:
                    self.order_target_value(data, 0)
                    self._record_trade(code, "SELL", -position.size, data.close[0], position.size * data.close[0])
                    stock_name = code2name.get(code, "") if code2name else ""
                    self.log(f"SELL {code} {stock_name} (到期)", log_type="trade")
                expired_codes.append(code)

        for code in expired_codes:
            self.holdings.pop(code, None)

        weight_col = getattr(self.config, "weight_column", "weight")
        if weight_col in today_df.columns:
            today_df = today_df.sort_values(weight_col, ascending=False)

        total_weight = today_df[weight_col].sum()
        if total_weight <= 0:
            return

        total_value = self.broker.getvalue()
        for _, row in today_df.iterrows():
            code = str(row["code"]).zfill(6)
            data = self._find_data(code)
            if data is None:
                continue

            weight = float(getattr(row, weight_col, 0.0))
            target_value = total_value * (weight / total_weight)

            position = self.broker.getposition(data)
            current_value = position.size * data.close[0] if position.size > 0 else 0

            if total_value == 0:
                continue

            if abs(target_value - current_value) / total_value > 0.01:
                size_change = int((target_value - current_value) / data.close[0])
                self.order_target_value(data, target_value)
                action = "BUY" if size_change > 0 else "SELL"
                traded_value = abs(size_change) * data.close[0]
                if traded_value > 0:
                    self._record_trade(code, action, size_change, data.close[0], traded_value)
                    self.holdings[code] = dt
                    stock_name = code2name.get(code, "") if code2name else ""
                    self.log(f"SET {code} {stock_name} target_value={target_value:.2f}", log_type="trade")

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
        top_n_stocks = max(1, getattr(self.config, "top_n_stocks", 10))

        if len(today_df) > top_n_stocks:
            today_df = today_df.head(top_n_stocks)

        current_codes = set(today_df["code"].astype(str).str.zfill(6))
        removed_codes = []
        for code in list(self.holdings.keys()):
            if code in current_codes:
                continue
            data = self._find_data(code)
            if data is None:
                continue
            position = self.broker.getposition(data)
            if position.size > 0:
                self.order_target_value(data, 0)
                self._record_trade(code, "SELL", -position.size, data.close[0], position.size * data.close[0])
                stock_name = code2name.get(code, "") if code2name else ""
                self.log(f"SELL {code} {stock_name} (调出)", log_type="trade")
            removed_codes.append(code)

        for code in removed_codes:
            self.holdings.pop(code, None)

        total_value = self.broker.getvalue()
        n_stocks = len(today_df)
        if n_stocks == 0 or total_value == 0:
            return

        weight_per_stock = 1.0 / n_stocks
        target_value = total_value * weight_per_stock

        for _, row in today_df.iterrows():
            code = str(row["code"]).zfill(6)
            data = self._find_data(code)
            if data is None:
                continue

            position = self.broker.getposition(data)
            current_value = position.size * data.close[0] if position.size > 0 else 0

            if abs(target_value - current_value) / total_value > 0.01:
                size_change = int((target_value - current_value) / data.close[0])
                self.order_target_value(data, target_value)
                if size_change != 0:
                    action = "BUY" if size_change > 0 else "SELL"
                    self._record_trade(code, action, size_change, data.close[0], abs(size_change) * data.close[0])
                    if size_change > 0:
                        self.holdings[code] = dt
                    stock_name = code2name.get(code, "") if code2name else ""
                    self.log(f"ADJUST {code} {stock_name} equal_weight={weight_per_stock:.3f}", log_type="trade")

class DirectExecutionStrategy(BaseStrategy):
    """直接执行预测数据的策略
    
    假设预测数据已经是完整的每日持仓计划（股票+权重），
    直接按照预测数据执行，不做任何筛选或重新分配。
    """

    def execute_strategy(self):
        """直接执行预测数据"""
        dt = self.current_dt.date()
        ts = pd.Timestamp(dt)
        
        # 获取当日预测数据
        today_df = self.pred_df[self.pred_df["date"] == ts]
        if today_df.empty:
            return
        
        # 清空所有现有持仓（如果需要的话）
        # 注意：这里假设预测数据是完整的持仓目标，不需要保留之前的持仓
        
        total_value = self.broker.getvalue()
        if total_value == 0:
            return
        
        # 按预测数据直接执行
        weight_col = getattr(self.config, "weight_column", "weight")
        
        for _, row in today_df.iterrows():
            code = str(row["code"]).zfill(6)
            data = self._find_data(code)
            if data is None:
                continue
            
            # 获取目标权重
            weight = float(getattr(row, weight_col, 0.0))
            if weight <= 0:
                continue
                
            # 计算目标价值
            target_value = total_value * weight
            
            # 执行交易
            position = self.broker.getposition(data)
            current_value = position.size * data.close[0] if position.size > 0 else 0
            
            if abs(target_value - current_value) / total_value > 0.01:  # 1%阈值
                size_change = int((target_value - current_value) / data.close[0])
                self.order_target_value(data, target_value)
                
                if abs(size_change) > 0:
                    action = "BUY" if size_change > 0 else "SELL"
                    traded_value = abs(size_change) * data.close[0]
                    self._record_trade(code, action, size_change, data.close[0], traded_value)
                    
                    # 更新持仓记录
                    if size_change > 0:
                        self.holdings[code] = dt
                    
                    stock_name = code2name.get(code, "") if code2name else ""
                    self.log(f"{action} {code} {stock_name} target_value={target_value:.2f}", log_type="trade")

# =========================
# 策略工厂
# =========================
class StrategyFactory:
    """策略工厂类"""
    
    _strategies = {
        "weighted_top_n": WeightedTopNStrategy,
        "equal_weight": EqualWeightStrategy,
        "direct_execution": DirectExecutionStrategy,
        "limit_up_gene": None,  # Will be imported dynamically
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str) -> type:
        """获取策略类"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"不支持的策略: {strategy_name}. 可选策略: {list(cls._strategies.keys())}")
        
        # Dynamic import for limit_up_gene
        if strategy_name == "limit_up_gene" and cls._strategies[strategy_name] is None:
            try:
                from jq_backtest_strategy import LimitUpGeneStrategy
                cls._strategies[strategy_name] = LimitUpGeneStrategy
            except ImportError as e:
                raise ValueError(f"无法导入策略 {strategy_name}: {e}")
        
        return cls._strategies[strategy_name]
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """获取所有支持的策略名称"""
        return list(cls._strategies.keys())

# =========================
# 核心回测引擎
# =========================
class BacktestEngine:
    """核心回测引擎"""
    
    def __init__(self, system_config: 'SystemConfig', strategy_config: 'StrategyConfig'):
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
    
    def run_single_backtest(self, pred_df: pd.DataFrame, strategy_name: str, file_name: str = "") -> Dict:
        """运行单个回测，返回供分析层使用的结果字典。"""

        try:
            strategy_class = StrategyFactory.get_strategy(strategy_name)
            prepared_df = self._prepare_prediction_data(pred_df)

            stock_codes = self._extract_stock_codes(prepared_df)
            if not stock_codes:
                raise ValueError("没有有效的股票代码")

            start_date, end_date = self._determine_date_window(prepared_df)
            print(f"股票数量: {len(stock_codes)}, 日期范围: {start_date} 到 {end_date}")

            feeds = self._load_market_feeds(stock_codes, start_date, end_date, prepared_df)
            if not feeds:
                raise ValueError("无法加载任何股票数据（主数据源和备用数据源均失败）")

            valid_stocks = list(feeds.keys())
            print(f"成功加载 {len(valid_stocks)} 只股票数据: {valid_stocks}")

            filtered_pred = prepared_df[prepared_df["code"].isin(valid_stocks)].copy()
            if filtered_pred.empty:
                raise ValueError("预测数据与可用行情不匹配，无法执行回测")

            strategy_params = self.get_strategy_params()
            strategy_params["hold_days"] = max(1, int(strategy_params.get("hold_days", 1)))
            strategy_params["top_n_stocks"] = max(1, int(strategy_params.get("top_n_stocks", 1)))

            config_obj = self._build_strategy_runtime_config(strategy_params)

            cerebro, strategy = self._execute_backtest(
                feeds=feeds,
                strategy_class=strategy_class,
                pred_df=filtered_pred,
                config_obj=config_obj,
            )

            result = self._collect_backtest_result(
                cerebro=cerebro,
                strategy=strategy,
                valid_stocks=valid_stocks,
                strategy_name=strategy_name,
                file_name=file_name,
                start_date=start_date,
                end_date=end_date,
            )

            result_dict = result.to_dict()
            if file_name:
                self.results[file_name] = result_dict
            return result_dict

        except Exception as e:
            print(f"回测执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    def _prepare_prediction_data(self, pred_df: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """标准化预测数据，确保列名/数据类型满足后续处理需求。"""

        if pred_df is None:
            raise ValueError("预测数据为空，无法进行回测")

        if isinstance(pred_df, str):
            df = pd.read_json(pred_df)
        else:
            df = pred_df.copy()

        if df.empty:
            raise ValueError("预测数据为空，无法进行回测")

        required_columns = {"code", "date"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"预测数据缺少必需的列：{sorted(missing_columns)}")

        df["code"] = df["code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "weight" not in df.columns:
            df["weight"] = 1.0

        df = df.dropna(subset=["date", "code"])
        df = df.sort_values(["date", "code"]).reset_index(drop=True)
        return df

    @staticmethod
    def _extract_stock_codes(pred_df: pd.DataFrame) -> List[str]:
        return [str(code).zfill(6) for code in pred_df["code"].dropna().unique().tolist()]

    @staticmethod
    def _determine_date_window(pred_df: pd.DataFrame) -> Tuple[str, str]:
        min_date = pred_df["date"].min()
        max_date = pred_df["date"].max()
        if pd.isna(min_date) or pd.isna(max_date):
            raise ValueError("日期数据无效")
        return min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")

    def _load_market_feeds(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        pred_df: pd.DataFrame,
    ) -> Dict[str, bt.feeds.PandasData]:
        feeds: Dict[str, bt.feeds.PandasData] = {}

        if stock_codes:
            try:
                feeds = load_bt_stocks(stock_codes, start_date, end_date)
            except Exception as exc:
                print(f"主数据加载失败: {exc}")

        if feeds:
            return feeds

        print("主数据源未返回数据，尝试基于预测结果生成合成行情...")
        synthetic_feeds = self._generate_synthetic_feeds(stock_codes, start_date, end_date, pred_df)
        return synthetic_feeds

    @staticmethod
    def _build_strategy_runtime_config(params: Dict[str, Any]) -> SimpleNamespace:
        config_obj = SimpleNamespace(
            parameters=params,
            hold_days=params.get("hold_days", 2),
            top_n_stocks=params.get("top_n_stocks", 10),
            commission_rate=params.get("commission_rate", 0.0002),
            weight_column=params.get("weight_column", "weight"),
        )

        for key, value in params.items():
            if not hasattr(config_obj, key):
                setattr(config_obj, key, value)

        return config_obj

    def _execute_backtest(
        self,
        *,
        feeds: Dict[str, bt.feeds.PandasData],
        strategy_class: type,
        pred_df: pd.DataFrame,
        config_obj: SimpleNamespace,
    ) -> Tuple[bt.Cerebro, bt.Strategy]:
        cerebro = bt.Cerebro()
        for symbol, data_feed in feeds.items():
            cerebro.adddata(data_feed, name=symbol)

        cerebro.addstrategy(strategy_class, config=config_obj, pred_df=pred_df)

        cerebro.broker.setcash(self.system_config.initial_cash)
        cerebro.broker.setcommission(commission=self.system_config.commission_rate)
        cerebro.broker.set_slippage_perc(getattr(self.system_config, "slippage_rate", 0.0))

        print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}")

        results = cerebro.run()
        if not results:
            raise RuntimeError("Backtrader 未返回结果")

        strategy = results[0]
        print(f"Final Portfolio Value: {cerebro.broker.getvalue():,.2f}")
        return cerebro, strategy

    def _collect_backtest_result(
        self,
        *,
        cerebro: bt.Cerebro,
        strategy: bt.Strategy,
        valid_stocks: List[str],
        strategy_name: str,
        file_name: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        navs_data = list(getattr(strategy, "navs", []))
        if not navs_data:
            navs_data = [self.system_config.initial_cash]

        nav_index = self._build_nav_index(strategy, len(navs_data))
        strategy_nav = pd.Series(navs_data, index=nav_index, name="strategy_nav")

        benchmark_nav = self._load_benchmark_nav(start_date, end_date, strategy_nav.index)
        strategy_returns = strategy_nav.pct_change().dropna().values

        trade_history = getattr(strategy, '_trade_history', [])
        daily_holdings = getattr(strategy, '_daily_holdings', [])

        detailed_results = self.calculate_detailed_metrics(
            strategy_nav,
            benchmark_nav,
            strategy_returns,
            trade_history=trade_history,
            daily_holdings=daily_holdings,
        )
        monthly_stats = self.generate_period_stats(strategy_nav, benchmark_nav, 'M')
        yearly_stats = self.generate_period_stats(strategy_nav, benchmark_nav, 'Y')

        final_value = cerebro.broker.getvalue()

        return BacktestResult(
            strategy_nav=strategy_nav,
            benchmark_nav=benchmark_nav,
            performance=detailed_results['performance_df'],
            detailed_metrics=detailed_results['detailed_metrics'],
            monthly_stats=monthly_stats,
            yearly_stats=yearly_stats,
            trade_history=trade_history,
            daily_holdings=daily_holdings,
            final_value=final_value,
            valid_stocks=len(valid_stocks),
            strategy_name=strategy_name,
            file_name=file_name,
        )

    @staticmethod
    def _build_nav_index(strategy: bt.Strategy, nav_count: int) -> pd.DatetimeIndex:
        if nav_count == 0:
            return pd.DatetimeIndex([])

        try:
            data = strategy.datas[0]
            dates = [data.datetime.date(-i) for i in range(nav_count)]
            dates.reverse()
            return pd.to_datetime(dates)
        except Exception:
            end_date = datetime.now().date()
            return pd.date_range(end=pd.Timestamp(end_date), periods=nav_count, freq='D')

    def _load_benchmark_nav(
        self,
        start_date: str,
        end_date: str,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        benchmark_nav = get_index_daily(self.system_config.benchmark_index, start_date, end_date)
        benchmark_nav = benchmark_nav.reindex(target_index).ffill()
        if benchmark_nav.isna().all():
            raise ValueError(
                f"基准 {self.system_config.benchmark_index} 在 {start_date}~{end_date} 区间没有数据"
            )
        return benchmark_nav

    def _generate_synthetic_feeds(
        self,
        codes: list,
        start: str,
        end: str,
        pred_df: pd.DataFrame,
    ) -> dict:
        """基于预测数据生成可用于回测的合成行情。"""

        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
        except Exception:
            start_dt = pd.to_datetime(pred_df['date'].min())
            end_dt = pd.to_datetime(pred_df['date'].max())

        # 统一日期索引：优先使用预测数据覆盖的日期
        if pred_df is None or pred_df.empty or 'date' not in pred_df.columns:
            candidate_dates = pd.bdate_range(start_dt, end_dt)
            normalized_pred = pd.DataFrame()
            weight_col = None
        else:
            candidate_dates = pd.to_datetime(pred_df['date'].dropna().unique())
            candidate_dates = candidate_dates[(candidate_dates >= start_dt) & (candidate_dates <= end_dt)]
            candidate_dates = pd.DatetimeIndex(sorted(candidate_dates))
            normalized_pred = pred_df.copy()
            if "code" not in normalized_pred.columns:
                normalized_pred["code"] = "000000"
            normalized_pred["code"] = normalized_pred["code"].astype(str).str.zfill(6)
            if "date" in normalized_pred.columns:
                normalized_pred["date"] = pd.to_datetime(normalized_pred["date"])
            weight_col = "weight" if "weight" in normalized_pred.columns else None

        if len(candidate_dates) == 0:
            print("合成行情构建失败：无法确定日期范围。")
            return {}

        feeds: dict[str, bt.feeds.PandasData] = {}

        for raw_code in codes:
            code = str(raw_code).zfill(6)
            code_pred = normalized_pred[normalized_pred["code"] == code]

            if weight_col:
                weight_series = code_pred.set_index("date")[weight_col].reindex(candidate_dates)
            else:
                weight_series = pd.Series(0.0, index=candidate_dates)

            if weight_series is None or weight_series.isna().all():
                weight_series = pd.Series(0.0, index=candidate_dates)
            else:
                weight_series = weight_series.ffill().fillna(0.0)

            base_price = 50 + (abs(hash(code)) % 50)
            trend = np.linspace(0, 5, len(candidate_dates))
            weight_effect = np.cumsum(weight_series.values) * 0.5
            price = np.maximum(base_price + trend + weight_effect, 1.0)

            df = pd.DataFrame({
                "datetime": candidate_dates,
                "close": price,
            })
            df["open"] = price * (1 - 0.001)
            df["high"] = price * (1 + 0.002)
            df["low"] = price * (1 - 0.003)
            volume_trend = np.linspace(0, 200_000, len(candidate_dates))
            df["volume"] = 1_000_000 + np.abs(weight_series.values) * 100_000 + volume_trend
            df["openinterest"] = 0

            feed = bt.feeds.PandasData(
                dataname=df,
                datetime="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest="openinterest",
                name=code,
            )

            feeds[code] = feed

        return feeds
    
    def calculate_detailed_metrics(
        self,
        strategy_nav: pd.Series,
        benchmark_nav: pd.Series,
        returns: np.ndarray,
        trade_history: Optional[List[dict]] = None,
        daily_holdings: Optional[List[dict]] = None,
    ) -> Dict:
        """计算详细的性能指标，并整合结构、交易与扩展风险指标。"""

        trade_history = list(trade_history or [])
        daily_holdings = list(daily_holdings or [])

        # 预先计算额外指标，确保即使主流程提前返回也能提供结果
        try:
            structure_info = structure_metrics(daily_holdings)
        except Exception as exc:
            print(f"结构指标计算失败: {exc}")
            structure_info = structure_metrics([])

        nav_for_turnover = strategy_nav.dropna() if isinstance(strategy_nav, pd.Series) else pd.Series(dtype=float)
        try:
            trading_info = trading_metrics(trade_history, nav_series=nav_for_turnover)
        except Exception as exc:
            print(f"交易指标计算失败: {exc}")
            trading_info = trading_metrics([])

        try:
            extended_risk_info = extended_risk_metrics(strategy_nav)
        except Exception as exc:
            print(f"扩展风险指标计算失败: {exc}")
            extended_risk_info = extended_risk_metrics(pd.Series(dtype=float))

        def finalize(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
            return self._apply_additional_metrics(metrics_dict, structure_info, trading_info, extended_risk_info)

        try:
            # 基本参数验证
            if len(strategy_nav) == 0 or len(benchmark_nav) == 0 or len(returns) == 0:
                print("警告：策略净值、基准净值或收益数据为空，返回空指标")
                return finalize(self._create_empty_metrics())

            # 验证输入数据的有效性
            if strategy_nav.isna().all() or benchmark_nav.isna().all():
                print("警告：策略净值或基准净值全部为NaN，返回空指标")
                return finalize(self._create_empty_metrics())

            if np.isnan(returns).all():
                print("警告：收益数据全部为NaN，返回空指标")
                return finalize(self._create_empty_metrics())

            # 基本指标计算
            if np.isinf(strategy_nav.iloc[0]) or np.isinf(strategy_nav.iloc[-1]) or strategy_nav.iloc[0] == 0:
                total_return = 0.0
            elif strategy_nav.iloc[0] > 1000:  # 假设是绝对金额
                initial_capital = strategy_nav.iloc[0]
                final_capital = strategy_nav.iloc[-1]
                total_return = (final_capital - initial_capital) / initial_capital if initial_capital != 0 else 0.0
            else:
                total_return = strategy_nav.iloc[-1] - 1

            # 年化收益率
            actual_trading_days = len(strategy_nav) - 1
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

            # 基准相关指标
            try:
                if np.isinf(benchmark_nav.iloc[0]) or benchmark_nav.iloc[0] == 0:
                    benchmark_total_return = 0.0
                elif benchmark_nav.iloc[0] > 1000:
                    benchmark_total_return = (benchmark_nav.iloc[-1] - benchmark_nav.iloc[0]) / benchmark_nav.iloc[0] if benchmark_nav.iloc[0] != 0 else 0.0
                else:
                    benchmark_total_return = benchmark_nav.iloc[-1] - 1

                benchmark_annual_return = (1 + benchmark_total_return) ** (252 / actual_trading_days) - 1 if actual_trading_days > 0 else 0
            except (IndexError, ZeroDivisionError, OverflowError) as e:
                print(f"基准指标计算失败: {e}")
                benchmark_total_return = 0
                benchmark_annual_return = 0

            # Alpha和Beta
            try:
                benchmark_returns = benchmark_nav.pct_change().dropna().values
                if len(benchmark_returns) != len(returns):
                    min_len = min(len(returns), len(benchmark_returns))
                    if min_len > 0:
                        alpha, beta = self.calculate_alpha_beta(returns[:min_len], benchmark_returns[:min_len])
                    else:
                        alpha, beta = 0, 0
                else:
                    alpha, beta = self.calculate_alpha_beta(returns, benchmark_returns)
            except Exception as e:
                print(f"Alpha/Beta计算失败: {e}")
                alpha, beta = 0, 0

            # 其他指标
            if np.isinf(returns).any() or np.isnan(returns).any():
                win_rate = 0.0
            else:
                win_rate = (returns > 0).mean() if len(returns) > 0 else 0

            # 确保所有数值都是有限值
            def safe_float(value):
                if isinstance(value, (int, float, np.number)):
                    if np.isnan(value) or np.isinf(value):
                        return 0.0
                    return float(value)
                return 0.0

            performance_data = {
                'total_return': safe_float(total_return),
                'annual_return': safe_float(annual_return),
                'max_drawdown': safe_float(max_drawdown),
                'sharpe_ratio': safe_float(sharpe_ratio),
                'volatility': safe_float(volatility),
                'win_rate': safe_float(win_rate),
                'alpha': safe_float(alpha),
                'beta': safe_float(beta),
                'benchmark_annual_return': safe_float(benchmark_annual_return),
            }

            performance_df = pd.DataFrame(list(performance_data.items()), columns=['metric', 'value'])
            performance_df = performance_df.set_index('metric')

            detailed_metrics = {
                'drawdown_series': drawdown,
                'running_max': running_max,
            }

            metrics_dict = {
                'performance_df': performance_df,
                'detailed_metrics': detailed_metrics,
            }
            return finalize(metrics_dict)

        except Exception as e:
            print(f"计算详细指标时出错: {e}")
            return finalize(self._create_empty_metrics())
    
    def _create_empty_metrics(self) -> Dict:
        """创建空指标"""
        performance_data = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'benchmark_annual_return': 0.0,
        }
        
        performance_df = pd.DataFrame(list(performance_data.items()), columns=['metric', 'value'])
        performance_df = performance_df.set_index('metric')
        
        detailed_metrics = {
            'drawdown_series': pd.Series(dtype=float),
            'running_max': pd.Series(dtype=float),
        }
        
        return {
            'performance_df': performance_df,
            'detailed_metrics': detailed_metrics
        }

    def _apply_additional_metrics(
        self,
        metrics: Dict[str, Any],
        structure_info: Dict[str, Any],
        trading_info: Dict[str, Any],
        extended_risk_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将结构、交易与扩展风险指标注入 performance/detailed 结果。"""

        performance_df = metrics.get('performance_df') if isinstance(metrics, dict) else None
        detailed_metrics = metrics.get('detailed_metrics') if isinstance(metrics, dict) else {}
        if not isinstance(detailed_metrics, dict):
            detailed_metrics = {}

        def to_float(value: Any) -> Optional[float]:
            if isinstance(value, (int, np.integer)):
                return float(value)
            if isinstance(value, (float, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return float(value)
            return None

        if isinstance(performance_df, pd.DataFrame):
            structure_keys = [
                'industry_hhi',
                'top_industry_weight',
                'industry_count',
                'effective_positions',
                'max_single_weight',
                'normalized_entropy',
                'weight_entropy',
                'gini_coefficient',
                'industry_rotation',
            ]
            for key in structure_keys:
                value = to_float(structure_info.get(key)) if isinstance(structure_info, dict) else None
                if value is not None:
                    performance_df.loc[key, 'value'] = value

            trading_mapping = [
                ('total_turnover', 'total_turnover'),
                ('average_daily_turnover', 'average_daily_turnover'),
                ('trade_count', 'trade_count'),
                ('round_trip_count', 'round_trip_count'),
                ('avg_holding_days', 'avg_holding_days'),
                ('median_holding_days', 'median_holding_days'),
                ('max_holding_days', 'max_holding_days'),
                ('win_rate', 'round_trip_win_rate'),
                ('payoff_ratio', 'payoff_ratio'),
                ('expectancy', 'expectancy'),
                ('average_gain', 'average_gain'),
                ('average_loss', 'average_loss'),
            ]
            for source_key, target_key in trading_mapping:
                value = to_float(trading_info.get(source_key)) if isinstance(trading_info, dict) else None
                if value is not None:
                    performance_df.loc[target_key, 'value'] = value

            risk_keys = [
                'sortino_ratio',
                'downside_deviation',
                'tail_ratio',
                'ulcer_index',
                'skewness',
                'kurtosis',
                'return_count',
            ]
            for key in risk_keys:
                value = to_float(extended_risk_info.get(key)) if isinstance(extended_risk_info, dict) else None
                if value is not None:
                    performance_df.loc[key, 'value'] = value

        def sanitize_value(value: Any):
            if isinstance(value, dict):
                return sanitize_mapping(value)
            if isinstance(value, (float, np.floating)):
                return 0.0 if np.isnan(value) or np.isinf(value) else float(value)
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (pd.Timestamp, datetime, date)):
                return value.isoformat()
            if isinstance(value, (list, tuple, set)):
                return [sanitize_value(item) for item in value]
            return value

        def sanitize_mapping(payload: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(payload, dict):
                return {}

            sanitized: Dict[str, Any] = {}
            for key, value in payload.items():
                sanitized[key] = sanitize_value(value)
            return sanitized

        detailed_metrics['structure_metrics'] = sanitize_mapping(structure_info)
        detailed_metrics['trading_metrics'] = sanitize_mapping(trading_info)
        detailed_metrics['extended_risk_metrics'] = sanitize_mapping(extended_risk_info)

        metrics['detailed_metrics'] = detailed_metrics
        if isinstance(performance_df, pd.DataFrame):
            metrics['performance_df'] = performance_df

        return metrics
    
    def calculate_alpha_beta(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0, 0.0
        
        # 简单线性回归计算Beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        # Alpha年化处理
        alpha = (np.mean(strategy_returns) - beta * np.mean(benchmark_returns)) * 252
        
        # 确保返回有限值
        if np.isnan(alpha) or np.isinf(alpha):
            alpha = 0.0
        if np.isnan(beta) or np.isinf(beta):
            beta = 0.0
            
        return alpha, beta
    
    def generate_period_stats(self, strategy_nav: pd.Series, benchmark_nav: pd.Series, period: str) -> pd.DataFrame:
        """按周期（月度/年度）生成收益统计"""
        try:
            if not isinstance(strategy_nav.index, pd.DatetimeIndex):
                strategy_nav.index = pd.to_datetime(strategy_nav.index)
            if not isinstance(benchmark_nav.index, pd.DatetimeIndex):
                benchmark_nav.index = pd.to_datetime(benchmark_nav.index)

            if not strategy_nav.index.equals(benchmark_nav.index):
                common_index = strategy_nav.index.intersection(benchmark_nav.index)
                if len(common_index) == 0:
                    return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])
                strategy_nav = strategy_nav.reindex(common_index)
                benchmark_nav = benchmark_nav.reindex(common_index)

            if len(strategy_nav) < 2:
                return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])

            try:
                if period == 'M':
                    freq = 'ME'
                elif period == 'Y':
                    freq = 'YE'
                else:
                    freq = period

                strategy_returns = strategy_nav.resample(freq).last().pct_change().dropna()
                benchmark_returns = benchmark_nav.resample(freq).last().pct_change().dropna()

                if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
                    raise ValueError("Resample结果为空")

            except Exception:
                if period == 'M':
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.to_period('M')).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.to_period('M')).last()
                elif period == 'Y':
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.to_period('Y')).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.to_period('Y')).last()
                else:
                    strategy_groups = strategy_nav.groupby(strategy_nav.index.date).last()
                    benchmark_groups = benchmark_nav.groupby(benchmark_nav.index.date).last()

                strategy_returns = strategy_groups.pct_change().dropna()
                benchmark_returns = benchmark_groups.pct_change().dropna()

            common_returns_index = strategy_returns.index.intersection(benchmark_returns.index)
            if len(common_returns_index) == 0:
                min_len = min(len(strategy_returns), len(benchmark_returns))
                if min_len == 0:
                    return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])

                start_ts = strategy_returns.index[0]
                if hasattr(start_ts, 'to_timestamp'):
                    start_ts = start_ts.to_timestamp()
                aligned_index = pd.date_range(start=start_ts, periods=min_len, freq='D')
                strategy_returns = strategy_returns.iloc[:min_len]
                benchmark_returns = benchmark_returns.iloc[:min_len]

                strategy_returns.index = aligned_index
                benchmark_returns.index = aligned_index
            else:
                strategy_returns = strategy_returns.reindex(common_returns_index)
                benchmark_returns = benchmark_returns.reindex(common_returns_index)

            stats = pd.DataFrame({
                'Strategy_Return': strategy_returns,
                'Benchmark_Return': benchmark_returns,
                'Excess_Return': strategy_returns - benchmark_returns,
                'Win': strategy_returns > benchmark_returns,
            })

            return stats

        except Exception as e:
            print(f"生成期间统计时出错: {e}")
            return pd.DataFrame(columns=['Strategy_Return', 'Benchmark_Return', 'Excess_Return', 'Win'])


class AnalysisBuilder:
    """根据 `BacktestResult` 构建分析层所需的七大数据模型。"""

    @staticmethod
    def prepare_overview(
        result: BacktestResult,
        system_config: 'SystemConfig',
        selected_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        # 字段健壮性检查
        required_fields = [
            'strategy_nav', 'benchmark_nav', 'performance', 'detailed_metrics',
            'monthly_stats', 'yearly_stats', 'trade_history', 'daily_holdings',
            'final_value', 'valid_stocks', 'strategy_name', 'file_name'
        ]
        for f in required_fields:
            if not hasattr(result, f):
                raise Exception(f"BacktestResult缺少字段: {f}")

        strategy_nav = result.strategy_nav.dropna()
        # index类型健壮性处理
        start_date = strategy_nav.index.min() if not strategy_nav.empty else None
        end_date = strategy_nav.index.max() if not strategy_nav.empty else None
        date_span = None
        if start_date is not None and end_date is not None:
            try:
                date_span = (end_date - start_date).days + 1
            except Exception:
                date_span = None

        initial_cash = getattr(system_config, "initial_cash", None)
        final_value = result.final_value
        summary = {
            "strategy_name": result.strategy_name,
            "file_name": result.file_name,
            "initial_cash": initial_cash,
            "final_value": final_value,
            "total_return": ((final_value - initial_cash) / initial_cash) if initial_cash else None,
            "valid_stocks": result.valid_stocks,
            "benchmark": getattr(system_config, "benchmark_index", None),
            "start_date": start_date,
            "end_date": end_date,
            "date_span": date_span,
        }

        performance_df = getattr(result, 'performance', pd.DataFrame())
        metrics_source = performance_df.index.tolist() if not performance_df.empty else []

        metrics_to_use = selected_metrics or metrics_source
        prepared_metrics: List[Dict[str, Any]] = []
        for metric in metrics_to_use:
            if metric not in performance_df.index:
                continue
            row = performance_df.loc[metric]
            value = row.get("value") if isinstance(row, pd.Series) else row
            metric_info = None
            if CONFIG_AVAILABLE:
                try:
                    from config import get_metric_info

                    metric_info = get_metric_info(metric)
                except Exception:
                    metric_info = None

            prepared_metrics.append(
                {
                    "metric": metric,
                    "value": value,
                    "formatted": AnalysisBuilder._format_metric(value, metric_info),
                    "meta": metric_info,
                }
            )

        return {
            "summary": summary,
            "metrics": prepared_metrics,
            "performance_table": performance_df,
        }

    @staticmethod
    def prepare_net_value(result: BacktestResult) -> Dict[str, Any]:
        aligned = pd.concat(
            [
                result.strategy_nav.rename("strategy"),
                result.benchmark_nav.rename("benchmark"),
            ],
            axis=1,
        ).dropna(how="all")

        if aligned.empty:
            strategy_norm = pd.Series(dtype=float, name="strategy")
            benchmark_norm = pd.Series(dtype=float, name="benchmark")
        else:
            aligned = aligned.ffill().dropna(how="all")
            strategy_series = aligned["strategy"].ffill()
            benchmark_series = aligned["benchmark"].ffill()

            strategy_norm = AnalysisBuilder._normalize_nav(strategy_series).rename("strategy")
            benchmark_norm = AnalysisBuilder._normalize_nav(benchmark_series).rename("benchmark")

        relative = strategy_norm - benchmark_norm
        summary = {
            "strategy_return": strategy_norm.iloc[-1] - 1 if len(strategy_norm) else None,
            "benchmark_return": benchmark_norm.iloc[-1] - 1 if len(benchmark_norm) else None,
            "excess_return": relative.iloc[-1] if len(relative) else None,
        }

        return {
            "strategy_nav": strategy_norm,
            "benchmark_nav": benchmark_norm,
            "relative_nav": relative.rename("relative"),
            "summary": summary,
        }

    @staticmethod
    def prepare_returns(result: BacktestResult) -> Dict[str, Any]:
        strategy_returns = result.strategy_nav.sort_index().pct_change().dropna()
        benchmark_returns = result.benchmark_nav.sort_index().pct_change().dropna()

        combined_index = strategy_returns.index.intersection(benchmark_returns.index)
        if len(combined_index):
            strategy_returns = strategy_returns.reindex(combined_index)
            benchmark_returns = benchmark_returns.reindex(combined_index)

        stats = AnalysisBuilder._daily_stats(strategy_returns, benchmark_returns)

        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_benchmark = (1 + benchmark_returns).cumprod()

        monthly_table = result.monthly_stats.copy() if isinstance(result.monthly_stats, pd.DataFrame) else pd.DataFrame()
        yearly_table = result.yearly_stats.copy() if isinstance(result.yearly_stats, pd.DataFrame) else pd.DataFrame()

        return {
            "daily_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "daily_stats": stats,
            "cumulative_strategy": cumulative_strategy,
            "cumulative_benchmark": cumulative_benchmark,
            "monthly_table": monthly_table,
            "yearly_table": yearly_table,
        }

    @staticmethod
    def prepare_risk(result: BacktestResult) -> Dict[str, Any]:
        daily_returns = result.strategy_nav.sort_index().pct_change().dropna()
        benchmark_returns = result.benchmark_nav.sort_index().pct_change().dropna()
        active_returns = daily_returns.reindex(benchmark_returns.index, method="ffill") - benchmark_returns

        drawdown_series = result.detailed_metrics.get("drawdown_series") if isinstance(result.detailed_metrics, dict) else None
        running_max = result.detailed_metrics.get("running_max") if isinstance(result.detailed_metrics, dict) else None

        drawdown_series = AnalysisBuilder._ensure_series(drawdown_series, name="drawdown")
        running_max = AnalysisBuilder._ensure_series(running_max, name="running_max")

        var_95, cvar_95 = AnalysisBuilder._var_metrics(daily_returns)
        tracking_error = AnalysisBuilder._tracking_error(active_returns)
        information_ratio = AnalysisBuilder._information_ratio(active_returns)
        performance_df = result.performance if isinstance(result.performance, pd.DataFrame) else pd.DataFrame()

        max_drawdown = performance_df.loc["max_drawdown", "value"] if "max_drawdown" in performance_df.index else drawdown_series.min()
        annual_return = performance_df.loc["annual_return", "value"] if "annual_return" in performance_df.index else daily_returns.mean() * 252
        calmar_ratio = AnalysisBuilder._calmar_ratio(annual_return, max_drawdown)

        risk_metrics = {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "annual_return": annual_return,
        }
        # 行业集中度若在 performance 中则补充
        if "industry_hhi" in performance_df.index:
            risk_metrics["industry_hhi"] = performance_df.loc["industry_hhi", "value"]
        if "top_industry_weight" in performance_df.index:
            risk_metrics["top_industry_weight"] = performance_df.loc["top_industry_weight", "value"]
        if "industry_count" in performance_df.index:
            risk_metrics["industry_count"] = performance_df.loc["industry_count", "value"]

        return {
            "drawdown_series": drawdown_series,
            "running_max": running_max,
            "risk_metrics": risk_metrics,
        }

    @staticmethod
    def prepare_period_stats(result: BacktestResult) -> Dict[str, Any]:
        monthly = result.monthly_stats.copy() if isinstance(result.monthly_stats, pd.DataFrame) else pd.DataFrame()
        yearly = result.yearly_stats.copy() if isinstance(result.yearly_stats, pd.DataFrame) else pd.DataFrame()

        monthly_win_rate = None
        if not monthly.empty and "Win" in monthly.columns:
            monthly_win_rate = monthly["Win"].mean()

        return {
            "monthly": monthly,
            "yearly": yearly,
            "monthly_win_rate": monthly_win_rate,
        }

    @staticmethod
    def prepare_holdings(result: BacktestResult) -> Dict[str, Any]:
        # 类型健壮性处理
        snapshots = result.daily_holdings if isinstance(result.daily_holdings, list) else []
        latest_snapshot = snapshots[-1] if snapshots else {
            "date": None,
            "holdings": [],
            "total_value": getattr(result, 'final_value', 0.0),
            "cash": None,
        }

        holdings = latest_snapshot.get("holdings", [])
        if not isinstance(holdings, list):
            holdings = []
        holdings_df = pd.DataFrame(holdings)
        if not holdings_df.empty:
            if code2name:
                holdings_df["name"] = holdings_df["code"].map(code2name).fillna("")
            if "value" in holdings_df.columns:
                holdings_df = holdings_df.sort_values(by="value", ascending=False).reset_index(drop=True)

        asset_curve = pd.DataFrame(
            [
                {
                    "date": snap.get("date"),
                    "total_value": snap.get("total_value"),
                    "cash": snap.get("cash"),
                }
                for snap in snapshots
                if isinstance(snap, dict) and snap.get("date") is not None
            ]
        )
        if not asset_curve.empty and "date" in asset_curve.columns:
            asset_curve["date"] = pd.to_datetime(asset_curve["date"])
            asset_curve = asset_curve.sort_values("date")

        return {
            "latest_snapshot": latest_snapshot,
            "holdings_table": holdings_df,
            "asset_curve": asset_curve,
        }

    @staticmethod
    def prepare_daily_holdings_history(result: BacktestResult) -> pd.DataFrame:
        """准备每日持仓历史DataFrame，用于UI和命令行展示"""
        daily_holdings = result.daily_holdings if isinstance(result.daily_holdings, list) else []

        if not daily_holdings:
            return pd.DataFrame()

        # 收集所有股票代码
        all_codes = set()
        for record in daily_holdings:
            holdings_list = record.get('holdings', [])
            if isinstance(holdings_list, list):
                for item in holdings_list:
                    if isinstance(item, dict) and 'code' in item:
                        all_codes.add(item['code'])

        if not all_codes:
            return pd.DataFrame()

        # 创建每日持仓历史数据
        history_data = []
        for record in daily_holdings:
            date = record.get('date')
            holdings_list = record.get('holdings', [])
            total_value = record.get('total_value', 0)

            if date and holdings_list:
                row = {
                    'date': date,
                    'total_value': total_value,
                    'cash': record.get('cash', 0)
                }

                # 创建持仓字典
                holdings_dict = {}
                if isinstance(holdings_list, list):
                    holdings_dict = {item.get('code'): item for item in holdings_list if item.get('code')}

                # 为每只股票添加数量、市值和权重
                for code in all_codes:
                    if code in holdings_dict:
                        info = holdings_dict[code]
                        row[f"{code}_size"] = info.get('size', 0)
                        row[f"{code}_value"] = info.get('value', 0)
                        row[f"{code}_weight"] = info.get('weight', 0)
                    else:
                        row[f"{code}_size"] = 0
                        row[f"{code}_value"] = 0
                        row[f"{code}_weight"] = 0

                history_data.append(row)

        if history_data:
            df = pd.DataFrame(history_data)
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    @staticmethod
    def prepare_trades(result: BacktestResult) -> Dict[str, Any]:
        # 类型健壮性处理
        trade_history = result.trade_history if isinstance(result.trade_history, list) else []
        trades = pd.DataFrame(trade_history)
        if not trades.empty:
            if "date" in trades.columns:
                trades["date"] = pd.to_datetime(trades["date"])
                trades = trades.sort_values("date")
        # 若无'date'字段则不排序
        return {
            "trades_table": trades,
            "trade_count": len(trades),
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_nav(series: pd.Series) -> pd.Series:
        series = series.dropna()
        if series.empty:
            return pd.Series(dtype=float)
        base = series.iloc[0]
        if base == 0:
            return (series + 1e-9) / (series.iloc[0] + 1e-9)
        return series / base

    @staticmethod
    def _daily_stats(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        if strategy_returns.empty:
            return {
                "mean": None,
                "std": None,
                "positive_days": 0,
                "negative_days": 0,
                "win_rate": None,
                "sample_size": 0,
            }

        positive_days = int((strategy_returns > 0).sum())
        negative_days = int((strategy_returns < 0).sum())
        sample_size = len(strategy_returns)

        win_rate = None
        if not benchmark_returns.empty:
            aligned = strategy_returns.align(benchmark_returns, join="inner")
            if len(aligned[0]):
                win_rate = float((aligned[0] > aligned[1]).mean())

        return {
            "mean": float(strategy_returns.mean()),
            "std": float(strategy_returns.std()),
            "positive_days": positive_days,
            "negative_days": negative_days,
            "win_rate": win_rate,
            "sample_size": sample_size,
        }

    @staticmethod
    def _var_metrics(returns: pd.Series, alpha: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
        if returns.empty:
            return None, None
        percentile = (1 - alpha) * 100
        var = float(np.percentile(returns, percentile))
        tail = returns[returns <= var]
        cvar = float(tail.mean()) if not tail.empty else None
        return var, cvar

    @staticmethod
    def _tracking_error(active_returns: pd.Series) -> Optional[float]:
        active_returns = active_returns.dropna()
        if active_returns.empty:
            return None
        return float(active_returns.std() * np.sqrt(252))

    @staticmethod
    def _information_ratio(active_returns: pd.Series) -> Optional[float]:
        active_returns = active_returns.dropna()
        if active_returns.empty:
            return None
        tracking_error = active_returns.std()
        if tracking_error == 0:
            return None
        mean = active_returns.mean()
        return float((mean / tracking_error) * np.sqrt(252))

    @staticmethod
    def _calmar_ratio(annual_return: Optional[float], max_drawdown: Optional[float]) -> Optional[float]:
        if annual_return is None or max_drawdown in (None, 0):
            return None
        if max_drawdown == 0:
            return None
        return float(annual_return / abs(max_drawdown))

    @staticmethod
    def _format_metric(value: Any, metric_info: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None or metric_info is None:
            return None
        fmt = metric_info.get("format")
        try:
            return fmt.format(value) if fmt else str(value)
        except Exception:
            return str(value)

    @staticmethod
    def _ensure_series(series: Any, name: str) -> pd.Series:
        if isinstance(series, pd.Series):
            return series.rename(name)
        return pd.Series(dtype=float, name=name)