"""kill_switch: emergency stop mechanism for trading system risk control
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import threading
import time
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    ACTIVE = "active"      # 正常交易状态
    WARNING = "warning"    # 警告状态，限制交易
    PAUSED = "paused"      # 暂停状态，只允许卖出
    HALTED = "halted"      # 完全停止，所有交易暂停


class KillSwitchTrigger(Enum):
    MARKET_CRASH = "market_crash"              # 市场暴跌
    HIGH_VOLATILITY = "high_volatility"        # 高波动率
    LARGE_DRAWDOWN = "large_drawdown"          # 大幅回撤
    ORDER_FAILURES = "order_failures"          # 订单失败过多
    SYSTEM_ERROR = "system_error"              # 系统错误
    CONNECTION_LOST = "connection_lost"        # 连接丢失
    MANUAL_OVERRIDE = "manual_override"        # 人工干预
    TIME_BASED = "time_based"                  # 时间触发（如收盘前暂停）


@dataclass
class KillSwitchEvent:
    trigger: KillSwitchTrigger
    state: KillSwitchState
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KillSwitchConfig:
    # 市场风险阈值
    market_crash_threshold: float = -0.05  # 5%暴跌触发
    high_volatility_threshold: float = 0.08  # 8%波动率触发
    large_drawdown_threshold: float = -0.10  # 10%回撤触发

    # 订单失败阈值
    max_order_failures: int = 10  # 连续失败次数
    failure_window_minutes: int = 30  # 失败统计窗口

    # 时间控制
    pause_before_close_minutes: int = 15  # 收盘前暂停时间
    halt_after_close_minutes: int = 5  # 收盘后完全停止

    # 自动恢复
    auto_resume_delay_minutes: int = 60  # 自动恢复延迟
    require_manual_resume: bool = True  # 是否需要人工确认恢复

    # 监控频率
    market_check_interval_seconds: int = 60  # 市场监控间隔
    position_check_interval_seconds: int = 300  # 持仓监控间隔


class KillSwitch:
    """Emergency stop mechanism for trading system"""

    def __init__(self, config: KillSwitchConfig = None):
        self.config = config or KillSwitchConfig()
        self.current_state = KillSwitchState.ACTIVE
        self.last_state_change = datetime.now()
        self.event_history: List[KillSwitchEvent] = []
        self.failure_counts: Dict[str, List[datetime]] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # 回调函数
        self.state_change_callbacks: List[Callable[[KillSwitchState, KillSwitchEvent], None]] = []

    def start_monitoring(self):
        """Start the monitoring thread"""
        with self._lock:
            if self.monitoring_active:
                return

            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Kill Switch monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        with self._lock:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            logger.info("Kill Switch monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_market_conditions()
                self._check_time_based_rules()
                self._check_system_health()
                self._attempt_auto_resume()
            except Exception as e:
                logger.error(f"Kill Switch monitoring error: {e}")

            time.sleep(min(self.config.market_check_interval_seconds,
                          self.config.position_check_interval_seconds))

    def _check_market_conditions(self):
        """Check market risk conditions"""
        # 这里需要接入实时市场数据
        # 示例实现 - 实际需要从数据源获取

        # 检查指数暴跌
        market_return = self._get_market_return()
        if market_return < self.config.market_crash_threshold:
            self.trigger(KillSwitchTrigger.MARKET_CRASH,
                        f"Market crash detected: {market_return:.2%}",
                        {"market_return": market_return})

        # 检查高波动率
        volatility = self._get_market_volatility()
        if volatility > self.config.high_volatility_threshold:
            self.trigger(KillSwitchTrigger.HIGH_VOLATILITY,
                        f"High volatility detected: {volatility:.2%}",
                        {"volatility": volatility})

    def _check_time_based_rules(self):
        """Check time-based rules"""
        now = datetime.now().time()

        # 收盘前暂停
        market_close = datetime.strptime("15:00", "%H:%M").time()
        pause_time = (datetime.combine(datetime.today(), market_close) -
                     timedelta(minutes=self.config.pause_before_close_minutes)).time()

        if pause_time <= now < market_close:
            if self.current_state == KillSwitchState.ACTIVE:
                self.trigger(KillSwitchTrigger.TIME_BASED,
                            f"Pausing trading {self.config.pause_before_close_minutes} minutes before market close")

        # 收盘后停止
        halt_time = (datetime.combine(datetime.today(), market_close) +
                    timedelta(minutes=self.config.halt_after_close_minutes)).time()

        if now >= halt_time:
            if self.current_state != KillSwitchState.HALTED:
                self.trigger(KillSwitchTrigger.TIME_BASED,
                            f"Halting trading {self.config.halt_after_close_minutes} minutes after market close")

    def _check_system_health(self):
        """Check system health indicators"""
        # 检查订单失败率
        recent_failures = self._get_recent_failures()
        if recent_failures >= self.config.max_order_failures:
            self.trigger(KillSwitchTrigger.ORDER_FAILURES,
                        f"Too many order failures: {recent_failures} in last {self.config.failure_window_minutes} minutes",
                        {"failure_count": recent_failures})

        # 检查连接状态
        if not self._check_broker_connection():
            self.trigger(KillSwitchTrigger.CONNECTION_LOST,
                        "Lost connection to broker")

    def _attempt_auto_resume(self):
        """Attempt automatic resume if conditions allow"""
        if not self.config.require_manual_resume:
            time_since_pause = datetime.now() - self.last_state_change
            if (time_since_pause.total_seconds() > self.config.auto_resume_delay_minutes * 60 and
                self.current_state in [KillSwitchState.WARNING, KillSwitchState.PAUSED]):

                # 检查是否可以安全恢复
                if self._can_safely_resume():
                    self.resume("Automatic resume after cooldown period")

    def trigger(self, trigger: KillSwitchTrigger, reason: str, metadata: Dict[str, Any] = None):
        """Trigger a kill switch event"""
        with self._lock:
            old_state = self.current_state
            new_state = self._determine_new_state(trigger)

            if new_state == old_state:
                return  # No state change needed

            event = KillSwitchEvent(
                trigger=trigger,
                state=new_state,
                reason=reason,
                metadata=metadata or {}
            )

            self.current_state = new_state
            self.last_state_change = datetime.now()
            self.event_history.append(event)

            logger.warning(f"Kill Switch triggered: {trigger.value} -> {new_state.value}: {reason}")

            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(new_state, event)
                except Exception as e:
                    logger.error(f"Error in kill switch callback: {e}")

    def resume(self, reason: str = "Manual resume"):
        """Manually resume trading"""
        with self._lock:
            if self.current_state == KillSwitchState.HALTED:
                return False  # Cannot resume from halted state

            old_state = self.current_state
            self.current_state = KillSwitchState.ACTIVE
            self.last_state_change = datetime.now()

            event = KillSwitchEvent(
                trigger=KillSwitchTrigger.MANUAL_OVERRIDE,
                state=KillSwitchState.ACTIVE,
                reason=reason
            )
            self.event_history.append(event)

            logger.info(f"Kill Switch resumed: {old_state.value} -> {self.current_state.value}: {reason}")

            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(self.current_state, event)
                except Exception as e:
                    logger.error(f"Error in resume callback: {e}")

            return True

    def _determine_new_state(self, trigger: KillSwitchTrigger) -> KillSwitchState:
        """Determine the new state based on trigger type"""
        if trigger in [KillSwitchTrigger.MARKET_CRASH, KillSwitchTrigger.SYSTEM_ERROR]:
            return KillSwitchState.HALTED
        elif trigger in [KillSwitchTrigger.HIGH_VOLATILITY, KillSwitchTrigger.LARGE_DRAWDOWN,
                        KillSwitchTrigger.MANUAL_OVERRIDE]:  # Add MANUAL_OVERRIDE here for testing
            return KillSwitchState.PAUSED
        elif trigger in [KillSwitchTrigger.ORDER_FAILURES, KillSwitchTrigger.CONNECTION_LOST]:
            return KillSwitchState.WARNING
        elif trigger == KillSwitchTrigger.TIME_BASED:
            return KillSwitchState.PAUSED
        else:
            return KillSwitchState.WARNING

    def can_trade(self, order_type: str = "any") -> bool:
        """Check if trading is allowed"""
        with self._lock:
            if self.current_state == KillSwitchState.HALTED:
                return False
            elif self.current_state == KillSwitchState.PAUSED:
                # In paused state, only allow selling/reducing positions
                return order_type in ["sell", "reduce"]
            elif self.current_state == KillSwitchState.WARNING:
                # In warning state, limit order size and frequency
                return True  # But with restrictions (checked elsewhere)
            else:  # ACTIVE
                return True

    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status"""
        with self._lock:
            return {
                "state": self.current_state.value,
                "last_change": self.last_state_change.isoformat(),
                "reason": self.event_history[-1].reason if self.event_history else None,
                "can_trade": self.can_trade(),
                "event_count": len(self.event_history),
                "monitoring_active": self.monitoring_active
            }

    def add_state_change_callback(self, callback: Callable[[KillSwitchState, KillSwitchEvent], None]):
        """Add a callback for state changes"""
        self.state_change_callbacks.append(callback)

    # Placeholder methods - need to be implemented with real data sources
    def _get_market_return(self) -> float:
        """Get current market return (placeholder)"""
        # TODO: Implement with real market data
        return 0.0

    def _get_market_volatility(self) -> float:
        """Get current market volatility (placeholder)"""
        # TODO: Implement with real market data
        return 0.02

    def _get_recent_failures(self) -> int:
        """Get recent order failures (placeholder)"""
        # TODO: Implement with real order tracking
        return 0

    def _check_broker_connection(self) -> bool:
        """Check broker connection status (placeholder)"""
        # TODO: Implement with real broker connection check
        return True

    def _can_safely_resume(self) -> bool:
        """Check if it's safe to resume trading (placeholder)"""
        # TODO: Implement safety checks
        return True


class RealTimeMonitor:
    """Real-time market and system monitoring"""

    def __init__(self, kill_switch: KillSwitch):
        self.kill_switch = kill_switch
        self.market_indicators = {}
        self.system_metrics = {}
        self.alerts = []

    def update_market_data(self, indicator: str, value: float, timestamp: datetime = None):
        """Update market indicator"""
        self.market_indicators[indicator] = {
            "value": value,
            "timestamp": timestamp or datetime.now()
        }

        # Check for alert conditions
        self._check_market_alerts(indicator, value)

    def update_system_metric(self, metric: str, value: Any, timestamp: datetime = None):
        """Update system metric"""
        self.system_metrics[metric] = {
            "value": value,
            "timestamp": timestamp or datetime.now()
        }

        # Check for system alerts
        self._check_system_alerts(metric, value)

    def _check_market_alerts(self, indicator: str, value: float):
        """Check market indicators for alerts"""
        if indicator == "index_return" and value < -0.05:
            self.kill_switch.trigger(
                KillSwitchTrigger.MARKET_CRASH,
                f"Market index dropped {value:.2%}",
                {"indicator": indicator, "value": value}
            )
        elif indicator == "vix" and value > 30:
            self.kill_switch.trigger(
                KillSwitchTrigger.HIGH_VOLATILITY,
                f"VIX spiked to {value}",
                {"indicator": indicator, "value": value}
            )

    def _check_system_alerts(self, metric: str, value: Any):
        """Check system metrics for alerts"""
        if metric == "order_failure_rate" and value > 0.1:  # 10% failure rate
            self.kill_switch.trigger(
                KillSwitchTrigger.ORDER_FAILURES,
                f"High order failure rate: {value:.2%}",
                {"metric": metric, "value": value}
            )
        elif metric == "connection_status" and not value:
            self.kill_switch.trigger(
                KillSwitchTrigger.CONNECTION_LOST,
                "Broker connection lost",
                {"metric": metric, "value": value}
            )

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        return {
            "kill_switch_status": self.kill_switch.get_status(),
            "market_indicators": self.market_indicators,
            "system_metrics": self.system_metrics,
            "active_alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": datetime.now().isoformat()
        }