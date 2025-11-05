"""test_kill_switch: comprehensive tests for Kill Switch mechanism
"""
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from live_trading.kill_switch import (
    KillSwitch, KillSwitchConfig, KillSwitchState,
    KillSwitchTrigger, KillSwitchEvent, RealTimeMonitor
)


class TestKillSwitch:
    """Test cases for Kill Switch mechanism"""

    def test_initial_state(self):
        """Test initial state is ACTIVE"""
        ks = KillSwitch()
        assert ks.current_state == KillSwitchState.ACTIVE
        assert ks.can_trade() == True

    def test_market_crash_trigger(self):
        """Test market crash triggers halt"""
        ks = KillSwitch()

        # Mock market crash
        with patch.object(ks, '_get_market_return', return_value=-0.06):
            ks._check_market_conditions()

        assert ks.current_state == KillSwitchState.HALTED
        assert ks.can_trade() == False

    def test_high_volatility_trigger(self):
        """Test high volatility triggers pause"""
        ks = KillSwitch()

        with patch.object(ks, '_get_market_volatility', return_value=0.10):
            ks._check_market_conditions()

        assert ks.current_state == KillSwitchState.PAUSED
        assert ks.can_trade("buy") == False  # Cannot buy in paused state
        assert ks.can_trade("sell") == True  # Can sell in paused state

    def test_order_failures_trigger(self):
        """Test excessive order failures trigger warning"""
        ks = KillSwitch()

        with patch.object(ks, '_get_recent_failures', return_value=15):
            ks._check_system_health()

        assert ks.current_state == KillSwitchState.WARNING
        assert ks.can_trade() == True  # Can still trade in warning state

    def test_time_based_pause(self):
        """Test time-based trading pause before market close"""
        ks = KillSwitch(KillSwitchConfig(pause_before_close_minutes=30))

        # Mock time to be 30 minutes before close (14:30)
        mock_now = datetime.combine(datetime.today(), datetime.strptime("14:30", "%H:%M").time())

        with patch('live_trading.kill_switch.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.combine = datetime.combine
            mock_datetime.strptime = datetime.strptime
            mock_datetime.today.return_value = datetime.today()

            ks._check_time_based_rules()

        assert ks.current_state == KillSwitchState.PAUSED

    def test_manual_resume(self):
        """Test manual resume from paused state"""
        ks = KillSwitch()

        # Trigger pause
        ks.trigger(KillSwitchTrigger.MANUAL_OVERRIDE, "Test pause")
        assert ks.current_state == KillSwitchState.PAUSED

        # Resume
        result = ks.resume("Manual resume")
        assert result == True
        assert ks.current_state == KillSwitchState.ACTIVE

    def test_cannot_resume_from_halted(self):
        """Test cannot resume from halted state"""
        ks = KillSwitch()

        # Trigger halt
        ks.trigger(KillSwitchTrigger.MARKET_CRASH, "Market crash")
        assert ks.current_state == KillSwitchState.HALTED

        # Try to resume
        result = ks.resume("Attempt resume")
        assert result == False
        assert ks.current_state == KillSwitchState.HALTED

    def test_state_change_callbacks(self):
        """Test state change callback notifications"""
        ks = KillSwitch()

        callback_calls = []
        def test_callback(new_state, event):
            callback_calls.append((new_state, event.trigger))

        ks.add_state_change_callback(test_callback)

        # Trigger state change
        ks.trigger(KillSwitchTrigger.MARKET_CRASH, "Test")

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == KillSwitchState.HALTED
        assert callback_calls[0][1] == KillSwitchTrigger.MARKET_CRASH

    def test_event_history(self):
        """Test event history tracking"""
        ks = KillSwitch()

        # Trigger multiple events
        ks.trigger(KillSwitchTrigger.MARKET_CRASH, "Event 1")
        ks.trigger(KillSwitchTrigger.HIGH_VOLATILITY, "Event 2")

        assert len(ks.event_history) == 2
        assert ks.event_history[0].trigger == KillSwitchTrigger.MARKET_CRASH
        assert ks.event_history[1].trigger == KillSwitchTrigger.HIGH_VOLATILITY

    def test_get_status(self):
        """Test status reporting"""
        ks = KillSwitch()

        status = ks.get_status()
        assert status["state"] == "active"
        assert status["can_trade"] == True
        assert "last_change" in status

    def test_monitoring_thread(self):
        """Test monitoring thread lifecycle"""
        ks = KillSwitch()

        # Start monitoring
        ks.start_monitoring()
        assert ks.monitoring_active == True

        # Wait a bit for thread to start
        time.sleep(0.1)

        # Stop monitoring
        ks.stop_monitoring()
        assert ks.monitoring_active == False


class TestRealTimeMonitor:
    """Test cases for Real-Time Monitor"""

    def test_market_indicator_update(self):
        """Test market indicator updates"""
        ks = KillSwitch()
        monitor = RealTimeMonitor(ks)

        # Update market indicator
        monitor.update_market_data("index_return", -0.03)

        assert "index_return" in monitor.market_indicators
        assert monitor.market_indicators["index_return"]["value"] == -0.03

    def test_system_metric_update(self):
        """Test system metric updates"""
        ks = KillSwitch()
        monitor = RealTimeMonitor(ks)

        # Update system metric
        monitor.update_system_metric("cpu_usage", 85.5)

        assert "cpu_usage" in monitor.system_metrics
        assert monitor.system_metrics["cpu_usage"]["value"] == 85.5

    def test_market_alert_trigger(self):
        """Test market alert triggers kill switch"""
        ks = KillSwitch()
        monitor = RealTimeMonitor(ks)

        # Trigger market crash alert
        monitor.update_market_data("index_return", -0.08)

        assert ks.current_state == KillSwitchState.HALTED

    def test_system_alert_trigger(self):
        """Test system alert triggers kill switch"""
        ks = KillSwitch()
        monitor = RealTimeMonitor(ks)

        # Trigger high failure rate alert
        monitor.update_system_metric("order_failure_rate", 0.15)

        assert ks.current_state == KillSwitchState.WARNING

    def test_monitoring_report(self):
        """Test comprehensive monitoring report"""
        ks = KillSwitch()
        monitor = RealTimeMonitor(ks)

        # Add some data
        monitor.update_market_data("vix", 25.0)
        monitor.update_system_metric("memory_usage", 78.5)

        report = monitor.get_monitoring_report()

        assert "kill_switch_status" in report
        assert "market_indicators" in report
        assert "system_metrics" in report
        assert report["market_indicators"]["vix"]["value"] == 25.0
        assert report["system_metrics"]["memory_usage"]["value"] == 78.5


class TestKillSwitchIntegration:
    """Integration tests for Kill Switch system"""

    def test_full_workflow(self):
        """Test complete kill switch workflow"""
        config = KillSwitchConfig(
            market_crash_threshold=-0.03,  # More sensitive for testing
            require_manual_resume=False,
            auto_resume_delay_minutes=0  # Immediate auto-resume for testing
        )
        ks = KillSwitch(config)

        # Start with active state
        assert ks.current_state == KillSwitchState.ACTIVE

        # Trigger warning
        ks.trigger(KillSwitchTrigger.ORDER_FAILURES, "Test warning")
        assert ks.current_state == KillSwitchState.WARNING

        # Trigger pause
        ks.trigger(KillSwitchTrigger.HIGH_VOLATILITY, "Test pause")
        assert ks.current_state == KillSwitchState.PAUSED

        # Auto resume should work
        time.sleep(0.1)  # Allow auto-resume check
        # Note: In real implementation, this would need the monitoring loop

        # Manual resume
        ks.resume("Manual intervention")
        assert ks.current_state == KillSwitchState.ACTIVE

    def test_concurrent_access(self):
        """Test thread-safe concurrent access"""
        ks = KillSwitch()
        results = []

        def worker(worker_id):
            for i in range(10):
                ks.trigger(KillSwitchTrigger.MANUAL_OVERRIDE, f"Test {worker_id}-{i}")
                results.append(f"worker_{worker_id}_trigger_{i}")
                status = ks.get_status()
                results.append(f"worker_{worker_id}_status_{status['state']}")

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify results
        assert len(results) == 60  # 3 workers * 20 operations each
        assert ks.current_state == KillSwitchState.PAUSED  # Last trigger state (MANUAL_OVERRIDE -> PAUSED)

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = KillSwitchConfig()
        ks = KillSwitch(config)
        assert ks.config.market_crash_threshold == -0.05

        # Custom config
        custom_config = KillSwitchConfig(
            market_crash_threshold=-0.03,
            max_order_failures=5,
            auto_resume_delay_minutes=30
        )
        ks_custom = KillSwitch(custom_config)
        assert ks_custom.config.market_crash_threshold == -0.03
        assert ks_custom.config.max_order_failures == 5