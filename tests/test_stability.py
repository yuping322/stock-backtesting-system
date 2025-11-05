"""tests for stability enhancements: advanced drift detection, prediction validation, and error recovery
"""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from live_trading.drift_detector import AdvancedDriftDetector, HealthStatus, ModelHealthMetrics
from live_trading.prediction_validator import PredictionValidator, ValidationSeverity, ValidationResult
from live_trading.pipeline_runner import ResilientPipelineRunner, RecoveryStrategy, ErrorContext


class TestAdvancedDriftDetector:
    """Test advanced model health monitoring"""

    def setup_method(self):
        self.detector = AdvancedDriftDetector(
            window=5,
            ic_floor=0.03,
            psi_threshold=0.1,
            ks_threshold=0.1,
            distribution_shift_threshold=0.15
        )

    def test_initial_state(self):
        """Test initial detector state"""
        summary = self.detector.get_health_summary()
        assert summary['total_observations'] == 0
        assert summary['baseline_established'] is False
        assert summary['ic_history_length'] == 0

    def test_ic_calculation(self):
        """Test IC calculation with mock data"""
        # Create mock predictions and returns
        predictions = pd.DataFrame({
            'model': ['model1'] * 10,
            'date': ['20240101'] * 10,
            'code': [f'00000{i}' for i in range(10)],
            'score': np.random.randn(10)
        })

        returns = pd.DataFrame({
            'date': ['20240101'] * 10,
            'code': [f'00000{i}' for i in range(10)],
            'ret': np.random.randn(10) * 0.02
        })

        metrics = self.detector.update_and_check(predictions, returns)

        assert isinstance(metrics, ModelHealthMetrics)
        assert metrics.ic is not None
        assert isinstance(metrics.health_status, HealthStatus)

    def test_distribution_drift_detection(self):
        """Test distribution drift detection after baseline establishment"""
        # First run to establish baseline
        predictions1 = pd.DataFrame({
            'model': ['model1'] * 20,
            'date': ['20240101'] * 20,
            'code': [f'00000{i}' for i in range(20)],
            'score': np.random.normal(0, 1, 20)  # Normal distribution
        })

        # Update multiple times to establish baseline
        for _ in range(6):
            self.detector.update_and_check(predictions1)

        # Second run with shifted distribution
        predictions2 = pd.DataFrame({
            'model': ['model1'] * 20,
            'date': ['20240102'] * 20,
            'code': [f'00000{i}' for i in range(20)],
            'score': np.random.normal(0.5, 1.2, 20)  # Shifted distribution
        })

        metrics = self.detector.update_and_check(predictions2)

        # Should detect some drift
        assert metrics.psi_score is not None or metrics.ks_statistic is not None

    def test_health_status_assessment(self):
        """Test health status assessment logic"""
        # Create metrics that should trigger warnings
        metrics = ModelHealthMetrics(
            ic=-0.1,  # Negative IC
            rolling_ic=0.01,  # Below threshold
            outlier_ratio=0.08  # High outliers
        )

        status = self.detector._assess_health_status(metrics)
        assert status in [HealthStatus.WARNING, HealthStatus.CRITICAL]

    def test_baseline_reset(self):
        """Test baseline reset functionality"""
        # Establish some history
        predictions = pd.DataFrame({
            'model': ['model1'] * 5,
            'date': ['20240101'] * 5,
            'code': [f'00000{i}' for i in range(5)],
            'score': [0.1, 0.2, 0.3, 0.4, 0.5]
        })

        self.detector.update_and_check(predictions)
        assert len(self.detector.history_predictions) > 0

        # Reset baseline
        self.detector.reset_baseline()
        summary = self.detector.get_health_summary()
        assert summary['total_observations'] == 0
        assert summary['baseline_established'] is False


class TestPredictionValidator:
    """Test prediction data validation"""

    def setup_method(self):
        self.validator = PredictionValidator()

    def test_valid_data_validation(self):
        """Test validation of valid prediction data"""
        valid_df = pd.DataFrame({
            'model': ['model1', 'model2'] * 5,
            'date': ['20240101'] * 10,
            'code': [f'00000{i}' for i in range(10)],
            'score': np.random.uniform(-1, 1, 10)
        })

        report = self.validator.validate(valid_df)

        assert report.is_valid
        assert report.passed_checks > 0
        assert len(report.errors) == 0

    def test_missing_columns_validation(self):
        """Test validation with missing required columns"""
        invalid_df = pd.DataFrame({
            'model': ['model1'] * 5,
            'date': ['20240101'] * 5,
            # Missing 'code' and 'score' columns
        })

        report = self.validator.validate(invalid_df)

        assert not report.is_valid
        assert len(report.errors) > 0
        assert any("Missing required columns" in error.message for error in report.errors)

    def test_score_range_validation(self):
        """Test score range validation"""
        out_of_range_df = pd.DataFrame({
            'model': ['model1'] * 5,
            'date': ['20240101'] * 5,
            'code': [f'00000{i}' for i in range(5)],
            'score': [1.5, -2.0, 0.5, 0.8, 1.2]  # Some outside [-1, 1] range
        })

        report = self.validator.validate(out_of_range_df)

        # Check that score range validation failed
        score_range_checks = [r for r in report.errors + report.warnings
                            if r.check_name == 'score_range']
        assert len(score_range_checks) > 0
        assert not score_range_checks[0].passed

    def test_duplicate_detection(self):
        """Test duplicate prediction detection"""
        duplicate_df = pd.DataFrame({
            'model': ['model1'] * 6,
            'date': ['20240101'] * 6,
            'code': ['000001', '000002', '000001', '000002', '000003', '000001'],  # Duplicates
            'score': [0.1, 0.2, 0.1, 0.2, 0.3, 0.1]
        })

        report = self.validator.validate(duplicate_df)

        # Should detect duplicates but might not fail validation depending on ratio
        duplicate_checks = [r for r in report.errors + report.warnings
                          if "duplicate" in r.message.lower()]
        assert len(duplicate_checks) > 0

    def test_outlier_detection(self):
        """Test outlier detection in predictions"""
        # Create data with clear outliers (more than 5% to trigger warning)
        normal_scores = np.random.normal(0, 0.1, 90)  # Normal scores
        outlier_scores = [5.0, -3.0, 4.0, -4.0, 6.0, 7.0, -5.0, 8.0]  # 8 outliers = 8%
        all_scores = np.concatenate([normal_scores, outlier_scores])

        outlier_df = pd.DataFrame({
            'model': ['model1'] * 98,
            'date': ['20240101'] * 98,
            'code': [f'00000{i:03d}' for i in range(98)],
            'score': all_scores
        })

        report = self.validator.validate(outlier_df)

        # Should detect high outlier ratio (check_name="outliers" or message contains "outlier")
        outlier_checks = [r for r in report.warnings + report.errors + report.info
                         if "outlier" in r.check_name.lower() or "outlier" in r.message.lower()]
        assert len(outlier_checks) > 0, "Should detect outliers in the validation report"

    def test_validation_history(self):
        """Test validation history tracking"""
        test_df = pd.DataFrame({
            'model': ['model1'] * 3,
            'date': ['20240101'] * 3,
            'code': ['000001', '000002', '000003'],
            'score': [0.1, 0.2, 0.3]
        })

        # Run multiple validations
        for _ in range(3):
            self.validator.validate(test_df)

        history = self.validator.get_validation_history()
        assert len(history) == 3

        summary = self.validator.get_validation_summary()
        assert summary['total_validations'] == 3


class TestResilientPipelineRunner:
    """Test error recovery in pipeline runner"""

    def setup_method(self):
        # Create mocks for all dependencies
        self.data_provider = Mock()
        self.prediction_loader = Mock()
        self.portfolio_builder = Mock()
        self.risk_engine = Mock()
        self.execution_engine = Mock()
        self.state_store = Mock()
        self.drift_detector = Mock()

        self.runner = ResilientPipelineRunner(
            data_provider=self.data_provider,
            prediction_loader=self.prediction_loader,
            portfolio_builder=self.portfolio_builder,
            risk_engine=self.risk_engine,
            execution_engine=self.execution_engine,
            state_store=self.state_store,
            drift_detector=self.drift_detector,
            max_retries=2,
            enable_degraded_mode=True
        )

    def test_successful_pipeline_run(self):
        """Test successful pipeline execution"""
        # Setup mocks to return valid data
        self.data_provider.load_universe.return_value = pd.DataFrame({'code': ['000001', '000002']})
        self.data_provider.fetch_basic_panel.return_value = pd.DataFrame({
            'code': ['000001', '000002'],
            'price': [10.0, 20.0]
        })
        self.data_provider.fetch_suspension.return_value = pd.DataFrame()
        self.data_provider.build_blacklist.return_value = pd.DataFrame()
        self.data_provider.fetch_industry.return_value = pd.DataFrame()

        self.prediction_loader.load_from_df.return_value = None
        aggregated_df = pd.DataFrame({
            'code': ['000001', '000002'],
            'mean_score': [0.1, 0.2]
        })
        self.prediction_loader.aggregate_mean.return_value = aggregated_df

        weights_df = pd.DataFrame({
            'code': ['000001', '000002'],
            'weight': [0.6, 0.4]
        })
        self.portfolio_builder.build.return_value = weights_df

        self.risk_engine.run.return_value = {
            'weights': weights_df,
            'risk_logs': []
        }

        self.state_store.snapshot_positions.return_value = {}
        self.execution_engine.compute_diff.return_value = [{'code': '000001', 'weight': 0.6}]
        self.execution_engine.submit_orders.return_value = None

        self.drift_detector.update_and_check.return_value = {}

        predictions_df = pd.DataFrame({
            'model': ['model1'],
            'date': ['20240101'],
            'code': ['000001'],
            'score': [0.1]
        })

        result = self.runner.run('20240101', predictions_df)

        assert result['pipeline_status'] == 'success'
        assert 'recovery_info' in result
        assert result['recovery_info']['errors_encountered'] == []

    def test_data_fetch_error_recovery(self):
        """Test recovery from data fetch errors"""
        # Make data fetch fail
        self.data_provider.load_universe.side_effect = Exception("Network error")
        self.data_provider.fetch_basic_panel.side_effect = Exception("API error")

        # Setup successful prediction processing
        self.prediction_loader.load_from_df.return_value = None
        self.prediction_loader.aggregate_mean.return_value = pd.DataFrame({
            'code': ['000001'],
            'mean_score': [0.1]
        })

        predictions_df = pd.DataFrame({
            'model': ['model1'],
            'date': ['20240101'],
            'code': ['000001'],
            'score': [0.1]
        })

        result = self.runner.run('20240101', predictions_df)

        # Should still succeed due to recovery
        assert result['pipeline_status'] == 'success'
        assert len(result['recovery_info']['errors_encountered']) > 0
        assert len(result['recovery_info']['recoveries_attempted']) > 0

    def test_prediction_validation_error_handling(self):
        """Test handling of prediction validation errors"""
        # Create invalid predictions (missing required columns)
        invalid_predictions = pd.DataFrame({
            'model': ['model1'],
            'date': ['20240101'],
            # Missing 'code' and 'score'
        })

        result = self.runner.run('20240101', invalid_predictions)

        # Should handle gracefully in degraded mode
        assert result['pipeline_status'] == 'success'
        assert result['recovery_info']['validation_report'] is not None
        assert not result['recovery_info']['validation_report'].is_valid

    @patch('time.sleep')  # Mock sleep to speed up test
    def test_retry_logic(self, mock_sleep):
        """Test retry logic for failed operations"""
        # Make operation fail twice then succeed
        self.data_provider.load_universe.side_effect = [
            Exception("Attempt 1 failed"),
            Exception("Attempt 2 failed"),
            pd.DataFrame({'code': ['000001']})  # Success on third try
        ]

        # Setup other mocks to succeed
        self.data_provider.fetch_basic_panel.return_value = pd.DataFrame({
            'code': ['000001'],
            'price': [10.0]
        })
        self.data_provider.fetch_suspension.return_value = pd.DataFrame()
        self.data_provider.build_blacklist.return_value = pd.DataFrame()
        self.data_provider.fetch_industry.return_value = pd.DataFrame()

        self.prediction_loader.load_from_df.return_value = None
        self.prediction_loader.aggregate_mean.return_value = pd.DataFrame({
            'code': ['000001'],
            'mean_score': [0.1]
        })

        predictions_df = pd.DataFrame({
            'model': ['model1'],
            'date': ['20240101'],
            'code': ['000001'],
            'score': [0.1]
        })

        result = self.runner.run('20240101', predictions_df)

        assert result['pipeline_status'] == 'success'
        # Should have recorded the errors from failed attempts
        assert len(result['recovery_info']['errors_encountered']) >= 2  # At least 2 failures

    def test_health_status_reporting(self):
        """Test system health status reporting"""
        health_status = self.runner.get_health_status()

        assert 'degraded_components' in health_status
        assert 'recent_errors' in health_status
        assert 'system_status' in health_status

    def test_error_state_reset(self):
        """Test error state reset functionality"""
        # Simulate some errors
        self.data_provider.load_universe.side_effect = Exception("Test error")
        predictions_df = pd.DataFrame({
            'model': ['model1'],
            'date': ['20240101'],
            'code': ['000001'],
            'score': [0.1]
        })

        self.runner.run('20240101', predictions_df)

        # Check errors were recorded
        health_before = self.runner.get_health_status()
        assert health_before['total_errors'] > 0

        # Reset error state
        self.runner.reset_error_state()

        # Check errors were cleared
        health_after = self.runner.get_health_status()
        assert health_after['total_errors'] == 0


class TestStabilityIntegration:
    """Integration tests for stability features working together"""

    def test_full_pipeline_with_all_stability_features(self):
        """Test complete pipeline with all stability enhancements"""
        # This would be a comprehensive integration test
        # For now, just ensure all components can be imported and instantiated
        from live_trading.drift_detector import AdvancedDriftDetector
        from live_trading.prediction_validator import PredictionValidator
        from live_trading.pipeline_runner import ResilientPipelineRunner

        detector = AdvancedDriftDetector()
        validator = PredictionValidator()

        # Create minimal mocks for pipeline
        data_provider = Mock()
        prediction_loader = Mock()
        portfolio_builder = Mock()
        risk_engine = Mock()
        execution_engine = Mock()
        state_store = Mock()

        runner = ResilientPipelineRunner(
            data_provider=data_provider,
            prediction_loader=prediction_loader,
            portfolio_builder=portfolio_builder,
            risk_engine=risk_engine,
            execution_engine=execution_engine,
            state_store=state_store,
            drift_detector=detector,
            prediction_validator=validator
        )

        assert runner is not None
        assert hasattr(runner, 'run')
        assert hasattr(runner, 'get_health_status')