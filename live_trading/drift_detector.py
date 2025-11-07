"""drift_detector: advanced model health monitoring with comprehensive drift detection
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ModelHealthMetrics:
    """Comprehensive model health metrics"""
    ic: Optional[float] = None
    rolling_ic: Optional[float] = None
    ic_trend: Optional[float] = None
    psi_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    distribution_shift: Optional[float] = None
    prediction_stability: Optional[float] = None
    outlier_ratio: Optional[float] = None
    health_status: HealthStatus = HealthStatus.HEALTHY
    alerts: List[str] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class AdvancedDriftDetector:
    """Advanced model health monitoring with multiple drift detection methods"""

    def __init__(self,
                 window: int = 20,
                 ic_floor: float = 0.03,
                 psi_threshold: float = 0.1,
                 ks_threshold: float = 0.1,
                 distribution_shift_threshold: float = 0.15,
                 stability_window: int = 5):
        """
        Args:
            window: Rolling window size for trend analysis
            ic_floor: Minimum acceptable IC value
            psi_threshold: PSI threshold for population stability
            ks_threshold: KS statistic threshold for distribution comparison
            distribution_shift_threshold: Threshold for distribution shift detection
            stability_window: Window for prediction stability calculation
        """
        self.window = window
        self.ic_floor = ic_floor
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.distribution_shift_threshold = distribution_shift_threshold
        self.stability_window = stability_window

        # Historical data storage
        self.history_ic: List[float] = []
        self.history_predictions: List[pd.Series] = []
        self.baseline_predictions: Optional[pd.Series] = None
        self.baseline_returns: Optional[pd.Series] = None

        # Health monitoring
        self.consecutive_warnings = 0
        self.consecutive_criticals = 0

    def update_and_check(self,
                        predictions_df: pd.DataFrame,
                        actual_returns_df: Optional[pd.DataFrame] = None) -> ModelHealthMetrics:
        """
        Update model health metrics and check for drift

        Args:
            predictions_df: DataFrame with columns [model, date, code, score]
            actual_returns_df: Optional DataFrame with [date, code, ret]

        Returns:
            ModelHealthMetrics with comprehensive health assessment
        """
        metrics = ModelHealthMetrics()

        try:
            # Calculate IC if actual returns available
            if actual_returns_df is not None:
                ic_metrics = self._calculate_ic_metrics(predictions_df, actual_returns_df)
                metrics.ic = ic_metrics.get('ic')
                metrics.rolling_ic = ic_metrics.get('rolling_ic')
                metrics.ic_trend = ic_metrics.get('ic_trend')

            # Calculate prediction quality metrics
            quality_metrics = self._calculate_prediction_quality(predictions_df)
            metrics.prediction_stability = quality_metrics.get('stability')
            metrics.outlier_ratio = quality_metrics.get('outlier_ratio')

            # Calculate distribution drift metrics
            if len(self.history_predictions) >= 2:
                drift_metrics = self._calculate_drift_metrics(predictions_df)
                metrics.psi_score = drift_metrics.get('psi')
                metrics.ks_statistic = drift_metrics.get('ks')
                metrics.distribution_shift = drift_metrics.get('distribution_shift')

            # Determine overall health status
            metrics.health_status = self._assess_health_status(metrics)
            metrics.alerts = self._generate_alerts(metrics)

            # Update historical data
            self._update_history(predictions_df, actual_returns_df)

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            metrics.health_status = HealthStatus.CRITICAL
            metrics.alerts = [f"Drift detection error: {str(e)}"]

        return metrics

    def _calculate_ic_metrics(self, predictions_df: pd.DataFrame,
                            actual_returns_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Information Coefficient metrics"""
        # Aggregate predictions by taking mean across models
        pred_agg = predictions_df.groupby(['date', 'code'])['score'].mean().reset_index()

        # Merge with actual returns
        merged = pred_agg.merge(actual_returns_df, on=["date", "code"], how="inner")

        if merged.empty:
            return {}

        # Calculate IC
        ic = merged["score"].corr(merged["ret"])
        self.history_ic.append(ic)

        # Rolling IC
        rolling_ic = None
        if len(self.history_ic) >= 3:
            rolling = pd.Series(self.history_ic).rolling(self.window, min_periods=1).mean()
            rolling_ic = rolling.iloc[-1]

        # IC trend (slope of recent ICs)
        ic_trend = None
        if len(self.history_ic) >= 5:
            recent_ics = self.history_ic[-5:]
            ic_trend = np.polyfit(range(len(recent_ics)), recent_ics, 1)[0]

        return {
            'ic': float(ic) if pd.notnull(ic) else None,
            'rolling_ic': float(rolling_ic) if rolling_ic is not None and pd.notnull(rolling_ic) else None,
            'ic_trend': float(ic_trend) if ic_trend is not None else None
        }

    def _calculate_prediction_quality(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction quality metrics"""
        # Extract prediction scores
        scores = predictions_df['score'].values

        # Calculate outlier ratio (beyond 3 standard deviations)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        outlier_threshold = 3 * std_score
        outlier_ratio = np.sum(np.abs(scores - mean_score) > outlier_threshold) / len(scores)

        # Calculate prediction stability (coefficient of variation of recent predictions)
        stability = None
        if len(self.history_predictions) >= self.stability_window:
            recent_scores = [pred.values for pred in self.history_predictions[-self.stability_window:]]
            if recent_scores:
                # Calculate coefficient of variation for each stock across time
                recent_df = pd.DataFrame(recent_scores).T
                cv_scores = recent_df.std(axis=1) / recent_df.mean(axis=1).abs()
                stability = 1.0 / (1.0 + cv_scores.mean())  # Convert to stability score (0-1)

        return {
            'stability': float(stability) if stability is not None else None,
            'outlier_ratio': float(outlier_ratio)
        }

    def _calculate_drift_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate distribution drift metrics"""
        current_scores = predictions_df['score'].values

        # PSI (Population Stability Index)
        psi_score = self._calculate_psi(current_scores, self.baseline_predictions)

        # KS statistic (simplified version without scipy)
        ks_statistic = self._calculate_ks_statistic(current_scores, self.baseline_predictions)

        # Distribution shift (difference in means and variances)
        distribution_shift = None
        if self.baseline_predictions is not None:
            mean_shift = abs(np.mean(current_scores) - np.mean(self.baseline_predictions.values))
            var_shift = abs(np.var(current_scores) - np.var(self.baseline_predictions.values))
            distribution_shift = (mean_shift + var_shift) / 2

        return {
            'psi': psi_score,
            'ks': ks_statistic,
            'distribution_shift': distribution_shift
        }

    def _calculate_psi(self, current_scores: np.ndarray, baseline_scores: pd.Series) -> float:
        """Calculate Population Stability Index"""
        def _get_bins(scores: np.ndarray, bins: int = 10) -> np.ndarray:
            """Get bin edges for PSI calculation"""
            return np.histogram(scores, bins=bins)[1]

        try:
            bins = _get_bins(baseline_scores.values)
            expected_dist, _ = np.histogram(baseline_scores.values, bins=bins, density=True)
            actual_dist, _ = np.histogram(current_scores, bins=bins, density=True)

            # Avoid division by zero
            expected_dist = np.where(expected_dist == 0, 1e-6, expected_dist)
            actual_dist = np.where(actual_dist == 0, 1e-6, actual_dist)

            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            return float(psi)

        except Exception:
            return 0.0

    def _calculate_ks_statistic(self, current_scores: np.ndarray, baseline_scores: pd.Series) -> float:
        """Calculate simplified KS statistic without scipy"""
        try:
            # Sort both arrays
            current_sorted = np.sort(current_scores)
            baseline_sorted = np.sort(baseline_scores.values)

            # Calculate empirical CDFs
            n1, n2 = len(current_sorted), len(baseline_sorted)

            # Find all unique values
            all_values = np.concatenate([current_sorted, baseline_sorted])
            all_values = np.unique(all_values)
            all_values.sort()

            # Calculate CDF values
            cdf1 = np.searchsorted(current_sorted, all_values, side='right') / n1
            cdf2 = np.searchsorted(baseline_sorted, all_values, side='right') / n2

            # Calculate KS statistic
            ks_stat = np.max(np.abs(cdf1 - cdf2))
            return float(ks_stat)

        except Exception:
            return 0.0

    def _assess_health_status(self, metrics: ModelHealthMetrics) -> HealthStatus:
        """Assess overall model health status"""
        critical_conditions = []
        warning_conditions = []

        # IC-based conditions
        if metrics.rolling_ic is not None and metrics.rolling_ic < self.ic_floor:
            critical_conditions.append("IC below threshold")

        if metrics.ic_trend is not None and metrics.ic_trend < -0.01:
            warning_conditions.append("IC trending downward")

        # Distribution drift conditions
        if metrics.psi_score is not None and metrics.psi_score > self.psi_threshold:
            warning_conditions.append("Population stability index high")

        if metrics.ks_statistic is not None and metrics.ks_statistic > self.ks_threshold:
            warning_conditions.append("KS statistic indicates distribution shift")

        if metrics.distribution_shift is not None and metrics.distribution_shift > self.distribution_shift_threshold:
            critical_conditions.append("Significant distribution shift detected")

        # Prediction quality conditions
        if metrics.outlier_ratio is not None and metrics.outlier_ratio > 0.05:
            warning_conditions.append("High outlier ratio in predictions")

        if metrics.prediction_stability is not None and metrics.prediction_stability < 0.7:
            warning_conditions.append("Low prediction stability")

        # Determine status
        if critical_conditions:
            self.consecutive_criticals += 1
            self.consecutive_warnings = 0
            return HealthStatus.CRITICAL
        elif warning_conditions:
            self.consecutive_warnings += 1
            self.consecutive_criticals = 0
            return HealthStatus.WARNING
        else:
            self.consecutive_warnings = 0
            self.consecutive_criticals = 0
            return HealthStatus.HEALTHY

    def _generate_alerts(self, metrics: ModelHealthMetrics) -> List[str]:
        """Generate specific alerts based on metrics"""
        alerts = []

        if metrics.ic is not None and metrics.ic < 0:
            alerts.append(f"IC is negative: {metrics.ic:.3f}")

        if metrics.rolling_ic is not None and metrics.rolling_ic < self.ic_floor:
            alerts.append(f"Rolling IC below threshold: {metrics.rolling_ic:.3f}")

        if metrics.psi_score is not None and metrics.psi_score > self.psi_threshold:
            alerts.append(f"PSI score high: {metrics.psi_score:.3f}")

        if metrics.ks_statistic is not None and metrics.ks_statistic > self.ks_threshold:
            alerts.append(f"KS statistic high: {metrics.ks_statistic:.3f}")

        if metrics.distribution_shift is not None and metrics.distribution_shift > self.distribution_shift_threshold:
            alerts.append(f"Distribution shift detected: {metrics.distribution_shift:.3f}")

        if metrics.outlier_ratio is not None and metrics.outlier_ratio > 0.05:
            alerts.append(f"High outlier ratio: {metrics.outlier_ratio:.3f}")

        return alerts

    def _update_history(self, predictions_df: pd.DataFrame,
                       actual_returns_df: Optional[pd.DataFrame] = None):
        """Update historical data storage"""
        # Store prediction scores
        scores = predictions_df.groupby('code')['score'].mean()
        self.history_predictions.append(scores)

        # Maintain baseline (first window of data)
        if self.baseline_predictions is None and len(self.history_predictions) >= self.window:
            baseline_scores = []
            for pred in self.history_predictions[:self.window]:
                baseline_scores.extend(pred.values)
            self.baseline_predictions = pd.Series(baseline_scores)

        # Store baseline returns if available
        if self.baseline_returns is None and actual_returns_df is not None:
            self.baseline_returns = actual_returns_df.set_index('code')['ret']

        # Limit history size
        if len(self.history_predictions) > self.window * 2:
            self.history_predictions = self.history_predictions[-self.window:]

        if len(self.history_ic) > self.window * 2:
            self.history_ic = self.history_ic[-self.window:]

    def reset_baseline(self):
        """Reset baseline data for recalibration"""
        self.baseline_predictions = None
        self.baseline_returns = None
        self.history_predictions.clear()
        self.history_ic.clear()
        self.consecutive_warnings = 0
        self.consecutive_criticals = 0

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        return {
            'total_observations': len(self.history_predictions),
            'consecutive_warnings': self.consecutive_warnings,
            'consecutive_criticals': self.consecutive_criticals,
            'baseline_established': self.baseline_predictions is not None,
            'ic_history_length': len(self.history_ic)
        }


# Backward compatibility
class DriftDetector(AdvancedDriftDetector):
    """Legacy DriftDetector class for backward compatibility"""

    def update_and_check(self, predicted: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, Any]:
        """Legacy interface - returns dict instead of ModelHealthMetrics"""
        metrics = super().update_and_check(predicted, actual)

        # Convert to legacy format
        result = {
            "ic": metrics.ic,
            "rolling_ic": metrics.rolling_ic,
            "flag_retrain": metrics.health_status == HealthStatus.CRITICAL
        }

        # Add additional metrics
        if metrics.psi_score is not None:
            result["psi_score"] = metrics.psi_score
        if metrics.ks_statistic is not None:
            result["ks_statistic"] = metrics.ks_statistic

        return result