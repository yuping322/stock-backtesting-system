"""pipeline_runner: orchestrates the full live_trading pipeline with error recovery
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, Set
import pandas as pd
import time
import logging
from dataclasses import dataclass
from enum import Enum
from live_trading.data_provider import DataProvider
from live_trading.prediction_loader import PredictionLoader
from live_trading.portfolio_builder import PortfolioBuilder
from live_trading.risk_manager import CompositeRiskEngine
from live_trading.execution_engine import ExecutionEngine
from live_trading.state_store import StateStore
from live_trading.drift_detector import DriftDetector
from live_trading.prediction_validator import PredictionValidator, ValidationReport

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    RETRY = "retry"  # Retry the failed operation
    SKIP = "skip"     # Skip the failed component
    FALLBACK = "fallback"  # Use fallback/default values
    DEGRADED = "degraded"  # Continue with reduced functionality


@dataclass
class ErrorContext:
    """Context information for error recovery"""
    component: str
    operation: str
    error: Exception
    attempt: int
    max_attempts: int
    timestamp: float
    context_data: Dict[str, Any]


@dataclass
class RecoveryResult:
    """Result of error recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    fallback_data: Optional[Any]
    error_message: str
    recovery_time: float


class ResilientPipelineRunner:
    """Enhanced pipeline runner with comprehensive error recovery and stability features"""

    def __init__(self,
                 data_provider: DataProvider,
                 prediction_loader: PredictionLoader,
                 portfolio_builder: PortfolioBuilder,
                 risk_engine: CompositeRiskEngine,
                 execution_engine: ExecutionEngine,
                 state_store: StateStore,
                 drift_detector: DriftDetector,
                 prediction_validator: Optional[PredictionValidator] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_degraded_mode: bool = True):
        """
        Args:
            data_provider: Data provider for market data
            prediction_loader: Loader for model predictions
            portfolio_builder: Portfolio construction logic
            risk_engine: Risk management engine
            execution_engine: Order execution engine
            state_store: State persistence store
            drift_detector: Model health monitoring
            prediction_validator: Optional prediction data validator
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retries in seconds
            enable_degraded_mode: Whether to enable degraded mode operation
        """
        self.data_provider = data_provider
        self.prediction_loader = prediction_loader
        self.portfolio_builder = portfolio_builder
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.state_store = state_store
        self.drift_detector = drift_detector
        self.prediction_validator = prediction_validator or PredictionValidator()

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_degraded_mode = enable_degraded_mode

        # Recovery tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[RecoveryResult] = []
        self.degraded_components: Set[str] = set()

        # Fallback data
        self._fallback_universe = pd.DataFrame()
        self._fallback_panel = pd.DataFrame()
        self._fallback_predictions = pd.DataFrame()

    def run(self, date: str, predictions_df: pd.DataFrame,
            actual_returns_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run the full pipeline with error recovery and validation

        Args:
            date: trading date (YYYYMMDD)
            predictions_df: DataFrame with columns [model, date, code, score]
            actual_returns_df: optional DataFrame with [date, code, ret] for drift detection

        Returns:
            dict with pipeline results and recovery information
        """
        start_time = time.time()
        pipeline_result = {}
        recovery_info = {
            "errors_encountered": [],
            "recoveries_attempted": [],
            "degraded_components": list(self.degraded_components),
            "validation_report": None
        }

        try:
            # Step 1: Validate predictions
            validation_report = self._validate_predictions(predictions_df)
            recovery_info["validation_report"] = validation_report

            if not validation_report.is_valid and not self.enable_degraded_mode:
                raise RuntimeError(f"Prediction validation failed: {len(validation_report.errors)} errors")

            # Step 2: Data fetch with recovery
            data_result = self._fetch_data_with_recovery(date)
            pipeline_result.update(data_result["data"])
            recovery_info["errors_encountered"].extend(data_result["errors"])
            recovery_info["recoveries_attempted"].extend(data_result["recoveries"])

            # Step 3: Process predictions with recovery
            prediction_result = self._process_predictions_with_recovery(predictions_df, validation_report)
            pipeline_result.update(prediction_result["data"])
            recovery_info["errors_encountered"].extend(prediction_result["errors"])
            recovery_info["recoveries_attempted"].extend(prediction_result["recoveries"])

            # Step 4: Build portfolio with recovery
            portfolio_result = self._build_portfolio_with_recovery(pipeline_result)
            pipeline_result.update(portfolio_result["data"])
            recovery_info["errors_encountered"].extend(portfolio_result["errors"])
            recovery_info["recoveries_attempted"].extend(portfolio_result["recoveries"])

            # Step 5: Execute orders with recovery
            execution_result = self._execute_orders_with_recovery(pipeline_result, date)
            pipeline_result.update(execution_result["data"])
            recovery_info["errors_encountered"].extend(execution_result["errors"])
            recovery_info["recoveries_attempted"].extend(execution_result["recoveries"])

            # Step 6: Model health monitoring
            drift_result = self._monitor_model_health(predictions_df, actual_returns_df)
            pipeline_result["drift_result"] = drift_result

            # Add recovery information
            pipeline_result["recovery_info"] = recovery_info
            pipeline_result["execution_time"] = time.time() - start_time
            pipeline_result["pipeline_status"] = "success"

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_result.update({
                "pipeline_status": "failed",
                "error": str(e),
                "recovery_info": recovery_info,
                "execution_time": time.time() - start_time
            })

        return pipeline_result

    def _validate_predictions(self, predictions_df: pd.DataFrame) -> ValidationReport:
        """Validate prediction data quality"""
        try:
            return self.prediction_validator.validate(predictions_df)
        except Exception as e:
            logger.warning(f"Prediction validation failed: {e}")
            # Return minimal valid report on validation failure
            return ValidationReport(
                timestamp=pd.Timestamp.now(),
                total_checks=1,
                passed_checks=0,
                failed_checks=1,
                errors=[],
                warnings=[],
                info=[],
                is_valid=False,
                summary={"validation_error": str(e)}
            )

    def _fetch_data_with_recovery(self, date: str) -> Dict[str, Any]:
        """Fetch market data with error recovery"""
        result = {"data": {}, "errors": [], "recoveries": []}

        # Try to fetch universe
        universe_result = self._execute_with_recovery(
            "data_fetch", "universe",
            lambda: self.data_provider.load_universe()
        )
        result["data"]["universe"] = universe_result["result"]
        result["errors"].extend(universe_result["errors"])
        result["recoveries"].extend(universe_result["recoveries"])

        # Try to fetch panel
        panel_result = self._execute_with_recovery(
            "data_fetch", "panel",
            lambda: self.data_provider.fetch_basic_panel()
        )
        panel = panel_result["result"]
        result["data"]["panel"] = panel
        result["errors"].extend(panel_result["errors"])
        result["recoveries"].extend(panel_result["recoveries"])

        # If panel is empty or failed, this is critical
        if panel is None or panel.empty:
            if not self.enable_degraded_mode:
                raise RuntimeError("Critical data fetch failed and degraded mode disabled")
            result["data"]["panel"] = self._fallback_panel

        # Try to fetch suspension data
        if panel is not None and not panel.empty:
            suspension_result = self._execute_with_recovery(
                "data_fetch", "suspension",
                lambda: self.data_provider.fetch_suspension(panel.code.tolist())
            )
            suspension = suspension_result["result"]
            result["errors"].extend(suspension_result["errors"])
            result["recoveries"].extend(suspension_result["recoveries"])

            # Build blacklist
            blacklist_result = self._execute_with_recovery(
                "data_fetch", "blacklist",
                lambda: self.data_provider.build_blacklist(panel, suspension if suspension is not None else pd.DataFrame())
            )
            result["data"]["blacklist"] = blacklist_result["result"]
            result["errors"].extend(blacklist_result["errors"])
            result["recoveries"].extend(blacklist_result["recoveries"])
        else:
            result["data"]["blacklist"] = pd.DataFrame()

        # Try to fetch industry data
        industry_result = self._execute_with_recovery(
            "data_fetch", "industry",
            lambda: self.data_provider.fetch_industry()
        )
        result["data"]["industry"] = industry_result["result"] if industry_result["result"] is not None else pd.DataFrame()
        result["errors"].extend(industry_result["errors"])
        result["recoveries"].extend(industry_result["recoveries"])

        return result

    def _process_predictions_with_recovery(self, predictions_df: pd.DataFrame,
                                        validation_report: ValidationReport) -> Dict[str, Any]:
        """Process predictions with error recovery"""
        result = {"data": {}, "errors": [], "recoveries": []}

        # Load predictions
        load_result = self._execute_with_recovery(
            "prediction_processing", "load",
            lambda: self.prediction_loader.load_from_df(predictions_df)
        )
        result["errors"].extend(load_result["errors"])
        result["recoveries"].extend(load_result["recoveries"])

        # Aggregate predictions
        aggregate_result = self._execute_with_recovery(
            "prediction_processing", "aggregate",
            lambda: self.prediction_loader.aggregate_mean().rename(columns={"mean_score": "score"})
        )
        aggregated = aggregate_result["result"]
        result["data"]["aggregated_predictions"] = aggregated
        result["errors"].extend(aggregate_result["errors"])
        result["recoveries"].extend(aggregate_result["recoveries"])

        # Use fallback if aggregation failed
        if aggregated is None or aggregated.empty:
            logger.warning("Prediction aggregation failed, using fallback")
            result["data"]["aggregated_predictions"] = self._fallback_predictions

        return result

    def _build_portfolio_with_recovery(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build portfolio with error recovery"""
        result = {"data": {}, "errors": [], "recoveries": []}

        aggregated = pipeline_data.get("aggregated_predictions")
        blacklist = pipeline_data.get("blacklist", pd.DataFrame())
        panel = pipeline_data.get("panel", pd.DataFrame())
        industry = pipeline_data.get("industry", pd.DataFrame())

        if aggregated is None or aggregated.empty:
            result["data"]["target_weights"] = pd.DataFrame()
            result["data"]["filtered_weights"] = pd.DataFrame()
            result["data"]["risk_logs"] = ["No predictions available for portfolio building"]
            return result

        # Build target portfolio
        picks = aggregated.rename(columns={"code": "code", "score": "score"})
        portfolio_result = self._execute_with_recovery(
            "portfolio_building", "target_weights",
            lambda: self.portfolio_builder.build(picks, cash_target=0.1)
        )
        target_weights = portfolio_result["result"] if portfolio_result["result"] is not None else pd.DataFrame()
        result["data"]["target_weights"] = target_weights
        result["errors"].extend(portfolio_result["errors"])
        result["recoveries"].extend(portfolio_result["recoveries"])

        # Apply risk management
        ctx = {
            "weights": target_weights.copy(),
            "blacklist": blacklist,
            "panel": panel,
            "industry": industry
        }
        risk_result = self._execute_with_recovery(
            "portfolio_building", "risk_management",
            lambda: self.risk_engine.run(ctx)
        )
        risk_out = risk_result["result"] if risk_result["result"] is not None else {}
        result["data"]["filtered_weights"] = risk_out.get("weights", target_weights)
        result["data"]["risk_logs"] = risk_out.get("risk_logs", [])
        result["errors"].extend(risk_result["errors"])
        result["recoveries"].extend(risk_result["recoveries"])

        return result

    def _execute_orders_with_recovery(self, pipeline_data: Dict[str, Any], date: str) -> Dict[str, Any]:
        """Execute orders with error recovery"""
        result = {"data": {}, "errors": [], "recoveries": []}

        filtered_weights = pipeline_data.get("filtered_weights", pd.DataFrame())

        if filtered_weights.empty:
            result["data"]["orders"] = []
            result["data"]["fills"] = []
            return result

        # Get current positions
        positions_result = self._execute_with_recovery(
            "execution", "get_positions",
            lambda: self.state_store.snapshot_positions()
        )
        current_positions = positions_result["result"] if positions_result["result"] is not None else {}
        result["errors"].extend(positions_result["errors"])
        result["recoveries"].extend(positions_result["recoveries"])

        # Compute orders
        orders_result = self._execute_with_recovery(
            "execution", "compute_orders",
            lambda: self.execution_engine.compute_diff(filtered_weights, current_positions)
        )
        orders = orders_result["result"] if orders_result["result"] is not None else []
        result["data"]["orders"] = orders
        result["errors"].extend(orders_result["errors"])
        result["recoveries"].extend(orders_result["recoveries"])

        # Submit orders
        submit_result = self._execute_with_recovery(
            "execution", "submit_orders",
            lambda: self.execution_engine.submit_orders(orders)
        )
        result["errors"].extend(submit_result["errors"])
        result["recoveries"].extend(submit_result["recoveries"])

        # Simulate fills (in real system this would be async)
        fills = []
        for order in orders:
            fill_result = self._execute_with_recovery(
                "execution", f"fill_{order.get('code', 'unknown')}",
                lambda o=order: self._simulate_fill(o, date)
            )
            if fill_result["result"]:
                fills.append(fill_result["result"])
            result["errors"].extend(fill_result["errors"])
            result["recoveries"].extend(fill_result["recoveries"])

        result["data"]["fills"] = fills
        return result

    def _monitor_model_health(self, predictions_df: pd.DataFrame,
                            actual_returns_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Monitor model health with error recovery"""
        if actual_returns_df is None:
            return {}

        try:
            return self.drift_detector.update_and_check(predictions_df, actual_returns_df)
        except Exception as e:
            logger.warning(f"Model health monitoring failed: {e}")
            return {"health_check_error": str(e)}

    def _execute_with_recovery(self, component: str, operation: str,
                             func: Callable) -> Dict[str, Any]:
        """Execute function with recovery logic"""
        errors = []
        recoveries = []

        for attempt in range(self.max_retries + 1):
            try:
                result = func()
                return {"result": result, "errors": errors, "recoveries": recoveries}
            except Exception as e:
                error_ctx = ErrorContext(
                    component=component,
                    operation=operation,
                    error=e,
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1,
                    timestamp=time.time(),
                    context_data={}
                )
                self.error_history.append(error_ctx)
                errors.append(error_ctx)

                if attempt < self.max_retries:
                    # Wait before retry (don't attempt recovery until all retries exhausted)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # All retries failed, now attempt recovery
                    logger.warning(f"All retry attempts failed for {component}.{operation}, attempting recovery")
                    recovery_result = self._attempt_recovery(error_ctx)
                    recoveries.append(recovery_result)

                    if recovery_result.success:
                        logger.info(f"Recovery successful for {component}.{operation}")
                        return {
                            "result": recovery_result.fallback_data,
                            "errors": errors,
                            "recoveries": recoveries
                        }
                    else:
                        logger.error(f"Recovery also failed for {component}.{operation}")

        # All attempts and recovery failed
        return {"result": None, "errors": errors, "recoveries": recoveries}

    def _attempt_recovery(self, error_ctx: ErrorContext) -> RecoveryResult:
        """Attempt to recover from an error"""
        start_time = time.time()

        try:
            component = error_ctx.component
            operation = error_ctx.operation

            # Component-specific recovery strategies
            if component == "data_fetch":
                if operation == "universe":
                    # Use cached universe or empty fallback
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK,
                        fallback_data=self._fallback_universe,
                        error_message="",
                        recovery_time=time.time() - start_time
                    )
                elif operation in ["panel", "suspension", "blacklist", "industry"]:
                    # Use empty DataFrame as fallback
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK,
                        fallback_data=pd.DataFrame(),
                        error_message="",
                        recovery_time=time.time() - start_time
                    )

            elif component == "prediction_processing":
                if operation in ["load", "aggregate"]:
                    # Use fallback predictions
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK,
                        fallback_data=self._fallback_predictions,
                        error_message="",
                        recovery_time=time.time() - start_time
                    )

            elif component == "portfolio_building":
                if operation == "target_weights":
                    # Return empty weights
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK,
                        fallback_data=pd.DataFrame(),
                        error_message="",
                        recovery_time=time.time() - start_time
                    )

            elif component == "execution":
                if operation == "get_positions":
                    # Return empty positions
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.FALLBACK,
                        fallback_data={},
                        error_message="",
                        recovery_time=time.time() - start_time
                    )
                elif operation.startswith("fill_"):
                    # Skip this fill
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.SKIP,
                        fallback_data=None,
                        error_message="",
                        recovery_time=time.time() - start_time
                    )

            # Default: cannot recover
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                fallback_data=None,
                error_message=f"No recovery strategy for {component}.{operation}",
                recovery_time=time.time() - start_time
            )

        except Exception as recovery_error:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                fallback_data=None,
                error_message=f"Recovery failed: {str(recovery_error)}",
                recovery_time=time.time() - start_time
            )

    def _simulate_fill(self, order: Dict[str, Any], date: str) -> Dict[str, Any]:
        """Simulate order fill for testing"""
        fill = {
            "ts": date,
            "order": order,
            "status": "filled"
        }
        self.state_store.append_fill(fill)
        return fill

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour

        return {
            "degraded_components": list(self.degraded_components),
            "recent_errors": len(recent_errors),
            "total_errors": len(self.error_history),
            "recovery_attempts": len(self.recovery_history),
            "successful_recoveries": sum(1 for r in self.recovery_history if r.success),
            "system_status": "degraded" if self.degraded_components else "healthy"
        }

    def reset_error_state(self):
        """Reset error tracking state"""
        self.error_history.clear()
        self.recovery_history.clear()
        self.degraded_components.clear()


# Backward compatibility
class PipelineRunner(ResilientPipelineRunner):
    """Legacy PipelineRunner class for backward compatibility"""

    def __init__(self, data_provider: DataProvider, prediction_loader: PredictionLoader,
                 portfolio_builder: PortfolioBuilder, risk_engine: CompositeRiskEngine,
                 execution_engine: ExecutionEngine, state_store: StateStore,
                 drift_detector: DriftDetector):
        super().__init__(
            data_provider=data_provider,
            prediction_loader=prediction_loader,
            portfolio_builder=portfolio_builder,
            risk_engine=risk_engine,
            execution_engine=execution_engine,
            state_store=state_store,
            drift_detector=drift_detector,
            enable_degraded_mode=False  # Legacy behavior: fail fast
        )