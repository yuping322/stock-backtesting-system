"""prediction_validator: comprehensive prediction data validation and quality assurance
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    errors: List[ValidationResult]
    warnings: List[ValidationResult]
    info: List[ValidationResult]
    is_valid: bool
    summary: Dict[str, Any]

    def __post_init__(self):
        self.timestamp = datetime.now()


class PredictionValidator:
    """Comprehensive prediction data validator for quality assurance"""

    def __init__(self,
                 required_columns: Optional[List[str]] = None,
                 expected_dtypes: Optional[Dict[str, str]] = None,
                 score_range: Tuple[float, float] = (-1.0, 1.0),
                 max_missing_ratio: float = 0.1,
                 max_outlier_ratio: float = 0.05,
                 min_unique_stocks: int = 100,
                 max_duplicate_ratio: float = 0.01):
        """
        Args:
            required_columns: List of required column names
            expected_dtypes: Expected data types for columns
            score_range: Valid range for prediction scores
            max_missing_ratio: Maximum allowed missing value ratio
            max_outlier_ratio: Maximum allowed outlier ratio
            min_unique_stocks: Minimum number of unique stocks required
            max_duplicate_ratio: Maximum allowed duplicate ratio
        """
        self.required_columns = required_columns or ['model', 'date', 'code', 'score']
        self.expected_dtypes = expected_dtypes or {
            'model': 'object',
            'date': 'object',
            'code': 'object',
            'score': 'float64'
        }
        self.score_range = score_range
        self.max_missing_ratio = max_missing_ratio
        self.max_outlier_ratio = max_outlier_ratio
        self.min_unique_stocks = min_unique_stocks
        self.max_duplicate_ratio = max_duplicate_ratio

        # Validation history
        self.validation_history: List[ValidationReport] = []

    def validate(self, predictions_df: pd.DataFrame,
                context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Perform comprehensive validation on prediction data

        Args:
            predictions_df: DataFrame to validate
            context: Optional context information (e.g., expected date, universe size)

        Returns:
            ValidationReport with detailed results
        """
        results = []
        context = context or {}

        # Basic structure validation
        results.extend(self._validate_structure(predictions_df))

        # Data type validation
        results.extend(self._validate_data_types(predictions_df))

        # Data quality validation
        results.extend(self._validate_data_quality(predictions_df))

        # Business logic validation
        results.extend(self._validate_business_logic(predictions_df, context))

        # Statistical validation
        results.extend(self._validate_statistics(predictions_df))

        # Generate report
        report = self._generate_report(results, predictions_df)

        # Store in history
        self.validation_history.append(report)

        return report

    def _validate_structure(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate DataFrame structure"""
        results = []

        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            results.append(ValidationResult(
                check_name="required_columns",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required columns: {list(missing_columns)}",
                details={"missing_columns": list(missing_columns)}
            ))
        else:
            results.append(ValidationResult(
                check_name="required_columns",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="All required columns present"
            ))

        # Check for empty DataFrame
        if df.empty:
            results.append(ValidationResult(
                check_name="non_empty",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message="DataFrame is empty",
                details={"row_count": 0}
            ))
        else:
            results.append(ValidationResult(
                check_name="non_empty",
                severity=ValidationSeverity.INFO,
                passed=True,
                message=f"DataFrame contains {len(df)} rows"
            ))

        return results

    def _validate_data_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate data types"""
        results = []

        for col, expected_dtype in self.expected_dtypes.items():
            if col not in df.columns:
                continue

            actual_dtype = str(df[col].dtype)

            # Handle nullable dtypes
            if expected_dtype == 'object' and 'string' in actual_dtype:
                actual_dtype = 'object'

            if expected_dtype == 'float64' and 'float' in actual_dtype:
                actual_dtype = 'float64'

            if actual_dtype != expected_dtype:
                results.append(ValidationResult(
                    check_name=f"dtype_{col}",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Column '{col}' has wrong dtype. Expected: {expected_dtype}, Got: {actual_dtype}",
                    details={"column": col, "expected": expected_dtype, "actual": actual_dtype}
                ))
            else:
                results.append(ValidationResult(
                    check_name=f"dtype_{col}",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"Column '{col}' has correct dtype: {expected_dtype}"
                ))

        return results

    def _validate_data_quality(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate data quality"""
        results = []

        # Check for missing values
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        max_missing_col = missing_summary.idxmax() if total_missing > 0 else None

        if total_missing > 0:
            missing_ratio = total_missing / (len(df) * len(df.columns))

            severity = ValidationSeverity.WARNING if missing_ratio <= self.max_missing_ratio else ValidationSeverity.ERROR

            results.append(ValidationResult(
                check_name="missing_values",
                severity=severity,
                passed=missing_ratio <= self.max_missing_ratio,
                message=f"Found {total_missing} missing values ({missing_ratio:.3%})",
                details={
                    "total_missing": int(total_missing),
                    "missing_ratio": float(missing_ratio),
                    "missing_by_column": missing_summary.to_dict(),
                    "worst_column": max_missing_col
                }
            ))
        else:
            results.append(ValidationResult(
                check_name="missing_values",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="No missing values found"
            ))

        # Check for duplicates
        if 'score' in df.columns:
            duplicates = df.duplicated(subset=['model', 'date', 'code'], keep=False)
            duplicate_count = duplicates.sum()
            duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0

            if duplicate_ratio > self.max_duplicate_ratio:
                results.append(ValidationResult(
                    check_name="duplicates",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"High duplicate ratio: {duplicate_ratio:.3%} ({duplicate_count} duplicates)",
                    details={"duplicate_count": int(duplicate_count), "duplicate_ratio": float(duplicate_ratio)}
                ))
            else:
                results.append(ValidationResult(
                    check_name="duplicates",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"Duplicate ratio acceptable: {duplicate_ratio:.3%}"
                ))

        return results

    def _validate_business_logic(self, df: pd.DataFrame, context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate business logic constraints"""
        results = []

        # Validate score range
        if 'score' in df.columns:
            scores = df['score'].dropna()
            out_of_range = ((scores < self.score_range[0]) | (scores > self.score_range[1])).sum()
            range_violations = out_of_range / len(scores) if len(scores) > 0 else 0

            if range_violations > 0:
                results.append(ValidationResult(
                    check_name="score_range",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"{out_of_range} scores outside valid range {self.score_range}",
                    details={
                        "out_of_range_count": int(out_of_range),
                        "violation_ratio": float(range_violations),
                        "valid_range": self.score_range
                    }
                ))
            else:
                results.append(ValidationResult(
                    check_name="score_range",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"All scores within valid range {self.score_range}"
                ))

        # Validate unique stocks
        if 'code' in df.columns:
            unique_stocks = df['code'].nunique()

            if unique_stocks < self.min_unique_stocks:
                results.append(ValidationResult(
                    check_name="unique_stocks",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Only {unique_stocks} unique stocks, minimum required: {self.min_unique_stocks}",
                    details={"unique_stocks": unique_stocks, "minimum_required": self.min_unique_stocks}
                ))
            else:
                results.append(ValidationResult(
                    check_name="unique_stocks",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message=f"Sufficient unique stocks: {unique_stocks}"
                ))

        # Validate date consistency
        if 'date' in df.columns:
            unique_dates = df['date'].nunique()

            if unique_dates > 1:
                results.append(ValidationResult(
                    check_name="date_consistency",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Multiple dates found: {unique_dates} unique dates",
                    details={"unique_dates": unique_dates, "date_counts": df['date'].value_counts().to_dict()}
                ))
            else:
                results.append(ValidationResult(
                    check_name="date_consistency",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="All predictions for single date"
                ))

        return results

    def _validate_statistics(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate statistical properties"""
        results = []

        if 'score' in df.columns:
            scores = df['score'].dropna()

            if len(scores) > 0:
                # Check for outliers using IQR method
                Q1 = scores.quantile(0.25)
                Q3 = scores.quantile(0.75)
                IQR = Q3 - Q1
                outlier_bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

                outliers = ((scores < outlier_bounds[0]) | (scores > outlier_bounds[1])).sum()
                outlier_ratio = outliers / len(scores)

                if outlier_ratio > self.max_outlier_ratio:
                    results.append(ValidationResult(
                        check_name="outliers",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"High outlier ratio: {outlier_ratio:.3%} ({outliers} outliers)",
                        details={
                            "outlier_count": int(outliers),
                            "outlier_ratio": float(outlier_ratio),
                            "iqr_bounds": outlier_bounds
                        }
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="outliers",
                        severity=ValidationSeverity.INFO,
                        passed=True,
                        message=f"Outlier ratio acceptable: {outlier_ratio:.3%}"
                    ))

                # Check distribution normality (basic check)
                skewness = scores.skew()
                kurtosis = scores.kurtosis()

                if abs(skewness) > 2 or abs(kurtosis) > 7:  # Rough thresholds
                    results.append(ValidationResult(
                        check_name="distribution",
                        severity=ValidationSeverity.INFO,
                        passed=True,
                        message=f"Non-normal distribution detected (skewness: {skewness:.3f}, kurtosis: {kurtosis:.3f})",
                        details={"skewness": float(skewness), "kurtosis": float(kurtosis)}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="distribution",
                        severity=ValidationSeverity.INFO,
                        passed=True,
                        message="Distribution appears normal"
                    ))

        return results

    def _generate_report(self, results: List[ValidationResult], df: pd.DataFrame) -> ValidationReport:
        """Generate comprehensive validation report"""
        errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL or r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        info = [r for r in results if r.severity == ValidationSeverity.INFO]

        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = total_checks - passed_checks

        # Overall validity: pass if no critical errors
        is_valid = not any(r.severity == ValidationSeverity.CRITICAL for r in results)

        # Generate summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "validation_summary": {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "errors": len(errors),
                "warnings": len(warnings)
            }
        }

        # Add basic statistics if score column exists
        if 'score' in df.columns:
            scores = df['score'].dropna()
            summary["score_statistics"] = {
                "count": len(scores),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "median": float(scores.median())
            }

        return ValidationReport(
            timestamp=datetime.now(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            info=info,
            is_valid=is_valid,
            summary=summary
        )

    def get_validation_history(self, limit: Optional[int] = None) -> List[ValidationReport]:
        """Get validation history"""
        if limit is None:
            return self.validation_history
        return self.validation_history[-limit:]

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations"""
        if not self.validation_history:
            return {"message": "No validations performed yet"}

        recent = self.validation_history[-1] if self.validation_history else None

        return {
            "total_validations": len(self.validation_history),
            "last_validation": {
                "timestamp": recent.timestamp.isoformat() if recent else None,
                "is_valid": recent.is_valid if recent else None,
                "passed_checks": recent.passed_checks if recent else None,
                "failed_checks": recent.failed_checks if recent else None
            },
            "validation_trends": {
                "avg_pass_rate": np.mean([r.passed_checks / r.total_checks for r in self.validation_history]) if self.validation_history else 0,
                "valid_rate": sum(1 for r in self.validation_history if r.is_valid) / len(self.validation_history) if self.validation_history else 0
            }
        }