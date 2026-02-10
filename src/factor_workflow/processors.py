"""Custom dataset processors with explicit DataFrame copying to avoid chained assignment warnings."""
from __future__ import annotations

import pandas as pd
from qlib.data.dataset.processor import (
    CSRankNorm as _BaseCSRankNorm,
    Fillna as _BaseFillna,
    RobustZScoreNorm as _BaseRobustZScoreNorm,
)


class _CopyMixin:
    """Ensure we always operate on a copy to avoid pandas chained-assignment warnings."""

    @staticmethod
    def _ensure_copy(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy(deep=False)


class SafeRobustZScoreNorm(_CopyMixin, _BaseRobustZScoreNorm):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_copy(df)
        return super().__call__(df)


class SafeFillna(_CopyMixin, _BaseFillna):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_copy(df)
        return super().__call__(df)


class SafeCSRankNorm(_CopyMixin, _BaseCSRankNorm):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_copy(df)
        return super().__call__(df)


__all__ = ["SafeRobustZScoreNorm", "SafeFillna", "SafeCSRankNorm"]
