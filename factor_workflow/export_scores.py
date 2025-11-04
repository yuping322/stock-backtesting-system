"""Export fused prediction scores into `date,code,weight` format.

Usage:
    python export_scores.py --provider-uri ~/.qlib/qlib_data --region cn --output backtest_output/scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .backtest_evaluation import PROVIDER_URI, REGION, generate_final_signal
    from .paths import SCORES_OUTPUT_FILE
except ImportError:  # allow running as ``python factor_workflow/export_scores.py``
    import sys

    _PKG_ROOT = Path(__file__).resolve().parent
    _REPO_ROOT = _PKG_ROOT.parent
    _REPO_PATH = str(_REPO_ROOT)
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)

    from factor_workflow.backtest_evaluation import PROVIDER_URI, REGION, generate_final_signal
    from factor_workflow.paths import SCORES_OUTPUT_FILE

DEFAULT_OUTPUT = SCORES_OUTPUT_FILE


def _normalize_scores(series: pd.Series) -> pd.Series:
    """Shift scores to be non-negative and normalize to sum to 1 within each date."""

    scores = series.astype(float)
    min_score = scores.min()
    shifted = scores - min_score
    total = shifted.sum()

    if total <= 0:
        # Fallback to equal weights when all scores are identical.
        uniform = 1.0 / len(scores) if len(scores) else 0.0
        return pd.Series([uniform] * len(scores), index=series.index)

    return shifted / total


def export_scores(provider_uri: str = PROVIDER_URI, region: str = REGION, output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Train suites (if needed) and persist final scores as CSV."""
    final_signal, _, _ = generate_final_signal(provider_uri, region)
    if final_signal.empty:
        raise ValueError("Final signal is empty; check dataset configuration or training output")

    df = final_signal.reset_index()
    df.columns = ["date", "code", "score"]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["code"] = df["code"].astype(str).str.split(".").str[0]

    weights = (
        df.groupby("date", group_keys=False)["score"].apply(_normalize_scores)
    )
    df = df.assign(weight=weights).drop(columns="score")

    df = df.sort_values(["date", "weight"], ascending=[True, False]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final model weights to CSV")
    parser.add_argument("--provider-uri", default=PROVIDER_URI, help="qlib data provider URI")
    parser.add_argument("--region", default=REGION, help="qlib region code")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV file (default: backtest_output/scores.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_scores(args.provider_uri, args.region, args.output)
    print(f"Exported scores to {output_path}")


if __name__ == "__main__":
    main()
