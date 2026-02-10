"""Build daily prediction CSVs for single-factor and composite strategies.

Usage examples:

# Single-factor only (top 10 per factor, equal weight)
python scripts/build_predictions.py \
  --factor-file data/factor_values_sample.csv \
  --factors operating_cost_ttm sales_growth net_operate_cash_flow_per_share \
  --output-dir data/predictions \
  --mode single

# Composite v0 (top 10 by weighted composite score)
python scripts/build_predictions.py \
  --factor-file data/factor_values_sample.csv \
  --factors operating_cost_ttm sales_growth net_operate_cash_flow_per_share retained_earnings MAC20 \
  --output-dir data/predictions \
  --mode composite --weights-source pass_count

Input factor file format (wide):
  date, code, <factor1>, <factor2>, ...
  - date: YYYY-MM-DD
  - code: stock code (6-digit or will be left-padded)

Minimal output format for BacktestEngine DataLoader:
  date, code, weight
Optional extended metadata (written to sidecar file in same folder):
  source, composite_score, factor_list, direction

Notes:
- This script does NOT perform industry neutralization or winsorization. Provide preprocessed factor values if needed.
- Composite v0 weights: w_i ∝ base_signal_i * reliability_i (both placeholder in this prototype).
- base_signal_i currently derived from pass-count or simple ranking if a metadata file is provided later.
- Reliability placeholder: assumed 0.7 for all factors (can be replaced by real-time metric).

Future extensions (planned):
- Integrate dynamic reliability & EWMA IC.
- Add reverse factor channel.
- Add cluster de-duplication before composite.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_factor_matrix(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Factor file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("Factor file must contain columns: date, code")
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    # pick the latest date's cross-section
    latest_date = df["date"].max()
    return df.loc[df["date"] == latest_date].copy(), latest_date


def build_single_factor_top10(cs: pd.DataFrame, factor: str, top_n: int = 10,
                               direction: int = 1) -> pd.DataFrame:
    if factor not in cs.columns:
        raise ValueError(f"Factor column missing: {factor}")
    series = cs.set_index("code")[factor]
    ranked = series * direction  # direction = 1 for positive, -1 for reverse factor
    top_codes = ranked.nlargest(top_n).index.tolist()
    weight_each = 1.0 / len(top_codes) if top_codes else 0.0
    rows = []
    for c in top_codes:
        rows.append({"code": c, "weight": weight_each})
    out = pd.DataFrame(rows)
    return out


def compute_composite(cs: pd.DataFrame, factors: List[str],
                      weight_mode: str = "pass_count",
                      pass_counts: Dict[str, int] | None = None) -> Tuple[pd.Series, Dict[str, float]]:
    # Extract factor submatrix
    sub = cs.set_index("code")[factors].copy()
    # z-score each factor (avoid div by zero)
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    # Placeholder reliability: fixed 0.7 (could load from meta later)
    reliability = {f: 0.7 for f in factors}

    # Base signal depending on mode
    weights_raw = {}
    for f in factors:
        if weight_mode == "pass_count" and pass_counts is not None:
            base = max(pass_counts.get(f, 0) / 9.0, 0)
        else:
            # fallback: use cross-sectional IC proxy = abs(mean(z)) (weak proxy)
            base = abs(z[f].mean())
        weights_raw[f] = base * reliability[f]

    total = sum(weights_raw.values())
    if total <= 0:
        # fallback to equal weight
        equal = 1.0 / len(factors) if factors else 0.0
        weights = {f: equal for f in factors}
    else:
        weights = {f: w / total for f, w in weights_raw.items()}

    composite_score = pd.Series(0.0, index=sub.index)
    for f in factors:
        composite_score += weights[f] * z[f]
    return composite_score, weights


def build_composite_top10(cs: pd.DataFrame, factors: List[str], top_n: int = 10,
                           weight_mode: str = "pass_count",
                           pass_counts: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    comp_score, weights = compute_composite(cs, factors, weight_mode, pass_counts)
    top_codes = comp_score.nlargest(top_n).index.tolist()
    weight_each = 1.0 / len(top_codes) if top_codes else 0.0
    rows = []
    for c in top_codes:
        rows.append({"code": c, "weight": weight_each, "composite_score": comp_score.loc[c]})
    df = pd.DataFrame(rows)
    return df, weights


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate prediction CSVs from factor matrix")
    parser.add_argument("--factor-file", required=True, help="CSV with date, code, factor columns")
    parser.add_argument("--factors", nargs="+", required=True, help="List of factor column names to use")
    parser.add_argument("--output-dir", default="data/predictions", help="Output directory for prediction CSVs")
    parser.add_argument("--mode", choices=["single", "composite", "both"], default="both",
                        help="Generation mode")
    parser.add_argument("--top-n", type=int, default=10, help="Number of stocks to select")
    parser.add_argument("--weight-mode", default="pass_count", choices=["pass_count", "proxy"],
                        help="Composite weighting basis")
    parser.add_argument("--pass-counts", help="Optional CSV: factor,pass_count for weighting")
    parser.add_argument("--reverse-factors", nargs="*", default=[], help="Treat these factors as reverse (negative direction)")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    df = load_factor_matrix(args.factor_file)
    cs, latest_date = latest_snapshot(df)

    pass_counts_map: Dict[str, int] | None = None
    if args.pass_counts:
        pc_df = pd.read_csv(args.pass_counts)
        if "factor" not in pc_df.columns or "pass_count" not in pc_df.columns:
            raise ValueError("pass-counts file must have columns: factor, pass_count")
        pass_counts_map = dict(zip(pc_df.factor, pc_df.pass_count))

    outputs_created = []

    # SINGLE mode
    if args.mode in ("single", "both"):
        for f in args.factors:
            direction = -1 if f in args.reverse_factors else 1
            try:
                sf_df = build_single_factor_top10(cs, f, top_n=args.top_n, direction=direction)
            except Exception as e:
                print(f"Skip factor {f}: {e}")
                continue
            if sf_df.empty:
                print(f"No selection for factor {f}")
                continue
            sf_df.insert(0, "date", latest_date.strftime("%Y-%m-%d"))
            # Minimal prediction file
            out_path = os.path.join(args.output_dir, f"predictions_single_{f}.csv")
            sf_df.to_csv(out_path, index=False)
            # Extended sidecar
            meta_path = os.path.join(args.output_dir, f"predictions_single_{f}_meta.csv")
            meta = pd.DataFrame({
                "factor": [f],
                "date": [latest_date.strftime("%Y-%m-%d")],
                "direction": ["reverse" if direction == -1 else "positive"],
                "top_n": [len(sf_df)],
            })
            meta.to_csv(meta_path, index=False)
            outputs_created.append(out_path)

    # COMPOSITE mode
    if args.mode in ("composite", "both"):
        try:
            comp_df, weights = build_composite_top10(cs, args.factors, top_n=args.top_n,
                                                     weight_mode=args.weight_mode, pass_counts=pass_counts_map)
        except Exception as e:
            print(f"Composite generation failed: {e}")
            comp_df = pd.DataFrame()
            weights = {}
        if not comp_df.empty:
            comp_df.insert(0, "date", latest_date.strftime("%Y-%m-%d"))
            out_path = os.path.join(args.output_dir, "predictions_composite_v0.csv")
            comp_df.to_csv(out_path, index=False)
            weights_path = os.path.join(args.output_dir, "composite_v0_weights.csv")
            pd.DataFrame({"factor": list(weights.keys()), "weight": list(weights.values())}).to_csv(weights_path, index=False)
            outputs_created.append(out_path)

    if not outputs_created:
        print("No prediction files generated.")
    else:
        print("Generated prediction files:")
        for p in outputs_created:
            print(" -", p)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # argparse exits
        raise
    except Exception as ex:
        print(f"Error: {ex}")
        sys.exit(1)
