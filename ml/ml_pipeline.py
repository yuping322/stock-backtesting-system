"""ML factor modeling pipeline (moved from scripts/ml_pipeline.py).
For usage details, see top-level docstring in original version or run:
    python -m ml.ml_pipeline --help
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # type: ignore
    import shap  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore
    shap = None  # type: ignore

try:
    from scipy.cluster.hierarchy import linkage, fcluster
except Exception:  # pragma: no cover
    linkage = None  # type: ignore
    fcluster = None  # type: ignore

from sklearn.model_selection import train_test_split

@dataclass
class WalkForwardConfig:
    train_window: int = 120
    test_window: int = 20
    min_train: int = 60

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_factor_data(path: str, date_col: str = "date", code_col: str = "code") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Factor file not found: {path}")
    df = pd.read_csv(path)
    if date_col not in df.columns or code_col not in df.columns:
        raise ValueError("Factor file must contain date and code columns")
    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    return df.sort_values(date_col).reset_index(drop=True)


def generate_synthetic_label(df: pd.DataFrame, factor_cols: List[str]) -> pd.Series:
    rnd = np.random.default_rng(42)
    base = df[factor_cols].mean(axis=1)
    noise = rnd.normal(0, 0.5, size=len(df))
    return (base * 0.3 + noise).astype(float)


def get_factor_columns(df: pd.DataFrame, label_col: str, date_col: str, code_col: str) -> List[str]:
    exclude = {label_col, date_col, code_col}
    return [c for c in df.columns if c not in exclude]


def build_walk_forward_windows(unique_dates: List[pd.Timestamp], cfg: WalkForwardConfig) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    windows: List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]] = []
    i = 0
    while True:
        train_slice = unique_dates[i : i + cfg.train_window]
        test_slice = unique_dates[i + cfg.train_window : i + cfg.train_window + cfg.test_window]
        if len(train_slice) < cfg.min_train or len(test_slice) == 0:
            break
        windows.append((train_slice, test_slice))
        i += cfg.test_window
    return windows


def train_xgb_regressor(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray) -> object:
    if xgb is None:
        raise RuntimeError("xgboost not available; install package first")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.3,
        objective="reg:squarederror",
        n_jobs=4,
        verbosity=0,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False,
    )
    return model


def compute_shap_importance(model, X_sample: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    if shap is None:
        raise RuntimeError("shap not available; install package first")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    abs_mean = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"factor": feature_names, "mean_abs_shap": abs_mean})
    df["rank"] = df["mean_abs_shap"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("rank")


def build_factor_groups(df: pd.DataFrame, shap_df: pd.DataFrame, factor_cols: List[str], corr_threshold: float, max_groups: int) -> Dict[str, int]:
    if linkage is None or fcluster is None:
        raise RuntimeError("scipy not available for clustering")
    top_limit = min(max_groups * 4, len(shap_df))
    top_factors = shap_df.sort_values("rank").head(top_limit)["factor"].tolist()
    use_cols = [c for c in top_factors if c in factor_cols]
    if len(use_cols) <= max_groups:
        return {f: i for i, f in enumerate(use_cols)}
    corr = df[use_cols].corr(method="spearman").fillna(0)
    dist = 1 - corr
    condensed = dist.values[np.triu_indices(len(use_cols), k=1)]
    Z = linkage(condensed, method="average")
    cutoff = 1 - corr_threshold
    labels = fcluster(Z, t=cutoff, criterion="distance")
    mapping = {f: int(lbl) for f, lbl in zip(use_cols, labels)}
    if len(set(mapping.values())) > max_groups:
        cluster_imp: Dict[int, float] = {}
        shap_map = shap_df.set_index("factor")["mean_abs_shap"].to_dict()
        for f, g in mapping.items():
            cluster_imp[g] = cluster_imp.get(g, 0.0) + shap_map.get(f, 0.0)
        clusters_sorted = sorted(cluster_imp.items(), key=lambda x: x[1], reverse=True)
        keep = [c for c, _ in clusters_sorted[:max_groups]]
        remap = {old: i for i, old in enumerate(keep)}
        kept_factors = [f for f, g in mapping.items() if g in keep]
        kept_matrix = df[kept_factors].corr(method="spearman").fillna(0)
        final_mapping: Dict[str, int] = {}
        for f, g in mapping.items():
            if g in keep:
                final_mapping[f] = remap[g]
            else:
                corrs = kept_matrix[f].dropna()
                if corrs.empty:
                    final_mapping[f] = 0
                else:
                    best = corrs.abs().idxmax()
                    final_mapping[f] = final_mapping.get(best, 0)
        mapping = final_mapping
    return mapping


def train_group_models(df: pd.DataFrame, mapping: Dict[str, int], label_col: str, date_col: str, code_col: str, cfg: WalkForwardConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df[date_col].unique())
    windows = build_walk_forward_windows(unique_dates, cfg)
    blended_rows = []
    component_rows = []
    group_factors: Dict[int, List[str]] = {}
    for f, g in mapping.items():
        group_factors.setdefault(g, []).append(f)
    group_ic_history: Dict[int, List[float]] = {g: [] for g in group_factors}
    for train_dates, test_dates in windows:
        train_df = df[df[date_col].isin(train_dates)].copy()
        test_df = df[df[date_col].isin(test_dates)].copy()
        group_scores_test: Dict[int, pd.Series] = {}
        for g, f_list in group_factors.items():
            X_train = train_df[f_list].values
            y_train = train_df[label_col].values
            X_test = test_df[f_list].values
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            model = train_xgb_regressor(X_tr, y_tr, X_val, y_val)
            test_pred = model.predict(X_test)
            try:
                ic = pd.Series(test_pred).corr(pd.Series(y_train[: len(test_pred)]), method="spearman")
            except Exception:
                ic = 0.0
            group_ic_history[g].append(ic if not np.isnan(ic) else 0.0)
            group_scores_test[g] = pd.Series(test_pred, index=test_df.index)
        recent_ic = {g: np.mean(group_ic_history[g][-3:]) for g in group_factors}
        ic_plus = {g: max(v, 0) for g, v in recent_ic.items()}
        total = sum(ic_plus.values())
        if total <= 0:
            weights = {g: 1.0 / len(group_factors) for g in group_factors}
        else:
            weights = {g: ic_plus[g] / total for g in group_factors}
        test_df = test_df.copy()
        blended_score = np.zeros(len(test_df))
        for g in group_factors:
            g_scores = group_scores_test[g]
            component_rows.extend([
                {
                    "date": test_df.loc[idx, date_col],
                    "code": test_df.loc[idx, code_col],
                    "group_id": g,
                    "group_score": g_scores.loc[idx],
                    "group_weight": weights[g],
                }
                for idx in g_scores.index
            ])
            blended_score += weights[g] * g_scores.values
        test_df["blended_score"] = blended_score
        blended_rows.extend(test_df[[date_col, code_col, "blended_score"]].to_dict("records"))
    blended_df = pd.DataFrame(blended_rows)
    components_df = pd.DataFrame(component_rows)
    return blended_df, components_df


def build_prediction_weights(score_df: pd.DataFrame, date_col: str, code_col: str, score_col: str, top_n: int) -> pd.DataFrame:
    out_rows = []
    for d, sub in score_df.groupby(date_col):
        ranked = sub.sort_values(score_col, ascending=False).head(top_n)
        if ranked.empty:
            continue
        w_each = 1.0 / len(ranked)
        for _, row in ranked.iterrows():
            out_rows.append({"date": d.strftime("%Y-%m-%d"), "code": row[code_col], "weight": w_each, score_col: row[score_col]})
    return pd.DataFrame(out_rows)


def run_baseline(df: pd.DataFrame, args, cfg: WalkForwardConfig, factor_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    date_col, code_col, label_col = args.date_column, args.code_column, args.label_column
    unique_dates = sorted(df[date_col].unique())
    windows = build_walk_forward_windows(unique_dates, cfg)
    rows = []
    shap_importances: List[pd.DataFrame] = []
    for train_dates, test_dates in windows:
        train_df = df[df[date_col].isin(train_dates)].copy()
        test_df = df[df[date_col].isin(test_dates)].copy()
        X = train_df[factor_cols].values
        y = train_df[label_col].values
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_xgb_regressor(X_tr, y_tr, X_val, y_val)
        test_pred = model.predict(test_df[factor_cols].values)
        sample_idx = np.random.choice(len(X_val), size=min(200, len(X_val)), replace=False)
        shap_df = compute_shap_importance(model, X_val[sample_idx], factor_cols)
        shap_importances.append(shap_df)
        rows.extend([
            {
                "date": test_df.loc[i, date_col],
                "code": test_df.loc[i, code_col],
                "pred_score": test_pred[j],
            }
            for j, i in enumerate(test_df.index)
        ])
    pred_df = pd.DataFrame(rows)
    shap_full = pd.concat(shap_importances)
    shap_agg = shap_full.groupby("factor")["mean_abs_shap"].mean().reset_index()
    shap_agg["rank"] = shap_agg["mean_abs_shap"].rank(ascending=False, method="dense").astype(int)
    shap_agg = shap_agg.sort_values("rank")
    return pred_df, shap_agg


def main():
    parser = argparse.ArgumentParser(description="ML pipeline for factor modeling (package version)")
    parser.add_argument("--factor-file", required=True)
    parser.add_argument("--output-dir", default="data/ml")
    parser.add_argument("--mode", choices=["baseline", "groups", "submodels", "all"], default="baseline")
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--code-column", default="code")
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--test-window", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--corr-threshold", type=float, default=0.6)
    parser.add_argument("--max-groups", type=int, default=5)
    parser.add_argument("--min-shap", type=float, default=0.0)
    parser.add_argument("--synthetic-label", action="store_true", help="Generate synthetic label if absent")
    args = parser.parse_args()
    ensure_dir(args.output_dir)
    df = load_factor_data(args.factor_file, args.date_column, args.code_column)
    if args.label_column not in df.columns:
        if not args.synthetic_label:
            raise ValueError(f"Label column '{args.label_column}' missing. Use --synthetic-label to generate one.")
        factor_cols_tmp = get_factor_columns(df, args.label_column, args.date_column, args.code_column)
        df[args.label_column] = generate_synthetic_label(df, factor_cols_tmp)
    factor_cols = get_factor_columns(df, args.label_column, args.date_column, args.code_column)
    if not factor_cols:
        raise ValueError("No factor columns found for modeling")
    cfg = WalkForwardConfig(train_window=args.train_window, test_window=args.test_window)
    shap_path = os.path.join(args.output_dir, "shap_importance.csv")
    group_map_path = os.path.join(args.output_dir, "factor_groups.json")
    if args.mode in ("baseline", "all"):
        pred_df, shap_agg = run_baseline(df, args, cfg, factor_cols)
        weights_df = build_prediction_weights(pred_df, args.date_column, args.code_column, "pred_score", args.top_n)
        baseline_pred_file = os.path.join(args.output_dir, "baseline_predictions.csv")
        weights_df.to_csv(baseline_pred_file, index=False)
        pred_df.to_csv(os.path.join(args.output_dir, "baseline_full_scores.csv"), index=False)
        shap_agg.to_csv(shap_path, index=False)
        print(f"Baseline predictions written: {baseline_pred_file}")
        print(f"SHAP importance written: {shap_path}")
    if args.mode in ("groups", "all"):
        if not os.path.exists(shap_path):
            raise RuntimeError("SHAP importance file missing; run baseline first or use --mode all")
        shap_agg = pd.read_csv(shap_path)
        shap_agg = shap_agg[shap_agg["mean_abs_shap"] >= args.min_shap].copy()
        mapping = build_factor_groups(df, shap_agg, factor_cols, args.corr_threshold, args.max_groups)
        with open(group_map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Factor groups written: {group_map_path}")
    if args.mode in ("submodels", "all"):
        if not os.path.exists(group_map_path):
            raise RuntimeError("Factor group mapping missing; run groups step first or use --mode all")
        with open(group_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        blended_df, components_df = train_group_models(df, mapping, args.label_column, args.date_column, args.code_column, cfg)
        final_weights = build_prediction_weights(blended_df, args.date_column, args.code_column, "blended_score", args.top_n)
        final_pred_file = os.path.join(args.output_dir, "group_model_predictions.csv")
        final_weights.to_csv(final_pred_file, index=False)
        blended_df.to_csv(os.path.join(args.output_dir, "group_model_full_scores.csv"), index=False)
        components_df.to_csv(os.path.join(args.output_dir, "group_component_scores.csv"), index=False)
        print(f"Group model predictions written: {final_pred_file}")
    print("Pipeline complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
