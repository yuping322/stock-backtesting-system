"""Factor update script: compute rolling IC stats and promote/demote factors.
Run after you have daily IC DataFrame for each factor.
"""
from pathlib import Path
import pickle
import pandas as pd

try:
    from .config_factors import core_factor_list, candidate_factor_list
    from .paths import IC_FILE
except ImportError:  # support running as script
    import sys

    _PKG_ROOT = Path(__file__).resolve().parent
    _REPO_ROOT = _PKG_ROOT.parent
    _REPO_PATH = str(_REPO_ROOT)
    if _REPO_PATH not in sys.path:
        sys.path.insert(0, _REPO_PATH)

    from factor_workflow.config_factors import core_factor_list, candidate_factor_list
    from factor_workflow.paths import IC_FILE

FACTOR_IC_FILE = IC_FILE  # expect DataFrame columns=factors, index=date
OUTPUT_PICKLE = Path(__file__).parent / "factor_lists_update.pkl"

# --- statistics functions ---

def compute_factor_stats(factor_ic: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for f in factor_ic.columns:
        # use trading day counts (approx) 63 ~ 3m, 252 ~ 12m
        ic_3m = factor_ic[f].iloc[-63:].mean()
        ic_12m = factor_ic[f].iloc[-252:].mean()
        ir_12m = ic_12m / (factor_ic[f].iloc[-252:].std() + 1e-8)
        stats[f] = {"ic_3m": ic_3m, "ic_12m": ic_12m, "ir_12m": ir_12m}
    return pd.DataFrame(stats).T


def decide_updates(stats_df: pd.DataFrame, ic_long_thr=0.02, ic_short_thr=0.03):
    promote = [
        f for f, row in stats_df.iterrows()
        if f in candidate_factor_list and row["ic_12m"] > ic_long_thr and row["ic_3m"] > ic_short_thr
    ]
    demote = [
        f for f, row in stats_df.iterrows()
        if f in core_factor_list and row["ic_3m"] < 0  # simple rule: recent IC deteriorated
    ]
    return promote, demote


def update_lists(promote, demote):
    new_core = list(set(core_factor_list + promote) - set(demote))
    new_candidate = list(set(candidate_factor_list) - set(promote))
    return new_core, new_candidate


def main():
    if not FACTOR_IC_FILE.exists():
        raise FileNotFoundError(f"Missing factor IC file: {FACTOR_IC_FILE}")
    factor_ic = pickle.loads(FACTOR_IC_FILE.read_bytes())
    if not isinstance(factor_ic, pd.DataFrame):
        raise TypeError("factor_ic_daily.pkl must store a pandas DataFrame")
    stats = compute_factor_stats(factor_ic)
    promote, demote = decide_updates(stats)
    new_core, new_candidate = update_lists(promote, demote)
    print("Promote:", promote)
    print("Demote:", demote)
    print("Core count ->", len(new_core), "Candidate count ->", len(new_candidate))
    pickle.dump({"core": new_core, "candidate": new_candidate}, OUTPUT_PICKLE.open("wb"))
    print("Updated factor lists saved to", OUTPUT_PICKLE)


if __name__ == "__main__":
    main()
