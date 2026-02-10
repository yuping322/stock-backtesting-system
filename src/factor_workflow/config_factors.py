"""Factor configuration tailored for the real sample_data dataset.

This module lists the financial statement based factors that already exist in
``formatted_data_all.csv`` and therefore in the generated static feature
panel.  Downstream utilities (``update_factors.py`` and documentation
snippets) import ``core_factor_list`` and the optional ``candidate`` list to
know which columns to analyse.

The historic implementation exposed qlib expression strings that were used to
materialise synthetic demo factors.  With the real dataset the feature values
are provided directly, so the "expression" layer is reduced to simple column
names.  We still expose ``core_feature_expr`` with the same two-element
structure ``[expressions, names]`` expected by some helper scripts; in this
context an "expression" is simply the column name itself.

Users can optionally drop a ``factor_lists_update.pkl`` beside this file
containing a pickle-serialised dict ``{"core": [...], "candidate": [...]}``.
Only factor names that exist in ``AVAILABLE_FACTORS`` will be retained to
ensure the static dataset contains all requested columns.
"""
from __future__ import annotations

from pathlib import Path
import pickle

# ---------------------------------------------------------------------------
# 🔧 示例：直接列出长期、短期因子
# ---------------------------------------------------------------------------
#
# - 你拥有的全部列只需要在以下两个列表中出现即可（可以有交集）。
# - 不再维护单独的 AVAILABLE_FACTORS，全量集合由 core + candidate 的并集决定。
# - 替换示例名称为你真实的数据列名即可。
# ---------------------------------------------------------------------------


def _validate(names: list[str] | None, universe: set[str]) -> list[str]:
    if not names:
        return []
    return [name for name in names if name in universe]


def _feature_config(names: list[str]) -> list[list[str]]:
    # For static datasets the "expression" is just the column name itself.
    return [names, names]


# ---------------------------------------------------------------------------
# 📌 示例：长期/短期子集
# ---------------------------------------------------------------------------
# - core_factor_list: 你筛选出的“长期稳定有效”因子（例：50 个）。
# - candidate_factor_list: 最近 3 个月等短期有效的因子（例：30 个）。
#   两者可以有交集，不必手工去重。
#
# 下面提供示例，便于你参考格式 —— 实际使用时直接替换即可。
# ---------------------------------------------------------------------------
core_factor_list = [
    "CUSTOM_MA_RATIO", "VOL10", "qlib_alpha158_BETA10", "qlib_alpha158_BETA20", "qlib_alpha158_BETA30",
    "qlib_alpha158_BETA5", "qlib_alpha158_BETA60", "qlib_alpha158_CNTD10", "qlib_alpha158_CNTD20", "qlib_alpha158_CNTD30",
    "qlib_alpha158_CNTD5", "qlib_alpha158_CNTD60", "qlib_alpha158_CNTN10", "qlib_alpha158_CNTN20", "qlib_alpha158_CNTN30",
    "qlib_alpha158_CNTN5", "qlib_alpha158_CNTN60", "qlib_alpha158_CNTP10", "qlib_alpha158_CNTP20", "qlib_alpha158_CNTP30",
    "qlib_alpha158_CNTP5", "qlib_alpha158_CNTP60", "qlib_alpha158_CORD10", "qlib_alpha158_CORD20", "qlib_alpha158_CORD30",
    "qlib_alpha158_CORD5", "qlib_alpha158_CORD60", "qlib_alpha158_CORR10", "qlib_alpha158_CORR20", "qlib_alpha158_CORR30",
    "qlib_alpha158_CORR5", "qlib_alpha158_CORR60", "qlib_alpha158_HIGH0", "qlib_alpha158_IMAX10", "qlib_alpha158_IMAX20",
    "qlib_alpha158_IMAX30", "qlib_alpha158_IMAX5", "qlib_alpha158_IMAX60", "qlib_alpha158_IMIN10", "qlib_alpha158_IMIN20",
    "qlib_alpha158_IMIN30", "qlib_alpha158_IMIN5", "qlib_alpha158_IMIN60", "qlib_alpha158_IMXD10", "qlib_alpha158_IMXD20",
    "qlib_alpha158_IMXD30", "qlib_alpha158_IMXD5", "qlib_alpha158_IMXD60", "qlib_alpha158_KLEN", "qlib_alpha158_KLOW",
    "qlib_alpha158_KLOW2", "qlib_alpha158_KMID", "qlib_alpha158_KMID2", "qlib_alpha158_KSFT", "qlib_alpha158_KSFT2",
    "qlib_alpha158_KUP", "qlib_alpha158_KUP2", "qlib_alpha158_LOW0", "qlib_alpha158_MA10", "qlib_alpha158_MA20",
    "qlib_alpha158_MA30", "qlib_alpha158_MA5", "qlib_alpha158_MA60", "qlib_alpha158_MAX10", "qlib_alpha158_MAX20",
    "qlib_alpha158_MAX30", "qlib_alpha158_MAX5", "qlib_alpha158_MAX60", "qlib_alpha158_MIN10", "qlib_alpha158_MIN20",
    "qlib_alpha158_MIN30", "qlib_alpha158_MIN5", "qlib_alpha158_MIN60", "qlib_alpha158_OPEN0", "qlib_alpha158_QTLD10",
    "qlib_alpha158_QTLD20", "qlib_alpha158_QTLD30", "qlib_alpha158_QTLD5", "qlib_alpha158_QTLD60", "qlib_alpha158_QTLU10",
    "qlib_alpha158_QTLU20", "qlib_alpha158_QTLU30", "qlib_alpha158_QTLU5", "qlib_alpha158_QTLU60", "qlib_alpha158_RANK10",
    "qlib_alpha158_RANK20", "qlib_alpha158_RANK30", "qlib_alpha158_RANK5", "qlib_alpha158_RANK60", "qlib_alpha158_RESI10",
    "qlib_alpha158_RESI20", "qlib_alpha158_RESI30", "qlib_alpha158_RESI5", "qlib_alpha158_RESI60", "qlib_alpha158_ROC10",
    "qlib_alpha158_ROC20", "qlib_alpha158_ROC30", "qlib_alpha158_ROC5", "qlib_alpha158_ROC60", "qlib_alpha158_RSQR10"
]

candidate_factor_list: list[str] = [
    "operating_cost_ttm","net_operate_cash_flow_per_share","total_operating_revenue_ttm","np_parent_company_owners_ttm","cash_and_equivalents_per_share","total_operating_revenue_per_share_ttm","capital_reserve_fund_per_share","interest_free_current_liability","retained_profit_per_share","administration_expense_ttm","total_profit_ttm","operating_profit_per_share","cash_flow_to_price_ratio","cashflow_per_share_ttm","market_cap","net_working_capital","gross_profit_ttm","np_parent_company_owners_growth_rate","goods_sale_and_service_render_cash_ttm","operating_revenue_per_share_ttm","net_asset_per_share","OperateNetIncome","retained_earnings","retained_earnings_per_share","total_operating_cost_ttm","financial_liability","asset_impairment_loss_ttm","net_profit_ttm","current_asset_turnover_rate","EBITDA","operating_revenue_per_share","non_recurring_gain_loss","total_asset_growth_rate","operating_revenue_ttm","circulating_market_cap","eps_ttm","operating_liability","EBIT","interest_carry_current_liability","LVGI","growth","natural_log_of_market_cap","size","price_no_fq","SGI"
]

# ---------------------------------------------------------------------------
# 📁 高级用法：通过外部 pickle 覆盖
# ---------------------------------------------------------------------------
# 如果你想让筛因脚本自动维护最新名单，可以在同一目录下生成
#   factor_lists_update.pkl
# 内容示例：
#   overrides = {
#       "core": ["因子A", "因子B", ...],
#       "candidate": ["因子X", "因子Y", ...],
#   }
#   with open("factor_lists_update.pkl", "wb") as f:
#       pickle.dump(overrides, f)
#
# 覆盖逻辑会先取当前 core + candidate 的并集作为“可用因子宇宙”，
# 然后只保留交集，避免误写不存在的列名。
# ---------------------------------------------------------------------------
UPDATE_PICKLE = Path(__file__).parent / "factor_lists_update.pkl"
if UPDATE_PICKLE.exists():
    try:
        data = pickle.loads(UPDATE_PICKLE.read_bytes())
        universe = set(core_factor_list) | set(candidate_factor_list)
        override_core = _validate(data.get("core"), universe)
        override_candidate = _validate(data.get("candidate"), universe)
        if override_core:
            core_factor_list = override_core
        if override_candidate:
            candidate_factor_list = override_candidate
    except Exception:
        # Fall back to defaults quietly; downstream scripts still work.
        pass

core_feature_expr = _feature_config(core_factor_list)
candidate_feature_expr = _feature_config(candidate_factor_list)

__all__ = [
    "core_factor_list",
    "candidate_factor_list",
    "core_feature_expr",
    "candidate_feature_expr",
]
