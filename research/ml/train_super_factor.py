#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练超级因子模型
运行：python ml/train_super_factor.py
"""
import pandas as pd
import numpy as np
from scipy.stats import winsorize
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import data

# ---------- 参数区 ----------
MODEL_DIR = "ml/models"
ROLL_WIN = 50
LGB_ETA = 0.05
LGB_ITER = 300
LGB_MD = 3

# 使用已验证的有效因子
VALID_FACTORS = [
    "sales_growth", "operating_revenue_growth_rate", "total_profit_growth_rate",
    "np_parent_company_owners_growth_rate", "total_asset_growth_rate",
    "operating_profit_growth_rate", "growth", "SGI",
    "operating_cost_ttm", "total_operating_revenue_ttm", "gross_profit_ttm",
    "total_profit_ttm", "net_profit_ttm", "EBITDA", "EBIT", "eps_ttm",
    "operating_profit_per_share", "retained_earnings", "retained_earnings_per_share",
    "non_recurring_gain_loss", "OperateNetIncome", "np_parent_company_owners_ttm",
    "total_operating_cost_ttm", "administration_expense_ttm",
    "cashflow_per_share_ttm", "cash_flow_to_price_ratio",
    "net_operate_cash_flow_per_share", "cash_and_equivalents_per_share",
    "goods_sale_and_service_render_cash_ttm",
    "market_cap", "circulating_market_cap", "size", "natural_log_of_market_cap",
    "raw_beta", "beta", "boll_down", "MAC20", "MAC10", "EMAC10", "EMAC12",
]

NEUTRAL_STYLE = ["market_cap", "beta"]

def read_data(start_date: str, end_date: str, stock_pool: str = "small") -> pd.DataFrame:
    """从数据模块读取因子数据"""
    print(f"正在加载数据: {start_date} ~ {end_date}")
    
    if stock_pool == "small":
        stocks = data.get_index_stocks("000510")
    else:
        stocks = data.get_index_stocks(stock_pool)
    
    print(f"股票池: {len(stocks)} 只股票")
    
    df = data.read_factor_data(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        factors=VALID_FACTORS,
        base_path="uploads"
    )
    
    if df.empty:
        raise ValueError("未读取到因子数据")
    
    print(f"数据形状: {df.shape}")
    
    # 读取价格并计算收益率
    print("正在计算收益率...")
    prices = data.load_oss_stocks(stocks, start=start_date, end=end_date)
    
    if prices.empty:
        raise ValueError("未读取到价格数据")
    
    rets = prices.pct_change(10).shift(-10)
    rets_long = rets.stack().reset_index()
    rets_long.columns = ['date', 'code', 'next_ret']
    rets_long['code'] = rets_long['code'].astype(str).str.replace('.XSHG', '').str.replace('.XSHE', '')
    rets_long = rets_long.set_index(['date', 'code'])
    
    df = df.join(rets_long, how='left')
    
    if 'market_cap' in df.columns:
        df['market_cap'] = df['market_cap'].fillna(df['market_cap'].median())
    if 'beta' not in df.columns:
        df['beta'] = 1.0
    
    return df

def clean_factor(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """单日横截面清洗"""
    def _winsor(x): 
        try:
            return winsorize(x, limits=[0.05, 0.05])
        except:
            return x
    
    date = s.name[0]
    sub = df.loc[date]
    
    s_winsor = _winsor(s)
    
    if 'market_cap' in sub.columns:
        X = sub[['market_cap']].fillna(0)
        y = s_winsor.reindex(X.index)
        valid_mask = y.notna() & X['market_cap'].notna()
        if valid_mask.sum() > 10:
            beta = np.linalg.lstsq(X[valid_mask], y[valid_mask], rcond=None)[0]
            res = y - X @ beta
        else:
            res = y
    else:
        res = s_winsor
    
    res_std = res.std()
    if res_std > 0:
        return (res - res.mean()) / res_std
    else:
        return res - res.mean()

def roll_ic_weight(factors: pd.DataFrame, ret: pd.Series, window: int = 50) -> pd.Series:
    """滚动IC加权合成"""
    ic_by_date = {}
    dates = sorted(factors.index.get_level_values(0).unique())
    
    for date in dates:
        date_idx = factors.index.get_level_values(0) == date
        day_factors = factors.loc[date_idx]
        day_ret = ret.loc[date_idx]
        
        if len(day_factors) > 10 and day_ret.notna().sum() > 10:
            valid_mask = day_ret.notna()
            corrs = day_factors.loc[valid_mask].corrwith(day_ret.loc[valid_mask])
            ic_by_date[date] = corrs
    
    ic_df = pd.DataFrame(ic_by_date).T
    
    ic_rolling = ic_df.rolling(window=window, min_periods=10).mean()
    ic_abs_sum = ic_rolling.abs().sum(axis=1)
    weight_by_date = ic_rolling.div(ic_abs_sum, axis=0).fillna(0)
    
    super_factor = pd.Series(index=factors.index, dtype=float)
    
    for date in dates:
        date_idx = factors.index.get_level_values(0) == date
        day_factors = factors.loc[date_idx]
        
        if date in weight_by_date.index:
            day_weights = weight_by_date.loc[date]
            super_factor.loc[date_idx] = (day_factors * day_weights).sum(axis=1)
        else:
            super_factor.loc[date_idx] = day_factors.mean(axis=1)
    
    return super_factor, weight_by_date  # 返回权重用于保存

def neutralize(factor: pd.Series, df: pd.DataFrame, style_cols: list) -> pd.Series:
    """风格中性化"""
    res = pd.Series(index=factor.index, dtype=float)
    
    for date in df.index.get_level_values(0).unique():
        date_idx = df.index.get_level_values(0) == date
        y = factor.loc[date_idx]
        sub = df.loc[date_idx]
        
        X = sub[style_cols].fillna(0)
        valid_idx = y.index.intersection(X.index)
        
        if len(valid_idx) > 10:
            y_sub = y.loc[valid_idx]
            X_sub = X.loc[valid_idx]
            
            try:
                beta = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0]
                res.loc[valid_idx] = y_sub - X_sub @ beta
            except:
                res.loc[valid_idx] = y_sub
        else:
            res.loc[date_idx] = y
    
    return res

def train_lgb_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    """训练LightGBM模型"""
    print("  Fold 1/1 (全量训练)")
    
    model = lgb.LGBMRegressor(
        n_estimators=LGB_ITER,
        learning_rate=LGB_ETA,
        max_depth=LGB_MD,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

def main():
    # 训练参数
    start_date = "2024-01-01"  # 使用更长历史训练
    end_date = "2024-10-24"
    stock_pool = "small"
    
    print("=" * 60)
    print("训练超级因子模型")
    print("=" * 60)
    print(f"使用因子数: {len(VALID_FACTORS)}")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"股票池: {stock_pool}")
    print("=" * 60)
    
    # 1. 读取数据
    df = read_data(start_date, end_date, stock_pool)
    
    factor_cols = [f for f in VALID_FACTORS if f in df.columns]
    print(f"\n实际加载到 {len(factor_cols)} 个因子")
    
    factors = df[factor_cols]
    ret = df["next_ret"]
    
    # 2. 清洗因子
    print("\nStep-1 清洗因子 ...")
    factors_clean = factors.groupby(level=0).apply(
        lambda sub: sub.apply(lambda s: clean_factor(s, df))
    ).fillna(0)
    
    # 3. 滚动IC加权
    print("Step-2 滚动IC加权 ...")
    super_linear, ic_weights = roll_ic_weight(factors_clean, ret, ROLL_WIN)
    
    # 4. 风格中性化
    print("Step-3 风格中性化 ...")
    super_linear = neutralize(super_linear, df, NEUTRAL_STYLE)
    
    # 5. 训练LightGBM
    print("Step-4 训练LightGBM模型 ...")
    X = super_linear.to_frame("linear")
    y = ret.dropna()
    valid_idx = X.index.intersection(y.index)
    
    model = train_lgb_model(X.loc[valid_idx], y.loc[valid_idx])
    
    # 6. 保存模型和参数
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 保存LightGBM模型
    model_path = os.path.join(MODEL_DIR, "super_factor_model.txt")
    model.booster_.save_model(model_path)
    print(f"  ✓ 模型已保存: {model_path}")
    
    # 保存IC权重
    ic_weights_path = os.path.join(MODEL_DIR, "ic_weights.parquet")
    ic_weights.to_parquet(ic_weights_path)
    print(f"  ✓ IC权重已保存: {ic_weights_path}")
    
    # 保存因子列表
    factors_info = {
        "factor_list": factor_cols,
        "roll_window": ROLL_WIN,
        "lgb_params": {
            "eta": LGB_ETA,
            "iter": LGB_ITER,
            "max_depth": LGB_MD
        },
        "neutral_style": NEUTRAL_STYLE,
        "train_date_range": f"{start_date} ~ {end_date}",
        "stock_pool": stock_pool
    }
    
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(factors_info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 模型信息已保存: {info_path}")
    
    print("\n✅ 训练完成！")
    print(f"模型保存在: {MODEL_DIR}")

if __name__ == "__main__":
    main()

